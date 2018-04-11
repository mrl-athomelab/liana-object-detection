#!/usr/bin/env python

"""An object detection demo"""

import numpy as np
import os
import sys
import tensorflow as tf
import rospkg
import cv2

from collections import defaultdict
from io import StringIO
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError(
        'Please upgrade your tensorflow installation to v1.4.* or later!')

from utils import label_map_util

from utils import visualization_utils as vis_util


def run_inference_for_cam_stream(device_num, graph):
    cap = cv2.VideoCapture(device_num)
    # Capture frame-by-frame
    ret, image_np = cap.read()
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates
                # and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                    real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                    real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            stop = False
            while not stop:
                # Capture frame-by-frame
                ret, image_np = cap.read()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image_np, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(
                    output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=8)

                # Display the resulting frame
                cv2.imshow('frame', image_np)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    stop = True
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


class ObjectDetection:
    def __init__(self, model):
        self.__initialized = False
        self.output_dict = {}
        # What model to use.
        self.MODEL_NAME = model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        rp = rospkg.RosPack()
        SCRIPT_PATH = os.path.join(rp.get_path(
            "liana_object_detection"), "src", "object_detection")
        MODEL_PATH = os.path.join(SCRIPT_PATH, 'models', 'ssd_mobilenet_v2_coco', self.MODEL_NAME)

        # Path to frozen detection graph. This is the actual model
        # that is used for the object detection.
        os.path.join(MODEL_PATH, 'frozen_inference_graph.pb')
        PATH_TO_CKPT = os.path.join(
            MODEL_PATH, 'frozen_inference_graph.pb')

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join(
            SCRIPT_PATH, 'data', 'mscoco_label_map.pbtxt')

        self.NUM_CLASSES = 90

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(
            self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def __initialize_graph(self, image):
        if self.__initialized:
            return
        else:
            self.__initialized = True
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {
                    output.name for op in ops for output in op.outputs}
                self.tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        self.tensor_dict[key] = tf.get_default_graph(
                        ).get_tensor_by_name(tensor_name)
                if 'detection_masks' in self.tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(
                        self.tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(
                        self.tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates
                    # to image coordinates and fit the image size.
                    real_num_detection = tf.cast(
                        self.tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [
                        real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                        real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    self.tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            self.sess = tf.Session(graph=self.detection_graph, config=self.config)

    def detect(self, image):
        if not self.__initialized:
            self.__initialize_graph(image)

        image_np = np.copy(image)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Run inference
        output_dict = self.sess.run(self.tensor_dict, feed_dict={
            self.image_tensor: np.expand_dims(image_np, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(
            np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]

        detection_classes = np.copy(output_dict['detection_classes'])
        names = [item.display_name for _class in output_dict['detection_classes']
                 for item in self.label_map.item if _class == item.id]

        output_dict['detection_classes'] = np.array(names)

        idx_to_remove = []
        self.output_dict = {}
        for i in range(len(output_dict['detection_scores'])):
            if output_dict['detection_scores'][i] < 0.5:
                idx_to_remove.append(i)
        for key, value in output_dict.iteritems():
            if type(value).__module__ == 'numpy':
                self.output_dict[key] = np.delete(value, idx_to_remove, 0)

        self.output_dict['num_detections'] = self.output_dict['detection_classes'].shape[0]
        print self.output_dict['detection_classes']

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            self.output_dict['detection_boxes'],
            detection_classes,
            self.output_dict['detection_scores'],
            self.category_index,
            instance_masks=self.output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        return image_np, self.output_dict


if __name__ == "__main__":
    # run_inference_for_cam_stream(0, detection_graph)

    cap = cv2.VideoCapture(0)
    model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
    object_detection = ObjectDetection(model_name)

    stop = False
    while not stop:
        # Capture frame-by-frame
        ret, image_np = cap.read()

        image, detection_dict = object_detection.detect(image_np)

        # print(detection_dict['detection_boxes'], '\ndetection_boxes\n')
        print(detection_dict['detection_classes'], '\ndetection_classes\n')
        # print(detection_dict['detection_scores'], '\ndetection_scores\n')

        # Display the resulting frame
        cv2.imshow('frame', image)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            stop = True

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()