#!/usr/bin/env python

import roslib

roslib.load_manifest('liana_object_detection')
import sys
import rospy
from liana_object_detection.srv import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import json
import numpy as np
import time
# Ros Messages
from sensor_msgs.msg import CompressedImage


# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError


def imgmsg_to_cv2(imgmsg):
    bridge = CvBridge()
    try:
        cv2_image = bridge.imgmsg_to_cv2(
            imgmsg, desired_encoding="passthrough")
        return cv2_image
    except CvBridgeError as e:
        print(e)


def cv2_to_imgmsg(cv2_image):
    bridge = CvBridge()
    try:
        imgmsg = bridge.cv2_to_imgmsg(cv2_image, encoding="passthrough")
        return imgmsg
    except CvBridgeError as e:
        print(e)


def json_str_to_dict(json_str):
    data = json.loads(json_str)
    objects_dict = dict()
    objects_dict['num_detections'] = int(data['num_detections'])
    objects_dict['detection_boxes'] = np.array(data['detection_boxes'])
    objects_dict['detection_classes'] = np.array(data['detection_classes'])
    objects_dict['detection_scores'] = np.array(data['detection_scores'])
    return objects_dict


def what_am_i_looking_at_client(image):
    rospy.wait_for_service('what_am_i_looking_at')
    try:
        what_am_i_looking_at = rospy.ServiceProxy(
            'what_am_i_looking_at', WhatAmILookingAt)
        input_image_message = cv2_to_imgmsg(image)
        resp = what_am_i_looking_at(input_image=input_image_message)
        output_image_message = resp.output_image
        objects_json_str = resp.objects_json
        output_image = imgmsg_to_cv2(output_image_message)
        objects_dict = json_str_to_dict(objects_json_str.data)
        return output_image, objects_dict
    except rospy.ServiceException, e:
        print("Service call failed: {}".format(e))


class image_converter:

    def __init__(self, topic):
        self.image_pub = rospy.Publisher("/liana_object_detection/image_raw", Image)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback, queue_size=1, buff_size=2 ** 24)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        output_image, objects_dict = what_am_i_looking_at_client(cv_image)
        print objects_dict['detection_boxes'], '\ndetection_boxes\n'
        print objects_dict['detection_classes'], '\ndetection_classes\n'
        print objects_dict['detection_scores'], '\ndetection_scores\n'

        cv2.imshow("Image window", output_image)
        cv2.waitKey(3)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(output_image, "bgr8"))
        except CvBridgeError as e:
            print(e)


VERBOSE = True


class image_feature:

    def __init__(self, topic):
        """Initialize ros publisher, ros subscriber"""
        # topic where we publish
        self.image_pub = rospy.Publisher("/saam_object_detection/image_raw/compressed", CompressedImage)
        # self.bridge = CvBridge()

        # subscribed Topic
        self.subscriber = rospy.Subscriber(topic, CompressedImage, self.callback, queue_size=1, buff_size=2 ** 24)
        if VERBOSE:
            print "subscribed to {}".format(topic)
        self.image_np = None

    def callback(self, ros_data):
        '''Callback function of subscribed topic.
        Here images get converted and features detected'''
        if VERBOSE:
            print 'received image of type: "%s"' % ros_data.format

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        # self.image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        self.image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # OpenCV >= 3.0:

        output_image, objects_dict = what_am_i_looking_at_client(self.image_np)
        print objects_dict['detection_boxes'], '\ndetection_boxes\n'
        print objects_dict['detection_classes'], '\ndetection_classes\n'
        print objects_dict['detection_scores'], '\ndetection_scores\n'

        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', output_image)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)

        # self.subscriber.unregister()

        # Display the resulting frame
        # cv2.imshow('frame', output_image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):


if __name__ == "__main__":
    topic = '/camera/rgb/image_raw/compressed'
    ic = image_feature(topic)

    rospy.init_node('what_am_i_looking_at_client', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()