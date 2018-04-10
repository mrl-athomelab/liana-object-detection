#!/usr/bin/env python

import roslib
roslib.load_manifest('liana_object_detection')
from liana_object_detection.srv import *
import rospy
from std_msgs.msg import String
import detection
from cv_bridge import CvBridge, CvBridgeError
import json
import os


os.environ["CUDA_VISIBLE_DEVICES"] = ""
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
#model_name = 'faster_rcnn_resnet50_coco_2018_01_28'
detector = detection.ObjectDetection(model_name)


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


def dict_to_json_str(data):
    json_data = dict()
    for key, value in data.iteritems():
        if isinstance(value, list):  # for lists
            value = [json.dumps(item) if isinstance(item, dict)
                     else item for item in value]
        if isinstance(value, dict):  # for nested lists
            value = json.dumps(value)
        if isinstance(key, int):  # if key is integer: > to string
            key = str(key)
        if type(value).__module__ == 'numpy':  # if value is numpy.*: > to python list
            value = value.tolist()
        json_data[key] = value
    return json.dumps(json_data, indent=4)


def handle_what_am_i_looking_at(req):
    input_image = imgmsg_to_cv2(req.input_image)
    output_image, objects_dict = detector.detect(input_image)
    output_image_message = cv2_to_imgmsg(output_image)
    objects_json_str = dict_to_json_str(objects_dict)
    #print type(objects_json_str)
    #print objects_json_str
    objects_message = String(data=objects_json_str)
    return WhatAmILookingAtResponse(output_image=output_image_message, objects_json=objects_message)


def what_am_i_looking_at_server():
    rospy.init_node('what_am_i_looking_at_server')
    rospy.Service('what_am_i_looking_at', WhatAmILookingAt,
                      handle_what_am_i_looking_at)
    print("Ready to tell you what you're looking at.")
    rospy.spin()


if __name__ == "__main__":
    what_am_i_looking_at_server()