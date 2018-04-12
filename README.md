# Liana Object Detection

This object detection package has been developed and tested on ROS Kinetic.
It is based on the Tensorflow object detection API and runs a 
Single Shot Detector with MobileNets, pretrained on the COCO dataset.

Please refer to the Tensorflow object detection API installation guide 
and install the API before using this package.

usage:

rosrun liana_object_detection what_am_i_looking_at_server.py

rosrun liana_object_detection what_am_i_looking_at_client.py
