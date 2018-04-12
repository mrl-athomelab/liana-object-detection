#!/usr/bin/env python
import rospy
import roslib
import sys
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

roslib.load_manifest('liana_object_detection')


class ImageSubscriber:

    def __init__(self, image_topic):
        """Initialize ros subscriber"""
        # subscribed Topic
        self.subscriber = rospy.Subscriber(image_topic, Image, self.callback, queue_size=1)
        rospy.loginfo('subscribed to {}'.format(topic))
        self.bridge = CvBridge()
        self.frame = None

    def callback(self, ros_data):
        """Callback function of subscribed topic."""
        try:
            # Convert your ROS Image message to OpenCV2
            self.frame = self.bridge.imgmsg_to_cv2(ros_data, "bgr8")
        except CvBridgeError, e:
            print(e)
        if self.frame is None:
            rospy.loginfo('Skipped a frame...')
            return
        cv2.imshow('frame', self.frame)
        if cv2.waitKey(32) & 0xFF == ord('q'):
            shutdown_msg = 'User shut down the node.'
            rospy.loginfo(shutdown_msg)
            rospy.signal_shutdown(shutdown_msg)


def main(image_topic):
    rospy.init_node('grab_samples')
    rospy.loginfo('Starting to grab samples...')
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    image_subscriber = ImageSubscriber(image_topic)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down sample grabber module..."
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = rospy.myargv(argv=sys.argv)
    topic = '/camera/rgb/image_raw'
    if len(args) > 1:
        topic = args[1]
    main(topic)
