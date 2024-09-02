#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class KinectCaptureNode:
    def __init__(self):
        rospy.init_node('kinect_capture_node')
        self.pub = rospy.Publisher('/kinect_camera/image_raw', Image, queue_size=10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)  # ใช้ index 0 หรือ URL ที่ต้องการ
        self.publish_images()

    def publish_images(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if ret:
                ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                self.pub.publish(ros_image)
            rospy.sleep(0.1)

if __name__ == '__main__':
    KinectCaptureNode()

