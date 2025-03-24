#!/usr/bin/env python3
import rospy
import smach
import torch
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

class DetectAndGrabBag(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['bag_grabbed'], input_keys=['image_in'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/user/catkin_ws/src/project/Best,py-object/best.pt').to(self.device)
        self.pub = rospy.Publisher('Test', String, queue_size=10)
        self.sub = rospy.Subscriber('Test1', String, self.message_callback)
        self.g_received = False
        self.bridge = CvBridge()

    def execute(self, userdata):
        rospy.loginfo("Detecting the bag...")
        if userdata.image_in is not None:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(userdata.image_in, "bgr8")
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")
                return 'bag_grabbed'

            bag_detected = self.detect_bag(cv_image)
            if bag_detected:
                self.pub.publish("B")
                rospy.loginfo("Waiting for message 'G'...")
                rospy.sleep(1)
                while not self.g_received:
                    rospy.sleep(0.1)
                rospy.loginfo("Received 'G', transitioning to next state.")
                return 'bag_grabbed'
            else:
                rospy.logwarn("Bag not detected!")
                return 'bag_grabbed'
        else:
            rospy.logwarn("No image available in 'image_in'.")
            return 'bag_grabbed'

    def detect_bag(self, image):
        results = self.model(image)
        for *xyxy, conf, cls in results.xyxy[0]:
            trust = conf.item() * 100
            if trust > 60 and self.model.names[int(cls)] == 'bag':
                rospy.loginfo(f"Bag detected with confidence {trust:.2f}%")
                return True
        return False

    def message_callback(self, msg):
        if msg.data == "G":
            rospy.loginfo("Received message 'G'")
            self.g_received = True

