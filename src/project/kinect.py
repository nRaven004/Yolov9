#!/usr/bin/env python3
import rospy
import smach
from cv_bridge import CvBridge, CvBridgeError
import cv2
import mediapipe as mp

class DetectHand(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['left', 'right'], input_keys=['image_in'], output_keys=['hand_position'])
        self.bridge = CvBridge()
        self.hands = mp.solutions.hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    def execute(self, userdata):
        rospy.loginfo("Detecting hand...")
        if userdata.image_in is not None:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(userdata.image_in, "bgr8")
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")
                return 'left'

            hand_position = self.detect_hand(cv_image)
            if hand_position:
                userdata.hand_position = hand_position
                return hand_position
            else:
                rospy.logwarn("Hand position not detected!")
                return 'left'
        else:
            rospy.logwarn("No image available in 'image_in'.")
            return 'left'

    def detect_hand(self, image):
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * image.shape[1])
                return 'left' if x < image.shape[1] // 2 else 'right'
        return None

