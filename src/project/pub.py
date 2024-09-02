#!/usr/bin/env python3
import rospy
import smach
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import mediapipe as mp

class DetectHand():
    def __init__(self):
        smach.State.__init__(self, outcomes=['left', 'right'], input_keys=['image_in'], output_keys=['hand_position'])
        self.bridge = CvBridge()
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

    def execute(self, userdata):
        rospy.loginfo("Detecting hand...")
        image = userdata.image_in

        # Check if image is valid
        if image is None:
            rospy.logerr("Received None image!")
            return 'left'  # Default fallback

        hand_position = self.detect_hand(image)
        if hand_position:
            userdata.hand_position = hand_position
            return hand_position
        else:
            rospy.logwarn("Hand position not detected!")
            return 'left'  # Default fallback

    def detect_hand(self, image):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
            rospy.loginfo("Image converted to OpenCV format")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return None

        try:
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            rospy.loginfo("Image converted to RGB format")
        except cv2.error as e:
            rospy.logerr(f"OpenCV Error: {e}")
            return None
            
            

        # Process the image with MediaPipe
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(cv_image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Determine hand position (simplified example)
                index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * cv_image.shape[1])
                rospy.loginfo(f"Hand detected at x position: {x}")
                if x < cv_image.shape[1] // 2:
                    return 'left'
                else:
                    return 'right'

        return None



if __name__ == '__main__':
    DetectHand()
