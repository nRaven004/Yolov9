#!/usr/bin/env python3
import cv2
import mediapipe as mp
from cv_bridge import CvBridge, CvBridgeError
import rospy
import smach
import torch
import numpy as np

class DetectHand(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['left', 'right'], input_keys=['image_in'], output_keys=['hand_position', 'object_info'])
        self.bridge = CvBridge()
        self.hands = mp.solutions.hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

        model_path = '/home/user/catkin_ws/src/project/Best,py-object/best.pt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(self.device)
        self.model.eval()

    def execute(self, userdata):
        rospy.loginfo("Detecting hand and object...")
        if userdata.image_in is not None:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(userdata.image_in, "bgr8")
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")
                return 'left'

            hand_position, object_info = self.detect_hand_and_objects(cv_image)
            userdata.hand_position = hand_position
            userdata.object_info = object_info

            rospy.loginfo(f"Hand Position: {hand_position}")
            rospy.loginfo(f"Object Info: {object_info}")

            if hand_position:
                return hand_position
            else:
                rospy.logwarn("Hand position not detected!")
                return 'left'
        else:
            rospy.logwarn("No image available in 'image_in'.")
            return 'left'

    def detect_hand_and_objects(self, image):
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        hand_position = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * image.shape[1])
                hand_position = 'left' if x < image.shape[1] // 2 else 'right'

        img_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)) / 255.0).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            results = self.model(img_tensor)

        objects = []
        for det in results.xyxy[0]:
            if det[4] > 0.5:  # Confidence threshold
                x1, y1, x2, y2, conf, cls = det
                objects.append({
                    'class': int(cls),
                    'confidence': conf.item(),
                    'bbox': [x1.item(), y1.item(), x2.item(), y2.item()]
                })

        return hand_position, objects

