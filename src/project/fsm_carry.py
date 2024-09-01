#!/usr/bin/env python3
import rospy
import smach
import smach_ros
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib import SimpleActionClient
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import time
import warnings
import mediapipe as mp
import speech_recognition as sr
from gtts import gTTS
import os
import threading  # Add this if using threading
import roslaunch
# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/user/yolov5/runs/train/exp10/weights/best.pt').to(device)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Helper functions for text-to-speech and speech recognition
def speak_ready_to_start():
    """Convert the specific text to speech and play it."""
    text = "Are you ready to start?"
    tts = gTTS(text=text, lang='en')
    tts.save("/tmp/response.mp3")
    os.system("mpg321 /tmp/response.mp3")
class WaitForYes(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['yes'])
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def execute(self, userdata):
        rospy.loginfo("Waiting for 'Yes' command...")
        response = self.recognize_speech_from_mic()
        if response and "yes" in response:
            rospy.loginfo("Received 'Yes' command.")
            return 'yes'
        return 'yes'  # Default outcome to allow retry or handle as needed

    def recognize_speech_from_mic(self):
        """Recognize speech using the microphone."""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
        try:
            response = self.recognizer.recognize_google(audio)
            return response.lower()
        except sr.RequestError:
            rospy.logerr("Speech recognition request error.")
            return None
        except sr.UnknownValueError:
            rospy.logerr("Speech recognition could not understand audio.")
            return None  
class DetectHand(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['left', 'right'], input_keys=['image_in'], output_keys=['hand_position'])
        self.bridge = CvBridge()
        self.hands = hands  # Use the global MediaPipe hands instance
        self.mp_drawing = mp_drawing  # Use the global MediaPipe drawing utils instance
        self.pub = rospy.Publisher('/camera/processed_image_mouse_left', Image, queue_size=10)  # Publisher for processed image

    def execute(self, userdata):
        rospy.loginfo("Detecting hand...")
        image = userdata.image_in
        hand_position = self.detect_hand(image)
        
        if hand_position:
            userdata.hand_position = hand_position
            return hand_position
        else:
            rospy.logwarn("Hand position not detected!")
            return 'left'  # Default fallback; consider if this makes sense for your FSM logic

    def detect_hand(self, image):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return None

        # Process the image with MediaPipe
        frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(cv_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Determine hand position (simplified example)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * cv_image.shape[1])
                if x < cv_image.shape[1] // 2:
                    hand_position = 'left'
                else:
                    hand_position = 'right'
                
                # Publish the processed image
                try:
                    ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
                    self.pub.publish(ros_image)
                except CvBridgeError as e:
                    rospy.logerr(f"CvBridge Error: {e}")

                return hand_position

        # Publish the processed image even if no hand is detected
        try:
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.pub.publish(ros_image)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

        return None
class MoveToChair(smach.State):
    def __init__(self, left=None):
        smach.State.__init__(self, outcomes=['moved_to_chair'])
        self.client = SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        self.left = left  # Store the direction if provided
    def execute(self, userdata):
        rospy.loginfo("Moving to chair...")
        if self.left is not None:
            if self.left:
                goal_position = (0.353, 3.11)  # Coordinates for left
            else:
                goal_position = (0.307, 2.13)  # Coordinates for right (or adjust if different)      
        # Move to the chair
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = goal_position[0]
        goal.target_pose.pose.position.y = goal_position[1]
        goal.target_pose.pose.orientation.w = 1.0    
        self.client.send_goal(goal)
        self.client.wait_for_result()       
        return 'moved_to_chair'
class MoveToBag(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['moved_to_bag'])
        self.client = SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()

    def execute(self, userdata):
        rospy.loginfo("Moving to bag...")
        # Implement logic to move to the bag
        bag_position = (0, 0)  # Replace with actual bag position
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = bag_position[0]
        goal.target_pose.pose.position.y = bag_position[1]
        goal.target_pose.pose.orientation.w = 1.0   
        self.client.send_goal(goal)
        self.client.wait_for_result()
        return 'moved_to_bag'
class DetectAndGrabBag(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['bag_grabbed'], input_keys=['image_in'])
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/user/yolov5/runs/train/exp10/weights/best.pt').to(self.device)
        self.pub = rospy.Publisher('Test', String, queue_size=10)

    def execute(self, userdata):
        rospy.loginfo("Detecting the bag...")
        image = userdata.image_in
        bag_detected = self.detect_bag(image)
        if bag_detected:
            # Publish message "B" to indicate the bag is detected and ready to grab
            self.pub.publish("B")
            return 'bag_grabbed'
        else:
            rospy.logwarn("Bag not detected!")
            return 'bag_grabbed'  # You can implement retry logic if needed
    def detect_bag(self, image):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return False
        # Perform object detection
        results = self.model(cv_image)
        for *xyxy, conf, cls in results.xyxy[0]:
            trust = conf.item() * 100
            if trust > 60 and self.model.names[int(cls)] == 'bag':
                rospy.loginfo(f"Bag detected with confidence {trust:.2f}%")
                return True
        return False
class TurnAndFollow(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['following_done'])
        self.following = False
        self.stop_event = threading.Event()
        self.pub = rospy.Publisher('/test', String, queue_size=10)

    def execute(self, userdata):
        rospy.loginfo("Turning and following...")
        self.stop_event.clear()
        self.following = True
        
        # Start the follower node via roslaunch
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/user/Documents/additional_ws/src/turtlebot_apps/turtlebot_follower/launch/follower.launch"])
        launch.start()
        rospy.loginfo("Follower node launched.")

        while self.following:
            self.speak("Are you there?")
            response = self.recognize_speech_from_mic()
            if response and "yes" in response:
                self.stop_event.set()
                rospy.loginfo("Received 'Yes'. Moving to next state.")
                launch.shutdown()  # Stop the follower node
                self.pub.publish("A")
                return 'following_done'

            # Wait for 10 seconds before asking again
            time.sleep(10)

        # Ensure the follower node is stopped if exiting
        launch.shutdown()
        return 'following_done'

    def recognize_speech_from_mic(self):
        """Recognize speech using the microphone."""
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            response = recognizer.recognize_google(audio)
            return response.lower()
        except sr.RequestError:
            rospy.logerr("Speech recognition API unavailable or unresponsive.")
            return None
        except sr.UnknownValueError:
            rospy.loginfo("Unable to recognize speech.")
            return None

    def speak(self, text):
        """Convert text to speech and play it."""
        tts = gTTS(text=text, lang='en')
        tts.save("/tmp/response.mp3")
        os.system("mpg321 /tmp/response.mp3")
        os.remove("/tmp/response.mp3")  # Clean up the file
class ReturnToBase(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['completed'])
        self.client = SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()

    def execute(self, userdata):
        rospy.loginfo("Listening for 'Go' command...")
        while True:
            response = self.recognize_speech_from_mic()
            if response and "go" in response:
                rospy.loginfo("Command 'Go' received. Moving to base.")
                self.speak("Moving to base.")
                self.move_to_base()
                return 'completed'
            rospy.loginfo("Command not recognized. Listening again...")

    def recognize_speech_from_mic(self):
        """Recognize speech using the microphone."""
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            response = recognizer.recognize_google(audio)
            return response.lower()
        except sr.RequestError:
            rospy.logerr("API request error.")
            return None
        except sr.UnknownValueError:
            rospy.logerr("Unable to recognize speech.")
            return None

    def speak(self, text):
        """Convert text to speech and play it."""
        tts = gTTS(text=text, lang='en')
        tts.save("/tmp/response.mp3")
        os.system("mpg321 /tmp/response.mp3")
        os.remove("/tmp/response.mp3")  # Clean up the file

    def move_to_base(self):
        """Send a move_base goal to return to the base."""
        rospy.loginfo("Sending goal to move base...")
        base_position = (0, 0)  # Replace with actual base position coordinates
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = base_position[0]
        goal.target_pose.pose.position.y = base_position[1]
        goal.target_pose.pose.orientation.w = 1.0  # Default orientation

        self.client.send_goal(goal)
        self.client.wait_for_result()

        if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Successfully returned to base.")
        else:
            rospy.logwarn("Failed to return to base.")

def main():
    rospy.init_node('carry_my_luggage_fsm')

    # Create the state machine
    sm = smach.StateMachine(outcomes=['completed'])
    sm.userdata.hand_position = None

    with sm:
        smach.StateMachine.add('WAIT_FOR_YES', WaitForYes(), transitions={'yes': 'DETECT_HAND'})
        smach.StateMachine.add('DETECT_HAND', DetectHand(), transitions={'left': 'MOVE_TO_CHAIR_LEFT', 'right': 'MOVE_TO_CHAIR_RIGHT'})
        smach.StateMachine.add('MOVE_TO_CHAIR_LEFT', MoveToChair(left=True), transitions={'moved_to_chair': 'MOVE_TO_BAG'})
        smach.StateMachine.add('MOVE_TO_CHAIR_RIGHT', MoveToChair(left=False), transitions={'moved_to_chair': 'MOVE_TO_BAG'})
        smach.StateMachine.add('MOVE_TO_BAG', MoveToBag(), transitions={'moved_to_bag': 'DETECT_AND_GRAB_BAG'})
        smach.StateMachine.add('DETECT_AND_GRAB_BAG', DetectAndGrabBag(), transitions={'bag_grabbed': 'TURN_AND_FOLLOW'})
        smach.StateMachine.add('TURN_AND_FOLLOW', TurnAndFollow(), transitions={'following_done': 'RETURN_TO_BASE'})
        smach.StateMachine.add('RETURN_TO_BASE', ReturnToBase(), transitions={'completed': 'completed'})

    # Start the execution of the state machine
    outcome = sm.execute()

if __name__ == '__main__':
    main()



