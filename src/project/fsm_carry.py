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
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/user/catkin_ws/src/project/Best,py-object/best.pt').to(device)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
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
        speak_ready_to_start()
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load the YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/user/Downloads/additional_ws/src/yolov3/best.pt').to(self.device)
        # Publisher and Subscriber for messages
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
        launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/user/Downloads/additional_ws/src/turtlebot_apps/turtlebot_follower/launch/follower.launch"])
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

    outcome = sm.execute()


if __name__ == '__main__':
    main()



