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
import time
import speech_recognition as sr
from gtts import gTTS
import os
import threading
import roslaunch
from geometry_msgs.msg import Twist
from detect_hand import DetectHand
from detect_and_grab_bag import DetectAndGrabBag


def speak_ready_to_start():
    text = "Are you ready to start?"
    tts = gTTS(text=text, lang='en')
    tts.save("/tmp/response.mp3")
    rospy.loginfo("Audio file created at /tmp/response.mp3")
    os.system("mpg321 /tmp/response.mp3")
    rospy.loginfo("Audio playback attempted.")
    os.remove("/tmp/response.mp3")

class WaitForYes(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['yes'])
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
    def execute(self, userdata):
        rospy.loginfo("Waiting for 'Yes' command...")
        speak_ready_to_start()  # Call the function to speak the message
        response = self.recognize_speech_from_mic()
        if response and "yes" in response:
            rospy.loginfo("Received 'Yes' command.")
            return 'yes'
        return 'yes'
    def recognize_speech_from_mic(self):
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

class MoveToChair(smach.State):
    def __init__(self, left=None):
        smach.State.__init__(self, outcomes=['moved_to_chair'])
        self.client = SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        self.left = left

    def execute(self, userdata):
        rospy.loginfo("Moving to chair...")
        try:
            if self.left is not None:
                goal_position = (0.353, 3.11) if self.left else (0.307, 2.13)
            else:
                raise ValueError("Invalid direction specified for MoveToChair.")
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = goal_position[0]
            goal.target_pose.pose.position.y = goal_position[1]
            goal.target_pose.pose.orientation.w = 1.0
            self.client.send_goal(goal)
            self.client.wait_for_result()
            if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
                return 'moved_to_chair'
            else:
                rospy.logwarn("Failed to move to chair.")
                return 'moved_to_chair'
        except Exception as e:
            rospy.logerr(f"An error occurred: {e}")
            return 'moved_to_chair'

class TurnAndFollow(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['following_done'])
        self.following = False
        self.stop_event = threading.Event()
        self.pub = rospy.Publisher('/test', String, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)  # Publisher for robot movement
        self.bridge = CvBridge()  # Initialize CvBridge
        self.latest_image = None
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        self.image_lock = threading.Lock()  # For thread-safe image access

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.image_lock:
                self.latest_image = cv_image
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def detect_full_body(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bodies = self.body_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return len(bodies) > 0

    def execute(self, userdata):
        rospy.loginfo("Turning and following...")
        self.stop_event.clear()
        self.following = True
        
        try:
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/user/Downloads/additional_ws/src/turtlebot_apps/turtlebot_follower/launch/follower.launch"])
            launch.start()
            rospy.loginfo("Follower node launched.")
            
            while self.following:
                if self.latest_image is not None:
                    with self.image_lock:
                        full_body_detected = self.detect_full_body(self.latest_image)
                        if full_body_detected:
                            rospy.loginfo("Full body detected. Starting following...")
                            self.pub.publish("A")
                            self.speak("Are you there?")
                            response = self.recognize_speech_from_mic()
                            if response and "yes" in response:
                                self.stop_event.set()
                                rospy.loginfo("Received 'Yes'. Moving to next state.")
                                launch.shutdown()
                                return 'following_done'
                        
                        # Adding robot movement commands
                        self.turn_robot()

                time.sleep(1)  # Adjust sleep time as necessary

            launch.shutdown()
            return 'following_done'
        except Exception as e:
            rospy.logerr(f"An error occurred: {e}")
            return 'following_done'  # Ensure it returns only 'following_done'

    def turn_robot(self):
        twist = Twist()
        twist.angular.z = 0.5  # Set the angular velocity for turning
        self.cmd_vel_pub.publish(twist)
        rospy.loginfo("Turning the robot...")
        time.sleep(1)  # Adjust the sleep time as necessary
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def speak(self, text):
        rospy.loginfo(f"Speaking: {text}")
        tts = gTTS(text=text, lang='en')
        audio_path = "/tmp/response.mp3"
        tts.save(audio_path)
        os.system(f"mpg321 {audio_path}")
        os.remove(audio_path)

    def recognize_speech_from_mic(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            response = recognizer.recognize_google(audio)
            return response.lower()
        except sr.RequestError:
            rospy.logerr("Speech recognition request error.")
            return None
        except sr.UnknownValueError:
            rospy.logerr("Speech recognition could not understand audio.")
            return None
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
        tts = gTTS(text=text, lang='en')
        tts.save("/tmp/response.mp3")
        os.system("mpg321 /tmp/response.mp3")
        os.remove("/tmp/response.mp3")

    def move_to_base(self):
        rospy.loginfo("Sending goal to move base...")
        base_position = (0, 0)                             #set coordinate here
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = base_position[0]
        goal.target_pose.pose.position.y = base_position[1]
        goal.target_pose.pose.orientation.w = 1.0

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
    
    with sm:
        smach.StateMachine.add('WAIT_FOR_YES', WaitForYes(), transitions={'yes': 'DETECT_HAND'})
        smach.StateMachine.add('DETECT_HAND', DetectHand(), 
                       transitions={'left': 'MOVE_TO_CHAIR_LEFT',
                                    'right': 'MOVE_TO_CHAIR_RIGHT'})
        smach.StateMachine.add('MOVE_TO_CHAIR_LEFT', MoveToChair(left=True), transitions={
            'moved_to_chair': 'DETECT_AND_GRAB_BAG'
        })
        smach.StateMachine.add('MOVE_TO_CHAIR_RIGHT', MoveToChair(left=False), transitions={
            'moved_to_chair': 'DETECT_AND_GRAB_BAG'
        })
        smach.StateMachine.add('DETECT_AND_GRAB_BAG', DetectAndGrabBag(), transitions={
            'bag_grabbed': 'TURN_AND_FOLLOW'
        })
        smach.StateMachine.add('TURN_AND_FOLLOW', TurnAndFollow(), transitions={
            'following_done': 'RETURN_TO_BASE'
        })
        smach.StateMachine.add('RETURN_TO_BASE', ReturnToBase(), transitions={'completed': 'completed'})
    
    # Execute the state machine
    outcome = sm.execute()

if __name__ == '__main__':
    main()

