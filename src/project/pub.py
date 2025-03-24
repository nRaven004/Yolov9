#!/usr/bin/env python3

import rospy
import torch
import cv2
import mediapipe as mp
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import smach
import smach_ros
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from gtts import gTTS
import os
import tf
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import subprocess

# Function for speech using gTTS
def speak(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save('/tmp/speech.mp3')
    os.system('mpg321 /tmp/speech.mp3')

# Load YOLOv5 model
model = torch.hub.load('/home/user/yolov5', 'custom', path='/home/user/Yolov9/src/yolov3/best.pt', source='local')
bridge = CvBridge()

# Function to detect a bag using YOLOv5
def detect_bag(frame):
    results = model(frame)
    labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    n = len(labels)
    for i in range(n):
        row = coords[i]
        if row[4] >= 0.5:  # Check confidence threshold
            x1, y1, x2, y2 = int(row[0] * frame.shape[1]), int(row[1] * frame.shape[0]), int(row[2] * frame.shape[1]), int(row[3] * frame.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{int(labels[i])}: {row[4]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return True, frame

    return False, frame

# Class for navigating the robot and setting the home position
class RobotNavigator:
    def __init__(self):
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.move_base.wait_for_server(rospy.Duration(5))
        self.tf_listener = tf.TransformListener()
        self.home_position = {
            'x': 1.09,
            'y': 2.68,
            'theta': 0.00247
        }
        self.last_position = None  # ตัวแปรสำหรับบันทึกตำแหน่งสุดท้าย
        rospy.loginfo(f"Home position manually set to: {self.home_position}")

    def save_last_position(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            self.last_position = {
                'x': trans[0],
                'y': trans[1],
                'theta': tf.transformations.euler_from_quaternion(rot)[2]
            }
            rospy.loginfo(f"Saved last position: {self.last_position}")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Failed to get current position.")

    def go_to_position(self, position):
        if not position:
            rospy.logwarn("Position is not set!")
            return

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = position['x']
        goal.target_pose.pose.position.y = position['y']
        goal.target_pose.pose.orientation.z = np.sin(position['theta'] / 2)
        goal.target_pose.pose.orientation.w = np.cos(position['theta'] / 2)

        self.move_base.send_goal(goal)
        wait = self.move_base.wait_for_result()
        if wait:
            rospy.loginfo("Reached the target position.")
        else:
            rospy.logwarn("Failed to reach the target position.")

    def return_to_last_position(self):
        rospy.loginfo("Returning to last known position.")
        self.go_to_position(self.last_position)

    def return_to_home(self):
        rospy.loginfo("Returning to home position.")
        self.go_to_position(self.home_position)

class SelectBag(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['bag_selected'])
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.bridge = CvBridge()
        self.model = torch.hub.load('/home/user/yolov5', 'custom', path='/home/user/Yolov9/src/yolov3/best.pt', source='local')
        self.twist_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=10)

        # ปิดโหนด turtlebot_follower เพื่อหลีกเลี่ยงการใช้งานกล้องซ้ำซ้อน
        rospy.loginfo("Shutting down turtlebot_follower before selecting bag")
        subprocess.call(['rosnode', 'kill', '/turtlebot_follower'])

    def execute(self, userdata):
        rospy.loginfo('Selecting a bag...')
        selected_bag = None
        bag_position = None

        while selected_bag is None and not rospy.is_shutdown():
            frame = self.get_kinect_rgb_frame()
            bag_detected, frame = self.detect_bag(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                    h, w, _ = frame.shape
                    cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                    cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

                    if bag_detected and self.is_pointing_at_bag(cx, cy, w, h):
                        if cx < w // 2:
                            selected_bag = "left_bag"
                            bag_position = "left"
                            rospy.loginfo('Left bag selected.')
                        else:
                            selected_bag = "right_bag"
                            bag_position = "right"
                            rospy.loginfo('Right bag selected.')
                        break

            cv2.imshow('Select Bag', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        # Perform the corresponding action based on the selected bag
        self.turn_and_move(bag_position)

        return 'bag_selected'

    def get_kinect_rgb_frame(self):
        frame = rospy.wait_for_message('/camera/rgb/image_raw', Image)
        return self.bridge.imgmsg_to_cv2(frame, "bgr8")

    def detect_bag(self, frame):
        results = self.model(frame)
        labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        n = len(labels)
        for i in range(n):
            row = coords[i]
            if row[4] >= 0.5:
                x1, y1, x2, y2 = int(row[0] * frame.shape[1]), int(row[1] * frame.shape[0]), int(row[2] * frame.shape[1]), int(row[3] * frame.shape[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{int(labels[i])}: {row[4]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                return True, frame

        return False, frame

    def is_pointing_at_bag(self, cx, cy, w, h):
        if cx > w * 0.25 and cx < w * 0.75 and cy > h * 0.25 and cy < h * 0.75:
            return True
        return False

    def turn_and_move(self, bag_position):
        twist = Twist()
        if bag_position == "left":
            twist.angular.z = 0.5  # Turn left
        elif bag_position == "right":
            twist.angular.z = -0.5  # Turn right

        self.twist_pub.publish(twist)
        rospy.sleep(1.5)  # Adjust the sleep time to control how much the robot turns

        # Move forward after turning
        twist.angular.z = 0.0
        twist.linear.x = 0.2  # Move forward
        self.twist_pub.publish(twist)
        rospy.sleep(2)  # Adjust the sleep time to control how much the robot moves forward

        # Stop the robot
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.twist_pub.publish(twist)

class ReturnToLastPosition(smach.State):
    def __init__(self, navigator):
        smach.State.__init__(self, outcomes=['returned'])
        self.navigator = navigator

    def execute(self, userdata):
        rospy.loginfo('Returning to the last known position...')
        self.navigator.return_to_last_position()
        return 'returned'

class FollowPerson(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['following', 'stop_following'])
        self.twist_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=10)
        self.bridge = CvBridge()
        rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.person_detected = False
        self.person_position = None
        self.image_width = 640  # Assumed width of the image
        self.following_duration = 60  # กำหนดเวลาติดตามเป็น 60 วินาที (1 นาที)

        # เปิดการใช้งานโหนด `turtlebot_follower`
        rospy.loginfo("Launching turtlebot_follower for person following")
        subprocess.Popen(['roslaunch', 'turtlebot_follower', 'follower.launch'])

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Use Mediapipe to detect the person in the RGB image
        mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        results = mp_pose.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            h, w, _ = cv_image.shape
            # Use the midpoint between the shoulders as the "person position"
            shoulder_left = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            shoulder_right = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            cx = int((shoulder_left.x + shoulder_right.x) / 2 * w)
            self.person_position = cx
            self.person_detected = True
        else:
            self.person_detected = False
            self.person_position = None

    def execute(self, userdata):
        rospy.loginfo('Following the person...')
        speak('I am following you.')

        # ตั้ง timer 60 วินาที (1 นาที)
        end_time = rospy.Time.now() + rospy.Duration(self.following_duration)

        rate = rospy.Rate(10)  # ตั้งอัตราการทำงานที่ 10Hz
        while not rospy.is_shutdown():
            if rospy.Time.now() >= end_time:
                rospy.loginfo("Time is up! Stopping follow after 1 minute")
                self.stop_follower()
                return 'stop_following'

            if self.person_detected and self.person_position is not None:
                twist = Twist()
                
                # Adjust direction based on the position of the person in the image
                if self.person_position < self.image_width / 3:
                    twist.angular.z = 0.5  # Turn left
                elif self.person_position > 2 * self.image_width / 3:
                    twist.angular.z = -0.5  # Turn right
                else:
                    twist.angular.z = 0.0  # Move forward

                twist.linear.x = 0.2  # Move forward
                rospy.loginfo("Person detected: Moving forward")
                self.twist_pub.publish(twist)
            else:
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                rospy.loginfo("Person not detected: Stopping")
                self.twist_pub.publish(twist)
            rate.sleep()

        return 'following'

    def stop_follower(self):
        rospy.loginfo("Shutting down turtlebot_follower")
        subprocess.call(['rosnode', 'kill', '/turtlebot_follower'])

class RequestPickup(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['requested'])
        self.arduino_pub = rospy.Publisher('Test', String, queue_size=10)
    
    def execute(self, userdata):
        rospy.loginfo('Requesting pickup...')
        speak('I will pick up the bag now.')
        self.arduino_pub.publish("B")  # Initial pickup action
        rospy.sleep(6) 
        self.arduino_pub.publish("D")  # Arm move to pickup
        rospy.sleep(3) 
        self.arduino_pub.publish("C")  # Close gripper
        rospy.sleep(3) 
        self.arduino_pub.publish("E")  # Retract arm
        rospy.sleep(3)   # Wait for the arm to complete its movement

        return 'requested'

class DeliverBag(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['delivered', 'timeout'])
        self.arduino_pub = rospy.Publisher('Test', String, queue_size=10)
        self.received_flag = False

    def execute(self, userdata):
        rospy.loginfo('Delivering the bag...')
        speak('Here is the bag. Please take it.')
        self.arduino_pub.publish("B")  # Extend the arm
        rospy.sleep(5)
        self.arduino_pub.publish("D")  # Open the gripper
        rospy.sleep(5)  # Wait for the person to take the bag
        
        return 'delivered'

class RetractArm(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['retracted'])
        self.arduino_pub = rospy.Publisher('Test', String, queue_size=10)

    def execute(self, userdata):
        rospy.loginfo('Retracting the arm...')
        speak('Thank you. Retracting the arm now.')
        self.arduino_pub.publish("E")  # Retract the arm
        rospy.sleep(3)
        return 'retracted'

class ReturnToStart(smach.State):
    def __init__(self, navigator):
        smach.State.__init__(self, outcomes=['returned'])
        self.navigator = navigator
    
    def execute(self, userdata):
        rospy.loginfo('Returning to the starting point...')
        self.navigator.return_to_home()
        speak('Mission accomplished. Returning to the start.')
        return 'returned'

def main():
    rospy.init_node('turtlebot_fsm')

    navigator = RobotNavigator()

    sm = smach.StateMachine(outcomes=['MISSION_COMPLETED'])
    sm.userdata.attempts = 0

    with sm:
        smach.StateMachine.add('SELECT_BAG', SelectBag(), 
                               transitions={'bag_selected':'NAVIGATE_TO_BAG'})
        
        smach.StateMachine.add('NAVIGATE_TO_BAG', NavigateToBag(), 
                               transitions={'arrived':'REQUEST_PICKUP'})
        
        smach.StateMachine.add('REQUEST_PICKUP', RequestPickup(), 
                               transitions={'requested':'FOLLOW_PERSON'})
        
        smach.StateMachine.add('FOLLOW_PERSON', FollowPerson(), 
                               transitions={'following':'DELIVER_BAG', 'stop_following':'DELIVER_BAG'})
        
        smach.StateMachine.add('DELIVER_BAG', DeliverBag(), 
                               transitions={'delivered':'RETRACT_ARM', 'timeout':'RETRACT_ARM'})
        
        smach.StateMachine.add('RETRACT_ARM', RetractArm(), 
                               transitions={'retracted':'RETURN_TO_LAST_POSITION'})
        
        smach.StateMachine.add('RETURN_TO_LAST_POSITION', ReturnToLastPosition(navigator), 
                               transitions={'returned':'RETURN_TO_START'})
        
        smach.StateMachine.add('RETURN_TO_START', ReturnToStart(navigator), 
                               transitions={'returned':'MISSION_COMPLETED'})

    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()

    outcome = sm.execute()

    rospy.spin()
    sis.stop()

if __name__ == '__main__':
    main()

