#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import tf
import torch

class ImageDisplayAndPose:
    def __init__(self):
        # Initialize ROS Node
        rospy.init_node('image_display_and_pose', anonymous=True)

        # Camera Intrinsic Parameters (Replace these with your actual values)
        self.fx = 800  # Focal length in x direction (in pixels)
        self.fy = 800  # Focal length in y direction (in pixels)
        self.cx = 320  # Principal point x (in pixels)
        self.cy = 240  # Principal point y (in pixels)
        self.camera_matrix = np.array([[self.fx, 0, self.cx],
                                      [0, self.fy, self.cy],
                                      [0, 0, 1]])
        self.dist_coeffs = np.array([0, 0, 0, 0, 0])  # Distortion coefficients (if any)

        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/user/yolov5/runs/train/exp10/weights/best.pt')
        self.model.eval()

        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Create a subscriber to the image topic
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        
        # Create a window to display images
        cv2.namedWindow("Image Window", cv2.WINDOW_AUTOSIZE)

        # Publisher for pose messages
        self.pose_pub = rospy.Publisher('/robot/target_pose', PoseStamped, queue_size=10)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Check if the image is empty
            if cv_image is None or cv_image.size == 0:
                rospy.logwarn("Received an empty image.")
                return
            
            # Perform object detection using YOLOv5
            detection_result = self.detect_bag(cv_image)

            if detection_result:
                x_center, y_center, width, height = detection_result

                # Draw bounding box on the image
                cv2.rectangle(cv_image, (int(x_center - width / 2), int(y_center - height / 2)),
                              (int(x_center + width / 2), int(y_center + height / 2)), (0, 255, 0), 2)
                cv2.putText(cv_image, 'Bag Detected', (int(x_center - width / 2), int(y_center - height / 2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Convert image coordinates to real-world coordinates
                real_world_position = self.convert_to_real_world(x_center, y_center)

                # Publish target pose
                self.publish_pose(real_world_position)

            # Display the image
            cv2.imshow("Image Window", cv_image)
            cv2.waitKey(1)  # Display the image until a key is pressed

        except Exception as e:
            rospy.logerr("Error in image callback: %s", str(e))

    def detect_bag(self, image):
        # Perform detection using YOLOv5
        results = self.model(image)

        # Extract bounding boxes and labels
        boxes = results.xyxy[0].cpu().numpy()  # Bounding boxes
        if len(boxes) == 0:
            rospy.logwarn("No objects detected.")
            return None

        # Assuming the first detected object is the bag
        x_min, y_min, x_max, y_max, conf, cls = boxes[0]
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        return (x_center, y_center, width, height)

    def convert_to_real_world(self, x_center, y_center):
        # Example conversion using camera intrinsic parameters
        u = x_center
        v = y_center

        # Compute real-world coordinates (assuming a fixed depth)
        z = 1.0  # Example depth in meters
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        return (x, y, z)

    def publish_pose(self, position):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        pose.pose.orientation.w = 1.0  # No rotation

        self.pose_pub.publish(pose)

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = ImageDisplayAndPose()
        node.run()
    except rospy.ROSInterruptException:
        pass

