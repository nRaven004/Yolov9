#!/usr/bin/env python
import rospy
import torch
from std_msgs.msg import String
import cv2

def detect_and_display():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov8', 'custom', path='/home/user/catkin_ws/src/yolov3/dataset/yolov8_test/best.pt').to(device)

    # Create a publisher for the "Test" topic
    pub = rospy.Publisher('Test', String, queue_size=10)

    # Use OpenCV to capture video from the camera (0 is the default camera)
    cap = cv2.VideoCapture(0)

    rospy.init_node('yolo_detection_node')

    if not cap.isOpened():
        rospy.logerr("Error: Could not open video capture.")
        return

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logerr("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 480))  # Resize for faster processing

        results = model(frame)
        detected = False

        for *xyxy, conf, cls in results.xyxy[0]:
            trust = conf * 100
            if trust > 60:
                detected = True
                rospy.loginfo("Detected bag")

            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if detected:
            # Publish message "B" to the "Test" topic
            pub.publish("B")

        # Display the frame
        cv2.imshow("YOLOv5 Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown('User requested shutdown')

    # Release the video capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_and_display()
