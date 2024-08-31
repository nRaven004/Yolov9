#!/usr/bin/env python
import rospy
import cv2
import torch
import time
import warnings
from std_msgs.msg import String  # Import the String message type
from gtts import gTTS           # Import gTTS for text-to-speech
from playsound import playsound # Import playsound to play the TTS audio

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def detect_and_display():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/user/yolov5/runs/train/exp10/weights/best.pt').to(device)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 5)  # Reduce FPS to lower CPU/GPU usage
    frame_id = 0

    # ROS Publisher for the topic 'Test'
    pub = rospy.Publisher('Test', String, queue_size=10)

    while not rospy.is_shutdown():
        frame_id += 1
        ret, frame = cap.read()
        if not ret or frame_id % 2 != 0:  # Process every 2nd frame
            continue

        frame = cv2.resize(frame, (640, 480))  # Reduce frame size for faster processing
        results = model(frame)

        for *xyxy, conf, cls in results.xyxy[0]:
            trust = conf * 100
            if trust > 70:
                rospy.loginfo("Detected object with high confidence")
                
                # TTS saying "Please place the bag on my hand."
                tts = gTTS(text="Please place the bag on my hand.", lang='en')
                tts.save("bag_prompt.mp3")
                playsound("bag_prompt.mp3")  # Play the TTS audio

                pub.publish("B")  # Publish message 'B' to the 'Test' topic

            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("YOLOv5 Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)  # Reduce continuous frame processing for performance

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('yolo_detection_node')
    detect_and_display()

