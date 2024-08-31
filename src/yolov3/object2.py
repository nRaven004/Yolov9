#!/usr/bin/env python
import rospy
import cv2
import torch
import time
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def detect_and_display():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/user/yolov5/runs/train/exp10/weights/best.pt').to(device)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 5)  # ลด FPS ลงเพื่อลดการใช้งาน CPU/GPU
    frame_id = 0

    while not rospy.is_shutdown():
        frame_id += 1
        ret, frame = cap.read()
        if not ret or frame_id % 2 != 0:  # ประมวลผลทุกๆ 2 เฟรม
            continue

        frame = cv2.resize(frame, (640, 480))  # ลดขนาดภาพเพื่อประมวลผลได้เร็วขึ้น
        results = model(frame)

        for *xyxy, conf, cls in results.xyxy[0]:
            trust = conf * 100
            if trust > 60:
                rospy.loginfo("Detected bag")

            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("YOLOv5 Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)  # ลดการประมวลผลเฟรมอย่างต่อเนื่องเพื่อเพิ่มประสิทธิภาพ

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('yolo_detection_node')
    detect_and_display()
