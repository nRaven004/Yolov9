#!/usr/bin/env python3
import cv2
import face_recognition
import numpy as np
import os
import rospy
from std_msgs.msg import String
import time

# Initialize ROS node
rospy.init_node('face_recognition_node')
pub = rospy.Publisher('Servo', String, queue_size=10)

# Load a sample dataset (folders with images)
def load_dataset(dataset_path):
    known_face_encodings = []
    known_face_names = []

    for folder_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(folder_name)
                except IndexError:
                    rospy.logwarn(f"Skipping {image_path}: No face found.")
    return known_face_encodings, known_face_names

# Path to dataset
dataset_path = "/home/jetson/catkin_ws/src/stuel_show/dataset_face" # Replace with your dataset path
known_face_encodings, known_face_names = load_dataset(dataset_path)

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
last_published_time = 0 # To manage delay
state = {"current_state": "idle", "last_recognized": None}

# Start video capture
video_capture = cv2.VideoCapture(0)

while not rospy.is_shutdown():
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Process every other frame
    if process_this_frame:
        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Publish the recognized name and handle state transitions
    if face_names:
        recognized_name = face_names[0] # Only handle the first recognized face
        current_time = time.time()

        if recognized_name != "Unknown":
            if state["current_state"] == "idle" and state["last_recognized"] != recognized_name:
                rospy.loginfo("First recognition. Publishing: B, C, D")
                pub.publish("B")
                rospy.sleep(6) # Delay 6 seconds after B
                pub.publish("C")
                rospy.sleep(3) # Delay 3 seconds after C
                pub.publish("D")
                state["last_recognized"] = recognized_name
                state["current_state"] = "holding"
                last_published_time = current_time

            elif state["current_state"] == "holding" and (current_time - last_published_time) >= 5:
                rospy.loginfo("Re-recognition. Publishing: E, A, C")
                pub.publish("E")
                rospy.sleep(5)
                pub.publish("A")
                rospy.sleep(3)
                pub.publish("C")
                state["last_recognized"] = recognized_name
                state["current_state"] = "idle"
                last_published_time = current_time

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()
