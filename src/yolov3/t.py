#!/usr/bin/env python3
import rospy
import cv2
import os
import numpy as np
import face_recognition
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from gtts import gTTS
import subprocess
import speech_recognition as sr

# Path to store face encodings and names
dataset_path = "/home/user/catkin_ws/src/yolov3/data_set"  # create empty folder
known_face_encodings = []
known_face_names = []

# Initialize Speech Recognizer
recognizer = sr.Recognizer()

def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            for image_file in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_file)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)

                for face_encoding in face_encodings:
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person_name)

load_known_faces()

# Function to use Google Text-to-Speech
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("/tmp/voice.mp3")
    os.system("mpg321 /tmp/voice.mp3")

# Function to recognize speech and convert to text
def recognize_speech():
    with sr.Microphone() as source:
        print("Listening for the name...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

# Add new face to the dataset in real-time
def add_new_face(frame, user_name):
    global known_face_encodings, known_face_names

    # Detect face encoding for the new face
    face_encodings = face_recognition.face_encodings(frame)
    if len(face_encodings) > 0:
        new_face_encoding = face_encodings[0]

        # Save the encoding and the name to the list
        known_face_encodings.append(new_face_encoding)
        known_face_names.append(user_name)

        # Save multiple images and encoding persistently (to disk)
        new_person_dir = os.path.join(dataset_path, user_name)
        if not os.path.exists(new_person_dir):
            os.makedirs(new_person_dir)
        
        # Capture several frames
        video_capture = cv2.VideoCapture(0)
        for i in range(5):  # Adjust the number of frames as needed
            # Capture frame
            ret, frame = video_capture.read()
            if not ret:
                break

            # Detect face encoding
            face_encodings = face_recognition.face_encodings(frame)
            if len(face_encodings) > 0:
                cv2.imwrite(os.path.join(new_person_dir, f"{user_name}_{i}.jpg"), frame)

        video_capture.release()
        speak(f"Hello {user_name}, your face has been registered.")
    else:
        print("No face detected for learning.")

# Main registration function
def register_guests():
    rospy.init_node('register_guest_node')
    video_capture = cv2.VideoCapture(0)

    favorite_drinks = ["coffee", "orange", "apple"]
    guest_count = 0

    while guest_count < len(favorite_drinks):
        # Capture frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            continue

        # Recognize speech to get guest name
        speak("Please say your name.")
        guest_name = recognize_speech()
        if guest_name:
            # Register new face
            add_new_face(frame, guest_name)
            
            # Assign favorite drink
            drink = favorite_drinks[guest_count]
            rospy.loginfo(f"Registered {guest_name} with favorite drink: {drink}")

            guest_count += 1
        else:
            print("Failed to recognize the name. Please try again.")
    
    video_capture.release()
    rospy.signal_shutdown("Registration completed.")

if __name__ == '__main__':
    register_guests()

