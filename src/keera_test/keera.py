#!/usr/bin/env python3
import cv2
import numpy as np
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps
from gtts import gTTS  # Google Text-to-Speech
import os
from playsound import playsound

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("/home/user/catkin_ws/src/keera_test/keras_model.h5", compile=False)

# Load the labels
class_names = open("/home/user/catkin_ws/src/keera_test/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Start video capture from the webcam (change 0 to your camera index if needed)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Convert the frame to PIL Image for preprocessing
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    
    # Resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Turn the image into a numpy array
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array
    
    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Display the predictions on the frame
    label = f"Class: {class_name} | Confidence: {confidence_score:.2f}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Check if the class is 1, 2, 3, 4, or 5 and speak the number
    if class_name in ['1', '2', '3', '4', '5']:
        text_to_speak = f"We saw number {class_name}"
        print(text_to_speak)

        # Use gTTS to convert the text to speech
        tts = gTTS(text=text_to_speak, lang='en')
        tts.save("detected_number.mp3")

        # Play the generated speech audio
        playsound("detected_number.mp3")
        
        # Optionally, remove the audio file after playing
        os.remove("detected_number.mp3")
    
    # Show the frame with the predictions
    cv2.imshow('Real-Time Classification', frame)
    
    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

