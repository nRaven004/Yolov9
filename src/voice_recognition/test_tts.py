#!/usr/bin/env python3
from gtts import gTTS
from playsound import playsound
import os

# Define the text you want to convert to speech
text = "Hello, welcome to the ROS home reception task."

# Create a gTTS object
tts = gTTS(text=text, lang='en')

# Save the audio file
audio_file = 'output.mp3'
tts.save(audio_file)

# Play the audio file
playsound(audio_file)

# Optionally, remove the file after playing
os.remove(audio_file)

