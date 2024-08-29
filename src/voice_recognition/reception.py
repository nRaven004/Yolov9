#!/usr/bin/env python3
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
import os

def speak(text):
    tts = gTTS(text=text, lang='en')
    audio_file = 'temp.mp3'
    tts.save(audio_file)
    playsound(audio_file)
    os.remove(audio_file)

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            print("Audio captured, recognizing...")
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            speak("Sorry, I did not understand that.")
            return listen()
        except sr.RequestError:
            speak("Sorry, there was an error with the speech recognition service.")
            return None
        except sr.WaitTimeoutError:
            speak("No speech detected. Please try again.")
            return listen()

def confirm_and_get_name():
    while True:
        speak("Excuse me, what is your name?")
        name = listen()
        if name:
            speak(f"Is {name} your name?")
            response = listen()
            if response and 'yes' in response.lower():
                speak(f"Hello, {name}, nice to meet you.")
                return name
            else:
                speak("Let's try that again.")

def main():
    name = confirm_and_get_name()

    # Ask for favorite food
    speak("What is your favorite food?")
    favorite_food = listen()
    if favorite_food:
        speak(f"Do {favorite_food} is your favorite food?")
        response = listen()
        if response:
            if 'yes' in response.lower():
                speak(f"He’s {name} and his favorite food is {favorite_food}.")
            else:
                # Handle 'no' response and ask again
                while True:
                    speak("What is your favorite food?")
                    favorite_food = listen()
                    if favorite_food:
                        speak(f"Do {favorite_food} is your favorite food?")
                        response = listen()
                        if response and 'yes' in response.lower():
                            speak(f"He’s {name} and his favorite food is {favorite_food}.")
                            break

if __name__ == "__main__":
    main()

