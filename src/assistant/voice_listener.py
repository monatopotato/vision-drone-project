"""
Voice Command Listener for Autonomous Drone Assistant
Captures microphone audio and converts spoken natural language phrases to text.
"""

import time

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

class VoiceListener:
    def __init__(self):
        self.recognizer = None
        self.microphone = None
        self.available = False

        if HAS_SR:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                # Calibrate for ambient noise
                with self.microphone as source:
                    print(">> Calibrating microphone for ambient noise...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                self.available = True
                print(">> Voice Recognition Initialized Successfully!")
            except Exception as e:
                print(f"!! Microphone initialization error: {e}")
                self.available = False
        else:
            print("!! SpeechRecognition library not found. Install via 'pip install SpeechRecognition PyAudio'")

    def listen_for_command(self, timeout=5, phrase_time_limit=7):
        """
        Listens for a spoken phrase from the microphone and returns recognized text.
        Returns:
          recognized_text: str (or None if no speech detected or error)
        """
        if not self.available:
            return None

        try:
            print("\n[Voice Listener] Listening for spoken command... (Speak now)")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            print("[Voice Listener] Processing speech...")
            text = self.recognizer.recognize_google(audio)
            print(f"[Voice Listener] Recognized Speech: \"{text}\"")
            return text

        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            print("!! Could not understand audio phrase. Please try speaking clearly.")
            return None
        except sr.RequestError as e:
            print(f"!! Speech Recognition service error: {e}")
            return None
        except Exception as ex:
            print(f"!! Voice error: {ex}")
            return None

if __name__ == '__main__':
    listener = VoiceListener()
    if listener.available:
        cmd = listener.listen_for_command()
        print(f"Result: {cmd}")
