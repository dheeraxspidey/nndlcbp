import os
import platform
import threading
import time

class DrowsinessAlarm:
    """A class for handling drowsiness alarms."""
    
    def __init__(self, sound_file=None):
        """
        Initialize the drowsiness alarm.
        
        Args:
            sound_file: Path to the sound file to play. If None, use text-to-speech.
        """
        self.sound_file = sound_file
        self.alarm_on = False
        self.alarm_thread = None
        self.system = platform.system()
    
    def sound_alarm(self):
        """Play the alarm sound."""
        # Keep sounding alarm until alarm_on is False
        while self.alarm_on:
            if self.sound_file and os.path.exists(self.sound_file):
                self._play_sound_file()
            else:
                self._text_to_speech()
            
            # Wait before playing again
            time.sleep(2)
    
    def _play_sound_file(self):
        """Play a sound file."""
        try:
            from playsound import playsound
            playsound(self.sound_file)
        except Exception as e:
            print(f"Error playing sound: {e}")
            self._text_to_speech()  # Fallback to text-to-speech
    
    def _text_to_speech(self):
        """Use text-to-speech to alert the driver."""
        message = "Wake up! You are drowsy!"
        
        try:
            if self.system == "Windows":
                import winsound
                winsound.Beep(1000, 1000)  # Frequency, duration
            elif self.system == "Darwin":  # macOS
                os.system(f'say "{message}"')
            else:  # Linux and others
                os.system(f'echo "{message}" | espeak')
        except Exception as e:
            print(f"Alert! {message}")
            print(f"Error with audio alert: {e}")
    
    def start_alarm(self):
        """Start the alarm in a separate thread."""
        if not self.alarm_on:
            self.alarm_on = True
            self.alarm_thread = threading.Thread(target=self.sound_alarm)
            self.alarm_thread.daemon = True
            self.alarm_thread.start()
    
    def stop_alarm(self):
        """Stop the alarm."""
        self.alarm_on = False
        if self.alarm_thread and self.alarm_thread.is_alive():
            self.alarm_thread.join(1.0)  # Wait for thread to finish
            self.alarm_thread = None 