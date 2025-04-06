import os
import subprocess
import sys
import time

def check_requirements():
    """Check if the required packages are installed."""
    try:
        import cv2
        import numpy
        print("OpenCV and NumPy are installed.")
        return True
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install the required packages using 'pip install opencv-python numpy'")
        return False

def check_camera():
    """Check if a camera is available."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("Camera is working.")
                cap.release()
                return True
            else:
                print("Error: Could not read frame from camera.")
                cap.release()
                return False
        else:
            print("Error: Could not open camera.")
            return False
    except Exception as e:
        print(f"Error accessing camera: {e}")
        return False

def start_drowsiness_detection():
    """Start the simplified drowsiness detection program."""
    print("Starting drowsiness detection...")
    
    try:
        # Run the program
        subprocess.run([sys.executable, "simplified_detection.py"], check=True)
    except KeyboardInterrupt:
        print("\nStopping drowsiness detection...")
    except Exception as e:
        print(f"Error running drowsiness detection: {e}")

def main():
    print("=" * 50)
    print("Simple Driver Drowsiness Detection - Setup")
    print("=" * 50)
    
    # Check if the requirements are installed
    if not check_requirements():
        print("Installing required packages...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python", "numpy"], check=True)
            print("Packages installed successfully.")
        except Exception as e:
            print(f"Error installing packages: {e}")
            return
    
    # Check if the camera is working
    if not check_camera():
        return
    
    print("\nAll checks passed. Starting drowsiness detection...\n")
    
    # Start the drowsiness detection program
    start_drowsiness_detection()

if __name__ == "__main__":
    main() 