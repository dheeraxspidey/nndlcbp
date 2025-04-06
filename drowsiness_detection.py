import cv2
import dlib
import numpy as np
import time
import argparse
import os
from imutils import face_utils

# Import local modules
from utils import eye_aspect_ratio, get_eye_landmarks, draw_eye_landmarks, draw_text
from alarm import DrowsinessAlarm
from dashboard import DrowsinessDashboard

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection")
    parser.add_argument("--shape_predictor", type=str, default="shape_predictor_68_face_landmarks.dat",
                        help="Path to facial landmark predictor")
    parser.add_argument("--ear_threshold", type=float, default=0.25,
                        help="Threshold for eye aspect ratio below which eyes are considered closed")
    parser.add_argument("--consecutive_frames", type=int, default=30,
                        help="Number of consecutive frames the eye must be below threshold to trigger alarm")
    parser.add_argument("--alarm_file", type=str, default=None,
                        help="Path to alarm sound file (optional)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index")
    parser.add_argument("--show_dashboard", action="store_true",
                        help="Show the drowsiness dashboard")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Define constants for drowsiness detection
    EAR_THRESHOLD = args.ear_threshold
    CONSECUTIVE_FRAMES = args.consecutive_frames
    
    # Initialize counter and alarm status
    frame_counter = 0
    alarm_status = False
    
    # Initialize the alarm
    alarm = DrowsinessAlarm(args.alarm_file)
    
    # Initialize the dashboard if enabled
    dashboard = DrowsinessDashboard() if args.show_dashboard else None
    
    # Check if shape predictor file exists
    if not os.path.exists(args.shape_predictor):
        print(f"Error: Shape predictor file not found at {args.shape_predictor}")
        print("Please download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract it and place it in the project directory.")
        return
    
    # Initialize dlib's face detector and facial landmark predictor
    print("Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)
    
    # Get the indices for the left and right eyes
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    # Start video capture
    print(f"Starting video stream from camera {args.camera}...")
    video_capture = cv2.VideoCapture(args.camera)
    
    # Give the camera sensor time to warm up
    time.sleep(1.0)
    
    # Loop over frames from the video stream
    while True:
        # Grab a frame from the video stream
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        # Resize the frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = detector(gray, 0)
        
        # Default EAR if no face is detected
        ear = 0.0
        
        # Loop over each detected face
        for face in faces:
            # Determine the facial landmarks
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)
            
            # Extract the left and right eye coordinates
            leftEye = landmarks[lStart:lEnd]
            rightEye = landmarks[rStart:rEnd]
            
            # Calculate the eye aspect ratio for both eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            
            # Average the eye aspect ratio for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            
            # Draw the contours of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            # Check if the eye aspect ratio is below the threshold
            if ear < EAR_THRESHOLD:
                frame_counter += 1
                
                # If the eyes have been closed for enough frames, trigger the alarm
                if frame_counter >= CONSECUTIVE_FRAMES:
                    if not alarm_status:
                        alarm_status = True
                        alarm.start_alarm()
                    
                    # Draw an alert on the frame
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Reset the frame counter and alarm status
                frame_counter = 0
                if alarm_status:
                    alarm_status = False
                    alarm.stop_alarm()
            
            # Display the eye aspect ratio on the frame
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Update and show dashboard if enabled
        if dashboard:
            dashboard_frame = dashboard.update(ear, alarm_status)
            cv2.imshow("Drowsiness Dashboard", dashboard_frame)
        
        # Display the frame
        cv2.imshow("Drowsiness Detection", frame)
        
        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 