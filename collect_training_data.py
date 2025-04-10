import cv2
import numpy as np
import os
import time
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Collect Training Data for Drowsiness Detection")
    parser.add_argument("--output_dir", type=str, default="dataset",
                        help="Directory to save the collected images")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index")
    parser.add_argument("--class_name", type=str, required=True,
                        help="Class name for the collected images (alert or drowsy)")
    parser.add_argument("--num_images", type=int, default=100,
                        help="Number of images to collect")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    class_dir = os.path.join(args.output_dir, args.class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # Load face and eye cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Start video capture
    print(f"Starting video stream from camera {args.camera}...")
    video_capture = cv2.VideoCapture(args.camera)
    
    # Give the camera sensor time to warm up
    time.sleep(1.0)
    
    # Initialize counter for collected images
    collected_images = 0
    
    print(f"Collecting {args.num_images} images for class '{args.class_name}'...")
    print("Press 'c' to capture an image, 'q' to quit")
    
    # Loop over frames from the video stream
    while collected_images < args.num_images:
        # Grab a frame from the video stream
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        # Resize the frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Loop over each detected face
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face region of interest
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            # Process each detected eye
            for (ex, ey, ew, eh) in eyes:
                # Draw eye rectangle
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Extract eye region
                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                
                # Resize eye region to match model input size
                eye_roi = cv2.resize(eye_roi, (64, 64))
        
        # Display the frame
        cv2.imshow("Collect Training Data", frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # If 'c' is pressed, capture the image
        if key == ord('c'):
            # Check if eyes were detected
            if len(eyes) > 0:
                # Save the eye region
                timestamp = int(time.time() * 1000)
                image_path = os.path.join(class_dir, f"{args.class_name}_{timestamp}.jpg")
                cv2.imwrite(image_path, eye_roi)
                collected_images += 1
                print(f"Captured image {collected_images}/{args.num_images}")
            else:
                print("No eyes detected. Please try again.")
        
        # If 'q' is pressed, quit
        elif key == ord('q'):
            break
    
    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()
    
    print(f"Collection complete. {collected_images} images saved to {class_dir}")

if __name__ == "__main__":
    main() 