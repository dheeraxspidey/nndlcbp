import cv2
import numpy as np
import os
import time
import argparse
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from alarm import DrowsinessAlarm
from dashboard import DrowsinessDashboard

def extract_features(image):
    """Extract features from eye image."""
    # Resize image to standard size
    image = cv2.resize(image, (64, 64))
    
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Extract HOG features
    # This is a simpler alternative to deep learning features
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hog_features = hog.compute(gray)
    
    # Extract histogram features
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_features = hist.flatten() / np.sum(hist)  # Normalize
    
    # Combine features
    features = np.concatenate((hog_features.flatten(), hist_features))
    
    return features

def train_model(data_dir, model_path):
    """Train a Random Forest model on the dataset of eye images."""
    features = []
    labels = []
    classes = ['alert', 'drowsy']
    
    print("Extracting features from training images...")
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory {class_dir} not found")
            continue
            
        for image_name in os.listdir(class_dir):
            if not (image_name.endswith('.jpg') or image_name.endswith('.png')):
                continue
                
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue
                
            # Extract features from the image
            feature_vector = extract_features(image)
            features.append(feature_vector)
            labels.append(i)  # 0 for alert, 1 for drowsy
    
    if not features:
        print("Error: No valid images found for training")
        return None, None
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\nModel evaluation:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Save the model and scaler
    print(f"Saving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump((model, scaler), f)
    
    # Feature importance plot
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
        plt.title('Feature Importance')
        plt.savefig(os.path.join(os.path.dirname(model_path), 'feature_importance.png'))
        plt.close()
    
    return model, scaler

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Machine Learning Drowsiness Detection")
    parser.add_argument("--model", type=str, default="drowsiness_model.pkl",
                        help="Path to the trained model")
    parser.add_argument("--train", action="store_true",
                        help="Train a new model")
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Directory containing training data")
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
    
    # Train a new model if requested
    if args.train:
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory {args.data_dir} not found")
            print("Please create a dataset with the following structure:")
            print("dataset/")
            print("  alert/")
            print("    image1.jpg")
            print("    image2.jpg")
            print("    ...")
            print("  drowsy/")
            print("    image1.jpg")
            print("    image2.jpg")
            print("    ...")
            return
        
        print("Training new model...")
        model, scaler = train_model(args.data_dir, args.model)
        if model is None:
            return
    else:
        # Load the pre-trained model
        if not os.path.exists(args.model):
            print(f"Error: Model file {args.model} not found")
            print("Please train a model first with --train")
            return
        
        print(f"Loading model from {args.model}...")
        with open(args.model, 'rb') as f:
            model, scaler = pickle.load(f)
    
    # Initialize the alarm
    alarm = DrowsinessAlarm(args.alarm_file)
    
    # Initialize the dashboard if enabled
    dashboard = DrowsinessDashboard() if args.show_dashboard else None
    
    # Initialize drowsiness detection variables
    drowsy_counter = 0
    alert_counter = 0
    alarm_status = False
    consecutive_drowsy_frames = 0
    drowsy_threshold = 30  # Number of consecutive drowsy frames to trigger alarm
    
    # Start video capture
    print(f"Starting video stream from camera {args.camera}...")
    video_capture = cv2.VideoCapture(args.camera)
    
    # Load face and eye cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
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
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Default prediction if no face is detected
        prediction = "No face detected"
        confidence = 0.0
        
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
                
                # Extract features
                features = extract_features(eye_roi)
                
                # Scale features
                scaled_features = scaler.transform([features])
                
                # Make prediction
                pred_proba = model.predict_proba(scaled_features)[0]
                class_idx = np.argmax(pred_proba)
                confidence = pred_proba[class_idx]
                
                # Map class index to label
                if class_idx == 0:
                    prediction = "Alert"
                    alert_counter += 1
                    drowsy_counter = 0
                else:
                    prediction = "Drowsy"
                    drowsy_counter += 1
                    alert_counter = 0
                
                # Check for consecutive drowsy frames
                if prediction == "Drowsy":
                    consecutive_drowsy_frames += 1
                else:
                    consecutive_drowsy_frames = 0
                
                # Trigger alarm if drowsy for too long
                if consecutive_drowsy_frames >= drowsy_threshold:
                    if not alarm_status:
                        alarm_status = True
                        alarm.start_alarm()
                    
                    # Draw an alert on the frame
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    if alarm_status:
                        alarm_status = False
                        alarm.stop_alarm()
                
                # Display the prediction and confidence on the frame
                cv2.putText(frame, f"{prediction}: {confidence:.2f}", (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Update and show dashboard if enabled
        if dashboard:
            dashboard_frame = dashboard.update(confidence, alarm_status)
            cv2.imshow("Drowsiness Dashboard", dashboard_frame)
        
        # Display the frame
        cv2.imshow("ML Drowsiness Detection", frame)
        
        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 