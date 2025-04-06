import cv2
import numpy as np
import time
import argparse
import os
import threading
from datetime import datetime

class SimpleDrowsinessDetector:
    """A simplified drowsiness detector using OpenCV's built-in detectors instead of dlib."""
    
    def __init__(self, blink_threshold=3, eye_ar_threshold=0.3, eye_ar_consec_frames=30):
        """
        Initialize the drowsiness detector.
        
        Args:
            blink_threshold: Number of blinks in a short period to trigger alert
            eye_ar_threshold: Threshold for considering eyes closed
            eye_ar_consec_frames: Number of consecutive frames to trigger alert
        """
        self.blink_threshold = blink_threshold
        self.eye_ar_threshold = eye_ar_threshold
        self.eye_ar_consec_frames = eye_ar_consec_frames
        
        # Load pre-trained models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Initialize counters
        self.frame_counter = 0
        self.blink_counter = 0
        self.last_blink_time = time.time()
        self.drowsy = False
        
        # Dashboard data
        self.dashboard_width = 800
        self.dashboard_height = 200
        self.dashboard = np.zeros((self.dashboard_height, self.dashboard_width, 3), dtype=np.uint8)
        self.eye_open_ratios = []
        self.timestamps = []
        self.alert_timestamps = []
        self.time_window = 60  # seconds to display in dashboard
        
        # Colors
        self.grid_color = (50, 50, 50)      # Dark gray
        self.eye_ratio_color = (0, 255, 0)  # Green
        self.alert_color = (0, 0, 255)      # Red
        self.text_color = (255, 255, 255)   # White
        
        # Alarm system
        self.alarm_on = False
        self.alarm_thread = None
    
    def process_frame(self, frame):
        """
        Process a single frame to detect drowsiness.
        
        Args:
            frame: The frame to process
            
        Returns:
            The processed frame with annotations
        """
        # Create a grayscale copy of the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Default value if no eyes detected
        eye_open_ratio = 0.0
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face region of interest
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes within the face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) >= 2:  # At least two eyes detected
                # Calculate rough eye openness based on detected eye dimensions
                eye_open_ratio = self._calculate_eye_open_ratio(eyes)
                
                # Draw eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            else:
                # If eyes not detected, assume eyes might be closed
                eye_open_ratio = 0.2
        
        # Update dashboard data
        self._update_dashboard_data(eye_open_ratio)
        
        # Check for drowsiness
        if eye_open_ratio < self.eye_ar_threshold:
            self.frame_counter += 1
            
            if self.frame_counter >= self.eye_ar_consec_frames:
                if not self.drowsy:
                    self.drowsy = True
                    self.alert_timestamps.append(time.time())
                    self._start_alarm()
                
                # Draw drowsiness alert
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Reset counter if eyes are open
            self.frame_counter = 0
            if self.drowsy:
                self.drowsy = False
                self._stop_alarm()
        
        # Display eye openness ratio
        cv2.putText(frame, f"Eye Ratio: {eye_open_ratio:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def _calculate_eye_open_ratio(self, eyes):
        """
        Calculate a rough measure of eye openness based on detected eye dimensions.
        
        Args:
            eyes: List of detected eye rectangles
        
        Returns:
            A ratio representing eye openness
        """
        # Find the average height-to-width ratio of detected eyes
        ratios = []
        for (ex, ey, ew, eh) in eyes:
            ratio = eh / ew if ew > 0 else 0
            ratios.append(ratio)
        
        return sum(ratios) / len(ratios) if ratios else 0
    
    def _update_dashboard_data(self, eye_open_ratio):
        """
        Update dashboard data with the latest eye openness ratio.
        
        Args:
            eye_open_ratio: The current eye openness ratio
        """
        current_time = time.time()
        
        # Add new data point
        self.timestamps.append(current_time)
        self.eye_open_ratios.append(eye_open_ratio)
        
        # Remove old data points
        cutoff_time = current_time - self.time_window
        while self.timestamps and self.timestamps[0] < cutoff_time:
            self.timestamps.pop(0)
            self.eye_open_ratios.pop(0)
        
        while self.alert_timestamps and self.alert_timestamps[0] < cutoff_time:
            self.alert_timestamps.pop(0)
    
    def update_dashboard(self):
        """
        Update the dashboard visualization.
        
        Returns:
            The updated dashboard image
        """
        # Clear the dashboard
        self.dashboard.fill(0)
        
        # Draw grid lines
        self._draw_grid()
        
        # Draw eye ratio graph
        self._draw_eye_ratio_graph()
        
        # Draw alert events
        self._draw_alert_events()
        
        # Draw statistics
        self._draw_statistics()
        
        return self.dashboard
    
    def _draw_grid(self):
        """Draw grid lines on the dashboard."""
        # Draw horizontal lines
        for i in range(1, 4):
            y = int(self.dashboard_height * i / 4)
            cv2.line(self.dashboard, (0, y), (self.dashboard_width, y), self.grid_color, 1)
        
        # Draw vertical lines (every 10 seconds)
        current_time = time.time()
        for i in range(self.time_window, 0, -10):
            x = int(self.dashboard_width * (1 - i / self.time_window))
            cv2.line(self.dashboard, (x, 0), (x, self.dashboard_height), self.grid_color, 1)
            
            # Add time labels
            time_str = f"-{i}s"
            cv2.putText(self.dashboard, time_str, (x - 20, self.dashboard_height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1)
    
    def _draw_eye_ratio_graph(self):
        """Draw the eye openness ratio as a line graph."""
        if len(self.timestamps) < 2:
            return
        
        current_time = time.time()
        points = []
        
        for t, ratio in zip(self.timestamps, self.eye_open_ratios):
            # Convert time to x coordinate
            x = int(self.dashboard_width * (1 - (current_time - t) / self.time_window))
            
            # Convert eye ratio to y coordinate
            # Higher ratio = more open eyes = higher on graph = smaller y value
            y = int(self.dashboard_height * (1 - ratio))
            y = max(0, min(y, self.dashboard_height - 1))  # Clamp to dashboard height
            
            points.append((x, y))
        
        # Draw the line connecting the points
        for i in range(1, len(points)):
            cv2.line(self.dashboard, points[i-1], points[i], self.eye_ratio_color, 2)
        
        # Draw threshold line
        threshold_y = int(self.dashboard_height * (1 - self.eye_ar_threshold))
        cv2.line(self.dashboard, (0, threshold_y), (self.dashboard_width, threshold_y), 
                 (255, 255, 0), 1, cv2.LINE_4)
        cv2.putText(self.dashboard, "Threshold", (5, threshold_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def _draw_alert_events(self):
        """Draw the alert events as vertical lines."""
        current_time = time.time()
        
        for t in self.alert_timestamps:
            # Convert time to x coordinate
            x = int(self.dashboard_width * (1 - (current_time - t) / self.time_window))
            cv2.line(self.dashboard, (x, 0), (x, self.dashboard_height), self.alert_color, 1)
    
    def _draw_statistics(self):
        """Draw statistics on the dashboard."""
        # Calculate statistics
        alerts = len(self.alert_timestamps)
        
        if self.eye_open_ratios:
            current_ratio = self.eye_open_ratios[-1]
            min_ratio = min(self.eye_open_ratios)
            max_ratio = max(self.eye_open_ratios)
            avg_ratio = sum(self.eye_open_ratios) / len(self.eye_open_ratios)
        else:
            current_ratio = min_ratio = max_ratio = avg_ratio = 0
        
        # Draw statistics text
        cv2.putText(self.dashboard, f"Alerts: {alerts}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        cv2.putText(self.dashboard, f"Current: {current_ratio:.2f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        cv2.putText(self.dashboard, f"Min: {min_ratio:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        cv2.putText(self.dashboard, f"Max: {max_ratio:.2f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        cv2.putText(self.dashboard, f"Avg: {avg_ratio:.2f}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        # Draw current time
        current_time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(self.dashboard, current_time_str, (self.dashboard_width - 100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
    
    def _sound_alarm(self):
        """Play the alarm sound."""
        while self.alarm_on:
            try:
                # Use system beep on Windows
                import winsound
                winsound.Beep(1000, 1000)  # Frequency, duration
            except Exception:
                # Fallback method for other platforms
                print('\a')  # ASCII bell
            
            time.sleep(1)
    
    def _start_alarm(self):
        """Start the alarm in a separate thread."""
        if not self.alarm_on:
            self.alarm_on = True
            self.alarm_thread = threading.Thread(target=self._sound_alarm)
            self.alarm_thread.daemon = True
            self.alarm_thread.start()
    
    def _stop_alarm(self):
        """Stop the alarm."""
        self.alarm_on = False

def main():
    """Main function to run the drowsiness detection."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simple Driver Drowsiness Detection")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Eye openness ratio threshold")
    parser.add_argument("--frames", type=int, default=30,
                        help="Consecutive frames for alert")
    args = parser.parse_args()
    
    # Initialize the detector
    detector = SimpleDrowsinessDetector(
        eye_ar_threshold=args.threshold,
        eye_ar_consec_frames=args.frames
    )
    
    # Start video capture
    print(f"Starting video stream from camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Wait for camera to initialize
    time.sleep(1)
    
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from camera.")
            break
        
        # Process the frame
        frame = detector.process_frame(frame)
        
        # Update and display the dashboard
        dashboard = detector.update_dashboard()
        
        # Display the processed frame and dashboard
        cv2.imshow("Drowsiness Detection", frame)
        cv2.imshow("Dashboard", dashboard)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 