import cv2
import numpy as np
import time
from datetime import datetime

class DrowsinessDashboard:
    """
    A dashboard for displaying drowsiness metrics over time.
    """
    
    def __init__(self, width=800, height=200, time_window=60):
        """
        Initialize the drowsiness dashboard.
        
        Args:
            width: Width of the dashboard in pixels.
            height: Height of the dashboard in pixels.
            time_window: Time window in seconds to display.
        """
        self.width = width
        self.height = height
        self.time_window = time_window
        
        # Create a blank canvas for the dashboard
        self.dashboard = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Initialize data structures for storing metrics
        self.timestamps = []
        self.ear_values = []
        self.alert_events = []
        
        # Initialize start time
        self.start_time = time.time()
        
        # Colors
        self.grid_color = (50, 50, 50)  # Dark gray
        self.ear_color = (0, 255, 0)    # Green
        self.alert_color = (0, 0, 255)  # Red
        self.text_color = (255, 255, 255)  # White
    
    def update(self, ear, is_alert):
        """
        Update the dashboard with new data.
        
        Args:
            ear: Current eye aspect ratio.
            is_alert: Whether an alert is currently active.
        """
        current_time = time.time()
        
        # Add new data point
        self.timestamps.append(current_time)
        self.ear_values.append(ear)
        if is_alert:
            self.alert_events.append(current_time)
        
        # Remove old data points outside the time window
        cutoff_time = current_time - self.time_window
        while self.timestamps and self.timestamps[0] < cutoff_time:
            self.timestamps.pop(0)
            self.ear_values.pop(0)
        
        while self.alert_events and self.alert_events[0] < cutoff_time:
            self.alert_events.pop(0)
        
        # Clear the dashboard
        self.dashboard.fill(0)
        
        # Draw grid lines
        self._draw_grid()
        
        # Draw EAR values
        self._draw_ear_graph()
        
        # Draw alert events
        self._draw_alert_events()
        
        # Draw statistics
        self._draw_statistics()
        
        return self.dashboard
    
    def _draw_grid(self):
        """Draw grid lines on the dashboard."""
        # Draw horizontal lines
        for i in range(1, 4):
            y = int(self.height * i / 4)
            cv2.line(self.dashboard, (0, y), (self.width, y), self.grid_color, 1)
        
        # Draw vertical lines (every 10 seconds)
        current_time = time.time()
        for i in range(self.time_window, 0, -10):
            x = int(self.width * (1 - i / self.time_window))
            cv2.line(self.dashboard, (x, 0), (x, self.height), self.grid_color, 1)
            
            # Add time labels
            time_str = f"-{i}s"
            cv2.putText(self.dashboard, time_str, (x - 20, self.height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1)
    
    def _draw_ear_graph(self):
        """Draw the EAR values as a line graph."""
        if len(self.timestamps) < 2:
            return
        
        current_time = time.time()
        points = []
        
        for i, (t, ear) in enumerate(zip(self.timestamps, self.ear_values)):
            # Convert time to x coordinate
            x = int(self.width * (1 - (current_time - t) / self.time_window))
            
            # Convert EAR to y coordinate (assuming EAR is between 0 and 0.4)
            # Higher EAR = higher on screen = smaller y value
            y = int(self.height * (1 - ear / 0.4))
            y = max(0, min(y, self.height - 1))  # Clamp to dashboard height
            
            points.append((x, y))
        
        # Draw the line connecting the points
        for i in range(1, len(points)):
            cv2.line(self.dashboard, points[i-1], points[i], self.ear_color, 2)
        
        # Draw threshold line at 0.25 EAR (assumed threshold)
        threshold_y = int(self.height * (1 - 0.25 / 0.4))
        cv2.line(self.dashboard, (0, threshold_y), (self.width, threshold_y), 
                 (255, 255, 0), 1, cv2.LINE_DASH)
        cv2.putText(self.dashboard, "Threshold", (5, threshold_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def _draw_alert_events(self):
        """Draw the alert events as vertical lines."""
        current_time = time.time()
        
        for t in self.alert_events:
            # Convert time to x coordinate
            x = int(self.width * (1 - (current_time - t) / self.time_window))
            cv2.line(self.dashboard, (x, 0), (x, self.height), self.alert_color, 1)
    
    def _draw_statistics(self):
        """Draw statistics on the dashboard."""
        # Calculate statistics
        uptime = time.time() - self.start_time
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        
        alert_count = len(self.alert_events)
        
        if self.ear_values:
            current_ear = self.ear_values[-1]
            min_ear = min(self.ear_values) if self.ear_values else 0
            max_ear = max(self.ear_values) if self.ear_values else 0
            avg_ear = sum(self.ear_values) / len(self.ear_values) if self.ear_values else 0
        else:
            current_ear = min_ear = max_ear = avg_ear = 0
        
        # Draw statistics text
        cv2.putText(self.dashboard, f"Uptime: {uptime_str}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        cv2.putText(self.dashboard, f"Alerts: {alert_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        cv2.putText(self.dashboard, f"Current EAR: {current_ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        cv2.putText(self.dashboard, f"Min EAR: {min_ear:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        cv2.putText(self.dashboard, f"Max EAR: {max_ear:.2f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        cv2.putText(self.dashboard, f"Avg EAR: {avg_ear:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        # Draw current time
        current_time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(self.dashboard, current_time_str, (self.width - 100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1) 