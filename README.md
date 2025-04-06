# Driver Drowsiness Detection System

This system detects driver drowsiness using computer vision techniques to monitor eye closure patterns and alert the driver when signs of drowsiness are detected.

## Features

- Real-time face detection
- Eye state monitoring using Eye Aspect Ratio (EAR)
- Drowsiness alerts when eyes remain closed for too long
- Visual and audio alerts

## Requirements

- Python 3.7+
- Webcam
- Dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download the shape predictor file:
   - Download the 68-point facial landmark predictor from:
   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   - Extract and place it in the project directory

## Usage

Run the main script:
```
python simple_start.py
```

## How It Works

The system calculates the Eye Aspect Ratio (EAR) which is the ratio of the height and width of the eye. When the EAR falls below a certain threshold for a specified duration, the system determines that the driver is drowsy and triggers an alert.

## Configuration

You can adjust the following parameters in the script:
- `EAR_THRESHOLD`: The threshold below which the eye is considered closed
- `CONSECUTIVE_FRAMES`: Number of consecutive frames the eye must be closed to trigger an alert 
