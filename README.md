# Driver Drowsiness Detection System

This system detects driver drowsiness using computer vision techniques to monitor eye closure patterns and alert the driver when signs of drowsiness are detected.

## Features

- Real-time face detection
- Eye state monitoring using Eye Aspect Ratio (EAR)
- Drowsiness alerts when eyes remain closed for too long
- Visual and audio alerts
- **Machine Learning-based detection using HOG features and Random Forest**
- **CNN-based detection using PyTorch**
- **Data collection tools for training custom models**
- **Performance comparison with metrics for all detection methods**

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
3. For the traditional method, download the shape predictor file:
   - Download the 68-point facial landmark predictor from:
   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   - Extract and place it in the project directory

## Usage

### Traditional Method

Run the main script with the simple mode:
```
python simple_start.py --mode simple
```

### Machine Learning Method

1. Collect training data:
   ```
   python simple_start.py --mode collect
   ```
   This will collect 50 images each for "alert" and "drowsy" classes.

2. Train and run the machine learning model:
   ```
   python simple_start.py --mode ml-train
   ```

3. Run with an existing trained model:
   ```
   python simple_start.py --mode ml
   ```

### CNN Method (Deep Learning)

1. Collect training data (if not already done):
   ```
   python simple_start.py --mode collect
   ```

2. Train and run the CNN model:
   ```
   python simple_start.py --mode cnn-train
   ```

3. Run with an existing trained CNN model:
   ```
   python simple_start.py --mode cnn
   ```

### Comparing All Methods

After you've trained the ML and CNN models and collected data, you can compare all three methods:

```
python simple_start.py --mode compare
```

This will generate performance metrics and visualizations in the `comparison_results` directory, including:
- Accuracy, precision, recall, and F1 score for each method
- Confusion matrices
- Processing time comparisons
- CSV file with all metrics for easy import into reports/presentations

See the [comparison_results/README.md](comparison_results/README.md) for more details on the metrics and how to use them in your report.

## Version Control

This project includes a `.gitignore` file configured to:
- Ignore large model files (*.pkl, *.pth) that should be generated locally
- Ignore dataset files that may contain private information
- Keep important comparison results in the `comparison_results/` directory
- Ignore other temporary and system files

When sharing this project, you only need to share the code, and others can generate their own models and data.

## How It Works

### Traditional Method

The system calculates the Eye Aspect Ratio (EAR) which is the ratio of the height and width of the eye. When the EAR falls below a certain threshold for a specified duration, the system determines that the driver is drowsy and triggers an alert.

### Machine Learning Method

The machine learning approach uses feature extraction and a Random Forest classifier to determine drowsiness:

1. **Feature Extraction**:
   - Histogram of Oriented Gradients (HOG) - Captures edge and gradient structure
   - Histogram features - Represents intensity distribution
   
2. **Classification**:
   - Random Forest classifier with 100 decision trees
   - Trained on custom dataset of eye images
   - Provides probability score for drowsiness detection

3. **Monitoring**:
   - Tracks consecutive drowsy predictions
   - Triggers alert if drowsiness persists beyond threshold

### CNN Method (Deep Learning)

The CNN approach uses a deep learning model with PyTorch:

1. **Neural Network Architecture**:
   - Three convolutional layers with ReLU activation
   - Max pooling layers for dimensionality reduction
   - Fully connected layers with dropout for regularization
   - Softmax classifier for alert/drowsy prediction

2. **Learning Process**:
   - End-to-end learning from raw eye images
   - Automatic feature extraction via convolutional layers
   - Training with backpropagation and Adam optimizer

3. **Advantages**:
   - No manual feature engineering needed
   - Potentially higher accuracy with sufficient training data
   - More robust to variations in eye appearance

## Metrics and Evaluation

The comparison tool evaluates all three methods using:

1. **Classification Metrics**:
   - Accuracy: Overall correctness (correct predictions / total predictions)
   - Precision: Reliability of positive predictions (true positives / predicted positives)
   - Recall: Ability to find all positive instances (true positives / actual positives)
   - F1 Score: Harmonic mean of precision and recall

2. **Confusion Matrix**:
   - True positives, false positives, true negatives, false negatives
   - Visualized as a heatmap

3. **Performance Metrics**:
   - Processing time per frame
   - Memory usage
   - Model size

## Configuration

You can adjust the following parameters in the script:
- `EAR_THRESHOLD`: The threshold below which the eye is considered closed
- `CONSECUTIVE_FRAMES`: Number of consecutive frames the eye must be closed to trigger an alert 

You can adjust the following parameters in the scripts:
- `EAR_THRESHOLD`: The threshold below which the eye is considered closed (traditional method)
- `CONSECUTIVE_FRAMES`: Number of consecutive frames the eye must be closed to trigger an alert
- `DROWSY_THRESHOLD`: Number of consecutive drowsy frames to trigger an alert (ML/CNN methods)
- `N_ESTIMATORS`: Number of trees in the Random Forest (default: 100)
- `NUM_EPOCHS`: Number of training epochs for the CNN (default: 10) 