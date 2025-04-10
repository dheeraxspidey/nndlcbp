import os
import subprocess
import sys
import time
import argparse

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

def check_ml_requirements():
    """Check if the machine learning packages are installed."""
    try:
        import sklearn
        import matplotlib
        print("Scikit-learn and Matplotlib are installed.")
        return True
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install the required packages using 'pip install scikit-learn matplotlib'")
        return False

def check_cnn_requirements():
    """Check if the CNN packages (PyTorch) are installed."""
    try:
        import torch
        import torchvision
        print("PyTorch and torchvision are installed.")
        return True
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install the required packages using 'pip install torch torchvision'")
        return False

def check_viz_requirements():
    """Check if the visualization packages are installed."""
    try:
        import seaborn
        import pandas
        print("Seaborn and Pandas are installed.")
        return True
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install the required packages using 'pip install seaborn pandas'")
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

def start_ml_detection():
    """Start the machine learning drowsiness detection program."""
    print("Starting machine learning drowsiness detection...")
    
    try:
        # Run the program
        subprocess.run([sys.executable, "ml_detection.py"], check=True)
    except KeyboardInterrupt:
        print("\nStopping drowsiness detection...")
    except Exception as e:
        print(f"Error running drowsiness detection: {e}")

def start_ml_detection_with_training():
    """Start the machine learning drowsiness detection program with training."""
    print("Starting machine learning drowsiness detection with training...")
    
    try:
        # Run the program with the train flag
        subprocess.run([sys.executable, "ml_detection.py", "--train"], check=True)
    except KeyboardInterrupt:
        print("\nStopping drowsiness detection...")
    except Exception as e:
        print(f"Error running drowsiness detection: {e}")

def start_cnn_detection():
    """Start the CNN drowsiness detection program."""
    print("Starting CNN drowsiness detection...")
    
    try:
        # Run the program
        subprocess.run([sys.executable, "cnn_detection.py"], check=True)
    except KeyboardInterrupt:
        print("\nStopping drowsiness detection...")
    except Exception as e:
        print(f"Error running drowsiness detection: {e}")

def start_cnn_detection_with_training():
    """Start the CNN drowsiness detection program with training."""
    print("Starting CNN drowsiness detection with training...")
    
    try:
        # Run the program with the train flag
        subprocess.run([sys.executable, "cnn_detection.py", "--train"], check=True)
    except KeyboardInterrupt:
        print("\nStopping drowsiness detection...")
    except Exception as e:
        print(f"Error running drowsiness detection: {e}")

def compare_methods():
    """Compare all three drowsiness detection methods."""
    print("Comparing drowsiness detection methods...")
    
    try:
        # Run the comparison script
        subprocess.run([sys.executable, "compare_methods.py"], check=True)
    except KeyboardInterrupt:
        print("\nStopping comparison...")
    except Exception as e:
        print(f"Error running comparison: {e}")

def collect_training_data():
    """Collect training data for the machine learning model."""
    print("Starting data collection...")
    
    try:
        # Run the program
        subprocess.run([sys.executable, "collect_training_data.py", "--class_name", "alert", "--num_images", "50"], check=True)
        subprocess.run([sys.executable, "collect_training_data.py", "--class_name", "drowsy", "--num_images", "50"], check=True)
    except KeyboardInterrupt:
        print("\nStopping data collection...")
    except Exception as e:
        print(f"Error collecting data: {e}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection")
    parser.add_argument("--mode", type=str, 
                        choices=["simple", "ml", "ml-train", "cnn", "cnn-train", "collect", "compare"], 
                        default="simple",
                        help="Detection mode: simple (traditional), ml (machine learning), ml-train (machine learning with training), cnn (neural network), cnn-train (neural network with training), collect (training data), or compare (compare all methods)")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    print("=" * 50)
    print("Driver Drowsiness Detection - Setup")
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
    
    # Check if the camera is working (not needed for comparison mode)
    if args.mode != "compare" and not check_camera():
        return
    
    # If ML mode is selected, check for ML requirements
    if args.mode in ["ml", "ml-train", "compare"]:
        if not check_ml_requirements():
            print("Installing machine learning packages...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn", "matplotlib", "imutils", "pillow"], check=True)
                print("Machine learning packages installed successfully.")
            except Exception as e:
                print(f"Error installing machine learning packages: {e}")
                return
    
    # If CNN mode is selected, check for CNN requirements
    if args.mode in ["cnn", "cnn-train", "compare"]:
        if not check_cnn_requirements():
            print("Installing CNN packages...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision"], check=True)
                print("CNN packages installed successfully.")
            except Exception as e:
                print(f"Error installing CNN packages: {e}")
                return
    
    # If comparison mode is selected, check for visualization requirements
    if args.mode == "compare":
        if not check_viz_requirements():
            print("Installing visualization packages...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "seaborn", "pandas"], check=True)
                print("Visualization packages installed successfully.")
            except Exception as e:
                print(f"Error installing visualization packages: {e}")
                return
    
    print("\nAll checks passed.\n")
    
    # Start the appropriate program based on the mode
    if args.mode == "simple":
        print("Starting simplified drowsiness detection...\n")
        start_drowsiness_detection()
    elif args.mode == "ml":
        print("Starting machine learning drowsiness detection...\n")
        start_ml_detection()
    elif args.mode == "ml-train":
        print("Starting machine learning drowsiness detection with training...\n")
        start_ml_detection_with_training()
    elif args.mode == "cnn":
        print("Starting CNN drowsiness detection...\n")
        start_cnn_detection()
    elif args.mode == "cnn-train":
        print("Starting CNN drowsiness detection with training...\n")
        start_cnn_detection_with_training()
    elif args.mode == "collect":
        print("Starting data collection for model training...\n")
        collect_training_data()
    elif args.mode == "compare":
        print("Starting comparison of all methods...\n")
        compare_methods()

if __name__ == "__main__":
    main() 