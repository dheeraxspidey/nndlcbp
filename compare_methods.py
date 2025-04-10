import cv2
import numpy as np
import os
import time
import argparse
import pickle
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from ml_detection import extract_features
from cnn_detection import DrowsinessDetectionCNN
import torchvision.transforms as transforms
from PIL import Image

def load_ml_model(model_path):
    """Load the machine learning model."""
    try:
        with open(model_path, 'rb') as f:
            model, scaler = pickle.load(f)
        print(f"Loaded ML model from {model_path}")
        return model, scaler
    except Exception as e:
        print(f"Error loading ML model: {e}")
        return None, None

def load_cnn_model(model_path):
    """Load the CNN model."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DrowsinessDetectionCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded CNN model from {model_path}")
        return model, device
    except Exception as e:
        print(f"Error loading CNN model: {e}")
        return None, None

def predict_traditional(eye_roi, ear_threshold=0.25):
    """
    Predict drowsiness using traditional method (EAR).
    
    Args:
        eye_roi: The eye region image
        ear_threshold: EAR threshold for drowsiness
    
    Returns:
        prediction: 0 for alert, 1 for drowsy
        confidence: EAR value
    """
    # Convert to grayscale if needed
    if len(eye_roi.shape) == 3:
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = eye_roi
    
    # Calculate eye aspect ratio using contour area and width/height ratio
    # as a simplified approximation (since we don't have landmarks)
    _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour (the eye)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate simplified EAR as height/width ratio (smaller when eyes are closing)
        ear = h / w if w > 0 else 0
    else:
        # No contours found (likely closed eye)
        ear = 0.1  # Assign a low EAR value
    
    # Determine prediction
    if ear < ear_threshold:
        prediction = 1  # drowsy
    else:
        prediction = 0  # alert
    
    return prediction, ear

def predict_ml(eye_roi, model, scaler):
    """
    Predict drowsiness using machine learning model.
    
    Args:
        eye_roi: The eye region image
        model: The ML model
        scaler: Feature scaler
    
    Returns:
        prediction: 0 for alert, 1 for drowsy
        confidence: Prediction probability
    """
    # Extract features
    features = extract_features(eye_roi)
    
    # Scale features
    scaled_features = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    confidence = model.predict_proba(scaled_features)[0][prediction]
    
    return prediction, confidence

def predict_cnn(eye_roi, model, device):
    """
    Predict drowsiness using CNN model.
    
    Args:
        eye_roi: The eye region image
        model: The CNN model
        device: The device (CPU/GPU)
    
    Returns:
        prediction: 0 for alert, 1 for drowsy
        confidence: Prediction probability
    """
    # Transform image for CNN
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess eye region for the model
    eye_tensor = transform(eye_roi).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(eye_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        prediction = torch.argmax(probabilities).item()
        confidence = probabilities[prediction].item()
    
    return prediction, confidence

def evaluate_model(model_name, predict_func, test_data, labels):
    """
    Evaluate model performance.
    
    Args:
        model_name: Name of the model
        predict_func: Prediction function
        test_data: Test images
        labels: Ground truth labels
    
    Returns:
        metrics: Dictionary of performance metrics
    """
    predictions = []
    confidences = []
    times = []
    
    for i, image in enumerate(test_data):
        start_time = time.time()
        prediction, confidence = predict_func(image)
        end_time = time.time()
        
        predictions.append(prediction)
        confidences.append(confidence)
        times.append(end_time - start_time)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    avg_time = np.mean(times) * 1000  # ms
    
    # Create confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Store all metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_time': avg_time,
        'predictions': predictions,
        'confidences': confidences,
        'confusion_matrix': cm
    }
    
    return metrics

def load_test_data(data_dir):
    """
    Load test data from directory.
    
    Args:
        data_dir: Directory containing test data
    
    Returns:
        images: List of images
        labels: List of labels
    """
    images = []
    labels = []
    
    # Load alert images
    alert_dir = os.path.join(data_dir, 'alert')
    if os.path.exists(alert_dir):
        for filename in os.listdir(alert_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(alert_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    images.append(img)
                    labels.append(0)  # 0 for alert
    
    # Load drowsy images
    drowsy_dir = os.path.join(data_dir, 'drowsy')
    if os.path.exists(drowsy_dir):
        for filename in os.listdir(drowsy_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(drowsy_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    images.append(img)
                    labels.append(1)  # 1 for drowsy
    
    return images, labels

def visualize_results(metrics_list):
    """
    Visualize comparison results.
    
    Args:
        metrics_list: List of metrics dictionaries
    """
    # Create directory for saving results
    os.makedirs('comparison_results', exist_ok=True)
    
    # Extract model names and metrics
    model_names = [m['model_name'] for m in metrics_list]
    accuracies = [m['accuracy'] for m in metrics_list]
    precisions = [m['precision'] for m in metrics_list]
    recalls = [m['recall'] for m in metrics_list]
    f1_scores = [m['f1'] for m in metrics_list]
    avg_times = [m['avg_time'] for m in metrics_list]
    
    # Performance metrics comparison
    plt.figure(figsize=(12, 8))
    bar_width = 0.15
    index = np.arange(len(model_names))
    
    plt.bar(index, accuracies, bar_width, label='Accuracy')
    plt.bar(index + bar_width, precisions, bar_width, label='Precision')
    plt.bar(index + 2*bar_width, recalls, bar_width, label='Recall')
    plt.bar(index + 3*bar_width, f1_scores, bar_width, label='F1 Score')
    
    plt.xlabel('Method')
    plt.ylabel('Score')
    plt.title('Performance Comparison of Drowsiness Detection Methods')
    plt.xticks(index + 1.5*bar_width, model_names)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('comparison_results/performance_comparison.png')
    
    # Processing time comparison
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, avg_times, color='skyblue')
    plt.xlabel('Method')
    plt.ylabel('Average Processing Time (ms)')
    plt.title('Processing Time Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('comparison_results/time_comparison.png')
    
    # Confusion matrices
    for metrics in metrics_list:
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Alert', 'Drowsy'], yticklabels=['Alert', 'Drowsy'])
        plt.title(f'Confusion Matrix - {metrics["model_name"]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'comparison_results/confusion_matrix_{metrics["model_name"].replace(" ", "_").lower()}.png')
    
    # Save metrics to CSV
    with open('comparison_results/metrics_summary.csv', 'w') as f:
        f.write('Method,Accuracy,Precision,Recall,F1 Score,Avg Time (ms)\n')
        for i, name in enumerate(model_names):
            f.write(f'{name},{accuracies[i]:.4f},{precisions[i]:.4f},{recalls[i]:.4f},{f1_scores[i]:.4f},{avg_times[i]:.2f}\n')
    
    print(f"Results saved to comparison_results directory")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compare Driver Drowsiness Detection Methods")
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Directory containing test data")
    parser.add_argument("--ml_model", type=str, default="drowsiness_model.pkl",
                        help="Path to ML model")
    parser.add_argument("--cnn_model", type=str, default="drowsiness_cnn.pth",
                        help="Path to CNN model")
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from {args.data_dir}...")
    test_images, test_labels = load_test_data(args.data_dir)
    
    if len(test_images) == 0:
        print("Error: No test data found. Please collect data first.")
        return
    
    print(f"Loaded {len(test_images)} test images")
    print(f"Alert images: {test_labels.count(0)}")
    print(f"Drowsy images: {test_labels.count(1)}")
    
    # Initialize metrics list
    metrics_list = []
    
    # Evaluate traditional method
    print("\nEvaluating Traditional Method...")
    traditional_predict = lambda img: predict_traditional(img)
    traditional_metrics = evaluate_model("Traditional (EAR)", traditional_predict, test_images, test_labels)
    metrics_list.append(traditional_metrics)
    print(f"Accuracy: {traditional_metrics['accuracy']:.4f}")
    
    # Evaluate ML method
    print("\nEvaluating Machine Learning Method...")
    ml_model, scaler = load_ml_model(args.ml_model)
    if ml_model is not None:
        ml_predict = lambda img: predict_ml(img, ml_model, scaler)
        ml_metrics = evaluate_model("Machine Learning", ml_predict, test_images, test_labels)
        metrics_list.append(ml_metrics)
        print(f"Accuracy: {ml_metrics['accuracy']:.4f}")
    
    # Evaluate CNN method
    print("\nEvaluating CNN Method...")
    cnn_model, device = load_cnn_model(args.cnn_model)
    if cnn_model is not None:
        cnn_predict = lambda img: predict_cnn(img, cnn_model, device)
        cnn_metrics = evaluate_model("CNN", cnn_predict, test_images, test_labels)
        metrics_list.append(cnn_metrics)
        print(f"Accuracy: {cnn_metrics['accuracy']:.4f}")
    
    # Visualize results
    if metrics_list:
        print("\nVisualizing results...")
        visualize_results(metrics_list)
    
    print("\nComparison complete.")

if __name__ == "__main__":
    main() 