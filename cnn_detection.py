import cv2
import numpy as np
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from alarm import DrowsinessAlarm
from dashboard import DrowsinessDashboard

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the CNN model
class DrowsinessDetectionCNN(nn.Module):
    def __init__(self):
        super(DrowsinessDetectionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # First convolutional block
        x = self.pool(self.relu(self.conv1(x)))
        
        # Second convolutional block
        x = self.pool(self.relu(self.conv2(x)))
        
        # Third convolutional block
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# Define dataset class
class EyeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = ['alert', 'drowsy']
        
        # Load images and labels
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} not found")
                continue
                
            for img_name in os.listdir(class_dir):
                if not (img_name.endswith('.jpg') or img_name.endswith('.png')):
                    continue
                    
                img_path = os.path.join(class_dir, img_name)
                self.images.append(img_path)
                self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_model(data_dir, model_path, num_epochs=10, batch_size=32):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    dataset = EyeDataset(data_dir, transform=transform)
    
    # Check if dataset is empty
    if len(dataset) == 0:
        print(f"Error: No valid images found in {data_dir}")
        return None
        
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = DrowsinessDetectionCNN().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Starting training...")
    
    # Lists to store losses and accuracies
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_dataset)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_loss = running_loss / len(val_dataset)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Save the model
    print(f"Saving model to {model_path}...")
    torch.save(model.state_dict(), model_path)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(model_path), 'training_curves.png'))
    plt.close()
    
    return model

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CNN Drowsiness Detection")
    parser.add_argument("--model", type=str, default="drowsiness_cnn.pth",
                        help="Path to the trained model")
    parser.add_argument("--train", action="store_true",
                        help="Train a new model")
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Directory containing training data")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
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
        model = train_model(args.data_dir, args.model, args.epochs, args.batch_size)
        if model is None:
            return
    else:
        # Load the pre-trained model
        if not os.path.exists(args.model):
            print(f"Error: Model file {args.model} not found")
            print("Please train a model first with --train")
            return
        
        print(f"Loading model from {args.model}...")
        model = DrowsinessDetectionCNN().to(device)
        model.load_state_dict(torch.load(args.model))
        model.eval()
    
    # Initialize the alarm
    alarm = DrowsinessAlarm(args.alarm_file)
    
    # Initialize the dashboard if enabled
    dashboard = DrowsinessDashboard() if args.show_dashboard else None
    
    # Initialize transform for inference
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
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
                
                # Skip if eye region is too small
                if eye_roi.shape[0] < 10 or eye_roi.shape[1] < 10:
                    continue
                
                # Preprocess eye region for the model
                eye_tensor = transform(eye_roi).unsqueeze(0).to(device)
                
                # Make prediction
                with torch.no_grad():
                    outputs = model(eye_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    class_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[class_idx].item()
                
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
        cv2.imshow("CNN Drowsiness Detection", frame)
        
        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 