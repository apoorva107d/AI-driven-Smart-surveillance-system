
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Constants
data_dir = "violence"  # Path to your video dataset folder
output_dir = "violence_frames"  # Path to save extracted frames
categories = ["aggressive", "non_aggressive"]  # Categories in your dataset
img_size = (128, 128)  # Size to resize images for training

# Create directories to save extracted frames
os.makedirs(output_dir, exist_ok=True)
for category in categories:
    os.makedirs(os.path.join(output_dir, category), exist_ok=True)

# Function to extract frames from videos and save them as images
def extract_frames(data_dir, categories):
    for category in categories:
        input_path = os.path.join(data_dir, category)
        output_path = os.path.join(output_dir, category)
        
        for video_name in os.listdir(input_path):
            video_path = os.path.join(input_path, video_name)
            cap = cv2.VideoCapture(video_path)
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(output_path, f"{os.path.splitext(video_name)[0]}_frame{frame_idx}.jpg"), frame)
                frame_idx += 1
            cap.release()

# Function to load frames and labels
def load_images(data_dir, categories, img_size=(128, 128)):
    X = []
    y = []
    
    for category in categories:
        label = categories.index(category)
        category_path = os.path.join(data_dir, category)
        
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(label)
    
    return np.array(X), np.array(y)

# Extract frames from videos and save them
print("Extracting frames from videos...")
extract_frames(data_dir, categories)

# Load frames and labels
print("Loading frames and labels...")
X, y = load_images(output_dir, categories, img_size)

# Normalize image data
X = X / 255.0

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten images
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Train the model using Random Forest
print("Training model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_flat, y_train)

# Save the trained model (to avoid retraining)
joblib.dump(clf, "violence_detector_model.pkl")
print("Model trained and saved.")

# Evaluate the model
print("Evaluating the model...")
from sklearn.metrics import accuracy_score, classification_report
y_pred = clf.predict(X_test_flat)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

print("Training and evaluation completed.")
