import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("absl").setLevel(logging.ERROR)
import contextlib
import cv2
import torch
import mediapipe as mp
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# CONFIGURATION
SEQUENCE_LENGTH = 30
DATASET_PATH = "dataset"
MODEL_PATH = "fall_model.pth"
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# MODEL
class FallLSTM(nn.Module):
    def __init__(self, input_size=99, hidden_size=128, num_classes=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


# POSE INITIALIZATION

mp_pose = mp.solutions.pose

# Suppress MediaPipe backend logs
with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
    pose = mp_pose.Pose()


# KEYPOINT EXTRACTION

def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            frames.append(keypoints)

    cap.release()

    frames = np.array(frames)

    sequences = []
    for i in range(len(frames) - SEQUENCE_LENGTH):
        sequences.append(frames[i:i+SEQUENCE_LENGTH])

    return np.array(sequences)
# TRAIN MODEL

def train_model():

    print("Loading dataset...")
    video_data = []
    video_labels = []

    for label, folder in enumerate(["normal", "fall"]):
        folder_path = os.path.join(DATASET_PATH, folder)

        for file in os.listdir(folder_path):
            if file.lower().endswith((".mp4", ".mov", ".avi")):
                video_path = os.path.join(folder_path, file)
                print("Processing:", video_path)

                sequences = extract_keypoints(video_path)

                if len(sequences) > 0:
                    video_data.append(sequences)
                    video_labels.append(label)

    # Split by video 
    train_videos, test_videos, train_labels, test_labels = train_test_split(
        video_data, video_labels, test_size=0.2, random_state=42
    )

    X_train, y_train = [], []
    X_test, y_test = [], []

    for sequences, label in zip(train_videos, train_labels):
        for seq in sequences:
            X_train.append(seq)
            y_train.append(label)

    for sequences, label in zip(test_videos, test_labels):
        for seq in sequences:
            X_test.append(seq)
            y_test.append(label)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Normalize per sequence
    X_train = (X_train - X_train.mean(axis=2, keepdims=True)) / \
              (X_train.std(axis=2, keepdims=True) + 1e-6)

    X_test = (X_test - X_test.mean(axis=2, keepdims=True)) / \
             (X_test.std(axis=2, keepdims=True) + 1e-6)

    X_train = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)

    X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = FallLSTM().to(device)

    # Improve fall recall with class weighting
    class_weights = torch.tensor([1.0, 1.3]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # TRAINING LOOP

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")
    # EVALUATION

    model.eval()
    with torch.inference_mode():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_test).float().mean()

    print("\nReal Test Accuracy:", accuracy.item())

    y_true = y_test.cpu().numpy()
    y_pred = predictions.cpu().numpy()

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved.")

# REAL-TIME DEMO

def realtime_demo():

    model = FallLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)
    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam. Please ensure your terminal has Camera permissions in System Settings > Privacy & Security > Camera.")
            break

        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]

            if len(sequence) == SEQUENCE_LENGTH:

                input_data = np.array(sequence)

                input_data = (input_data - input_data.mean(axis=1, keepdims=True)) / \
                             (input_data.std(axis=1, keepdims=True) + 1e-6)

                input_data = np.expand_dims(
                    input_data.astype(np.float32), axis=0
                )

                input_tensor = torch.from_numpy(input_data).to(device)

                with torch.inference_mode():
                    prediction = model(input_tensor)
                    predicted_class = torch.argmax(prediction, dim=1).item()

                if predicted_class == 1:
                    cv2.putText(frame, "FALL DETECTED",
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, "NORMAL",
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 3)

        cv2.imshow("Fall Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# MAIN

if __name__ == "__main__":
    print("1 - Train Model")
    print("2 - Run Real-time Demo")

    choice = input("Select option: ")

    if choice == "1":
        train_model()
    elif choice == "2":
        realtime_demo()
