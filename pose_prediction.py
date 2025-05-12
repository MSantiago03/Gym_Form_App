import mediapipe as mp
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple
import matplotlib.pyplot as plt
import os

# -------------------------------
# DEVICE CONFIGURATION
# -------------------------------
def get_device(use_gpu: bool = True) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        print("✅ Using GPU (CUDA)")
        return torch.device("cuda")
    else:
        print("⚠️ Using CPU")
        return torch.device("cpu")

# -------------------------------
# MODEL
# -------------------------------
class FormRNN(nn.Module):
    def __init__(self, input_size: int = 99, hidden_size: int = 64, num_layers: int = 1, num_classes: int = 2) -> None:
        super(FormRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        return self.classifier(out)

# -------------------------------
# POSE DATA EXTRACTION
# -------------------------------
def extract_normalized_keypoints(results) -> np.ndarray:
    keypoints = [
        [lm.x, lm.y, lm.z]
        for lm in results.pose_landmarks.landmark
    ]
    keypoints = np.array(keypoints)
    mean = keypoints.mean(axis=0)
    std = keypoints.std(axis=0) + 1e-6
    normalized = (keypoints - mean) / std
    return normalized.flatten()

# -------------------------------
# DATA COLLECTION FROM VIDEO
# -------------------------------
def collect_pose_sequences_from_video(video_path: str, sequence_length: int = 30, label: int = 1) -> Tuple[List[np.ndarray], List[int]]:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    pose_sequences, X_data, y_data = [], [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            frame_vector = extract_normalized_keypoints(results)
            pose_sequences.append(frame_vector)

            if len(pose_sequences) == sequence_length:
                sample = np.array(pose_sequences)
                X_data.append(sample)
                y_data.append(label)
                pose_sequences = []

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Pose Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return X_data, y_data

# -------------------------------
# TRAINING
# -------------------------------
def train_rnn_model(X: np.ndarray, y: np.ndarray, device: torch.device, epochs: int = 10, batch_size: int = 8, lr: float = 1e-3) -> FormRNN:
    model = FormRNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    tensor_x = torch.tensor(X, dtype=torch.float32).to(device)
    tensor_y = torch.tensor(y, dtype=torch.long).to(device)
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (xb, yb) in enumerate(loader):
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

            if epoch == 0 and batch_idx == 0:
                probs = torch.nn.functional.softmax(preds, dim=1)
                print("🔍 Example predictions:")
                for i in range(min(3, len(probs))):
                    print(f"  Predicted: {predicted[i].item()} | Confidence: {probs[i][predicted[i]].item():.4f} | True: {yb[i].item()}")

        accuracy = correct / total
        losses.append(total_loss)
        accuracies.append(accuracy)
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {accuracy:.2%}")

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")

    plt.tight_layout()
    # plt.show()

    return model

# -------------------------------
# DATA LOADING (SIMPLIFIED)
# -------------------------------
def load_labeled_data_from_dir(data_dir: str, label: int) -> Tuple[List[np.ndarray], List[int]]:
    X_total, y_total = [], []

    if not os.path.isdir(data_dir):
        return X_total, y_total

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".mp4"):
            video_path = os.path.join(data_dir, filename)
            print(f"📂 Processing: {video_path}")
            X_data, y_data = collect_pose_sequences_from_video(video_path, label=label)
            if len(X_data) == 0:
                print(f"⚠️ Warning: No data extracted from {video_path}")
                continue
            X_total.extend(X_data)
            y_total.extend(y_data)

    return X_total, y_total

# -------------------------------
# EXERCISE TRAINING WRAPPER
# ------------------------------- 
def train_exercise_model(exercise_name: str, good_dir: str, bad_dir: str, device: torch.device) -> None:
    print(f"\n--- Training {exercise_name.capitalize()} Model ---")
    X_good, y_good = load_labeled_data_from_dir(good_dir, 1)
    X_bad, y_bad = load_labeled_data_from_dir(bad_dir, 0)
    X_total = X_good + X_bad
    y_total = y_good + y_bad

    if len(X_total) == 0:
        raise ValueError(f"No training data found for {exercise_name}.")

    model = train_rnn_model(np.array(X_total), np.array(y_total), device)
    model_path = f"form_rnn_{exercise_name.lower()}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ {exercise_name.capitalize()} model saved as '{model_path}'")

# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    device = get_device(use_gpu=True)

    # Push-up model
    # train_exercise_model(
    #     exercise_name="pushup",
    #     good_dir="Videos/Push_Up/good_push_ups",
    #     bad_dir="Videos/Push_Up/bad_push_ups",
    #     device=device
    # )

    # Squat model
    train_exercise_model(
        exercise_name="squat",
        good_dir="Videos/Squat/good_squats",
        bad_dir="Videos/Squat/bad_squats",
        device=device
    )

if __name__ == "__main__":
    main()
