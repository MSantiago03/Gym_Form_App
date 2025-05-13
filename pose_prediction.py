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
    """Returns GPU if available and desired, otherwise falls back to CPU."""
    if use_gpu and torch.cuda.is_available():
        print("✅ Using GPU (CUDA)")
        return torch.device("cuda")
    else:
        print("⚠️ Using CPU")
        return torch.device("cpu")

# -------------------------------
# RNN MODEL DEFINITION
# -------------------------------
class FormRNN(nn.Module):
    """LSTM-based binary classifier for pose sequences."""
    def __init__(self, input_size=99, hidden_size=64, num_layers=1, num_classes=2):
        super(FormRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # Take the final hidden state
        out = self.dropout(hn[-1])  # Apply dropout to last layer output
        return self.classifier(out)  # Pass through linear layer

# -------------------------------
# KEYPOINT NORMALIZATION
# -------------------------------
def extract_normalized_keypoints(results) -> np.ndarray:
    """
    Extracts pose landmarks and applies per-frame normalization
    (zero-mean, unit-variance) to make the model invariant to position and scale.
    """
    keypoints = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
    keypoints = np.array(keypoints)
    mean = keypoints.mean(axis=0)
    std = keypoints.std(axis=0) + 1e-6  # Prevent division by zero
    normalized = (keypoints - mean) / std
    return normalized.flatten()

# -------------------------------
# DATA COLLECTION FROM VIDEO
# -------------------------------
def collect_pose_sequences_from_video(video_path: str, sequence_length: int = 30, label: int = 1) -> Tuple[List[np.ndarray], List[int]]:
    """
    Reads a video, extracts pose sequences of a fixed length,
    and returns them along with their corresponding label.
    """
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

            # Once we have a full sequence, store it
            if len(pose_sequences) == sequence_length:
                sample = np.array(pose_sequences)
                X_data.append(sample)
                y_data.append(label)
                pose_sequences = []  # Reset for next sample

            # Optionally visualize pose (not needed for headless training)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Pose Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return X_data, y_data

# -------------------------------
# TRAINING LOOP
# -------------------------------
def train_rnn_model(X: np.ndarray, y: np.ndarray, device: torch.device, epochs: int = 10, batch_size: int = 8, lr: float = 1e-3) -> FormRNN:
    """
    Trains the RNN on the provided dataset.
    """
    model = FormRNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare data for PyTorch
    tensor_x = torch.tensor(X, dtype=torch.float32).to(device)
    tensor_y = torch.tensor(y, dtype=torch.long).to(device)
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses, accuracies = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

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

            # Show example predictions on first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                probs = torch.nn.functional.softmax(preds, dim=1)
                print("🔍 Example predictions:")
                for i in range(min(3, len(probs))):
                    print(f"  Predicted: {predicted[i].item()} | Confidence: {probs[i][predicted[i]].item():.4f} | True: {yb[i].item()}")

        # Epoch summary
        accuracy = correct / total
        losses.append(total_loss)
        accuracies.append(accuracy)
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {accuracy:.2%}")

    # Plot training history
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
    # plt.show()  # Uncomment to display plots

    return model

# -------------------------------
# LOAD DATA FROM DIRECTORY
# -------------------------------
def load_labeled_data_from_dir(data_dir: str, label: int) -> Tuple[List[np.ndarray], List[int]]:
    """
    Loads and processes all .mp4 videos in the directory into sequences + labels.
    """
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
# TRAINING WRAPPER PER EXERCISE
# -------------------------------
def train_exercise_model(exercise_name: str, good_dir: str, bad_dir: str, device: torch.device) -> None:
    """
    Trains and saves an RNN model for a specific exercise (e.g., pushup or squat).
    """
    print(f"\n--- Training {exercise_name.capitalize()} Model ---")
    X_good, y_good = load_labeled_data_from_dir(good_dir, 1)
    X_bad, y_bad = load_labeled_data_from_dir(bad_dir, 0)

    # Combine positive and negative examples
    X_total = X_good + X_bad
    y_total = y_good + y_bad

    if len(X_total) == 0:
        raise ValueError(f"No training data found for {exercise_name}.")

    model = train_rnn_model(np.array(X_total), np.array(y_total), device)
    model_path = f"form_rnn_{exercise_name.lower()}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ {exercise_name.capitalize()} model saved as '{model_path}'")

# -------------------------------
# MAIN ENTRY POINT
# -------------------------------
def main():
    """
    Entry point to train one or more exercise models.
    """
    device = get_device(use_gpu=True)

    # Uncomment this block to train the pushup model
    # train_exercise_model(
    #     exercise_name="pushup",
    #     good_dir="Videos/Push_Up/good_push_ups",
    #     bad_dir="Videos/Push_Up/bad_push_ups",
    #     device=device
    # )

    # Train the squat model
    train_exercise_model(
        exercise_name="squat",
        good_dir="Videos/Squat/good_squats",
        bad_dir="Videos/Squat/bad_squats",
        device=device
    )

if __name__ == "__main__":
    main()
