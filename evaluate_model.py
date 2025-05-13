import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from typing import List

# -------------------------------
# Constants
# -------------------------------
SEQUENCE_LENGTH = 30  # Number of frames to feed into the model
LABEL_MAP = {0: "Bad Form", 1: "Good Form"}  # Output labels for classification

# -------------------------------
# Model Definition
# -------------------------------
class FormRNN(nn.Module):
    """Simple LSTM-based model for classifying exercise form sequences."""
    def __init__(self, input_size=99, hidden_size=64, num_layers=1, num_classes=2):
        super(FormRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.classifier(hn[-1])  # Use last hidden state

# -------------------------------
# Helper Functions
# -------------------------------
def extract_normalized_keypoints(results) -> np.ndarray:
    """
    Extracts and flattens the normalized 3D pose landmarks into a single vector.
    """
    keypoints = [
        [lm.x, lm.y, lm.z]
        for lm in results.pose_landmarks.landmark
    ]
    return np.array(keypoints).flatten()

def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Loads the RNN model from file and sends it to the correct device.
    """
    model = FormRNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def setup_pose_estimator():
    """
    Initializes MediaPipe's pose estimation pipeline.
    """
    return mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# -------------------------------
# Main Evaluation Function
# -------------------------------
def evaluate_video(video_path: str, model_path: str, label_map: dict = LABEL_MAP) -> None:
    """
    Evaluates a video file using pose estimation and an RNN model to classify form quality.
    Displays annotated video in a pop-up window.
    """
    # Setup device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    # Initialize MediaPipe Pose and drawing utils
    pose = setup_pose_estimator()
    mp_drawing = mp.solutions.drawing_utils

    # Open video
    cap = cv2.VideoCapture(video_path)
    pose_sequence: List[np.ndarray] = []
    last_prediction = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Extract keypoints if detected
        if results.pose_landmarks:
            keypoints = extract_normalized_keypoints(results)
            pose_sequence.append(keypoints)

            # Maintain sliding window of SEQUENCE_LENGTH
            if len(pose_sequence) > SEQUENCE_LENGTH:
                pose_sequence.pop(0)

            frame_count += 1

            # Predict once full sequence is ready (every 10 frames)
            if len(pose_sequence) == SEQUENCE_LENGTH and frame_count % 10 == 0:
                sequence_tensor = torch.tensor([pose_sequence], dtype=torch.float32).to(device)
                with torch.no_grad():
                    output = model(sequence_tensor)
                    prediction = torch.argmax(output, dim=1).item()
                    last_prediction = label_map[prediction]
                    pose_sequence = []  # Reset after prediction

        # Draw results
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        if last_prediction:
            color = (0, 255, 0) if last_prediction == "Good Form" else (0, 0, 255)
            cv2.putText(frame, last_prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # Display frame
        cv2.imshow('Evaluation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# Entry Point for Manual Testing
# -------------------------------
if __name__ == "__main__":
    test_model_path = "models/form_rnn_squat.pth"

    test_videos = [
        "Videos/Squat/test_squats/test_squat_good_1.mp4",
        "Videos/Squat/test_squats/test_squat_good_2.mp4",
        "Videos/Squat/test_squats/test_squat_bad_1.mp4",
        "Videos/Squat/test_squats/test_squat_bad_2.mp4"
    ]

    for test_video_path in test_videos:
        evaluate_video(test_video_path, test_model_path)
