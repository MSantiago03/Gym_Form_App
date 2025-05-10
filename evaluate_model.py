import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List
import mediapipe as mp

# Constants
SEQUENCE_LENGTH = 30
MODEL_PATH = "form_rnn_pushup.pth"  # Update as needed
LABEL_MAP = {0: "Bad Form", 1: "Good Form"}

# -------------------------------
# MODEL DEFINITION
# -------------------------------
class FormRNN(nn.Module):
    def __init__(self, input_size: int = 99, hidden_size: int = 64, num_layers: int = 1, num_classes: int = 2) -> None:
        super(FormRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        return self.classifier(hn[-1])

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def extract_normalized_keypoints(results) -> np.ndarray:
    keypoints = [
        [lm.x, lm.y, lm.z]
        for lm in results.pose_landmarks.landmark
    ]
    return np.array(keypoints).flatten()

def evaluate_video(video_path: str, model_path: str = MODEL_PATH, label_map: dict = LABEL_MAP) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FormRNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    pose_sequence: List[np.ndarray] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            keypoints = extract_normalized_keypoints(results)
            pose_sequence.append(keypoints)

            if len(pose_sequence) == SEQUENCE_LENGTH:
                sequence_tensor = torch.tensor([pose_sequence], dtype=torch.float32).to(device)
                with torch.no_grad():
                    output = model(sequence_tensor)
                    prediction = torch.argmax(output, dim=1).item()
                    label = label_map[prediction]

                cv2.putText(frame, f"{label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 0) if prediction == 1 else (0, 0, 255), 3)
                pose_sequence = []

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Evaluation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    # Replace with your actual test video path
    test_video_path = "Videos/Push_Up/test_push_up/test_good_push_up.mp4"
    test_model_path = "form_rnn_pushup_good.pth"
    evaluate_video(test_video_path, test_model_path)
    test_video_path = "Videos/Push_Up/test_push_up/test_bad_push_up.mp4"
    evaluate_video(test_video_path, test_model_path)
