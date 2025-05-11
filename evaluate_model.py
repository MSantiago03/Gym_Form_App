import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List
import mediapipe as mp

from collections import deque, Counter


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
    frame_count = 0
    last_prediction = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            keypoints = extract_normalized_keypoints(results)
            pose_sequence.append(keypoints)

            if len(pose_sequence) > SEQUENCE_LENGTH:
                pose_sequence.pop(0)  # sliding window

            frame_count += 1

            # Predict every N frames
            if len(pose_sequence) == SEQUENCE_LENGTH and frame_count % 10 == 0:
                sequence_tensor = torch.tensor([pose_sequence], dtype=torch.float32).to(device)
                with torch.no_grad():
                    output = model(sequence_tensor)
                    prediction = torch.argmax(output, dim=1).item()
                    last_prediction = label_map[prediction]

            # Only display if we've made at least one prediction
            if last_prediction:
                color = (0, 255, 0) if last_prediction == "Good Form" else (0, 0, 255)
                cv2.putText(frame, last_prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

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
    test_video_path = "Videos/Push_Up/test_push_up/test_bad_push_up_2.mp4"
    test_model_path = "form_rnn_pushup99.pth"
    evaluate_video(test_video_path, test_model_path)
    test_video_path = "Videos/Push_Up/test_push_up/test_push_up_bad_3.mp4"
    evaluate_video(test_video_path, test_model_path)
    test_video_path = "Videos/Push_Up/test_push_up/test_good_3.mp4"
    evaluate_video(test_video_path, test_model_path)
    test_video_path = "Videos/Push_Up/test_push_up/test_good_4.mp4"
    evaluate_video(test_video_path, test_model_path)
    test_video_path = "Videos/Push_Up/test_push_up/test_good_6.mp4"
    evaluate_video(test_video_path, test_model_path)


# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# from typing import List
# import mediapipe as mp

# from collections import deque, Counter

# # Constants
# SEQUENCE_LENGTH = 30
# MODEL_PATH = "form_rnn_pushup.pth"  # Update as needed
# LABEL_MAP = {0: "Bad Form", 1: "Good Form"}

# # -------------------------------
# # MODEL DEFINITION
# # -------------------------------
# class FormRNN(nn.Module):
#     def __init__(self, input_size: int = 6, hidden_size: int = 64, num_layers: int = 1, num_classes: int = 2) -> None:
#         super(FormRNN, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.classifier = nn.Linear(hidden_size, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         _, (hn, _) = self.lstm(x)
#         return self.classifier(hn[-1])

# # -------------------------------
# # ANGLE-BASED FEATURE EXTRACTION
# # -------------------------------
# def get_angle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#     ba = a - b
#     bc = c - b
#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
#     return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# def extract_form_features(results) -> np.ndarray:
#     landmarks = results.pose_landmarks.landmark
#     get_point = lambda i: [landmarks[i].x, landmarks[i].y, landmarks[i].z]

#     left_elbow_angle = get_angle(get_point(11), get_point(13), get_point(15))
#     right_elbow_angle = get_angle(get_point(12), get_point(14), get_point(16))
#     left_knee_angle = get_angle(get_point(23), get_point(25), get_point(27))
#     right_knee_angle = get_angle(get_point(24), get_point(26), get_point(28))
#     back_alignment = get_angle(get_point(11), get_point(23), get_point(27))

#     shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
#     wrist_y = (landmarks[15].y + landmarks[16].y) / 2
#     shoulder_to_wrist_dist = abs(shoulder_y - wrist_y)

#     return np.array([
#         left_elbow_angle, right_elbow_angle,
#         left_knee_angle, right_knee_angle,
#         back_alignment, shoulder_to_wrist_dist
#     ])

# # -------------------------------
# # EVALUATE VIDEO
# # -------------------------------
# def evaluate_video(video_path: str, model_path: str = MODEL_PATH, label_map: dict = LABEL_MAP) -> None:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = FormRNN()
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval().to(device)

#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#     mp_drawing = mp.solutions.drawing_utils

#     cap = cv2.VideoCapture(video_path)
#     pose_sequence: List[np.ndarray] = []
#     frame_count = 0
#     last_prediction = None

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)

#         if results.pose_landmarks:
#             keypoints = extract_form_features(results)
#             pose_sequence.append(keypoints)

#             if len(pose_sequence) > SEQUENCE_LENGTH:
#                 pose_sequence.pop(0)

#             frame_count += 1

#             if len(pose_sequence) == SEQUENCE_LENGTH and frame_count % 10 == 0:
#                 sequence_tensor = torch.tensor([pose_sequence], dtype=torch.float32).to(device)
#                 with torch.no_grad():
#                     output = model(sequence_tensor)
#                     prediction = torch.argmax(output, dim=1).item()
#                     last_prediction = label_map[prediction]

#             if last_prediction:
#                 color = (0, 255, 0) if last_prediction == "Good Form" else (0, 0, 255)
#                 cv2.putText(frame, last_prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

#             mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         cv2.imshow('Evaluation', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # -------------------------------
# # ENTRY POINT
# # -------------------------------
# if __name__ == "__main__":
#     test_video_path = "Videos/Push_Up/test_push_up/test_bad_push_up_2.mp4"
#     test_model_path = "form_rnn_pushup.pth"
#     evaluate_video(test_video_path, test_model_path)

#     test_video_path = "Videos/Push_Up/test_push_up/test_good_2.mp4"
#     evaluate_video(test_video_path, test_model_path)
