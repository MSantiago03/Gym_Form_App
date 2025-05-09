import mediapipe as mp
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("************")
print(device)

SEQUENCE_LENGTH = 30

# -------------------------------
# MODEL
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
# POSE DATA EXTRACTION
# -------------------------------
def extract_normalized_keypoints(results) -> np.ndarray:
    keypoints = [
        [lm.x, lm.y, lm.z]
        for lm in results.pose_landmarks.landmark
    ]
    return np.array(keypoints).flatten()

# -------------------------------
# DATA COLLECTION FROM VIDEO
# -------------------------------
def collect_pose_sequences_from_video(video_path: str, sequence_length: int = SEQUENCE_LENGTH, label: int = 1) -> Tuple[List[np.ndarray], List[int]]:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    pose_sequences, X_data, y_data = [], [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)

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
def train_rnn_model(X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 8, lr: float = 1e-3) -> FormRNN:
    model = FormRNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    tensor_x = torch.tensor(X, dtype=torch.float32).to(device)
    tensor_y = torch.tensor(y, dtype=torch.long).to(device)
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    return model

# -------------------------------
# EVALUATION
# -------------------------------
def evaluate_video(video_path: str, model_path: str, label_map: dict = {0: "Bad Form", 1: "Good Form"}) -> None:
    model = FormRNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    pose_sequence = []

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
# MAIN PIPELINE
# -------------------------------
def main():
    # Pushup training videos with labels
    pushup_videos = [
        ("Videos/Push_Up/good_push_up.mp4", 1),
        ("Videos/Push_Up/bad_push_up.mp4", 0),
    ]

    # Collect training data from both
    X_total, y_total = [], []
    for video_path, label in pushup_videos:
        X_data, y_data = collect_pose_sequences_from_video(video_path, label=label)
        X_total.extend(X_data)
        y_total.extend(y_data)

    # Save data for inspection/debugging
    np.save("X_data_pushup.npy", np.array(X_total))
    np.save("y_data_pushup.npy", np.array(y_total))

    # Train one model for pushup classification
    model = train_rnn_model(np.array(X_total), np.array(y_total))
    torch.save(model.state_dict(), "form_rnn_pushup.pth")

    # Test it on new videos
    print("\nEvaluating trained pushup model...\n")
    evaluate_video("Videos/Push_Up/test_good_push_up.mp4", "form_rnn_pushup.pth")
    evaluate_video("Videos/Push_Up/test_bad_push_up.mp4", "form_rnn_pushup.pth")

if __name__ == "__main__":
    main()

# import mediapipe as mp
# import numpy as np
# import cv2
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# from typing import List, Tuple

# # Device config
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("************")
# print(device)

# SEQUENCE_LENGTH = 30

# # -------------------------------
# # MODEL (Per-exercise)
# # -------------------------------
# class FormRNN(nn.Module):
#     def __init__(self, input_size: int = 99, hidden_size: int = 64, num_layers: int = 1, num_classes: int = 2) -> None:
#         super(FormRNN, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.classifier = nn.Linear(hidden_size, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         _, (hn, _) = self.lstm(x)
#         return self.classifier(hn[-1])

# # -------------------------------
# # DATA COLLECTION
# # -------------------------------
# def extract_normalized_keypoints(results) -> np.ndarray:
#     """Extract normalized (x, y, z) keypoints from MediaPipe results."""
#     keypoints = [
#         [lm.x, lm.y, lm.z]
#         for lm in results.pose_landmarks.landmark
#     ]
#     return np.array(keypoints).flatten()  # Shape: (99,)

# def collect_pose_sequences_from_video(video_path: str, sequence_length: int = SEQUENCE_LENGTH, label: int = 1) -> Tuple[List[np.ndarray], List[int]]:
#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#     mp_drawing = mp.solutions.drawing_utils

#     cap = cv2.VideoCapture(video_path)
#     pose_sequences, X_data, y_data = [], [], []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame)

#         if results.pose_landmarks:
#             frame_vector = extract_normalized_keypoints(results)
#             pose_sequences.append(frame_vector)

#             if len(pose_sequences) == sequence_length:
#                 sample = np.array(pose_sequences)
#                 X_data.append(sample)
#                 y_data.append(label)
#                 pose_sequences = []  # Reset

#             mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         cv2.imshow('Pose Capture', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return X_data, y_data

# # -------------------------------
# # TRAINING
# # -------------------------------
# def train_rnn_model(X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 8, lr: float = 1e-3) -> FormRNN:
#     model = FormRNN().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     tensor_x = torch.tensor(X, dtype=torch.float32).to(device)
#     tensor_y = torch.tensor(y, dtype=torch.long).to(device)
#     dataset = TensorDataset(tensor_x, tensor_y)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0.0
#         for xb, yb in loader:
#             optimizer.zero_grad()
#             preds = model(xb)
#             loss = criterion(preds, yb)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

#     return model

# # -------------------------------
# # EVALUATION
# # -------------------------------
# def evaluate_video(video_path: str, model_path: str, label_map: dict = {0: "Bad Form", 1: "Good Form"}) -> None:
#     model = FormRNN()
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#     model.to(device)

#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#     mp_drawing = mp.solutions.drawing_utils

#     cap = cv2.VideoCapture(video_path)
#     pose_sequence = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)

#         if results.pose_landmarks:
#             keypoints = extract_normalized_keypoints(results)
#             pose_sequence.append(keypoints)

#             if len(pose_sequence) == SEQUENCE_LENGTH:
#                 sequence_tensor = torch.tensor([pose_sequence], dtype=torch.float32).to(device)
#                 with torch.no_grad():
#                     output = model(sequence_tensor)
#                     prediction = torch.argmax(output, dim=1).item()
#                     label = label_map[prediction]

#                 cv2.putText(frame, f"{label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                             1.5, (0, 255, 0) if prediction == 1 else (0, 0, 255), 3)
#                 pose_sequence = []

#             mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         cv2.imshow('Evaluation', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # -------------------------------
# # MAIN PIPELINE
# # -------------------------------
# def main():
#     # Train for multiple exercises separately
#     exercises = {
#         "pushup_good": ("Videos/Push_Up/good_push_up.mp4", 1),
#         "pushup_bad": ("Videos/Push_Up/bad_push_up.mp4", 0),
#     }

#     for name, (video_path, label) in exercises.items():
#         print(f"Collecting data for {name}")
#         X_data, y_data = collect_pose_sequences_from_video(video_path, label=label)
#         np.save(f"X_data_{name}.npy", np.array(X_data))
#         np.save(f"y_data_{name}.npy", np.array(y_data))

#         model = train_rnn_model(np.array(X_data), np.array(y_data))
#         torch.save(model.state_dict(), f"form_rnn_{name}.pth")

#     # Evaluation example
#     # Provide the path to evaluation video and the corresponding model
#     eval_model_path = "form_rnn_pushup_good.pth"  # change this as needed
#     eval_video_path = "Videos/Push_Up/test_good_push_up.mp4"  # change this as needed
#     evaluate_video(eval_video_path, eval_model_path)
#     eval_video_path_2 = "Videos/Push_Up/squat_test.mp4"
#     evaluate_video(eval_video_path_2, eval_model_path)

# if __name__ == "__main__":
#     main()


# Important Landmarks:
# Shoulders: L - 11, R - 12
# Elbows:L - 13, R - 14
# Wrists: L - 15, R - 16
# Waist: L - 23, R - 24


# # LIVE FEED VERSION
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     # read frame
#     _, frame = cap.read()
#     try:
#         # resize the frame for portrait video
#         # frame = cv2.resize(frame, (350, 600))
#         # convert to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # process the frame for pose detection
#         pose_results = pose.process(frame_rgb)
#         print("##### Printing Landmarks Below #####")
#         print(pose_results.pose_landmarks)

#         if pose_results.pose_landmarks:
#             landmarks = []
#             h, w, _ = frame.shape  # image dimensions

#             for landmark in pose_results.pose_landmarks.landmark:
#                 x_px = landmark.x * w
#                 y_px = landmark.y * h
#                 z = landmark.z  # z is still relative (can be left as is)
#                 landmarks.append([x_px, y_px, z])

#             landmarks_array = np.array(landmarks)

#             shoulder = landmarks_array[11]  # LEFT SHOULDER
#             elbow = landmarks_array[13]     # LEFT ELBOW

#             distance = np.linalg.norm(shoulder - elbow)
#             print(f"Distance between shoulder and elbow: {distance} pixels")

#             print(landmarks_array.shape)  # Should be (33, 3) -> 33 landmarks, x, y, z each
#             print(landmarks_array)  # Shows all landmarks

        
#         # draw skeleton on the frame
#         mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         # display the frame
#         cv2.imshow('Output', frame)
#     except:
#         break
        
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()