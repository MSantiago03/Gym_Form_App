import mediapipe as mp
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms


from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("************")
print(device)

SEQUENCE_LENGTH = 30
pose_sequences = []
labels = []  # Use this for supervised training (0 = bad, 1 = good)




def calculate_angle(a, b, c):
    """
    Calculates the angle at point b given points a, b, and c.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)



# Define transformations for image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')


## initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Feed Video MP4
cap = cv2.VideoCapture('/Users/manuelsantiago/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/DeepLearning/Final_Project/Gym_Form_App/Videos/jack_demo.mp4')

while cap.isOpened():
    # read frame
    _, frame = cap.read()
    try:
        # resize the frame for portrait video
        frame = cv2.resize(frame, (350, 600))
        # convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # process the frame for pose detection
        pose_results = pose.process(frame_rgb)
        # print("##### Printing Landmarks Below #####")
        # print(pose_results.pose_landmarks)

        if pose_results.pose_landmarks:
            landmarks = []
            h, w, _ = frame.shape  # image dimensions

            for landmark in pose_results.pose_landmarks.landmark:
                x_px = landmark.x * w
                y_px = landmark.y * h
                z = landmark.z  # z is still relative (can be left as is)
                landmarks.append([x_px, y_px, z])

            landmarks_array = np.array(landmarks)

            shoulder = landmarks_array[11]  # LEFT SHOULDER
            elbow = landmarks_array[13]     # LEFT ELBOW
            waist = landmarks_array[23]

            distance = np.linalg.norm(shoulder - elbow)
            angle = calculate_angle(elbow, shoulder, waist)

            print(f"Distance between shoulder and elbow: {distance} pixels")
            print(f"Angle between shoulder and elbow: {angle} degrees")

            print(landmarks_array.shape)  # Should be (33, 3) -> 33 landmarks, x, y, z each
            # print(landmarks_array)  # Shows all landmarks - landmarks stores landmark coords
        
        # draw skeleton on the frame
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # display the frame
        cv2.imshow('Output', frame)
    except:
        break
        
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




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
