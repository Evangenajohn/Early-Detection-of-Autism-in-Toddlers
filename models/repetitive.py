import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp

# Load the pretrained repetitive gesture model
model = load_model('models/repetitive_gesture_model.h5')

mp_pose = mp.solutions.pose

def extract_keypoints_from_video(video_path, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False)
    keypoints_seq = []

    while cap.isOpened() and len(keypoints_seq) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = []
            for idx in [15, 16, 13, 14]:  # wrists and elbows
                lm = landmarks[idx]
                keypoints.extend([lm.x, lm.y, lm.z])
            keypoints_seq.append(keypoints)
        else:
            keypoints_seq.append([0]*12)

    cap.release()
    pose.close()

    # Pad or trim to max_frames
    seq = np.array(keypoints_seq)
    if len(seq) < max_frames:
        padding = np.zeros((max_frames - len(seq), seq.shape[1]))
        seq = np.vstack([seq, padding])
    else:
        seq = seq[:max_frames]
    return np.expand_dims(seq, axis=0)  # (1, seq_len, features)

def predict_repetitive_movement(video_path):
    keypoints_seq = extract_keypoints_from_video(video_path)
    pred = model.predict(keypoints_seq)
    return float(pred[0][0])  # probability between 0 and 1
