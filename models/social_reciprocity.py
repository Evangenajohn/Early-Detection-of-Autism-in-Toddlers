import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def detect_smile(landmarks):
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]

    mouth_width = np.linalg.norm(np.array([left_mouth.x, left_mouth.y]) - np.array([right_mouth.x, right_mouth.y]))
    mouth_height = np.linalg.norm(np.array([top_lip.x, top_lip.y]) - np.array([bottom_lip.x, bottom_lip.y]))
    smile_ratio = mouth_width / mouth_height if mouth_height > 0 else 0
    return smile_ratio > 2.0  # Tune this threshold if needed

def analyze_social_reciprocity(video_path, response_time_threshold=30):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    frame_count = 0
    smile_frames = 0
    head_turn_responses = 0
    prev_head_turn = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Detect smile
            if detect_smile(landmarks):
                smile_frames += 1

            # Detect head turn based on nose tip x coord deviation from center (~0.5)
            nose_tip = landmarks[1]
            head_turn = (nose_tip.x < 0.4 or nose_tip.x > 0.6)

            # Count head turn responses (simple logic: if head_turn changes from previous)
            if head_turn and not prev_head_turn:
                head_turn_responses += 1
            prev_head_turn = head_turn

    cap.release()
    face_mesh.close()

    social_response_score = (head_turn_responses + smile_frames / frame_count) / 2  # simple avg
    if social_response_score > 0.3:
        level = "High"
    elif social_response_score > 0.1:
        level = "Medium"
    else:
        level = "Low"

    return {
        "social_response_score": social_response_score,
        "social_reciprocity_level": level,
        "total_frames": frame_count,
        "head_turn_responses": head_turn_responses,
        "smiling_frames": smile_frames
    }
