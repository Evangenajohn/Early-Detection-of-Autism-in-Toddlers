import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh

def estimate_gaze(landmarks, width, height):
    left_eye = landmarks[33]   # Left eye outer corner
    right_eye = landmarks[263] # Right eye outer corner
    nose_tip = landmarks[1]    # Nose tip

    eye_center_x = (left_eye.x + right_eye.x) / 2
    gaze_vector = eye_center_x - nose_tip.x
    gaze_ratio = gaze_vector + 0.5  # Normalize to ~0–1
    return gaze_ratio

def analyze_eye_contact(video_path, visualize=False):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    eye_contact_count = 0
    frame_count = 0
    gaze_ratios = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            gaze_ratio = estimate_gaze(landmarks, w, h)
            gaze_ratios.append(gaze_ratio)

            if 0.3 <= gaze_ratio <= 0.6:
                eye_contact_count += 1
        else:
            gaze_ratios.append(None)

    cap.release()
    face_mesh.close()

    eye_contact_percent = (eye_contact_count / frame_count) * 100 if frame_count else 0

    if visualize:
        gaze_plot_data = [gr if gr is not None else 0 for gr in gaze_ratios]
        plt.figure(figsize=(12, 4))
        plt.plot(gaze_plot_data, label='Gaze Ratio')
        plt.axhline(y=0.3, color='r', linestyle='--', label='Low Threshold (0.3)')
        plt.axhline(y=0.6, color='g', linestyle='--', label='High Threshold (0.6)')
        plt.title("Gaze Ratio Over Time")
        plt.xlabel("Frame")
        plt.ylabel("Gaze Ratio")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "eye_contact_percent": round(eye_contact_percent, 2),
        "gaze_ratios": gaze_ratios,
        "frame_count": frame_count
    }
