import os
import re
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

# ===============================
# MediaPipe setup
# ===============================
mp_face = mp.solutions.face_mesh

def eye_aspect_ratio(pts):
    A = np.linalg.norm(pts[1]-pts[5])
    B = np.linalg.norm(pts[2]-pts[4])
    C = np.linalg.norm(pts[0]-pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def mouth_aspect_ratio(outer):
    A = np.linalg.norm(outer[2]-outer[6])
    B = np.linalg.norm(outer[3]-outer[5])
    C = np.linalg.norm(outer[0]-outer[4])
    return (A + B) / (2.0 * C + 1e-6)

# ===============================
# Etiket çıkarımı (dosya adına göre)
# ===============================
def extract_emotion_id(filename):
    match = re.match(r"\d{2}-\d{2}-(\d{2})-", filename)
    return int(match.group(1)) if match else None

def map_to_stress(emotion_id):
    if emotion_id in [1, 2]:
        return 0  # calm
    elif emotion_id in [3, 4, 8]:
        return 1  # medium
    else:
        return 2  # stress

# ===============================
# Özellik çıkarım fonksiyonu
# ===============================
def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    with mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                pts = np.array([[p.x, p.y] for p in lm])

                left_eye_idx = [33, 160, 158, 133, 153, 144]
                right_eye_idx = [362, 385, 387, 263, 373, 380]
                mouth_idx = [61, 81, 91, 146, 321, 311, 308, 375]

                LE = eye_aspect_ratio(pts[left_eye_idx])
                RE = eye_aspect_ratio(pts[right_eye_idx])
                MAR = mouth_aspect_ratio(pts[mouth_idx])
                head_tilt = pts[33,1] - pts[263,1]

                frames.append([LE, RE, MAR, head_tilt])

    cap.release()
    if not frames:
        return None

    arr = np.array(frames)
    features = np.hstack([
        arr.mean(axis=0),
        arr.std(axis=0),
        np.percentile(arr, 95, axis=0)
    ])
    return features

# ===============================
# Ana döngü
# ===============================
root = "face_data"
all_features = []

for actor_dir in tqdm(sorted(os.listdir(root))):
    actor_path = os.path.join(root, actor_dir)
    if not os.path.isdir(actor_path):
        continue

    for fname in os.listdir(actor_path):
        if not fname.endswith(".mp4"):
            continue
        fpath = os.path.join(actor_path, fname)
        emotion_id = extract_emotion_id(fname)
        stress_level = map_to_stress(emotion_id)
        feats = extract_features(fpath)
        if feats is not None:
            all_features.append([*feats, stress_level])

df = pd.DataFrame(all_features, columns=[
    "LE_mean", "RE_mean", "MAR_mean", "tilt_mean",
    "LE_std", "RE_std", "MAR_std", "tilt_std",
    "LE_95", "RE_95", "MAR_95", "tilt_95", "stress_level"
])

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/ravdess_features.csv", index=False)

print("✅ Özellik çıkarımı tamamlandı:", df.shape)
