import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# ==============================
# MODEL YÜKLEME
# ==============================
@st.cache_resource
def load_model():
    return joblib.load("models/face_stress_model.pkl")

model = load_model()

# ==============================
# MEDIA PIPE YÜZ AĞI AYARLARI
# ==============================
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Göz ve ağız oranları
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

# ==============================
# STREAMLIT ARAYÜZÜ
# ==============================
st.set_page_config(page_title="🎥 Live Stress Analyzer", page_icon="🧠", layout="centered")

st.title("🧠 Gerçek Zamanlı Yüz Mimiklerinden Stres Analizi")
st.markdown("Kameranı aç, yüzündeki **mesh noktalarını** ve modelin **anlık stres tahminini** izle 🔍")

run = st.toggle("📸 Kamerayı Aç / Kapat")

frame_window = st.image([])
label_placeholder = st.empty()
bar_placeholder = st.empty()

# ==============================
# CANLI GÖRÜNTÜ DÖNGÜSÜ
# ==============================
if run:
    cap = cv2.VideoCapture(0)

    with mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                st.warning("Kamera açılamadı. Lütfen kontrol et.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Noktaları çiz
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
                    )

                    # Özellikleri hesapla
                    lm = face_landmarks.landmark
                    pts = np.array([[p.x, p.y] for p in lm])

                    left_eye_idx = [33, 160, 158, 133, 153, 144]
                    right_eye_idx = [362, 385, 387, 263, 373, 380]
                    mouth_idx = [61, 81, 91, 146, 321, 311, 308, 375]

                    LE = eye_aspect_ratio(pts[left_eye_idx])
                    RE = eye_aspect_ratio(pts[right_eye_idx])
                    MAR = mouth_aspect_ratio(pts[mouth_idx])
                    head_tilt = pts[33,1] - pts[263,1]

                    feats = np.array([[LE, RE, MAR, head_tilt,
                                       0,0,0,0,0,0,0,0]])
                    pred = model.predict(feats)[0]
                    proba = model.predict_proba(feats)[0]

                    label_map = {
                        0: ("😌 Calm", (111, 217, 145)),
                        1: ("😐 Medium", (240, 200, 80)),
                        2: ("😣 Stress", (230, 85, 80))
                    }
                    label, color = label_map[pred]

                    # Çerçeveye yazı yaz
                    cv2.putText(frame, f"{label}", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                    # Göz ve ağız bölgelerini dairelerle göster
                    for i in left_eye_idx + right_eye_idx + mouth_idx:
                        x, y = int(pts[i][0]*frame.shape[1]), int(pts[i][1]*frame.shape[0])
                        cv2.circle(frame, (x, y), 2, color, -1)

                    # Streamlit etiket ve grafik
                    label_placeholder.markdown(
                        f"<h2 style='color:rgb{color};text-align:center;'>Tahmin: {label}</h2>",
                        unsafe_allow_html=True
                    )
                    bar_placeholder.bar_chart({
                        "Calm": [proba[0]],
                        "Medium": [proba[1]],
                        "Stress": [proba[2]]
                    })

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            time.sleep(0.05)

    cap.release()
else:
    st.info("Kamerayı başlatmak için üstteki anahtarı aç 🎥")
