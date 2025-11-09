import streamlit as st
import sounddevice as sd
import numpy as np
import pandas as pd
import joblib
import soundfile as sf
from python_speech_features import mfcc
import tempfile
import os

# ========================
# Model yükleme
# ========================
@st.cache_resource
def load_model():
    model = joblib.load("models/stress_model.pkl")
    return model

model = load_model()

# ========================
# MFCC çıkarım fonksiyonu
# ========================
def extract_features(audio, samplerate):
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    mfcc_feat = mfcc(audio, samplerate, numcep=20, nfilt=40, nfft=2048)
    mfcc_mean = np.mean(mfcc_feat, axis=0)
    return mfcc_mean.reshape(1, -1)

# ========================
# Streamlit Arayüzü
# ========================
st.set_page_config(page_title="🎧 Stress Analyzer", page_icon="🎙️", layout="centered")

st.title("🎧 Stress Analyzer")
st.write("Mikrofonla kısa bir ses kaydı al ve modelin stres seviyeni tahmin etmesine izin ver.")

duration = st.slider("Kayıt süresi (saniye)", 2, 10, 4)
if st.button("🎙️ Kaydı Başlat"):
    st.info("Kaydediliyor... Konuşmaya başla 🎤")
    fs = 16000  # örnekleme hızı
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    st.success("✅ Kayıt tamamlandı!")

    # Kaydı geçici dosyaya kaydet
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, audio, fs)
        temp_path = tmp.name

    # Özellik çıkarımı
    features = extract_features(audio, fs)
    prediction = model.predict(features)[0]

    # Tahmini göster
    levels = {0: ("Calm 😌", "#6fc276"), 1: ("Medium 😐", "#f4c542"), 2: ("Stress 😣", "#e74c3c")}
    label, color = levels[prediction]

    st.markdown(f"<h2 style='color:{color};text-align:center;'>🧠 Tahmin: {label}</h2>", unsafe_allow_html=True)

    # Kaydı çal
    st.audio(temp_path, format="audio/wav")
