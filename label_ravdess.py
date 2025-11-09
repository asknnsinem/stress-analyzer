import os
import pandas as pd
import shutil
from tqdm import tqdm

# RAVDESS klasör yolu (senin klasör adın bu)
RAVDESS_PATH = "Audio_Song_Actors_01-24"

# Hedef klasör yapısı
os.makedirs("data/raw/calm", exist_ok=True)
os.makedirs("data/raw/medium", exist_ok=True)
os.makedirs("data/raw/stress", exist_ok=True)

# Duygu kodlarını haritalama
emotion_map = {
    "01": ("neutral", 0),
    "02": ("calm", 0),
    "03": ("happy", 0),
    "04": ("sad", 1),
    "05": ("angry", 2),
    "06": ("fearful", 2),
    "07": ("disgust", 1),
    "08": ("surprised", 2)
}

labels = []

print("🎧 Dosyalar etiketleniyor...\n")

for root, dirs, files in os.walk(RAVDESS_PATH):
    for file in tqdm(files, desc="Processing"):
        if not file.endswith(".wav"):
            continue
        parts = file.split("-")
        if len(parts) < 7:
            continue

        emotion_code = parts[2]
        emotion_name, stress_level = emotion_map.get(emotion_code, ("unknown", None))
        if stress_level is None:
            continue

        # Hedef klasör belirle
        if stress_level == 0:
            target_folder = "data/raw/calm"
        elif stress_level == 1:
            target_folder = "data/raw/medium"
        else:
            target_folder = "data/raw/stress"

        # Dosyayı kopyala
        src = os.path.join(root, file)
        dest = os.path.join(target_folder, file)
        shutil.copy2(src, dest)

        labels.append({
            "filename": f"{os.path.basename(target_folder)}/{file}",
            "stress_level": stress_level
        })

# labels.csv oluştur
df = pd.DataFrame(labels)
os.makedirs("data", exist_ok=True)
df.to_csv("data/labels.csv", index=False)

print(f"\n✅ Etiketleme tamamlandı! {len(df)} ses dosyası işlendi.")
print("📄 Kaydedildi: data/labels.csv")
