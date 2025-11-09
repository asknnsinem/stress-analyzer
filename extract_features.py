import os
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from python_speech_features import mfcc
import warnings
warnings.filterwarnings("ignore")

# Veri dosyasını oku
labels_df = pd.read_csv("data/labels.csv")

features = []
print(f"🎧 {len(labels_df)} ses dosyası işlenecek...\n")

for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
    file_path = os.path.join("data/raw", row["filename"])
    try:
        y, sr = sf.read(file_path, dtype='float32')
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)  # stereo → mono
        mfcc_feat = mfcc(y, sr, numcep=20, nfilt=40, nfft=2048)
        mfcc_mean = np.mean(mfcc_feat, axis=0)
        features.append([*mfcc_mean, row["stress_level"]])
    except Exception as e:
        print(f"⚠️ Hata: {file_path} ({e})")

columns = [f"mfcc_{i}" for i in range(1, 21)] + ["stress_level"]
df = pd.DataFrame(features, columns=columns)
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/features.csv", index=False)

print(f"\n✅ Özellik çıkarımı tamamlandı! {len(df)} kayıt işlendi.")
