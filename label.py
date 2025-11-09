import os
import pandas as pd

base_path = "data/raw"
labels = []

for label_name, stress_level in zip(["calm", "medium", "stress"], [0, 1, 2]):
    folder = os.path.join(base_path, label_name)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            labels.append({
                "filename": f"{label_name}/{file}",
                "stress_level": stress_level
            })

df = pd.DataFrame(labels)
df.to_csv("data/labels.csv", index=False)
print(f"✅ labels.csv güncellendi. {len(df)} kayıt bulundu.")
