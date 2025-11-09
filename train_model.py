import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# 1️⃣ Veri yükle
df = pd.read_csv("data/processed/features.csv")
X = df.drop("stress_level", axis=1)
y = df["stress_level"]

# 2️⃣ Eğitim/Test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3️⃣ Model eğitimi
model = RandomForestClassifier(
    n_estimators=250,
    max_depth=20,
    min_samples_split=3,
    random_state=42
)
model.fit(X_train, y_train)

# 4️⃣ Performans raporu
y_pred = model.predict(X_test)
print("\n📊 Model Performansı:\n")
print(classification_report(y_test, y_pred, digits=3))

# 5️⃣ Karışıklık matrisi
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Calm", "Medium", "Stress"],
            yticklabels=["Calm", "Medium", "Stress"])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Stres Seviyesi Karışıklık Matrisi")
plt.tight_layout()
plt.show()

# 6️⃣ Model kaydet
os.makedirs("models", exist_ok=True)
model_path = "models/stress_model.pkl"
joblib.dump(model, model_path)

print(f"\n✅ Model başarıyla kaydedildi: {model_path}")
