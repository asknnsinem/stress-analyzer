import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

df = pd.read_csv("data/processed/ravdess_features.csv")

X = df.drop("stress_level", axis=1)
y = df["stress_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("📊 Model Performansı:")
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/face_stress_model.pkl")
print("✅ Model kaydedildi: models/face_stress_model.pkl")
