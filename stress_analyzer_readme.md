# 🧠 Stress Analyzer

**Stress Analyzer** is a machine learning project that analyzes human stress levels using **facial expressions** and **voice tone**.  
It can run in real-time through a webcam or use pre-recorded `.wav` / `.mp4` files for emotion recognition.

---

## 🚀 Features

- 🎤 **Audio-based stress detection** (trained with the RAVDESS dataset)
- 👁️ **Facial emotion recognition** using MediaPipe FaceMesh
- 🧩 **Random Forest model** trained on MFCC and facial landmark features
- 🧠 **3 stress levels detected:**
  ```bash
  0 → Calm
  1 → Neutral
  2 → Stressed
  ```
- 🛠️ Streamlit interface for real-time webcam visualization
- 📦 Modular folder structure for extending to new data or models

---

## 🗂️ Project Structure

```bash
stress-analyzer/
│
├── face_stress_analyzer/
│   ├── app_live.py                # Streamlit live webcam app
│   ├── extract_features.py        # Extract features (RAVDESS or custom)
│   ├── train_model.py             # Train RandomForest model
│   ├── models/
│   │   └── face_stress_model.pkl  # Saved ML model
│   ├── data/
│   │   ├── raw/                   # Raw audio/video data
│   │   └── processed/             # Extracted feature CSVs
│   └── utils/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧮 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/asknnsinem/stress-analyzer.git
cd stress-analyzer
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training

### Extract features
```bash
cd face_stress_analyzer
python extract_features.py
```

### Train the model
```bash
python train_model.py
```

The trained model will be saved as:
```bash
models/face_stress_model.pkl
```

---

## 🎥 Run Real-Time Detection (Webcam)
```bash
streamlit run app_live.py
```
Then open the URL (e.g. `http://localhost:8501`) in your browser.

---

## 🧾 Example Output

- Webcam feed with face landmarks visualized  
- Stress prediction (0–2) displayed live  
- Logs stored under `.streamlit` or `logs/` folder  

---

## ⚙️ Requirements

```bash
Python 3.10–3.11
TensorFlow 2.13+
MediaPipe 0.10.10+
OpenCV 4.12+
Scikit-learn 1.5+
Streamlit 1.51+
```

---

## 🧠 Dataset

The project uses the **RAVDESS** dataset for training:

[🎧 Ryerson Audio-Visual Database of Emotional Speech and Song](https://zenodo.org/record/1188976)

You can also record your own samples and label them manually under:
```bash
data/raw/{calm, medium, stress}/
```

---

## 🧑‍💻 Future Improvements

- Add temporal (blink rate, heart rate) analysis
- Support multilingual voice stress detection
- Add fine-tuned CNN/LSTM models for improved accuracy

---

## 📄 License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute with proper credit.

---

## ❤️ Author

[@asknnsinem](https://github.com/asknnsinem)

**Real-time Stress Analysis using AI and Computer Vision**

