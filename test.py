import soundfile as sf
data, samplerate = sf.read("data/raw/calm/calm_001.wav")
print("✅ Okundu:", len(data), "örnek,", samplerate, "Hz")