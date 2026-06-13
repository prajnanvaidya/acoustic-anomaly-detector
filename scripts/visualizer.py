import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# --- PATH SETUP ---
# Automatically finds the project root directory (one folder up from /scripts)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 1. Exact paths to your files
normal_file = os.path.join(BASE_DIR, "data", "normal", "id_00_00000349.wav")
anomalous_file = os.path.join(BASE_DIR, "data", "anomalous", "id_02_00000055.wav")

def get_mel_spectrogram(file_path):
    # Load audio at standard 16kHz
    y, sr = librosa.load(file_path, sr=16000)
    # Compute Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    # Convert to Decibels
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_db, sr

# Load both sounds
normal_db, sr_n = get_mel_spectrogram(normal_file)
anomalous_db, sr_a = get_mel_spectrogram(anomalous_file)

# Plot side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot Normal Valve
img1 = librosa.display.specshow(normal_db, sr=sr_n, x_axis='time', y_axis='mel', ax=axes[0])
fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
axes[0].set_title("Normal Valve Sound (id_00_00000349)")

# Plot Anomalous Valve
img2 = librosa.display.specshow(anomalous_db, sr=sr_a, x_axis='time', y_axis='mel', ax=axes[1])
fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
axes[1].set_title("Anomalous Valve Sound (id_02_00000055)")

plt.tight_layout()
plt.show()