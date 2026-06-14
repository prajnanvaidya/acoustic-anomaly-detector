import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import librosa
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import shutil

# --- PATH SETUP ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Pointing strictly to your newly isolated 0dB dataset
NORMAL_DIR = os.path.join(BASE_DIR, "data_0db", "normal")
ANOMALOUS_DIR = os.path.join(BASE_DIR, "data_0db", "anomalous") # FIXED THIS TO ANOMALOUS
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "valve_anomaly_resnet18_best.pth")

# --- 1. DATASET LOADER ---
class StressTestDataset(Dataset):
    def __init__(self, normal_dir, anomalous_dir): # FIXED VARIABLE NAME
        self.file_paths = []
        self.labels = [] 
        
        # Load Normal Files (Class 0)
        if os.path.exists(normal_dir):
            for f in os.listdir(normal_dir):
                if f.endswith('.wav'):
                    self.file_paths.append(os.path.join(normal_dir, f))
                    self.labels.append(0)
        else:
            print(f"⚠️ Could not find {normal_dir}")
            
        # Load Anomalous Files (Class 1) - FIXED THIS WHOLE SECTION
        if os.path.exists(anomalous_dir):
            for f in os.listdir(anomalous_dir):
                if f.endswith('.wav'):
                    self.file_paths.append(os.path.join(anomalous_dir, f))
                    self.labels.append(1)
        else:
            print(f"⚠️ Could not find {anomalous_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Exact same preprocessing pipeline as your Streamlit App
        y, sr = librosa.load(file_path, sr=16000, duration=10.0)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        image = np.stack([mel_db, mel_db, mel_db], axis=0)
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# --- 2. INITIALIZATION ---
print("\n🔍 INITIALIZING ZERO-SHOT STRESS TEST...")
stress_dataset = StressTestDataset(NORMAL_DIR, ANOMALOUS_DIR) # FIXED VARIABLE NAME

if len(stress_dataset) == 0:
    print("❌ Error: No audio files found in data_0db. Check your folder structure!")
    exit()

print(f"📁 Total 0_dB files to analyze: {len(stress_dataset)}")

# Batch size 16 to keep your VRAM safe
test_loader = DataLoader(stress_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Running on Neural Accelerator: {device}\n")

# --- 3. MODEL SETUP ---
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model = model.to(device)
model.eval()

# --- 4. EXECUTION LOOP ---
all_preds = []
all_labels = []

print("🚀 Executing Deep Feature Extraction & Classification:")
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Processing Audio Files", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- 5. ANALYTICS & REPORTING ---
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = 100.0 * np.sum(all_preds == all_labels) / len(all_labels)

# Compute Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
else:
    tn, fp, fn, tp = 0, 0, 0, 0
    print("⚠️ Warning: Only one class was present in the dataset.")

print("\n" + "="*50)
print(f"🎯 ZERO-SHOT ACCURACY ON UNSEEN 0_dB DATA: {accuracy:.2f}%")
print("="*50)

print("\n📊 --- CONFUSION MATRIX ---")
print(f"✅ True Negatives (Correctly Normal)   : {tn}")
print(f"✅ True Positives (Correctly Anomalous): {tp}") # FIXED LABEL
print(f"❌ False Positives (False Alarms)      : {fp}")
print(f"❌ False Negatives (Missed Anomalies)  : {fn}")

print("\n📈 --- DETAILED CLASSIFICATION REPORT ---")
print(classification_report(all_labels, all_preds, target_names=["Normal", "Anomalous"])) # FIXED LABEL

# --- 6. EXTRACTING HARD NEGATIVES ---
print("\n🔍 --- EXTRACTING HARD NEGATIVES (FALSE ALARMS) ---")
hard_negatives_dir = os.path.join(BASE_DIR, "hard_negatives_to_review")
os.makedirs(hard_negatives_dir, exist_ok=True)

false_alarm_count = 0

for i in range(len(all_preds)):
    actual = all_labels[i]
    prediction = all_preds[i]
    
    if actual == 0 and prediction == 1:
        tricky_file_path = stress_dataset.file_paths[i]
        file_name = os.path.basename(tricky_file_path)
        destination_path = os.path.join(hard_negatives_dir, file_name)
        shutil.copy2(tricky_file_path, destination_path)
        false_alarm_count += 1

print(f"📁 Successfully isolated {false_alarm_count} False Alarm files!")
print(f"Check the '{hard_negatives_dir}' folder in your project directory.")