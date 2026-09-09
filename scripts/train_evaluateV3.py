import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import librosa
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# --- PATH SETUP ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "valve_anomaly_resnet18_v3_leakfree.pth")

# Pulling from all 3 folders to triple the dataset and maximize robustness
SNR_FOLDERS = ["data_6db", "data_0db", "data_minus6db"]

# --- LEAK-FREE GROUPING LOGIC ---
def get_machine_id(filename):
    match = re.search(r'(id_\d+)', filename)
    if match:
        return match.group(1)
    return "unknown"

# 1. Gather all files across all SNR levels
all_files, all_labels, all_groups = [], [], []

for snr in SNR_FOLDERS:
    norm_dir = os.path.join(BASE_DIR, snr, "normal")
    anom_dir = os.path.join(BASE_DIR, snr, "anomalous")
    
    if os.path.exists(norm_dir):
        for f in os.listdir(norm_dir):
            if f.endswith('.wav'):
                all_files.append(os.path.join(norm_dir, f))
                all_labels.append(0)
                all_groups.append(get_machine_id(f))
                
    if os.path.exists(anom_dir):
        for f in os.listdir(anom_dir):
            if f.endswith('.wav'):
                all_files.append(os.path.join(anom_dir, f))
                all_labels.append(1)
                all_groups.append(get_machine_id(f))

all_files = np.array(all_files)
all_labels = np.array(all_labels)
all_groups = np.array(all_groups)

# 2. Perform GroupShuffleSplit to completely isolate machines
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_val_idx, test_idx = next(gss_test.split(all_files, all_labels, groups=all_groups))

train_val_files, test_files = all_files[train_val_idx], all_files[test_idx]
train_val_labels, test_labels = all_labels[train_val_idx], all_labels[test_idx]
train_val_groups = all_groups[train_val_idx]

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
try:
    train_idx, val_idx = next(gss_val.split(train_val_files, train_val_labels, groups=train_val_groups))
    train_files, val_files = train_val_files[train_idx], train_val_files[val_idx]
    train_labels, val_labels = train_val_labels[train_idx], train_val_labels[val_idx]
except ValueError:
    print("⚠️ WARNING: Not enough unique machines. Using test set for validation.")
    train_files, val_files = train_val_files, test_files
    train_labels, val_labels = train_val_labels, test_labels

print(f"\n📊 MIXED-SNR LEAK-FREE SPLIT COMPLETED:")
print(f"   Train Files: {len(train_files)} | Train Machines: {set([get_machine_id(f) for f in train_files])}")
print(f"   Val Files:   {len(val_files)} | Val Machines:   {set([get_machine_id(f) for f in val_files])}")
print(f"   Test Files:  {len(test_files)} | Test Machines:  {set([get_machine_id(f) for f in test_files])}\n")

# 3. Custom Dataset Implementation
class ValveAudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        y, sr = librosa.load(file_path, sr=16000, duration=10.0)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        image = np.stack([mel_db, mel_db, mel_db], axis=0)
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

train_dataset = ValveAudioDataset(train_files, train_labels)
val_dataset = ValveAudioDataset(val_files, val_labels)
test_dataset = ValveAudioDataset(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 4. Setup Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on device: {device}")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. The Training & Validation Loop
epochs = 20
best_val_loss = float('inf')

print("\n🏋️ Starting 20-Epoch V3 (Leak-Free) Training Loop...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, batch_labels in train_loader:
        inputs, batch_labels = inputs.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
    epoch_train_loss = running_loss / len(train_dataset)
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, batch_labels in val_loader:
            inputs, batch_labels = inputs.to(device), batch_labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item() * inputs.size(0)
            
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
            
    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = 100.0 * correct / total
    
    print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
    
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("   🌟 New best V3 model saved!")

# 6. Final Unseen Test Evaluation
print("\n⏳ Loading best V3 model for final test evaluation...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

eval_preds, eval_labels = [], []
with torch.no_grad():
    for inputs, batch_labels in test_loader:
        inputs, batch_labels = inputs.to(device), batch_labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        eval_preds.extend(predicted.cpu().numpy())
        eval_labels.extend(batch_labels.cpu().numpy())

eval_preds = np.array(eval_preds)
eval_labels = np.array(eval_labels)

test_accuracy = 100.0 * np.sum(eval_preds == eval_labels) / len(eval_labels)
print(f"\n🎯 FINAL TEST ACCURACY ON UNSEEN MACHINES: {test_accuracy:.2f}%")

cm = confusion_matrix(eval_labels, eval_preds)
tn, fp, fn, tp = cm.ravel()
print("\n📊 --- CONFUSION MATRIX RESULTS ---")
print(f"True Negatives (Normal flagged Normal):     {tn}")
print(f"False Positives (Normal flagged Anomaly):   {fp}")
print(f"False Negatives (Anomaly flagged Normal):   {fn}")
print(f"True Positives (Anomaly flagged Anomaly):   {tp}")

print("\n📈 --- CLASSIFICATION REPORT ---")
print(classification_report(eval_labels, eval_preds, target_names=['Normal', 'Anomaly']))

f1 = f1_score(eval_labels, eval_preds, pos_label=1)
print(f"🏆 FINAL F1-SCORE (ANOMALY CLASS): {f1:.4f}\n")