import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import librosa
import numpy as np
import soundfile as sf
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit

# --- PATH SETUP ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIRS = [
    os.path.join(BASE_DIR, "data_6db"),
    os.path.join(BASE_DIR, "data_0db"),
    os.path.join(BASE_DIR, "data_minus6db"),
    os.path.join(BASE_DIR, "data_2021") 
]
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "valve_anomaly_resnet18_v4_robust.pth")

# --- GLOBAL STANDARDIZATION ---
def compute_global_stats(file_paths):
    print("\n📊 Computing global Z-Score stats on a training sample...")
    sample_paths = np.random.choice(file_paths, min(250, len(file_paths)), replace=False)
    mel_dbs = []
    for f in sample_paths:
        y, _ = sf.read(f, dtype='float32')
        if y.ndim > 1: y = y.mean(axis=1, dtype=np.float32) # FORCE 32-bit
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128)
        mel_dbs.append(librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32))
    
    mel_dbs_flat = np.concatenate(mel_dbs, axis=1) 
    return np.mean(mel_dbs_flat), np.std(mel_dbs_flat)

# 1. Custom Dataset
class ValveAudioRobustDataset(Dataset):
    def __init__(self, file_paths, labels, global_mean, global_std, is_train=False):
        self.file_paths = file_paths
        self.labels = labels
        self.global_mean = global_mean
        self.global_std = global_std
        self.is_train = is_train

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        y, sr = sf.read(file_path, dtype='float32')
        
        if y.ndim > 1: 
            y = y.mean(axis=1, dtype=np.float32) # FORCE 32-bit
            
        if len(y) > 160000:
            y = y[:160000]
        elif len(y) < 160000:
            y = np.pad(y, (0, 160000 - len(y))).astype(np.float32)
        
        if self.is_train:
            noise_amp = 0.005 * np.random.uniform() * np.amax(y)
            # FORCE 32-bit noise to prevent NumPy from silently upscaling the array
            noise = np.random.normal(size=y.shape).astype(np.float32)
            y = y + noise_amp * noise
            
        mel_spec = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
        
        if self.is_train:
            n_mels, n_steps = mel_db.shape
            f_mask = np.random.randint(0, 15)
            f0 = np.random.randint(0, n_mels - f_mask)
            mel_db[f0:f0+f_mask, :] = mel_db.min()
            
            t_mask = np.random.randint(0, 30)
            t0 = np.random.randint(0, n_steps - t_mask)
            mel_db[:, t0:t0+t_mask] = mel_db.min()

        mel_norm = (mel_db - self.global_mean) / (self.global_std + 1e-6)
        image = np.stack([mel_norm, mel_norm, mel_norm], axis=0)
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 2. Data Collection & Group Extraction
all_file_paths = []
all_labels = []
all_groups = []

print("🔍 Scanning directories for all datasets...")
for d in DATA_DIRS:
    if not os.path.exists(d):
        continue
    for category in ["normal", "anomalous"]:
        cat_dir = os.path.join(d, category)
        if not os.path.exists(cat_dir):
            continue
            
        label = 0 if category == "normal" else 1
        for f in os.listdir(cat_dir):
            if f.endswith('.wav'):
                all_file_paths.append(os.path.join(cat_dir, f))
                all_labels.append(label)
                
                match = re.search(r'(id_\d+|section_\d+)', f)
                all_groups.append(match.group(1) if match else "unknown")

all_file_paths = np.array(all_file_paths)
all_labels = np.array(all_labels)
all_groups = np.array(all_groups)

# 3. Leak-Free Group Shuffle Split
gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
train_idx, temp_idx = next(gss.split(all_file_paths, all_labels, groups=all_groups))

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx_temp, test_idx_temp = next(gss_val.split(
    all_file_paths[temp_idx], 
    all_labels[temp_idx], 
    groups=all_groups[temp_idx]
))

val_idx = temp_idx[val_idx_temp]
test_idx = temp_idx[test_idx_temp]

train_groups = set(all_groups[train_idx])
val_groups = set(all_groups[val_idx])
test_groups = set(all_groups[test_idx])

print("\n📊 MIXED-SNR LEAK-FREE SPLIT COMPLETED:")
print(f"   Train Files: {len(train_idx)} | Train Machines: {train_groups}")
print(f"   Val Files:   {len(val_idx)} | Val Machines:   {val_groups}")
print(f"   Test Files:  {len(test_idx)} | Test Machines:  {test_groups}")

GLOBAL_MEAN, GLOBAL_STD = compute_global_stats(all_file_paths[train_idx])
print(f"   -> Global Mean: {GLOBAL_MEAN:.2f} dB, Global Std: {GLOBAL_STD:.2f} dB")

train_dataset = ValveAudioRobustDataset(all_file_paths[train_idx], all_labels[train_idx], GLOBAL_MEAN, GLOBAL_STD, is_train=True)
val_dataset = ValveAudioRobustDataset(all_file_paths[val_idx], all_labels[val_idx], GLOBAL_MEAN, GLOBAL_STD, is_train=False)
test_dataset = ValveAudioRobustDataset(all_file_paths[test_idx], all_labels[test_idx], GLOBAL_MEAN, GLOBAL_STD, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# 4. Setup Robust Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Training on device: {device}")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, 2)
)
model = model.to(device)

class_weights = torch.tensor([1.0, 5.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 5. Training Loop
epochs = 30 
best_val_f1 = -1.0
early_stop_patience = 6
patience_counter = 0

print("\n🏋️ Starting V4 (Leak-Free Robust) Training Loop...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
    epoch_train_loss = running_loss / len(train_dataset)
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            val_preds.extend(predicted.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
            
    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = 100.0 * correct / total
    epoch_val_f1 = f1_score(val_targets, val_preds, pos_label=1, zero_division=0)
    
    scheduler.step(epoch_val_loss)
    
    current_lr = [group['lr'] for group in optimizer.param_groups][0]
    
    print(f"Epoch {epoch+1:02d}/{epochs} -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val F1: {epoch_val_f1:.4f} | LR: {current_lr:.6f}")
    
    if epoch_val_f1 > best_val_f1:
        best_val_f1 = epoch_val_f1
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"   🌟 New best robust model saved! (Val F1: {best_val_f1:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"\n🛑 Early stopping triggered after {early_stop_patience} epochs without F1-Score improvement.")
            break

# 6. Final Test Evaluation
print("\n⏳ Loading best robust model for final test evaluation...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

test_accuracy = 100.0 * np.sum(all_preds == all_labels) / len(all_labels)
print(f"\n🎯 FINAL TEST ACCURACY ON UNSEEN MACHINES: {test_accuracy:.2f}%\n")

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()
print("📊 --- CONFUSION MATRIX RESULTS ---")
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")
print("\n📈 --- CLASSIFICATION REPORT ---")
print(classification_report(all_labels, all_preds, target_names=["Normal", "Anomaly"]))