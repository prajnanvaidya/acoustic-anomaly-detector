import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import librosa
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# --- PATH SETUP ---
# Automatically finds the project root directory (one folder up from /scripts)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NORMAL_DIR = os.path.join(BASE_DIR, "data", "normal")
ANOMALOUS_DIR = os.path.join(BASE_DIR, "data", "anomalous")

# UNIQUE V2 FILENAME - Protects your original model from being overwritten!
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "valve_anomaly_resnet18_v2_best.pth")

# 1. Custom Dataset
class ValveAudioDataset(Dataset):
    def __init__(self, normal_dir, anomalous_dir):
        self.file_paths = []
        self.labels = [] 
        
        for f in os.listdir(normal_dir):
            if f.endswith('.wav'):
                self.file_paths.append(os.path.join(normal_dir, f))
                self.labels.append(0) # 0 = Normal
                
        for f in os.listdir(anomalous_dir):
            if f.endswith('.wav'):
                self.file_paths.append(os.path.join(anomalous_dir, f))
                self.labels.append(1) # 1 = Anomalous

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

# 2. Initialize Data and 3-Way Split (70 / 15 / 15)
full_dataset = ValveAudioDataset(NORMAL_DIR, ANOMALOUS_DIR)

train_size = int(0.70 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

# --- SEED LOCKED HERE FOR REPRODUCIBILITY ---
torch.manual_seed(42)

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size]
)

# Batch size lowered to 16 to safely fit inside VRAM
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"📊 Dataset Split -> Train: {train_size} | Val: {val_size} | Test: {test_size}")

# 3. Setup Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on device: {device}")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. The Training & Validation Loop
epochs = 20
best_val_loss = float('inf')

print("\n🏋️ Starting 20-Epoch V2 Training Loop...")
for epoch in range(epochs):
    # --- TRAINING PHASE ---
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
    
    # --- VALIDATION PHASE ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = 100.0 * correct / total
    
    print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
    
    # --- SAVE BEST MODEL ---
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        # Safely save straight to the models directory as V2
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("   🌟 New best V2 model saved!")

# 5. Final Unseen Test Evaluation
print("\n⏳ Loading best V2 model for final test evaluation...")
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
print(f"\n🎯 FINAL TEST ACCURACY ON UNSEEN SPLIT: {test_accuracy:.2f}%\n")

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()
print("📊 --- CONFUSION MATRIX RESULTS ---")
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")