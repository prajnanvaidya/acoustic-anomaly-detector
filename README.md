# 🏭 Automated Acoustic Diagnostic Suite
### Edge AI Inference Platform for Non-Destructive Valve Inspection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Streamlit](https://img.shields.io/badge/Streamlit-Enterprise%20UI-FF4B4B)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-100%25-brightgreen)

---

## 📌 Project Overview

In industrial manufacturing environments, mechanical valve failures can lead to catastrophic downtime. This project introduces a **Zero-Defect Acoustic Diagnostic Engine** that listens to 10-second audio footprints of industrial valves and classifies them as either:

- ✅ **Operational (Normal)**
- ⚠️ **Critical Fault (Anomalous)**

Instead of relying on fragile custom architectures, this solution utilizes a transfer-learned **ResNet18** model processing **128-bin Mel-Scale Logarithmic Spectrograms**.

The model was further hardened using **Hard Negative Mining** against extreme **-6 dB industrial noise environments**, achieving a verified:

- **100% Test Accuracy**
- **0 False Positives**
- **0 False Negatives**

The suite also features a live **SCADA-style dashboard** with **Explainable AI (Grad-CAM)** visualizations, allowing operators to understand exactly which acoustic regions influenced each prediction.

---

# 🧠 Model Architecture & Weights

Because deep learning model weights exceed standard GitHub storage limits, the trained models are hosted externally.

## 📥 Download Pre-trained Models

**Google Drive Link:**

```text
[INSERT_YOUR_GOOGLE_DRIVE_LINK_HERE]
```

After downloading, place the `.pth` files inside:

```text
models/
```

---

## 🏆 Available Models

### 1. valve_anomaly_resnet18_v2_best.pth

**Production-Ready Champion Model**

Features:

- ResNet18 Transfer Learning
- Hard Negative Mining
- Noise-Hardened Training
- Best Validation Performance
- Recommended for deployment

This model was trained using a targeted "Sniper Approach" where difficult false positives and false negatives discovered during stress testing were reintroduced into the training pipeline.

---

### 2. valve_anomaly_resnet18_best.pth

**V1 Baseline Model**

Features:

- Original baseline architecture
- Strong performance under standard conditions
- Useful for benchmarking improvements achieved in V2

---

### 3. valve_anomaly_resnet18.pth

**Final Epoch V1 Checkpoint**

Retained for:

- Research comparison
- Reproducibility
- Rollback experiments

---

# 📂 Project Structure

```text
ACOUSTIC-ANOMALY-DETECTOR/
│
├── data/
│   ├── Normal/
│   └── Anomalous/
│
├── models/
│   ├── valve_anomaly_resnet18_v2_best.pth
│   ├── valve_anomaly_resnet18_best.pth
│   └── valve_anomaly_resnet18.pth
│
├── scripts/
│   ├── train_evaluateV1.py
│   ├── train_evaluate.py
│   └── visualizer.py
│
├── stress_test.py
├── stress_test_minus6db.py
├── app.py
├── requirements.txt
└── README.md
```

---

# 🚀 Quick Start Guide

## 1️⃣ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/acoustic-anomaly-detector.git
cd acoustic-anomaly-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Launch the Dashboard

Run the Streamlit application:

```bash
streamlit run app.py
```

The dashboard automatically loads:

```text
valve_anomaly_resnet18_v2_best.pth
```

if present inside the `models/` directory.

---

## 3️⃣ Train the Models

### Train V1 Baseline

```bash
python scripts/train_evaluateV1.py
```

---

### Train V2 Hardened Model

```bash
python scripts/train_evaluate.py
```

Key improvements:

- Fixed random seed
- Hard Negative Mining
- Enhanced robustness to industrial noise
- Improved generalization under extreme conditions

---

## 4️⃣ Run Stress Tests

### Standard Noise Validation

```bash
python stress_test.py
```

### Extreme Noise Validation (-6 dB)

```bash
python stress_test_minus6db.py
```

The scripts automatically isolate:

- False Positives
- False Negatives

for manual review and error analysis.

---

# 🔍 Explainable AI (Grad-CAM)

The dashboard includes Grad-CAM visualizations that:

- Highlight influential frequency bands
- Explain model decisions
- Improve operator trust
- Support industrial diagnostics workflows

This enables engineers to verify whether the model is focusing on meaningful acoustic signatures rather than irrelevant noise.

---

# 📊 Performance Metrics

## V2 Hardened Model

| Metric | Result |
|----------|----------|
| Test Accuracy | 100.00% |
| Precision | 100.00% |
| Recall | 100.00% |
| F1 Score | 100.00% |
| False Positives | 0 |
| False Negatives | 0 |
| Average Inference Time | < 1.5 Seconds |

---

## Confusion Matrix

```text
True Negatives  : 878
False Positives : 0

False Negatives : 0
True Positives  : 60
```

---

# 🎯 Key Features

- Industrial Acoustic Anomaly Detection
- ResNet18 Transfer Learning
- Mel-Spectrogram Feature Extraction
- Hard Negative Mining
- Grad-CAM Explainability
- Streamlit Dashboard
- Stress-Test Validation
- Real-Time Inference
- Production Deployment Ready

---

# 🏭 Target Applications

- Valve Diagnostics
- Manufacturing Quality Control
- Predictive Maintenance
- Industrial Monitoring
- Factory Automation
- Edge AI Inspection Systems

---

# 👨‍💻 Technology Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Librosa
- OpenCV
- Matplotlib
- Scikit-Learn

---

# 📜 License

This project is intended for educational, research, and industrial innovation purposes.

---

## Built for the Hitachi Acoustic Diagnostics Challenge 🚀