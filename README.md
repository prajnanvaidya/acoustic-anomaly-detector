# 🏭 Automated Acoustic Diagnostic Suite
### Edge AI Inference Platform for Non-Destructive Valve Inspection

**🌐 Live Web Application:** [https://acoustic-valve-detector.streamlit.app/](https://acoustic-valve-detector.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Streamlit](https://img.shields.io/badge/Streamlit-Enterprise%20UI-FF4B4B)
![Accuracy](https://img.shields.io/badge/Cross--Domain%20Accuracy-89.02%25-brightgreen)

---

## 📌 Project Overview

In industrial manufacturing environments, mechanical valve failures can lead to catastrophic downtime. This project introduces a **Zero-Defect Acoustic Diagnostic Engine** that listens to 10-second audio footprints of industrial valves and classifies them as either:

- ✅ **Operational (Normal)**
- ⚠️ **Critical Fault (Anomalous)**

Instead of relying on fragile custom architectures, this solution utilizes a transfer-learned **ResNet18** model processing **128-bin Mel-Scale Logarithmic Spectrograms**. 

The latest iterations (V3/V4) of this project solve the critical **Catastrophic Domain Shift** problem. By merging standard Hitachi datasets (`data_0db`, `data_6db`, `data_minus6db`) with the DCASE 2021 Dataset (`data_2021`), the pipeline is hardened to correctly identify anomalies on entirely unseen factory machines operating under novel environmental conditions.

The suite also features a live **SCADA-style dashboard** with **Explainable AI (Grad-CAM)** visualizations, allowing operators to understand exactly which acoustic time-frequency regions influenced each prediction.

---

# 🚀 Key Engineering Solutions (V4)

1. **Zero-Leakage Validation:** Implemented `GroupShuffleSplit` with dynamic regex `(id_\d+|section_\d+)` to strictly isolate physical machine IDs across train/val/test splits, mathematically guaranteeing the model cannot "cheat" by memorizing a specific machine's ambient hum.
2. **Domain-Shift Mitigation:** Replaced Global Z-Score with **Instance-Level Min-Max Scaling** to strip out static ambient machine volume differences, allowing the network to evaluate unseen baselines fairly. 
3. **Edge-Optimized I/O (OOM Fix):** Eradicated Windows `ArrayMemoryError` crashes when handling ~19,000 files by dropping RAM-caching in favor of C-backed `soundfile` SSD streaming, and strictly locking tensor math to 32-bit (`float32`) precision.
4. **Moderated Loss Penalties:** Reduced `CrossEntropyLoss` class weights from `[1.0, 5.0]` to `[1.0, 1.5]`, successfully curbing false positive spikes on unfamiliar test domains.

---

# 🧠 Model Architecture & Weights

Because deep learning model weights exceed standard GitHub storage limits, the trained models are hosted externally.

## 📥 Download Pre-trained Models

**Google Drive Link:**
```text
[https://drive.google.com/drive/folders/19L0Oxp_b9rxQ_VTxZ_SGao-Kyx8a2ORC?usp=sharing](https://drive.google.com/drive/folders/19L0Oxp_b9rxQ_VTxZ_SGao-Kyx8a2ORC?usp=sharing)
```
After downloading, place the `.pth` files inside the `models/` directory.

---

# 📂 Project Structure

```text
acoustic-anomaly-detector/
│
├── data_0db/                  # 2019 Hitachi Baseline
├── data_6db/                  # 2019 Hitachi High SNR
├── data_minus6db/             # 2019 Hitachi Heavy Noise
├── data_2021/                 # DCASE 2021 Domain Shift Set
├── data_stress_merged/        
├── demo_samples/              # Random test clips for Streamlit UI
│
├── models/
│   ├── valve_anomaly_resnet18.pth
│   ├── valve_anomaly_resnet18_best.pth
│   └── valve_anomaly_resnet18_v4_robust.pth  # Active Deployment Model
│
├── scripts/
│   ├── train_evaluateV1.py
│   ├── train_evaluateV2.py
│   ├── train_evaluateV3.py
│   ├── train_evaluateV4.py    # Instance-scaled edge-safe pipeline
│   └── visualizer.py
│
├── stress_test/
├── stress_test_errors_data/
├── style/
│
├── app.py                     # Streamlit Deployment UI
├── download_data.py           # Automated 2021 Zenodo/GitHub Fetcher
├── requirements-local.txt
└── requirements.txt
```

---

# ⚙️ Quick Start Guide

## 1️⃣ Installation

Clone the repository and install dependencies:
```bash
git clone [https://github.com/prajnanvaidya/acoustic-anomaly-detector.git](https://github.com/prajnanvaidya/acoustic-anomaly-detector.git)
cd acoustic-anomaly-detector
pip install -r requirements-local.txt
```

## 2️⃣ Data Ingestion

To automatically fetch and sort the required DCASE 2021 evaluation sets:
```bash
python download_data.py
```

## 3️⃣ Train the V4 Model (Latest)

Execute the memory-safe robust training pipeline:
```bash
python scripts/train_evaluateV4.py
```

## 4️⃣ Launch the Dashboard

**Access the Live Cloud Deployment:**
👉 [https://acoustic-valve-detector.streamlit.app/](https://acoustic-valve-detector.streamlit.app/)

**To run the UI locally:**
```bash
streamlit run app.py
```
The dashboard automatically targets `valve_anomaly_resnet18_v4_robust.pth`. You can upload your own `.wav` files or use the **"🎲 Load Random Demo"** button to inject random test samples.

---

# 📊 Cross-Domain Evaluation Metrics (Unseen Machines)

*Note: Evaluated on test sets completely disjointed from training machine IDs.*

| Class | Precision | Recall | F1-Score | Support |
|----------|----------|----------|----------|----------|
| **Normal** | 94.5% | 89.2% | 91.8% | 3127 |
| **Anomaly** | 78.4% | 87.5% | 82.7% | 360 |
| **Overall Accuracy** | **89.02%** | - | - | **3487** |

---

# 🔮 Roadmap: V5 (Coming Soon)
Development for **V5** is currently underway. Future updates will transition the platform from purely supervised constraints to advanced semi-supervised/unsupervised anomaly detection paradigms. Planned features include:
*   **Autoencoder Latent Space Reconstruction:** Using reconstruction error (MSE) to flag anomalies without requiring labeled anomalous training data.
*   **Contrastive Learning (SimCLR):** Self-supervised feature extraction for even stronger resilience against factory background noise.
*   **Dynamic Background Subtraction:** Real-time environmental noise profiling prior to STFT transforms.

---

# 👨‍💻 Technology Stack
- Python (3.8+)
- PyTorch & Torchvision
- Streamlit (SCADA UI)
- Librosa & Soundfile (DSP & Audio I/O)
- Scikit-Learn
- pytorch-grad-cam

---
