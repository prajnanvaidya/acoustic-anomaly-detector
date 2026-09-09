import io
import os
import random
import datetime

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Acoustic Diagnostic Suite - Hitachi",
    layout="wide",
    page_icon="🏭",
    initial_sidebar_state="collapsed"
)

def html(markup):
    lines = [line.strip() for line in markup.strip().splitlines()]
    return "".join(line for line in lines if line)

def render(markup):
    st.markdown(html(markup), unsafe_allow_html=True)

def format_size(num_bytes):
    size_kb = num_bytes / 1024
    if size_kb >= 1024:
        return f"{size_kb / 1024:.1f} MB"
    return f"{size_kb:.1f} KB"

# --- BULLETPROOF CSS LOADER ---
def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Interface styling missing: Could not find {file_path}")

if os.path.exists(os.path.join("style", "style.css")):
    load_css(os.path.join("style", "style.css"))
else:
    load_css("style.css")

render("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
""")

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor':   '#0B1120',
    'axes.facecolor':     '#0B1120',
    'axes.edgecolor':     '#1C2B45',
    'axes.labelcolor':    '#7A93B4',
    'xtick.color':        '#7A93B4',
    'xtick.labelcolor':   '#7A93B4',
    'ytick.color':        '#7A93B4',
    'ytick.labelcolor':   '#7A93B4',
    'grid.color':         '#1C2B45',
    'grid.alpha':         0.6,
    'text.color':         '#E2EAF4',
    'font.family':        'monospace',
})

if "history" not in st.session_state:
    st.session_state.history = []

if "demo_file" not in st.session_state:
    st.session_state.demo_file = None

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    
    # Must match V4 Architecture (Dropout layer added)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 2)
    )
    
    model_path = "models/valve_anomaly_resnet18_v4_robust.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.warning(f"Model weights not found at {model_path}. Running with untrained weights (UI demo mode).")
    
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()
hw_label = "CUDA · RTX" if device.type == "cuda" else "CPU · EDGE"

def process_and_plot_audio(source):
    y, sr = librosa.load(source, sr=16000, duration=10.0)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig_wave, ax_wave = plt.subplots(figsize=(6, 3.2))
    librosa.display.waveshow(y, sr=sr, ax=ax_wave, color="#0EA5A0", alpha=0.92, linewidth=1.2)
    ax_wave.set_xlabel("Time (s)", fontsize=7.5, labelpad=6)
    ax_wave.set_ylabel("Amplitude", fontsize=7.5, labelpad=6)
    ax_wave.tick_params(labelsize=7)
    ax_wave.grid(True, linestyle="--", linewidth=0.5)
    ax_wave.set_title("")
    fig_wave.tight_layout(pad=0.8)

    fig_spec, ax_spec = plt.subplots(figsize=(6, 3.2))
    img = librosa.display.specshow(
        mel_db, sr=sr, x_axis='time', y_axis='mel',
        ax=ax_spec, cmap='magma'   
    )
    ax_spec.set_xlabel("Time (s)", fontsize=7.5, labelpad=6)
    ax_spec.set_ylabel("Mel Freq (Hz)", fontsize=7.5, labelpad=6)
    ax_spec.tick_params(labelsize=7)
    ax_spec.set_title("")
    cbar = fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB", pad=0.02)
    cbar.ax.tick_params(labelsize=7, colors='#7A93B4')
    cbar.outline.set_edgecolor('#1C2B45')
    cbar.ax.yaxis.set_tick_params(color='#7A93B4')
    fig_spec.tight_layout(pad=0.8)

    # Restored Instance-Level Min-Max Scaling
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    image_tensor = np.stack([mel_norm, mel_norm, mel_norm], axis=0)
    image_tensor = torch.tensor(image_tensor, dtype=torch.float32).unsqueeze(0)

    return image_tensor, fig_wave, fig_spec

render(f"""
<div class="header-band">
<div class="header-left">
<div class="header-icon">🏭</div>
<div>
<p class="header-title">Acoustic Diagnostic Suite</p>
<p class="header-subtitle">Edge AI · Non-Destructive Valve Inspection · Hitachi</p>
</div>
</div>
<div class="header-right">
<div class="status-pill"><div class="pulse-dot"></div>SYSTEM NOMINAL</div>
<div class="hw-badge">{hw_label}</div>
</div>
</div>
""")

tab1, tab2 = st.tabs(["  🔍  Live Diagnostic Console  ", "  📋  Architecture & Benchmarks  "])

with tab1:
    with st.container(border=True):
        render('<p class="section-eyebrow">Step 1 - Input</p>')
        render('<p class="section-title">Asset Data Ingestion</p>')

        uploaded_file = st.file_uploader(
            "Upload valve acoustic footprint",
            type=["wav"],
            label_visibility="collapsed"
        )

        render('<p class="helper-text">No audio file to upload? Test the engine with a built-in factory sample:</p>')

        demo_col, _demo_spacer = st.columns([1, 2])
        with demo_col:
            load_demo = st.button("🎲 Load Random Demo", use_container_width=True, key="load_demo_btn")

        audio_bytes = None
        audio_path = None
        file_name_display = ""
        file_size_display = ""

        if uploaded_file is not None:
            st.session_state.demo_file = None
            audio_bytes = uploaded_file.getvalue()
            file_name_display = uploaded_file.name
            file_size_display = format_size(len(audio_bytes))
        else:
            if load_demo:
                demo_folder = "demo_samples"
                demo_files = (
                    [f for f in os.listdir(demo_folder) if f.lower().endswith(".wav")]
                    if os.path.isdir(demo_folder) else []
                )
                if demo_files:
                    chosen_file = random.choice(demo_files)
                    st.session_state.demo_file = (os.path.join(demo_folder, chosen_file), chosen_file)
                else:
                    st.session_state.demo_file = None
                    st.error("⚠️ The 'demo_samples' folder is missing or empty. Please create it and add .wav files.")

            if st.session_state.demo_file is not None:
                candidate_path, candidate_name = st.session_state.demo_file
                if os.path.exists(candidate_path):
                    audio_path = candidate_path
                    file_name_display = candidate_name
                    file_size_display = format_size(os.path.getsize(candidate_path))
                else:
                    st.session_state.demo_file = None

        has_audio = audio_bytes is not None or audio_path is not None

        if has_audio:
            audio_col, meta_col = st.columns([5, 3])
            with audio_col:
                st.audio(audio_bytes if audio_bytes is not None else audio_path, format='audio/wav')
            with meta_col:
                render(f"""
                <div class="file-meta">
                <div class="file-meta-item">File&nbsp;<span>{file_name_display}</span></div>
                <div class="file-meta-item">Size&nbsp;<span>{file_size_display}</span></div>
                <div class="file-meta-item">SR&nbsp;<span>16 kHz</span></div>
                </div>
                """)

    if has_audio:
        st.write("")
        if st.button("⚡ RUN DIAGNOSTIC", use_container_width=True, key="run_diagnostic_btn"):
            with st.spinner("Compiling acoustic profile · running inference · generating Grad-CAM"):

                source = io.BytesIO(audio_bytes) if audio_bytes is not None else audio_path
                input_tensor, fig_wave, fig_spec = process_and_plot_audio(source)
                input_tensor = input_tensor.to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    confidence, predicted_class = torch.max(probabilities, dim=0)

                pred_idx = predicted_class.item()
                conf_pct = confidence.item() * 100
                is_normal = (pred_idx == 0)

                status_text = "Normal" if is_normal else "Abnormal"
                st.session_state.history.insert(0, {
                    "File":       file_name_display,
                    "Result":     status_text,
                    "Confidence": f"{conf_pct:.1f}%",
                    "Time":       datetime.datetime.now().strftime("%H:%M:%S")
                })

                target_layers = [model.layer4[-1]]
                with GradCAM(model=model, target_layers=target_layers) as cam:
                    targets = [ClassifierOutputTarget(pred_idx)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

                rgb_img = input_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                fig_xai, ax_xai = plt.subplots(figsize=(14, 3.4))
                ax_xai.imshow(cam_image, aspect='auto')
                ax_xai.set_xlabel("Frame Progression -> Time", fontsize=8, labelpad=6)
                ax_xai.set_ylabel("Frequency Target", fontsize=8, labelpad=6)
                ax_xai.tick_params(labelsize=7)
                ax_xai.set_title("")
                for spine in ax_xai.spines.values():
                    spine.set_edgecolor('#1C2B45')
                fig_xai.tight_layout(pad=0.8)

                st.write("")
                if is_normal:
                    banner = html("""
                        <div class="status-normal">
                        <div class="status-icon">&#9989;</div>
                        <div>
                        <p class="status-label status-label-normal">Operational Status</p>
                        <p class="status-heading">Asset Nominal - No Faults Detected</p>
                        <p class="status-desc">Acoustic signature matches healthy baseline. No evidence of friction, cavitation, erosion, or structural degradation within the 10-second window.</p>
                        </div>
                        </div>
                    """)
                else:
                    banner = html("""
                        <div class="status-fault">
                        <div class="status-icon">&#9888;</div>
                        <div>
                        <p class="status-label status-label-fault">Critical Fault</p>
                        <p class="status-heading">Anomalous Signature - Immediate Inspection Required</p>
                        <p class="status-desc">Anomalous frequency deviations detected. High probability of internal erosion, cavitation, or mechanical misfire. See the Grad-CAM map below for the exact fault window.</p>
                        </div>
                        </div>
                    """)

                conf_class = "metric-value-fault" if not is_normal else ""
                metric = html(f"""
                    <div class="metric-card">
                    <p class="metric-label">AI Confidence Index</p>
                    <p class="metric-value {conf_class}">{conf_pct:.2f}<span class="metric-unit">%</span></p>
                    </div>
                """)

                st.markdown(
                    '<div class="panel">'
                    '<p class="section-eyebrow">Diagnostic Intelligence Report</p>'
                    '<div class="report-grid">' + banner + metric + '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.write("")
                with st.container(border=True):
                    render('<p class="section-eyebrow">Deep Signal Analytics</p>')
                    wave_col, spec_col = st.columns(2)

                    with wave_col:
                        st.pyplot(fig_wave, use_container_width=True)
                        render('<p class="plot-label">Plot A - Raw Audio Signal (Time Domain)</p>')

                    with spec_col:
                        st.pyplot(fig_spec, use_container_width=True)
                        render('<p class="plot-label">Plot B - Mel-Scale Spectrogram (Frequency Domain)</p>')

                plt.close(fig_wave)
                plt.close(fig_spec)

                st.write("")
                with st.container(border=True):
                    render('<p class="section-eyebrow">Explainable AI - Grad-CAM Attention Map</p>')

                    if is_normal:
                        render("""
                        <div class="xai-card-normal">
                        <p class="xai-eyebrow xai-eyebrow-normal">Interpretation · Nominal Pattern</p>
                        <p class="xai-text">Attention is distributed in diffuse vertical bands at regular intervals - the model is validating the steady, rhythmic mechanical heartbeat of the valve. No concentrated anomaly region found.</p>
                        </div>
                        """)
                    else:
                        render("""
                        <div class="xai-card-fault">
                        <p class="xai-eyebrow xai-eyebrow-fault">Interpretation · Fault Localized</p>
                        <p class="xai-text">The network has suppressed background noise and focused on a concentrated frequency cluster (bright region). This identifies the precise time-frequency coordinate of the mechanical failure, not a broadband signal artifact.</p>
                        </div>
                        """)

                    st.pyplot(fig_xai, use_container_width=True)

                plt.close(fig_xai)

    st.write("")
    with st.container(border=True):
        log_label_col, log_clear_col = st.columns([8, 2])
        with log_label_col:
            render('<p class="section-eyebrow">Recent Analyses Log</p>')
        with log_clear_col:
            if st.button("Clear", use_container_width=True, key="clear_history_btn"):
                st.session_state.history = []
                st.rerun()

        if st.session_state.history:
            rows = ""
            for row in st.session_state.history:
                tag_cls = "tag-normal" if row["Result"] == "Normal" else "tag-fault"
                rows += (
                    "<tr>"
                    f'<td><div class="file-cell" title="{row["File"]}">{row["File"]}</div></td>'
                    f'<td><span class="{tag_cls}">{row["Result"]}</span></td>'
                    f'<td><span class="conf-mono">{row["Confidence"]}</span></td>'
                    f'<td><span class="time-mono">{row["Time"]}</span></td>'
                    "</tr>"
                )
            st.markdown(
                '<div class="table-scroll"><table class="history-table">'
                '<thead><tr>'
                '<th>Source File</th><th>Result</th><th>Confidence</th><th>Timestamp</th>'
                '</tr></thead><tbody>' + rows + '</tbody></table></div>',
                unsafe_allow_html=True
            )
        else:
            render('<div class="empty-state">No analyses yet - upload a valve acoustic file above to begin.</div>')

with tab2:
    render("""
    <div class="panel">
    <p class="section-eyebrow">End-to-End Pipeline Architecture</p>
    <p class="section-title">Acoustic Inspection Engine Workflow</p>
    <div class="pipeline-grid">
    <div class="card-inset">
    <p class="pipeline-step-number">01 · Ingestion</p>
    <p class="pipeline-step-title">16 kHz WAV Stream</p>
    <p class="pipeline-step-desc">10-second continuous valve acoustic recordings loaded via Librosa at 16,000 samples/sec.</p>
    </div>
    <div class="card-inset">
    <p class="pipeline-step-number">02 · Robust DSP Transform</p>
    <p class="pipeline-step-title">128-Bin Mel-STFT & Min-Max Scaling</p>
    <p class="pipeline-step-desc">Windowed STFT mapped to 128 bins. Subject to SpecAugment (frequency/time masking) and instance-level Min-Max scaling.</p>
    </div>
    <div class="card-inset">
    <p class="pipeline-step-number">03 · Inference</p>
    <p class="pipeline-step-title">Regularized ResNet18</p>
    <p class="pipeline-step-desc">Transfer-learned CNN heavily regularized via Dropout and Weight Decay to prevent machine-ID memorization.</p>
    </div>
    <div class="card-inset">
    <p class="pipeline-step-number">04 · Explainability</p>
    <p class="pipeline-step-title">Grad-CAM Heatmap</p>
    <p class="pipeline-step-desc">Targeted backprop on layer4[-1] activations projects anomalies into time-frequency space.</p>
    </div>
    </div>
    </div>
    """)

    st.write("")

    render(f"""
    <div class="panel-grid">
    <div class="panel">
    <p class="section-eyebrow">Empirical Verification</p>
    <p class="section-title">Cross-Domain Evaluation Metrics (Unseen Machines)</p>
    <div class="table-scroll">
    <table class="history-table">
    <thead>
    <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
    </thead>
    <tbody>
    <tr>
    <td><strong>Normal</strong></td>
    <td>94.5%</td>
    <td>89.2%</td>
    <td>91.8%</td>
    <td><span class="conf-mono">3127</span></td>
    </tr>
    <tr>
    <td><strong>Anomaly</strong></td>
    <td>78.4%</td>
    <td>87.5%</td>
    <td>82.7%</td>
    <td><span class="conf-mono">360</span></td>
    </tr>
    <tr style="border-top: 2px solid #1C2B45;">
    <td><strong>Overall Accuracy</strong></td>
    <td colspan="3" style="text-align: center; color: #E2EAF4;"><strong>89.02%</strong></td>
    <td><span class="conf-mono">3487</span></td>
    </tr>
    </tbody>
    </table>
    </div>
    <p class="panel-caption">*Evaluated via GroupShuffleSplit across disjointed machine sections to eliminate data leakage.</p>
    </div>
    <div class="panel">
    <p class="section-eyebrow">Operational Envelope</p>
    <p class="section-title">Edge Deployment Constraints</p>
    <div class="table-scroll">
    <table class="history-table">
    <tbody>
    <tr><td>Execution Target</td><td><strong>{hw_label}</strong></td></tr>
    <tr><td>End-to-End Latency</td><td><span class="conf-mono">&lt; 1.50 s</span></td></tr>
    <tr><td>Analysis Window</td><td>10.0 seconds (160,000 samples)</td></tr>
    <tr><td>Input Audio Format</td><td>WAV (16 kHz, Mono/Stereo)</td></tr>
    <tr><td>Active Weights</td><td><code>valve_anomaly_resnet18_v4_robust.pth</code></td></tr>
    <tr><td>XAI Target Layer</td><td><code>model.layer4[-1]</code></td></tr>
    </tbody>
    </table>
    </div>
    </div>
    </div>
    """)