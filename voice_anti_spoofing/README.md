# AI-Based Voice Authentication System with Replay Attack Detection

**B.Tech Minor Project** | Real-Time Voice Biometrics & Anti-Spoofing Architecture

---

## 📌 Executive Summary

Modern voice biometric authentication systems are highly susceptible to **replay spoofing attacks**, where an attacker captures an authorized user's genuine voice recording and replays it using a loudspeaker (such as a mobile phone, laptop, or portable speaker) into the microphone.

This project delivers a **complete, working, live-demonstrable voice authentication platform** that operates using standard laptop hardware. It combines two parallel AI models:
1. **Pretrained Speaker Verification Model (ECAPA-TDNN)**: Computes deep voiceprint embeddings to verify speaker identity.
2. **Anti-Spoofing Replay Detection Model (Log-Mel Spectrogram 2D-CNN)**: Analyzes fine acoustic channel artifacts, non-linear speaker diaphragm distortion, and secondary room reverberation to detect replay attacks.

---

## 🏗️ System Architecture

```
                    ┌────────────────────────┐
                    │    MICROPHONE INPUT    │
                    └───────────┬────────────┘
                                │
                                v
                    ┌────────────────────────┐
                    │  AUDIO PREPROCESSING   │
                    │  (16kHz Mono / Norm)   │
                    └───────────┬────────────┘
                                │
            ┌───────────────────┴───────────────────┐
            │                                       │
            v                                       v
┌──────────────────────┐                ┌──────────────────────┐
│ SPEAKER VERIFIER     │                │ ANTI-SPOOF CNN       │
│ (ECAPA-TDNN Model)   │                │ (Log-Mel Spectrogram)│
└───────────┬──────────┘                └───────────┬──────────┘
            │                                       │
            v                                       v
     Speaker Similarity                      Replay Probability
     Score (0 - 100%)                       Score (0 - 100%)
            │                                       │
            └───────────────────┬───────────────────┘
                                │
                                v
                    ┌────────────────────────┐
                    │    DECISION ENGINE     │
                    └───────────┬────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 v                             v
       ┌──────────────────┐          ┌──────────────────┐
       │  ACCESS GRANTED  │          │  ACCESS DENIED   │
       └──────────────────┘          └─────────┬────────┘
                                               │
                                ┌──────────────┴──────────────┐
                                │                             │
                                v                             v
                      ┌──────────────────┐          ┌──────────────────┐
                      │   Unauthorized   │          │  Replay Attack   │
                      │     Speaker      │          │     Detected     │
                      └──────────────────┘          └──────────────────┘
```

---

## 🎯 4-Scenario Demonstration Matrix

The system explicitly evaluates and demonstrates the 4 critical security scenarios:

| Scenario | Speaker Identity | Voice Channel | Speaker Verification | Voice Liveness | Final Decision | Panel Significance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Scenario 1** | Authorized User | Live Mic | **PASS** | **LIVE** | 🟢 **ACCESS GRANTED** | Normal legitimate access |
| **Scenario 2** | Authorized User | Recorded Replay | **PASS** | 🔴 **SPOOF/REPLAY** | 🔴 **ACCESS DENIED** | **Crucial Demo**: Rejects attack even though speaker matches! |
| **Scenario 3** | Unauthorized User | Live Mic | 🔴 **FAIL** | **LIVE** | 🔴 **ACCESS DENIED** | Rejects impostor |
| **Scenario 4** | Unauthorized User | Recorded Replay | 🔴 **FAIL** | 🔴 **SPOOF/REPLAY** | 🔴 **ACCESS DENIED** | Rejects imposter replay |

---

## 🎬 5-MINUTE PROJECT DEMONSTRATION SCRIPT (FOR EVALUATION PANEL)

Follow these exact steps during your live college project evaluation:

### **STEP 1: Launch Application & Enroll Your Voice** (1 Minute)
1. Launch app: `python3 run_app.py` or `streamlit run app/app.py`.
2. Go to **"Authorized Speaker Enrollment"** tab.
3. Record 3 short 3-second samples of your voice (saying *"Open security system"*).
4. Click **"SAVE & ENROLL SPEAKER PROFILE"**. Your centroid profile is stored locally.

### **STEP 2: Demonstrate SCENARIO 1 — Authorized Live Voice** (1 Minute)
1. Go to **"Live Authentication Engine"** tab.
2. Select **"Record Microphone Audio"** and speak directly into the laptop microphone.
3. Click **"AUTHENTICATE VOICE SAMPLE"**.
4. Show panel:
   - Speaker Verification: **PASS** (Similarity ~85–95%)
   - Voice Liveness: **LIVE** (Replay Prob ~1–10%)
   - Final Decision: 🟢 **ACCESS GRANTED**

### **STEP 3: Demonstrate SCENARIO 2 — Authorized Replay Attack** (1.5 Minutes - MOST IMPORTANT)
1. Record your own voice on your mobile phone.
2. Hold your mobile phone speaker near the laptop microphone and play the recording.
3. Click **"AUTHENTICATE VOICE SAMPLE"**.
4. Show panel:
   - Speaker Verification: **PASS** (Recognizes your authorized voice identity)
   - Voice Liveness: 🔴 **SPOOF / REPLAY** (Replay Prob ~80–98%)
   - Final Decision: 🔴 **ACCESS DENIED**
   - Reason: *"REPLAY ATTACK DETECTED: Authorized voice identity matches, BUT audio exhibits playback speaker channel artifacts!"*

### **STEP 4: Demonstrate SCENARIO 3 & 4 — Unauthorized Impostor** (1 Minute.5)
1. Ask another person (or classmate) to speak into the microphone.
2. Show panel that the system immediately rejects the impostor with **ACCESS DENIED** (*Reason: UNAUTHORIZED SPEAKER*).

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9 - 3.12
- Laptop Microphone and Speaker

### Quick Setup Commands
```bash
# Clone or navigate to project directory
cd voice_anti_spoofing

# Install required dependencies
pip install -r requirements.txt

# Prepare dataset (Generates benchmark dataset if ASVspoof is not present)
PYTHONPATH=. python3 scripts/prepare_dataset.py

# Train Anti-Spoof CNN model (runs on GPU / Apple Silicon MPS / CPU)
PYTHONPATH=. python3 scripts/train.py

# Evaluate model and generate plots
PYTHONPATH=. python3 scripts/evaluate.py

# Launch Streamlit GUI Application
python3 run_app.py
```

---

## 📂 Project Structure

```
voice_anti_spoofing/
├── data/
│   ├── raw/                  # ASVspoof / raw audio files
│   ├── processed/            # Train/val/test splits & history
│   ├── features/             # Feature cache
│   └── enrolled/             # Enrolled user vector (enrolled_speaker.npy)
├── models/
│   ├── antispoof_model.pth   # Trained PyTorch Anti-Spoof CNN model
│   └── speaker_model_cache/  # SpeechBrain ECAPA-TDNN cache
├── results/                  # Confusion Matrix, ROC Curve, Report
├── src/
│   ├── config.py             # System parameters & thresholds
│   ├── audio.py              # Audio I/O, resampling, mic recording
│   ├── preprocessing.py      # Replay simulation & data augmentation
│   ├── features.py           # Log-Mel Spectrogram & MFCC extraction
│   ├── antispoof_model.py    # PyTorch 2D AntiSpoofCNN architecture
│   ├── train_antispoof.py    # AntiSpoof dataset loader & PyTorch trainer
│   ├── speaker_verification.py # Pretrained ECAPA-TDNN embedding verifier
│   ├── enrollment.py         # User enrollment centroid manager
│   ├── authentication.py     # End-to-end authentication decision engine
│   └── utils.py              # Metric computation & plotting
├── app/
│   └── app.py                # Streamlit GUI Web Application
├── scripts/
│   ├── prepare_dataset.py    # ASVspoof parser & benchmark dataset generator
│   ├── train.py              # Model training runner
│   └── evaluate.py           # Evaluation runner
├── requirements.txt          # Dependency specifications
├── README.md                 # Project documentation
└── run_app.py                # One-click launcher
```

---

## 📊 Dataset & Training

### ASVspoof Dataset Integration
To train on official ASVspoof 2017 PA or ASVspoof 2019 PA datasets:
1. Download dataset files from official ASVspoof repositories.
2. Extract audio `.wav` or `.flac` files and protocol text files into `data/raw/`.
3. Run `python3 scripts/prepare_dataset.py`. The parser automatically detects official protocol files.

### Benchmark Demonstration Dataset
If no external dataset is provided, `scripts/prepare_dataset.py` generates an out-of-the-box acoustic replay dataset containing genuine speech and realistic physical replay attacks (simulated phone transducer frequency response, diaphragm THD distortion, room impulse response reverb, and background acoustic noise).

---

## 🧪 Model Evaluation Metrics

After running `python3 scripts/evaluate.py`, plots and reports are saved to `results/`:
- **`confusion_matrix.png`**: Visual matrix of True vs Predicted labels.
- **`roc_curve.png`**: ROC Curve and Equal Error Rate (EER).
- **`training_history.png`**: Epoch Loss & Accuracy curves.
- **`evaluation_report.txt`**: Summary text report.

---

## ⚙️ Offline & Laptop Compatibility

- **Hardware Acceleration**: Automatically selects Apple Silicon MPS, NVIDIA CUDA GPU, or CPU.
- **Offline Operation**: Pretrained speaker models and anti-spoof CNN weights are stored locally. After initial setup, no internet connection is required during live demonstration.

---

## 📄 License & Credits
Developed for B.Tech Minor Project evaluation in AI/ML & Audio Signal Processing.
