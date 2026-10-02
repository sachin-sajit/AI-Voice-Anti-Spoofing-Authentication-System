import os
from pathlib import Path
import torch

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
ENROLLED_DIR = DATA_DIR / "enrolled"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
SPEAKER_MODEL_CACHE_DIR = MODELS_DIR / "speaker_model_cache"

# Ensure essential directories exist
for folder in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, ENROLLED_DIR, MODELS_DIR, RESULTS_DIR, SPEAKER_MODEL_CACHE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Audio Preprocessing Parameters
SAMPLE_RATE = 16000
AUDIO_DURATION = 3.0  # seconds
NUM_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)  # 48000 samples
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Anti-Spoof Model Path
ANTISPOOF_MODEL_PATH = MODELS_DIR / "antispoof_model.pth"

# Authentication Thresholds
# Cosine similarity >= SPEAKER_THRESHOLD implies same speaker (handles channel attenuation during replay)
SPEAKER_THRESHOLD = 0.35

# Replay probability >= SPOOF_THRESHOLD implies replay/spoof attack
SPOOF_THRESHOLD = 0.50

# Hardware Device Configuration
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
