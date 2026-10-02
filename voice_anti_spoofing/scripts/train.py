import os
import sys
import json
import torch
import platform
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import config
from src.train_antispoof import train_anti_spoof_model
from scripts import prepare_dataset

def check_environment():
    """Check hardware resources, Python environment, PyTorch device capabilities."""
    print("====================================================")
    print("      ENVIRONMENT & HARDWARE CAPABILITY CHECK")
    print("====================================================")
    print(f"OS Platform     : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Version  : {platform.python_version()}")
    print(f"PyTorch Version : {torch.__version__}")
    print(f"Selected Device : {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU Name        : {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"Apple Silicon   : MPS Metal GPU Acceleration Enabled")
    else:
        print("Hardware Status : Running on CPU (Optimized for Laptop)")
    print("====================================================\n")

def main():
    check_environment()
    
    train_json = config.PROCESSED_DATA_DIR / "train_split.json"
    val_json = config.PROCESSED_DATA_DIR / "val_split.json"
    
    if not train_json.exists() or not val_json.exists():
        print("[Train] Dataset splits not found. Triggering dataset preparation...")
        prepare_dataset.main()
        
    with open(train_json, "r") as f:
        train_samples = json.load(f)
    with open(val_json, "r") as f:
        val_samples = json.load(f)
        
    print(f"[Train] Loaded {len(train_samples)} training samples and {len(val_samples)} validation samples.")
    
    history = train_anti_spoof_model(
        train_samples=train_samples,
        val_samples=val_samples,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        lr=config.LEARNING_RATE,
        save_path=config.ANTISPOOF_MODEL_PATH,
        device=config.DEVICE
    )
    
    # Save training history JSON
    history_file = config.PROCESSED_DATA_DIR / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
        
    print(f"\n[Train] Model training successfully completed and saved to {config.ANTISPOOF_MODEL_PATH}")

if __name__ == "__main__":
    main()
