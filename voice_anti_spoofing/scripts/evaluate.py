import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import config
from src.antispoof_model import AntiSpoofCNN
from src.train_antispoof import AntiSpoofDataset
from src.utils import evaluate_and_save_results

def main():
    print("[Evaluate] Starting Anti-Spoofing Model Evaluation...")
    
    test_json = config.PROCESSED_DATA_DIR / "test_split.json"
    if not test_json.exists():
        print("[Evaluate] Error: test_split.json not found. Please run prepare_dataset.py and train.py first.")
        return
        
    with open(test_json, "r") as f:
        test_samples = json.load(f)
        
    if not config.ANTISPOOF_MODEL_PATH.exists():
        print(f"[Evaluate] Error: Trained model weights not found at {config.ANTISPOOF_MODEL_PATH}. Run train.py first.")
        return
        
    # Load model
    device = config.DEVICE
    model = AntiSpoofCNN().to(device)
    model.load_state_dict(torch.load(config.ANTISPOOF_MODEL_PATH, map_location=device))
    model.eval()
    
    # Load test data
    test_dataset = AntiSpoofDataset(test_samples, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    y_true = []
    y_pred_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            probs = model(inputs)
            
            y_true.extend(labels.squeeze(-1).numpy().tolist())
            y_pred_probs.extend(probs.squeeze(-1).cpu().numpy().tolist())
            
    y_true_arr = np.array(y_true)
    y_pred_probs_arr = np.array(y_pred_probs)
    
    # Load training history if exists
    history = None
    hist_json = config.PROCESSED_DATA_DIR / "training_history.json"
    if hist_json.exists():
        with open(hist_json, "r") as f:
            history = json.load(f)
            
    # Compute metrics and generate charts
    metrics = evaluate_and_save_results(
        y_true=y_true_arr,
        y_pred_probs=y_pred_probs_arr,
        history=history,
        output_dir=config.RESULTS_DIR
    )
    
    print("\n[Evaluate] Evaluation complete! Check the 'results/' folder for PNG charts and text report.")

if __name__ == "__main__":
    main()
