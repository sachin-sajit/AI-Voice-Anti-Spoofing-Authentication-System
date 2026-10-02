import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
from src import config

def compute_equal_error_rate(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the corresponding optimal decision threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    
    # EER is where FPR == FNR
    eer_idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    optimal_threshold = float(thresholds[eer_idx])
    
    return eer, optimal_threshold

def evaluate_and_save_results(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    history: Optional[Dict[str, List[float]]] = None,
    output_dir: Path = config.RESULTS_DIR
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics and save plots & reports to results/ directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred_binary = (y_pred_probs >= config.SPOOF_THRESHOLD).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred_binary))
    prec = float(precision_score(y_true, y_pred_binary, zero_division=0))
    rec = float(recall_score(y_true, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = float(auc(fpr, tpr))
    eer, optimal_thresh = compute_equal_error_rate(y_true, y_pred_probs)
    
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # 1. Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['BONAFIDE (LIVE)', 'SPOOF (REPLAY)'],
        yticklabels=['BONAFIDE (LIVE)', 'SPOOF (REPLAY)']
    )
    plt.title('Voice Anti-Spoofing Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # 2. Plot ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.axhline(y=1-eer, color='red', linestyle=':', label=f'EER = {eer*100:.2f}%')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) - Anti-Spoof Model')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    # 3. Plot Training History if provided
    if history and 'train_loss' in history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        epochs_range = range(1, len(history['train_loss']) + 1)
        
        ax1.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
        ax1.plot(epochs_range, history['val_loss'], 'r-s', label='Val Loss')
        ax1.set_title('Loss History')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('BCE Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs_range, history['train_acc'], 'b-o', label='Train Acc')
        ax2.plot(epochs_range, history['val_acc'], 'r-s', label='Val Acc')
        ax2.set_title('Accuracy History')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        hist_path = output_dir / "training_history.png"
        plt.savefig(hist_path, dpi=300)
        plt.close()
        
    # 4. Save Text Evaluation Report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("====================================================\n")
        f.write("  VOICE ANTI-SPOOFING MODEL EVALUATION REPORT\n")
        f.write("====================================================\n\n")
        f.write(f"Accuracy         : {acc * 100.0:.2f}%\n")
        f.write(f"Precision        : {prec * 100.0:.2f}%\n")
        f.write(f"Recall           : {rec * 100.0:.2f}%\n")
        f.write(f"F1-Score         : {f1 * 100.0:.2f}%\n")
        f.write(f"ROC-AUC          : {roc_auc:.4f}\n")
        f.write(f"Equal Error Rate : {eer * 100.0:.2f}%\n")
        f.write(f"Optimal Threshold: {optimal_thresh:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TN (Live correctly classified): {cm[0, 0]}\n")
        f.write(f"  FP (Live misclassified spoof) : {cm[0, 1]}\n")
        f.write(f"  FN (Replay misclassified live): {cm[1, 0]}\n")
        f.write(f"  TP (Replay correctly classified): {cm[1, 1]}\n")

    print(f"\n[Utils] Evaluation results successfully saved to {output_dir}")
    print(f"Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}% | EER: {eer*100:.2f}%")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "eer": eer
    }
