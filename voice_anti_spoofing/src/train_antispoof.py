import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from tqdm import tqdm
from src import config, audio, features, preprocessing
from src.antispoof_model import AntiSpoofCNN

class AntiSpoofDataset(Dataset):
    """
    PyTorch Dataset for Voice Anti-Spoofing / Replay Detection.
    Label 0.0 = BONAFIDE / LIVE
    Label 1.0 = SPOOF / REPLAY
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],  # List of (file_path, label_id)
        is_training: bool = True
    ):
        self.samples = samples
        self.is_training = is_training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, label = self.samples[idx]
        
        # Load audio waveform
        waveform = audio.load_audio(file_path, sr=config.SAMPLE_RATE, duration=config.AUDIO_DURATION)
        
        # Data augmentation during training
        if self.is_training:
            waveform = preprocessing.augment_audio_for_training(waveform, is_spoof=(label == 1))
            
        # Compute Log-Mel Spectrogram
        spectrogram = features.compute_log_mel_spectrogram(waveform, sr=config.SAMPLE_RATE)
        
        # Convert to tensor: shape (1, N_MELS, time_steps)
        spec_tensor = features.spectrogram_to_tensor(spectrogram).squeeze(0)
        label_tensor = torch.tensor([float(label)], dtype=torch.float32)
        
        return spec_tensor, label_tensor

def train_anti_spoof_model(
    train_samples: List[Tuple[str, int]],
    val_samples: List[Tuple[str, int]],
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    save_path: Path = config.ANTISPOOF_MODEL_PATH,
    device: torch.device = config.DEVICE
) -> Dict[str, List[float]]:
    """
    Train the AntiSpoofCNN model.
    """
    print(f"\n[TrainAntiSpoof] Starting training on device: {device}")
    print(f"Train set: {len(train_samples)} samples | Val set: {len(val_samples)} samples")
    
    train_dataset = AntiSpoofDataset(train_samples, is_training=True)
    val_dataset = AntiSpoofDataset(val_samples, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = AntiSpoofCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = (outputs >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += inputs.size(0)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0
        
        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = (outputs >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += inputs.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0
        
        scheduler.step(epoch_val_loss)
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best model checkpoint to {save_path}")
            
    print("[TrainAntiSpoof] Training complete!")
    return history
