import numpy as np
import librosa
import torch
from typing import Tuple
from src import config

def compute_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_mels: int = config.N_MELS,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    fmin: float = config.FMIN,
    fmax: float = config.FMAX,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute Log-Mel Spectrogram representation of audio signal.
    
    Returns:
        np.ndarray: 2D array of shape (N_MELS, time_steps), dtype float32.
    """
    # Mel Spectrogram computation
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    
    # Power to Decibels (Log-Mel scale)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Standard Normalization
    if normalize:
        mean = np.mean(log_mel)
        std = np.std(log_mel) + 1e-7
        log_mel = (log_mel - mean) / std
        
    return log_mel.astype(np.float32)

def compute_mfcc_features(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_mfcc: int = 20
) -> np.ndarray:
    """
    Compute MFCCs + Deltas + Delta-Deltas for lightweight speaker embedding fallback.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    combined = np.vstack([mfcc, delta, delta2])
    mean = np.mean(combined, axis=1)
    std = np.std(combined, axis=1) + 1e-7
    feature_vector = np.concatenate([mean, std])
    
    # L2 normalize
    norm = np.linalg.norm(feature_vector) + 1e-7
    return (feature_vector / norm).astype(np.float32)

def spectrogram_to_tensor(spectrogram: np.ndarray) -> torch.Tensor:
    """
    Convert 2D Spectrogram array (N_MELS, time_steps) to PyTorch tensor (1, 1, N_MELS, time_steps).
    """
    tensor = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    return tensor
