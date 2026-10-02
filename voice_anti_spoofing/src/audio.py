import os
import io
import numpy as np
import librosa
import soundfile as sf
from typing import Union, Tuple, Optional
from src import config

def load_audio(
    source: Union[str, io.BytesIO, np.ndarray],
    sr: int = config.SAMPLE_RATE,
    duration: float = config.AUDIO_DURATION,
    mono: bool = True
) -> np.ndarray:
    """
    Load an audio file or array, resample to target rate, and adjust length to fixed duration.
    
    Args:
        source: File path, BytesIO object, or numpy array.
        sr: Target sample rate.
        duration: Desired audio duration in seconds.
        mono: Convert stereo to mono if True.
        
    Returns:
        np.ndarray: Preprocessed 1D float32 audio waveform.
    """
    target_samples = int(sr * duration)
    
    if isinstance(source, np.ndarray):
        audio = source.astype(np.float32)
        current_sr = sr
    elif isinstance(source, (str, os.PathLike)):
        audio, current_sr = sf.read(str(source), dtype='float32')
    elif isinstance(source, io.BytesIO):
        source.seek(0)
        audio, current_sr = sf.read(source, dtype='float32')
    else:
        raise ValueError(f"Unsupported audio source type: {type(source)}")
    
    # Stereo to mono
    if audio.ndim > 1:
        if mono:
            audio = np.mean(audio, axis=1)
        else:
            audio = audio[:, 0]
            
    # Resample if needed
    if current_sr != sr:
        audio = librosa.resample(audio, orig_sr=current_sr, target_sr=sr)
        
    # Trim or pad to exact target_samples length
    audio = fix_audio_length(audio, target_samples)
    
    # Peak normalization
    audio = normalize_audio(audio)
    
    return audio.astype(np.float32)

def fix_audio_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Trim or pad audio array to exact target length.
    """
    if len(audio) == target_samples:
        return audio
    elif len(audio) > target_samples:
        # Trim from center or beginning
        start = (len(audio) - target_samples) // 2
        return audio[start:start + target_samples]
    else:
        # Pad with zeros (or tile if very short)
        pad_len = target_samples - len(audio)
        return np.pad(audio, (0, pad_len), mode='constant', constant_values=0.0)

def normalize_audio(audio: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Peak normalize audio signal to [-1.0, 1.0].
    """
    max_val = np.max(np.abs(audio))
    if max_val > eps:
        return audio / max_val
    return audio

def validate_audio_signal(audio: np.ndarray, min_rms: float = 0.003) -> Tuple[bool, str]:
    """
    Check if the audio signal is valid (not silent or zeroed).
    """
    if audio is None or len(audio) == 0:
        return False, "Audio data is empty."
    
    rms = np.sqrt(np.mean(audio**2))
    if rms < min_rms:
        return False, f"Audio volume too low (RMS: {rms:.5f}). Please speak louder or check your microphone."
        
    return True, "Audio valid."

def record_audio_sounddevice(
    duration: float = config.AUDIO_DURATION,
    sr: int = config.SAMPLE_RATE
) -> np.ndarray:
    """
    Record audio directly from system default microphone using sounddevice.
    """
    import sounddevice as sd
    num_samples = int(duration * sr)
    recording = sd.rec(num_samples, samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    waveform = recording.flatten()
    return load_audio(waveform, sr=sr, duration=duration)

def save_audio_file(audio: np.ndarray, file_path: Union[str, os.PathLike], sr: int = config.SAMPLE_RATE):
    """
    Save waveform to a WAV file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(str(file_path), audio, sr, subtype='PCM_16')
