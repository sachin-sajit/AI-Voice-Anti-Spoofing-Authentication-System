import numpy as np
import scipy.signal as signal
from typing import Optional
from src import config

def apply_phone_speaker_response(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Simulate phone/laptop micro-speaker frequency response:
    - High-pass filter cut-off at ~300 Hz (lacks deep bass)
    - Low-pass filter cut-off at ~6500 Hz (lacks crisp high-end)
    - Mid-range resonant peaking around 2 kHz - 4 kHz
    """
    # High-pass filter (Butterworth 4th order)
    sos_hp = signal.butter(4, 300 / (sr / 2), btype='highpass', output='sos')
    audio = signal.sosfilt(sos_hp, audio)
    
    # Low-pass filter (Butterworth 4th order)
    sos_lp = signal.butter(4, 6500 / (sr / 2), btype='lowpass', output='sos')
    audio = signal.sosfilt(sos_lp, audio)
    
    # Resonant peak at 2.5 kHz to simulate cheap transducer resonance
    b_peak, a_peak = signal.iirpeak(2500 / (sr / 2), Q=3.0)
    sos_peak = signal.tf2sos(b_peak, a_peak)
    audio = signal.sosfilt(sos_peak, audio)
    
    return audio.astype(np.float32)

def apply_speaker_harmonic_distortion(audio: np.ndarray, drive: float = 2.5) -> np.ndarray:
    """
    Simulate non-linear speaker diaphragm distortion (Soft clipping / Tanh overdrive).
    """
    distorted = np.tanh(drive * audio) / np.tanh(drive)
    return distorted.astype(np.float32)

def apply_room_reverberation(audio: np.ndarray, sr: int = config.SAMPLE_RATE, room_decay: float = 0.3) -> np.ndarray:
    """
    Simulate secondary acoustic room reverberation via exponential decaying noise impulse response.
    """
    rir_duration = 0.15  # 150 ms reverb tail
    rir_len = int(sr * rir_duration)
    t = np.linspace(0, rir_duration, rir_len)
    
    # Synthetic Room Impulse Response (RIR)
    noise_ir = np.random.randn(rir_len)
    decay_envelope = np.exp(-t / (room_decay * 0.1))
    rir = noise_ir * decay_envelope
    rir = rir / np.max(np.abs(rir))
    
    # Fast convolution
    reverberated = signal.fftconvolve(audio, rir, mode='full')[:len(audio)]
    
    # Mix direct sound + reverberation
    mixed = 0.75 * audio + 0.25 * reverberated
    return mixed.astype(np.float32)

def add_background_noise(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """
    Add white/colored acoustic background noise at specified Signal-to-Noise Ratio (SNR).
    """
    signal_power = np.mean(audio ** 2)
    if signal_power <= 1e-10:
        return audio
        
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    
    noisy_audio = audio + noise
    return noisy_audio.astype(np.float32)

def simulate_replay_attack(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Full pipeline to transform a bona-fide voice recording into a realistic synthetic replay attack:
    1. Phone speaker frequency response
    2. Non-linear speaker distortion
    3. Room reverberation (secondary acoustics)
    4. Background environmental noise
    """
    replay_audio = apply_phone_speaker_response(audio, sr=sr)
    replay_audio = apply_speaker_harmonic_distortion(replay_audio, drive=np.random.uniform(1.8, 3.5))
    replay_audio = apply_room_reverberation(replay_audio, sr=sr, room_decay=np.random.uniform(0.2, 0.4))
    replay_audio = add_background_noise(replay_audio, snr_db=np.random.uniform(18.0, 28.0))
    
    # Peak normalization
    max_val = np.max(np.abs(replay_audio))
    if max_val > 1e-7:
        replay_audio = replay_audio / max_val
        
    return replay_audio.astype(np.float32)

def augment_audio_for_training(audio: np.ndarray, is_spoof: bool, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Apply slight realistic data augmentation during training.
    """
    if is_spoof:
        # Additional variation for spoof class
        if np.random.rand() > 0.3:
            audio = apply_phone_speaker_response(audio, sr=sr)
        if np.random.rand() > 0.5:
            audio = add_background_noise(audio, snr_db=np.random.uniform(15.0, 30.0))
    else:
        # Slight clean augmentation for bona-fide class (gain variation, mild noise)
        gain = np.random.uniform(0.85, 1.15)
        audio = audio * gain
        if np.random.rand() > 0.7:
            audio = add_background_noise(audio, snr_db=np.random.uniform(25.0, 35.0))
            
    max_val = np.max(np.abs(audio))
    if max_val > 1e-7:
        audio = audio / max_val
        
    return audio.astype(np.float32)
