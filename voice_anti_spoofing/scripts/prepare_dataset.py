import os
import sys
import json
import random
import numpy as np
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import List, Tuple, Dict
from src import config, audio, preprocessing

def check_and_parse_asvspoof(raw_dir: Path) -> List[Tuple[str, int]]:
    """
    Search for ASVspoof protocol files (ASVspoof 2017 / 2019 PA / LA) and parse them.
    Returns list of (file_path_str, label_int).
    """
    dataset_samples = []
    
    # Search for protocol files
    protocol_files = list(raw_dir.glob("**/protocol*.txt")) + list(raw_dir.glob("**/*cm_protocols*.txt"))
    
    if not protocol_files:
        return []
        
    print(f"[PrepareDataset] Found official ASVspoof protocol file: {protocol_files[0]}")
    
    with open(protocol_files[0], "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            
            # ASVspoof format usually: SPEAKER_ID AUDIO_FILE_NAME - ATTACK_ID KEY
            # or: SPEAKER_ID AUDIO_FILE_NAME SYSTEM_ID KEY
            file_id = parts[1]
            key_str = parts[-1].lower()
            
            label = 0 if key_str in ["bonafide", "genuine"] else 1
            
            # Find matching file on disk
            matching_files = list(raw_dir.glob(f"**/{file_id}.wav")) + list(raw_dir.glob(f"**/{file_id}.flac"))
            if matching_files:
                dataset_samples.append((str(matching_files[0]), label))
                
    print(f"[PrepareDataset] Parsed {len(dataset_samples)} samples from official ASVspoof dataset.")
    return dataset_samples

def generate_benchmark_synthetic_dataset(
    output_dir: Path,
    num_speakers: int = 10,
    samples_per_speaker: int = 16
) -> List[Tuple[str, int]]:
    """
    Generate a high-quality, realistic synthetic baseline dataset for out-of-the-box student laptop training.
    Creates clean bona-fide speech waveforms and realistic acoustic replay attack counterparts.
    """
    print("[PrepareDataset] ASVspoof dataset directory empty. Generating realistic demonstration dataset...")
    
    bonafide_dir = output_dir / "bonafide"
    spoof_dir = output_dir / "spoof"
    os.makedirs(bonafide_dir, exist_ok=True)
    os.makedirs(spoof_dir, exist_ok=True)
    
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)
    
    sr = config.SAMPLE_RATE
    duration = config.AUDIO_DURATION
    num_samples = config.NUM_SAMPLES
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    dataset_records = []
    
    # Common speech phrase pitch patterns
    vowel_formants = [
        (300, 850, 2200),   # 'a' sound
        (500, 1500, 2500),  # 'e' sound
        (300, 2300, 3000),  # 'i' sound
        (400, 1000, 2400),  # 'o' sound
        (350, 800, 2200)    # 'u' sound
    ]
    
    # Distinct pitch & formant characteristics per speaker
    speaker_configs = {
        1: (95.0, (280, 850, 2200)),    # Deep male voice
        2: (230.0, (550, 1800, 2800)),  # High female voice
        3: (130.0, (400, 1300, 2400)),  # Baritone male voice
        4: (195.0, (480, 1600, 2700)),  # Female voice 2
        5: (110.0, (320, 1150, 2300)),  # Male voice 2
        6: (250.0, (600, 2000, 3000)),  # High soprano voice
        7: (150.0, (420, 1400, 2500)),  # Tenor voice
        8: (210.0, (500, 1700, 2750)),  # Alto voice
        9: (120.0, (350, 1200, 2350)),  # Low male voice
        10: (220.0, (520, 1750, 2850)), # Female voice 3
        11: (160.0, (440, 1450, 2550)), # Mid male voice
        12: (240.0, (580, 1900, 2950))  # High female voice 2
    }
    
    for spk_idx in range(1, num_speakers + 1):
        base_f0, formants = speaker_configs.get(spk_idx, (100.0 + spk_idx * 15.0, (400, 1400, 2400)))
        
        for sample_idx in range(1, samples_per_speaker + 1):
            # Generate synthetic speech harmonics with pitch modulation
            f0_mod = base_f0 + (10 + sample_idx) * np.sin(2 * np.pi * (1.2 + sample_idx * 0.1) * t)
            phase = 2 * np.pi * np.cumsum(f0_mod) / sr
            
            # Harmonic voice signal
            speech_signal = np.zeros(num_samples)
            for harmonic in range(1, 15):
                amplitude = 1.0 / (harmonic ** (1.1 + (spk_idx % 3) * 0.1))
                speech_signal += amplitude * np.sin(harmonic * phase)
                
            # Apply speaker-specific vowel formant envelope filter
            f1, f2, f3 = formants
            speech_signal += 0.5 * np.sin(2 * np.pi * f1 * t) + 0.35 * np.sin(2 * np.pi * f2 * t) + 0.2 * np.sin(2 * np.pi * f3 * t)
            
            # Apply speech rhythm amplitude modulation (syllables)
            envelope = 0.5 * (1 + np.sin(2 * np.pi * 3.0 * t)) ** 2
            speech_signal = speech_signal * envelope
            
            # Peak normalize clean bona-fide speech
            speech_signal = audio.normalize_audio(speech_signal.astype(np.float32))
            
            # 1. Save Bona-fide sample
            bona_file = bonafide_dir / f"spk_{spk_idx:02d}_sample_{sample_idx:02d}_bonafide.wav"
            audio.save_audio_file(speech_signal, bona_file, sr=sr)
            dataset_records.append((str(bona_file), 0))
            
            # 2. Save Replay Attack sample (transformed through phone speaker + room reverb + THD)
            replay_signal = preprocessing.simulate_replay_attack(speech_signal, sr=sr)
            spoof_file = spoof_dir / f"spk_{spk_idx:02d}_sample_{sample_idx:02d}_replay.wav"
            audio.save_audio_file(replay_signal, spoof_file, sr=sr)
            dataset_records.append((str(spoof_file), 1))
            
    print(f"[PrepareDataset] Successfully created {len(dataset_records)} audio samples ({len(dataset_records)//2} Bonafide, {len(dataset_records)//2} Replay).")
    return dataset_records

def main():
    raw_dir = config.RAW_DATA_DIR
    processed_dir = config.PROCESSED_DATA_DIR
    
    # Check for official ASVspoof first
    samples = check_and_parse_asvspoof(raw_dir)
    
    # If no ASVspoof found, generate benchmark synthetic replay dataset
    if not samples:
        samples = generate_benchmark_synthetic_dataset(raw_dir, num_speakers=12, samples_per_speaker=12)
        
    random.seed(config.RANDOM_SEED)
    random.shuffle(samples)
    
    total = len(samples)
    train_end = int(total * (1 - config.VAL_SPLIT - config.TEST_SPLIT))
    val_end = int(total * (1 - config.TEST_SPLIT))
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    print(f"[PrepareDataset] Dataset Splits -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # Save splits to JSON
    with open(processed_dir / "train_split.json", "w") as f:
        json.dump(train_samples, f, indent=2)
    with open(processed_dir / "val_split.json", "w") as f:
        json.dump(val_samples, f, indent=2)
    with open(processed_dir / "test_split.json", "w") as f:
        json.dump(test_samples, f, indent=2)
        
    print(f"[PrepareDataset] Splits successfully saved to {processed_dir}")

if __name__ == "__main__":
    main()
