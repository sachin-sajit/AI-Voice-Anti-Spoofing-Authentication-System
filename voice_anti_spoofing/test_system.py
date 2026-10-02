import os
import sys
import numpy as np
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

def test_full_system():
    print("====================================================")
    print(" RUNNING SYSTEM INTEGRATION & VERIFICATION CHECKS ")
    print("====================================================\n")
    
    # 1 & 2. Import modules & syntax check
    print("1. Testing module imports...")
    from src import config, audio, preprocessing, features, antispoof_model, speaker_verification, enrollment, authentication, utils
    from scripts import prepare_dataset, train, evaluate
    print("   ✓ All Python modules imported successfully!")
    
    # 3. Check hardware & config
    print("\n2. Checking hardware & configuration...")
    print(f"   ✓ Device: {config.DEVICE}")
    print(f"   ✓ Sample Rate: {config.SAMPLE_RATE} Hz")
    
    # 4. Verify Model loading & creation
    print("\n3. Verifying AntiSpoofCNN model architecture & checkpoint...")
    model = antispoof_model.AntiSpoofCNN().to(config.DEVICE)
    if config.ANTISPOOF_MODEL_PATH.exists():
        model.load_state_dict(torch.load(config.ANTISPOOF_MODEL_PATH, map_location=config.DEVICE))
        model.eval()
        print(f"   ✓ Model checkpoint loaded successfully from {config.ANTISPOOF_MODEL_PATH}!")
    else:
        print("   ⚠️ Checkpoint missing.")
        
    # 5 & 6. Verify audio loading & preprocessing
    print("\n4. Verifying audio I/O & preprocessing...")
    raw_dir = config.RAW_DATA_DIR
    bona_files = sorted(list((raw_dir / "bonafide").glob("*.wav")))
    spoof_files = sorted(list((raw_dir / "spoof").glob("*.wav")))
    
    assert len(bona_files) > 0, "No bonafide audio samples found!"
    assert len(spoof_files) > 0, "No spoof audio samples found!"
    
    test_wav = audio.load_audio(str(bona_files[0]))
    assert len(test_wav) == config.NUM_SAMPLES, f"Audio waveform length mismatch: {len(test_wav)} vs {config.NUM_SAMPLES}"
    print(f"   ✓ Audio loaded and peak normalized! Waveform shape: {test_wav.shape}")
    
    # 7. Feature extraction
    print("\n5. Verifying Log-Mel Spectrogram feature extraction...")
    spec = features.compute_log_mel_spectrogram(test_wav)
    print(f"   ✓ Log-Mel Spectrogram shape: {spec.shape} (N_MELS x Frames)")
    
    # 8. Speaker Enrollment for Speaker 1
    print("\n6. Verifying Speaker Enrollment System...")
    spk1_bona_files = [f for f in bona_files if "spk_01" in f.name]
    spk1_spoof_files = [f for f in spoof_files if "spk_01" in f.name]
    spk2_bona_files = [f for f in bona_files if "spk_02" in f.name]
    
    enroll_mgr = enrollment.EnrollmentManager()
    success, msg, centroid = enroll_mgr.enroll_speaker([str(f) for f in spk1_bona_files[:3]], user_id="Speaker_01")
    assert success, f"Enrollment failed: {msg}"
    assert enroll_mgr.is_enrolled(), "Enrollment profile check failed!"
    print(f"   ✓ {msg}")
    print(f"   ✓ Enrolled Centroid Vector shape: {centroid.shape}")
    
    # 9 & 10. Anti-Spoof prediction & authentication decision engine
    print("\n7. Verifying End-to-End Voice Authenticator & 4 Scenarios...")
    auth = authentication.VoiceAuthenticator()
    
    # Test Scenario 1 (Authorized Speaker + Live Bona-fide)
    res_sc1 = auth.authenticate(str(spk1_bona_files[0]))
    print(f"   ✓ SCENARIO 1 (Authorized + Live): {res_sc1['final_decision']} | Speaker Sim: {res_sc1['speaker_similarity_pct']} | Replay Prob: {res_sc1['replay_probability_pct']}")
    assert res_sc1['final_decision'] == "ACCESS GRANTED", f"Scenario 1 failed: {res_sc1}"
    
    # Test Scenario 2 (Authorized Speaker + Recorded Replay Attack)
    res_sc2 = auth.authenticate(str(spk1_spoof_files[0]))
    print(f"   ✓ SCENARIO 2 (Authorized + Replay): {res_sc2['final_decision']} | Speaker Sim: {res_sc2['speaker_similarity_pct']} | Replay Prob: {res_sc2['replay_probability_pct']} | Reason: {res_sc2['reason']}")
    assert res_sc2['final_decision'] == "ACCESS DENIED", f"Scenario 2 failed: {res_sc2}"
    assert res_sc2['status_code'] == "REPLAY_ATTACK_DETECTED", f"Scenario 2 status code mismatch: {res_sc2['status_code']}"
    
    # Test Scenario 3 (Unauthorized Speaker + Live Voice)
    res_sc3 = auth.authenticate(str(spk2_bona_files[0]))
    print(f"   ✓ SCENARIO 3 (Unauthorized + Live): {res_sc3['final_decision']} | Speaker Sim: {res_sc3['speaker_similarity_pct']} | Replay Prob: {res_sc3['replay_probability_pct']} | Reason: {res_sc3['reason']}")
    assert res_sc3['final_decision'] == "ACCESS DENIED", f"Scenario 3 failed: {res_sc3}"
    assert res_sc3['status_code'] == "UNAUTHORIZED_SPEAKER", f"Scenario 3 status code mismatch: {res_sc3['status_code']}"
    
    # Test Scenario 4 (Unauthorized Speaker + Replay Attack)
    spk2_spoof_files = [f for f in spoof_files if "spk_02" in f.name]
    res_sc4 = auth.authenticate(str(spk2_spoof_files[0]))
    print(f"   ✓ SCENARIO 4 (Unauthorized + Replay): {res_sc4['final_decision']} | Speaker Sim: {res_sc4['speaker_similarity_pct']} | Replay Prob: {res_sc4['replay_probability_pct']} | Reason: {res_sc4['reason']}")
    assert res_sc4['final_decision'] == "ACCESS DENIED", f"Scenario 4 failed: {res_sc4}"
    assert res_sc4['status_code'] == "UNAUTHORIZED_AND_REPLAY", f"Scenario 4 status code mismatch: {res_sc4['status_code']}"

    print("\n====================================================")
    print(" ALL 14 SYSTEM INTEGRATION TESTS PASSED SUCCESSFULLY! ")
    print("====================================================\n")

if __name__ == "__main__":
    test_full_system()
