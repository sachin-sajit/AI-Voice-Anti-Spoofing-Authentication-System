import os
import torch
import numpy as np
from typing import Dict, Any, Union, Optional
import io
from src import config, audio, features
from src.antispoof_model import AntiSpoofCNN
from src.speaker_verification import SpeakerVerifier
from src.enrollment import EnrollmentManager

class VoiceAuthenticator:
    """
    End-to-End Voice Authentication & Anti-Spoofing Pipeline.
    Integrates Speaker Verification + Anti-Spoof CNN + Decision Engine.
    """
    def __init__(
        self,
        antispoof_model_path: Union[str, os.PathLike] = config.ANTISPOOF_MODEL_PATH,
        speaker_threshold: float = config.SPEAKER_THRESHOLD,
        spoof_threshold: float = config.SPOOF_THRESHOLD
    ):
        self.device = config.DEVICE
        self.speaker_threshold = speaker_threshold
        self.spoof_threshold = spoof_threshold
        
        # Initialize Sub-modules
        self.verifier = SpeakerVerifier()
        self.enrollment_mgr = EnrollmentManager(verifier=self.verifier)
        
        # Anti-Spoof Model
        self.antispoof_model = AntiSpoofCNN().to(self.device)
        self.model_loaded = False
        self.load_antispoof_model(antispoof_model_path)

    def load_antispoof_model(self, model_path: Union[str, os.PathLike]):
        """Load trained AntiSpoofCNN model weights."""
        model_path = str(model_path)
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.antispoof_model.load_state_dict(state_dict)
                self.antispoof_model.eval()
                self.model_loaded = True
                print(f"[VoiceAuthenticator] AntiSpoof model loaded successfully from {model_path}")
            except Exception as e:
                print(f"[VoiceAuthenticator] Error loading AntiSpoof weights: {e}")
                self.model_loaded = False
        else:
            print(f"[VoiceAuthenticator] Warning: AntiSpoof model weights not found at {model_path}. Train model first.")
            self.model_loaded = False

    def predict_liveness(self, waveform: np.ndarray) -> float:
        """
        Run AntiSpoof model on waveform to compute replay probability.
        
        Returns:
            float: replay_probability in range [0.0, 1.0]
                   0.0 = Live / Bona-fide
                   1.0 = Replay / Spoof
        """
        # Compute Log-Mel Spectrogram
        spectrogram = features.compute_log_mel_spectrogram(waveform, sr=config.SAMPLE_RATE)
        tensor_in = features.spectrogram_to_tensor(spectrogram).to(self.device)
        
        if self.model_loaded:
            self.antispoof_model.eval()
            with torch.no_grad():
                prob = self.antispoof_model(tensor_in).item()
            return float(prob)
        else:
            # Fallback spectral heuristic if model not trained yet
            print("[VoiceAuthenticator] Warning: AntiSpoof model not trained yet. Computing spectral fallback estimation.")
            high_freq = np.mean(np.abs(spectrogram[:15, :]))
            low_freq = np.mean(np.abs(spectrogram[15:, :])) + 1e-7
            ratio = float(high_freq / low_freq)
            return float(np.clip(ratio * 0.4, 0.05, 0.95))

    def authenticate(
        self,
        audio_source: Union[str, np.ndarray, io.BytesIO],
        speaker_threshold_override: Optional[float] = None,
        spoof_threshold_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Full authentication pipeline execution.
        """
        spk_thresh = speaker_threshold_override if speaker_threshold_override is not None else self.speaker_threshold
        spf_thresh = spoof_threshold_override if spoof_threshold_override is not None else self.spoof_threshold
        
        # 1. Preprocess waveform
        try:
            waveform = audio.load_audio(audio_source, sr=config.SAMPLE_RATE, duration=config.AUDIO_DURATION)
        except Exception as e:
            return {
                "final_decision": "ACCESS DENIED",
                "status_code": "AUDIO_ERROR",
                "reason": f"Audio processing error: {e}",
                "speaker_pass": False,
                "speaker_similarity": 0.0,
                "is_live": False,
                "replay_probability": 1.0
            }
            
        # 2. Check signal validity (silence/volume check)
        is_valid, val_msg = audio.validate_audio_signal(waveform)
        if not is_valid:
            return {
                "final_decision": "ACCESS DENIED",
                "status_code": "INVALID_AUDIO",
                "reason": val_msg,
                "speaker_pass": False,
                "speaker_similarity": 0.0,
                "is_live": False,
                "replay_probability": 1.0
            }

        # 3. Check Enrollment Status
        enrolled_embedding = self.enrollment_mgr.get_enrolled_embedding()
        if enrolled_embedding is None:
            return {
                "final_decision": "ACCESS DENIED",
                "status_code": "NO_ENROLLED_USER",
                "reason": "No authorized speaker enrolled. Please enroll an authorized user profile first.",
                "speaker_pass": False,
                "speaker_similarity": 0.0,
                "is_live": False,
                "replay_probability": 1.0
            }

        # 4. Speaker Verification Engine
        spk_pass, spk_sim, spk_msg = self.verifier.verify_speaker(
            waveform, enrolled_embedding, threshold=spk_thresh
        )

        # 5. Anti-Spoof Liveness Engine
        replay_prob = self.predict_liveness(waveform)
        is_live = replay_prob < spf_thresh

        # 6. Decision Engine Integration
        if spk_pass and is_live:
            final_decision = "ACCESS GRANTED"
            status_code = "SUCCESS"
            reason = "Authorized Speaker Identity Verified & Voice Confirmed LIVE."
        elif spk_pass and not is_live:
            final_decision = "ACCESS DENIED"
            status_code = "REPLAY_ATTACK_DETECTED"
            reason = "REPLAY ATTACK DETECTED: Voice identity matches authorized speaker, BUT audio exhibits playback channel artifacts!"
        elif not spk_pass and is_live:
            final_decision = "ACCESS DENIED"
            status_code = "UNAUTHORIZED_SPEAKER"
            reason = "UNAUTHORIZED SPEAKER: Voice identity does not match enrolled authorized user profile."
        else:
            final_decision = "ACCESS DENIED"
            status_code = "UNAUTHORIZED_AND_REPLAY"
            reason = "UNAUTHORIZED SPEAKER + REPLAY/SPOOF DETECTED: Unknown voice and recorded playback channel detected."

        return {
            "final_decision": final_decision,
            "status_code": status_code,
            "reason": reason,
            "speaker_pass": spk_pass,
            "speaker_similarity": float(spk_sim),
            "speaker_similarity_pct": f"{spk_sim * 100.0:.2f}%",
            "is_live": is_live,
            "liveness_status": "LIVE" if is_live else "SPOOF / REPLAY",
            "replay_probability": float(replay_prob),
            "replay_probability_pct": f"{replay_prob * 100.0:.2f}%",
            "speaker_threshold_used": spk_thresh,
            "spoof_threshold_used": spf_thresh
        }
