import os
import json
import numpy as np
from datetime import datetime
from typing import List, Union, Tuple, Optional
import io
from src import config, audio
from src.speaker_verification import SpeakerVerifier

class EnrollmentManager:
    """
    Manages speaker enrollment: extracts embeddings from multiple audio samples,
    computes speaker centroid, and saves local profile.
    """
    def __init__(self, verifier: Optional[SpeakerVerifier] = None):
        self.verifier = verifier or SpeakerVerifier()
        self.profile_path = config.ENROLLED_DIR / "enrolled_speaker.npy"
        self.metadata_path = config.ENROLLED_DIR / "enrolled_speaker_meta.json"

    def is_enrolled(self) -> bool:
        """Check if an authorized speaker profile exists."""
        return self.profile_path.exists() and self.metadata_path.exists()

    def enroll_speaker(
        self,
        samples: List[Union[str, np.ndarray, io.BytesIO]],
        user_id: str = "Authorized_User"
    ) -> Tuple[bool, str, np.ndarray]:
        """
        Enroll a new speaker using 3-10 short voice samples.
        
        Args:
            samples: List of audio file paths, arrays, or BytesIO objects.
            user_id: Descriptive user tag.
            
        Returns:
            Tuple[bool (success), str (message), np.ndarray (centroid)]
        """
        if len(samples) < 1:
            return False, "Enrollment requires at least 1 voice sample (5-10 recommended).", np.array([])

        embeddings = []
        for idx, sample in enumerate(samples):
            try:
                emb = self.verifier.extract_embedding(sample)
                embeddings.append(emb)
            except Exception as e:
                return False, f"Failed to extract embedding from sample #{idx+1}: {e}", np.array([])

        # Calculate centroid (mean vector) across all samples
        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid) + 1e-7
        centroid = (centroid / norm).astype(np.float32)

        # Save profile array
        np.save(self.profile_path, centroid)

        # Save metadata
        metadata = {
            "user_id": user_id,
            "sample_count": len(samples),
            "enrolled_at": datetime.now().isoformat(),
            "embedding_dim": int(centroid.shape[0]),
            "model_type": "ECAPA-TDNN" if not self.verifier.use_fallback else "SpectralFallback"
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return True, f"Successfully enrolled '{user_id}' using {len(samples)} voice sample(s)!", centroid

    def get_enrolled_embedding(self) -> Optional[np.ndarray]:
        """Load the enrolled speaker centroid embedding vector."""
        if not self.is_enrolled():
            return None
        try:
            return np.load(self.profile_path)
        except Exception as e:
            print(f"[EnrollmentManager] Error loading enrolled embedding: {e}")
            return None

    def get_enrollment_info(self) -> Optional[dict]:
        """Get metadata of enrolled speaker."""
        if not self.metadata_path.exists():
            return None
        try:
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def clear_enrollment(self) -> bool:
        """Remove enrolled user profile to allow fresh re-enrollment."""
        if self.profile_path.exists():
            os.remove(self.profile_path)
        if self.metadata_path.exists():
            os.remove(self.metadata_path)
        return True
