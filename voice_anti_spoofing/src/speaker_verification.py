import os
import torch
import numpy as np
from typing import Tuple, Union, Optional
import io
from src import config, audio, features

class SpeakerVerifier:
    """
    Speaker Verification module using pretrained ECAPA-TDNN speaker embeddings.
    """
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = str(cache_dir or config.SPEAKER_MODEL_CACHE_DIR)
        self.device = config.DEVICE
        self.classifier = None
        self.use_fallback = False
        self._init_model()

    def _init_model(self):
        """
        Initialize SpeechBrain ECAPA-TDNN speaker encoder model, or set up fallback.
        """
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            print(f"[SpeakerVerifier] Loading pretrained ECAPA-TDNN model from cache: {self.cache_dir}...")
            
            # HuggingFace SpeechBrain ECAPA-TDNN model
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=self.cache_dir,
                run_opts={"device": str(self.device)}
            )
            print("[SpeakerVerifier] SpeechBrain ECAPA-TDNN model successfully loaded.")
        except Exception as e:
            print(f"[SpeakerVerifier] Warning: Failed to load SpeechBrain model ({e}). Using robust spectral embedding fallback.")
            self.use_fallback = True

    def extract_embedding(self, audio_source: Union[str, np.ndarray, io.BytesIO]) -> np.ndarray:
        """
        Extract speaker embedding vector for given audio sample.
        
        Returns:
            np.ndarray: 1D normalized float32 embedding vector.
        """
        waveform = audio.load_audio(audio_source, sr=config.SAMPLE_RATE, duration=config.AUDIO_DURATION)
        
        if not self.use_fallback and self.classifier is not None:
            try:
                # Convert waveform to PyTorch tensor [1, samples]
                wav_tensor = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embeddings = self.classifier.encode_batch(wav_tensor)
                    # Squeeze shape to 1D
                    emb = embeddings.squeeze().cpu().numpy()
                    
                # L2 normalization
                norm = np.linalg.norm(emb) + 1e-7
                return (emb / norm).astype(np.float32)
            except Exception as e:
                print(f"[SpeakerVerifier] Embedding extraction error: {e}. Falling back to spectral feature vector.")
        
        # Fallback spectral feature embedding
        return features.compute_mfcc_features(waveform, sr=config.SAMPLE_RATE)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute Cosine Similarity between two speaker embedding vectors.
        
        Returns:
            float: Cosine similarity score in range [-1.0, 1.0].
        """
        norm1 = np.linalg.norm(embedding1) + 1e-7
        norm2 = np.linalg.norm(embedding2) + 1e-7
        dot_product = np.dot(embedding1, embedding2)
        sim = dot_product / (norm1 * norm2)
        # Clip to [0.0, 1.0] for similarity percentage calculation
        return float(np.clip((sim + 1.0) / 2.0 if self.use_fallback else sim, 0.0, 1.0))

    def verify_speaker(
        self,
        candidate_audio: Union[str, np.ndarray, io.BytesIO],
        enrolled_embedding: np.ndarray,
        threshold: float = config.SPEAKER_THRESHOLD
    ) -> Tuple[bool, float, str]:
        """
        Verify if candidate audio matches enrolled speaker.
        
        Returns:
            Tuple[bool (PASS/FAIL), float (Similarity Score 0.0-1.0), str (Message)]
        """
        candidate_emb = self.extract_embedding(candidate_audio)
        similarity = self.compute_similarity(candidate_emb, enrolled_embedding)
        
        is_pass = similarity >= threshold
        status_msg = f"Speaker Match: PASS ({similarity*100:.2f}%)" if is_pass else f"Speaker Match: FAIL ({similarity*100:.2f}%)"
        
        return is_pass, similarity, status_msg
