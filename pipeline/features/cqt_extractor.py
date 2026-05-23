"""
CQT (Constant-Q Transform) Feature Extractor

Uses nnAudio for GPU acceleration, with librosa fallback for CPU.
"""

import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path


class CQTExtractor:
    """
    Extract CQT spectrograms from audio.
    
    GPU-accelerated via nnAudio when available.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_bins: int = 84,
        hop_length: int = 512,
        fmin: float = 32.7,  # C1
        device: str = "auto"
    ):
        self.sample_rate = sample_rate
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.fmin = fmin
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self._extractor = None
        self._backend = None
        self._init_extractor()
    
    def _init_extractor(self):
        """Initialize the CQT extractor (nnAudio or librosa)."""
        # Try nnAudio first (GPU)
        try:
            from nnAudio.features import CQT
            self._extractor = CQT(
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_bins=self.n_bins,
                fmin=self.fmin,
                output_format="Magnitude"
            ).to(self.device)
            self._backend = "nnAudio"
            print(f"[CQT] Using nnAudio backend on {self.device}")
        except ImportError:
            # Fallback to librosa (CPU)
            try:
                import librosa
                self._backend = "librosa"
                print(f"[CQT] Using librosa backend (CPU)")
            except ImportError:
                raise ImportError("Neither nnAudio nor librosa available. "
                                  "Install with: pip install nnAudio librosa")
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract CQT from audio array.
        
        Args:
            audio: 1D numpy array of audio samples
            
        Returns:
            2D numpy array of shape (n_bins, time_frames)
        """
        if self._backend == "nnAudio":
            return self._extract_nnaudio(audio)
        else:
            return self._extract_librosa(audio)
    
    def _extract_nnaudio(self, audio: np.ndarray) -> np.ndarray:
        """Extract using nnAudio (GPU)."""
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            cqt = self._extractor(audio_tensor)
        
        # Convert to numpy (n_bins, time)
        return cqt.squeeze(0).cpu().numpy()
    
    def _extract_librosa(self, audio: np.ndarray) -> np.ndarray:
        """Extract using librosa (CPU fallback)."""
        import librosa
        
        cqt = librosa.cqt(
            audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            fmin=self.fmin
        )
        
        # Return magnitude
        return np.abs(cqt)
    
    def extract_batch(self, audios: list) -> list:
        """Extract CQT for a batch of audio arrays."""
        return [self.extract(audio) for audio in audios]
    
    def extract_file(self, file_path: str) -> np.ndarray:
        """Load and extract CQT from a WAV file."""
        import librosa
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        return self.extract(audio)


if __name__ == "__main__":
    # Test
    import numpy as np
    
    # Generate 3 seconds of test audio
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    extractor = CQTExtractor(sample_rate=sr)
    cqt = extractor.extract(test_audio)
    
    print(f"Input shape: {test_audio.shape}")
    print(f"CQT shape: {cqt.shape}")
    print(f"Backend: {extractor._backend}")
