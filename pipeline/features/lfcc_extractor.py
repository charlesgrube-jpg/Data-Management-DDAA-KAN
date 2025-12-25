"""
LFCC (Linear Frequency Cepstral Coefficients) Feature Extractor

LFCCs preserve high-frequency detail better than MFCCs,
making them preferred for synthetic speech detection.
"""

import numpy as np
import torch
from typing import Optional
from scipy.fftpack import dct


class LFCCExtractor:
    """
    Extract LFCC features from audio.
    
    Uses linear filterbank (vs mel) to preserve high-frequency artifacts.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_lfcc: int = 60,
        n_filters: int = 128,
        n_fft: int = 512,
        hop_length: int = 256,
        device: str = "auto"
    ):
        self.sample_rate = sample_rate
        self.n_lfcc = n_lfcc
        self.n_filters = n_filters
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Build linear filterbank
        self._filterbank = self._build_linear_filterbank()
        self._backend = self._init_backend()
    
    def _init_backend(self) -> str:
        """Initialize spectrogram backend."""
        try:
            import torchaudio
            return "torchaudio"
        except ImportError:
            try:
                import librosa
                return "librosa"
            except ImportError:
                raise ImportError("Neither torchaudio nor librosa available.")
    
    def _build_linear_filterbank(self) -> np.ndarray:
        """
        Build a linear-spaced filterbank matrix.
        
        Unlike mel filterbanks, linear spacing preserves high-frequency resolution.
        """
        n_freqs = self.n_fft // 2 + 1
        
        # Linear spacing from 0 to Nyquist
        fmax = self.sample_rate / 2
        center_freqs = np.linspace(0, fmax, self.n_filters + 2)
        
        # Convert to FFT bin indices
        bin_freqs = np.floor((self.n_fft + 1) * center_freqs / self.sample_rate).astype(int)
        
        # Build triangular filters
        filterbank = np.zeros((self.n_filters, n_freqs))
        
        for i in range(self.n_filters):
            start = bin_freqs[i]
            center = bin_freqs[i + 1]
            end = bin_freqs[i + 2]
            
            # Rising slope
            if center > start:
                filterbank[i, start:center] = np.linspace(0, 1, center - start)
            
            # Falling slope
            if end > center:
                filterbank[i, center:end] = np.linspace(1, 0, end - center)
        
        return filterbank
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract LFCC from audio array.
        
        Args:
            audio: 1D numpy array of audio samples
            
        Returns:
            2D numpy array of shape (n_lfcc, time_frames)
        """
        # Step 1: Compute power spectrogram
        spec = self._compute_spectrogram(audio)
        
        # Step 2: Apply linear filterbank
        filtered = np.dot(self._filterbank, spec)
        
        # Step 3: Log compression
        log_filtered = np.log(filtered + 1e-10)
        
        # Step 4: DCT to get cepstral coefficients
        lfcc = dct(log_filtered, type=2, axis=0, norm='ortho')[:self.n_lfcc]
        
        return lfcc
    
    def _compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute power spectrogram."""
        if self._backend == "torchaudio":
            return self._spec_torchaudio(audio)
        else:
            return self._spec_librosa(audio)
    
    def _spec_torchaudio(self, audio: np.ndarray) -> np.ndarray:
        """Compute spectrogram using torchaudio."""
        import torchaudio
        
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0
        ).to(self.device)
        
        with torch.no_grad():
            spec = spec_transform(audio_tensor)
        
        return spec.cpu().numpy()
    
    def _spec_librosa(self, audio: np.ndarray) -> np.ndarray:
        """Compute spectrogram using librosa."""
        import librosa
        
        spec = np.abs(librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )) ** 2
        
        return spec
    
    def extract_batch(self, audios: list) -> list:
        """Extract LFCC for a batch of audio arrays."""
        return [self.extract(audio) for audio in audios]
    
    def extract_file(self, file_path: str) -> np.ndarray:
        """Load and extract LFCC from a WAV file."""
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
    
    extractor = LFCCExtractor(sample_rate=sr)
    lfcc = extractor.extract(test_audio)
    
    print(f"Input shape: {test_audio.shape}")
    print(f"LFCC shape: {lfcc.shape}")
    print(f"Backend: {extractor._backend}")
