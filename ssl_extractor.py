"""
SSL Feature Extractor using Wav2Vec2/WavLM.

Extracts state-of-the-art self-supervised learning embeddings
for audio deepfake detection. Replaces CQT spectrograms.

Usage:
    from pipeline.features.ssl_extractor import SSLExtractor
    
    extractor = SSLExtractor(model_name="facebook/wav2vec2-base-960h")
    features = extractor.extract_file("audio.wav")  # Returns (768,) or (T, 768)
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Union, Optional


class SSLExtractor:
    """
    Self-Supervised Learning feature extractor using Wav2Vec2 or WavLM.
    
    Models:
        - facebook/wav2vec2-base-960h (95M params, good baseline)
        - facebook/wav2vec2-large-960h (317M params, better but slower)
        - microsoft/wavlm-base (94M params, best for anti-spoofing)
        - microsoft/wavlm-large (317M params, SOTA but memory hungry)
    """
    
    SAMPLE_RATE = 16000  # All SSL models expect 16kHz
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: str = "cuda",
        pooling: str = "mean",  # "mean", "first", "last", or "none"
        layer: int = -1,  # Which transformer layer to extract from (-1 = last)
    ):
        self.model_name = model_name
        self.device = device
        self.pooling = pooling
        self.layer = layer
        
        print(f"[SSL] Loading {model_name}...")
        
        # Import transformers here to avoid import errors if not installed
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            from transformers import WavLMModel
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers"
            )
        
        # Load processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        if "wavlm" in model_name.lower():
            self.model = WavLMModel.from_pretrained(model_name)
        else:
            self.model = Wav2Vec2Model.from_pretrained(model_name)
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Freeze model (we only extract features, no fine-tuning here)
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"[SSL] Loaded {model_name} on {device}")
        print(f"[SSL] Feature dim: {self.model.config.hidden_size}")
    
    @property
    def feature_dim(self) -> int:
        """Return the feature dimension (typically 768 or 1024)."""
        return self.model.config.hidden_size
    
    def load_audio(self, file_path: Union[str, Path]) -> torch.Tensor:
        """Load and resample audio to 16kHz."""
        waveform, sr = torchaudio.load(str(file_path))
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sr != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0)  # Remove channel dim -> (T,)
    
    @torch.no_grad()
    def extract(self, waveform: torch.Tensor) -> np.ndarray:
        """
        Extract SSL features from waveform tensor.
        
        Args:
            waveform: Audio tensor of shape (T,) at 16kHz
            
        Returns:
            features: numpy array of shape (768,) if pooling, else (T', 768)
        """
        # Process through Wav2Vec2 processor
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        
        # Extract hidden states
        outputs = self.model(input_values, output_hidden_states=True)
        
        # Get features from specified layer
        if self.layer == -1:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[self.layer]
        
        # hidden_states shape: (batch=1, time, features=768)
        features = hidden_states.squeeze(0)  # (time, 768)
        
        # Apply pooling
        if self.pooling == "mean":
            features = features.mean(dim=0)  # (768,)
        elif self.pooling == "first":
            features = features[0]  # (768,)
        elif self.pooling == "last":
            features = features[-1]  # (768,)
        # else: "none" -> keep (time, 768)
        
        return features.cpu().numpy()
    
    def extract_file(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Extract SSL features from an audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            features: numpy array of shape (768,) or (T, 768)
        """
        waveform = self.load_audio(file_path)
        return self.extract(waveform)
    
    def extract_batch(self, file_paths: list) -> list:
        """Extract features for multiple files."""
        return [self.extract_file(fp) for fp in file_paths]


# Convenience function for quick extraction
def extract_ssl_features(
    file_path: str,
    model_name: str = "facebook/wav2vec2-base-960h",
    device: str = "cuda"
) -> np.ndarray:
    """One-liner for extracting SSL features."""
    extractor = SSLExtractor(model_name=model_name, device=device)
    return extractor.extract_file(file_path)


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        print("Usage: python ssl_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = SSLExtractor(device="cuda" if torch.cuda.is_available() else "cpu")
    features = extractor.extract_file(file_path)
    print(f"Extracted features shape: {features.shape}")
    print(f"Features dtype: {features.dtype}")
    print(f"Features range: [{features.min():.4f}, {features.max():.4f}]")
