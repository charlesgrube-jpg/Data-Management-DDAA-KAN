"""
Standardizer Module

Audio preprocessing and segmentation:
- Resampling to target sample rate
- Volume normalization (RMS)
- Silence trimming
- Fixed-duration chunking
"""

from .preprocessor import preprocess_audio
from .segmenter import segment_audio

__all__ = ["preprocess_audio", "segment_audio"]
