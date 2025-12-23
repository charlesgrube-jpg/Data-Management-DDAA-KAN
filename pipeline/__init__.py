"""
Audio Deepfake Detection Dataset Pipeline

Modules:
    - loader: HuggingFace dataset loading with metadata extraction
    - preprocessor: Resample, normalize, trim silence
    - segmenter: Chunk audio by duration
    - transcriber: Whisper-based transcription (optional)
    - synthesizer: TTS and Voice Conversion generation
    - effects: Channel degradation (compression, noise, reverb)
    - splitter: Speaker-disjoint train/val/test splits
    - validator: Quality checks and balance verification
    - exporter: Save audio files and metadata CSV
"""

from pipeline.config import load_config, Config

__version__ = "0.1.0"
__all__ = ["load_config", "Config"]
