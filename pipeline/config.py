"""
Configuration loader and dataclass definitions.

Loads config.yaml and provides typed access to all parameters.
Supports optional random seeding for reproducibility.
"""

import yaml
import random
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict


@dataclass
class SourceConfig:
    dataset_name: str
    language: str
    split: str
    max_samples: Optional[int]
    cache_dir: str
    max_size_mb: Optional[float] = None # Limit by total audio size in MB
    streaming: bool = True  # Stream to avoid downloading entire dataset
    data_path: Optional[str] = None # Custom path for data (e.g. for Colab)


@dataclass
class AudioConfig:
    target_sr: int
    normalize_db: float
    silence_threshold_db: float
    min_duration: float
    max_duration: float


@dataclass
class SegmentationConfig:
    chunk_duration: float
    overlap: float
    pad_last_chunk: bool


@dataclass
class SynthesisConfig:
    tts_models: List[str]
    vc_models: List[str]
    pick_strategy: str  # "random", "both", "tts_only", "vc_only"
    fallback_on_error: bool
    huggingface_models: Optional[List[Dict]] = None
    vc_device: str = "cpu"  # Device for VC: "cpu" or "cuda:0"


@dataclass
class EffectsConfig:
    enabled: bool
    quality_distribution: Dict[str, float]
    apply_same_to_pair: bool


@dataclass
class CodecCompressionConfig:
    enabled: bool
    apply_to: str  # "all", "synthetic_only", "none"
    codec_distribution: Dict[str, float]
    apply_after_effects: bool


@dataclass
class SplitsConfig:
    train: float
    val: float
    test: float


@dataclass
class OutputConfig:
    base_dir: str
    audio_format: str
    metadata_file: str
    save_transcripts: bool


@dataclass
class ValidationConfig:
    check_balance: bool
    balance_tolerance: float
    check_duplicates: bool
    check_speaker_leakage: bool
    min_samples_per_split: int


@dataclass
class CQTConfig:
    n_bins: int
    hop_length: int
    fmin: float


@dataclass
class LFCCConfig:
    n_lfcc: int
    n_filters: int
    n_fft: int
    hop_length: int


@dataclass
class FeaturesConfig:
    type: str
    device: str
    output_format: str
    cqt: CQTConfig
    lfcc: LFCCConfig


@dataclass
class ExecutionConfig:
    num_workers: int = 1


@dataclass
class Config:
    """Master configuration container."""
    source: SourceConfig
    audio: AudioConfig
    segmentation: SegmentationConfig
    synthesis: SynthesisConfig
    effects: EffectsConfig
    codec_compression: CodecCompressionConfig
    splits: SplitsConfig
    output: OutputConfig
    output: OutputConfig
    validation: ValidationConfig
    features: FeaturesConfig
    execution: ExecutionConfig
    seed: Optional[int]
    
    def set_seed(self):
        """Set random seeds for reproducibility if seed is specified."""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            print(f"[Config] Random seed set to {self.seed}")
    
    @property
    def output_path(self) -> Path:
        return Path(self.output.base_dir)
    
    def get_timestamped_output_path(self) -> Path:
        """Get output path with timestamp for unique run directories."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.output.base_dir.rstrip("/\\")
        return Path(f"{base}_{timestamp}")


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Config dataclass with all parameters
    """
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    
    config = Config(
        source=SourceConfig(**raw['source']),
        audio=AudioConfig(**raw['audio']),
        segmentation=SegmentationConfig(**raw['segmentation']),
        synthesis=SynthesisConfig(**raw['synthesis']),
        effects=EffectsConfig(**raw['effects']),
        codec_compression=CodecCompressionConfig(**raw['codec_compression']),
        splits=SplitsConfig(**raw['splits']),
        output=OutputConfig(**raw['output']),
        validation=ValidationConfig(**raw['validation']),
        features=FeaturesConfig(
            type=raw['features']['type'],
            device=raw['features']['device'],
            output_format=raw['features']['output_format'],
            cqt=CQTConfig(**raw['features']['cqt']),
            lfcc=LFCCConfig(**raw['features']['lfcc'])
        ),
        execution=ExecutionConfig(**raw.get('execution', {'num_workers': 1})),
        seed=raw.get('seed')
    )
    
    # Set seed immediately if provided
    config.set_seed()
    
    return config


if __name__ == "__main__":
    # Test loading
    cfg = load_config("../config.yaml")
    print(f"Dataset: {cfg.source.dataset_name}")
    print(f"Target SR: {cfg.audio.target_sr}")
    print(f"Chunk duration: {cfg.segmentation.chunk_duration}s")
