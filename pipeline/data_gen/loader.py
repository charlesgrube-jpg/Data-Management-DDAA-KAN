"""
Dataset Loader Module

Loads audio datasets from HuggingFace with metadata extraction.
Supports Mozilla Common Voice and other speech datasets.
"""

from typing import Iterator, Dict, Any, Optional
from datasets import load_dataset, Audio
from pipeline.config import Config


def load_source_dataset(config: Config) -> Iterator[Dict[str, Any]]:
    """
    Load dataset from HuggingFace and yield samples with metadata.
    
    Args:
        config: Pipeline configuration
        
    Yields:
        Dict with keys: audio, speaker_id, transcript, gender, accent, source_idx
    """
    print(f"[Loader] Loading {config.source.dataset_name} ({config.source.language})...")
    
    # Check if streaming mode (avoids downloading entire dataset)
    use_streaming = getattr(config.source, 'streaming', True)  # Default to streaming
    
    # Load dataset with audio resampled to target SR
    dataset = load_dataset(
        config.source.dataset_name,
        config.source.language,
        split=config.source.split,
        cache_dir=config.source.cache_dir,
        trust_remote_code=True,
        streaming=use_streaming  # Stream to avoid 80GB download
    )
    
    # Cast audio to target sample rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=config.audio.target_sr))
    
    # Limit samples if configured (for pilot runs)
    if config.source.max_samples:
        dataset = dataset.select(range(min(config.source.max_samples, len(dataset))))
        print(f"[Loader] Limited to {len(dataset)} samples (pilot mode)")
    
    print(f"[Loader] Loaded {len(dataset)} samples")
    
    # Yield samples with standardized metadata
    for idx, sample in enumerate(dataset):
        # Extract audio array and sample rate
        audio_data = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        
        # Extract metadata (Common Voice specific, with fallbacks)
        yield {
            "audio": audio_data,
            "sample_rate": sr,
            "source_idx": idx,
            "speaker_id": _extract_speaker_id(sample, idx),
            "transcript": sample.get("sentence", sample.get("text", "")),
            "gender": sample.get("gender", "unknown"),
            "accent": sample.get("accent", "unknown"),
            "age": sample.get("age", "unknown"),
            "locale": sample.get("locale", config.source.language),
        }


def _extract_speaker_id(sample: Dict, fallback_idx: int) -> str:
    """
    Extract speaker ID from sample metadata.
    
    Common Voice uses 'client_id' as speaker identifier.
    Falls back to index-based ID if not available.
    """
    # Common Voice uses client_id
    if "client_id" in sample:
        return sample["client_id"]
    
    # LibriSpeech uses speaker_id
    if "speaker_id" in sample:
        return str(sample["speaker_id"])
    
    # VoxCeleb uses id
    if "id" in sample:
        return sample["id"]
    
    # Fallback: treat each sample as unique speaker
    # WARNING: This defeats speaker-disjoint splits!
    print(f"[Loader] WARNING: No speaker_id found for sample {fallback_idx}, using index")
    return f"unknown_speaker_{fallback_idx}"


def get_dataset_info(config: Config) -> Dict[str, Any]:
    """
    Get dataset statistics without loading all samples.
    
    Returns:
        Dict with dataset info: num_samples, features, etc.
    """
    from datasets import load_dataset_builder
    
    builder = load_dataset_builder(
        config.source.dataset_name,
        config.source.language,
        cache_dir=config.source.cache_dir
    )
    
    return {
        "name": config.source.dataset_name,
        "language": config.source.language,
        "description": builder.info.description,
        "features": str(builder.info.features),
        "num_examples": builder.info.splits.get(config.source.split, {}).num_examples
    }


if __name__ == "__main__":
    from pipeline.config import load_config
    
    cfg = load_config("../config.yaml")
    
    # Test loading a few samples
    for i, sample in enumerate(load_source_dataset(cfg)):
        print(f"Sample {i}: speaker={sample['speaker_id'][:20]}..., "
              f"transcript={sample['transcript'][:30]}...")
        if i >= 2:
            break
