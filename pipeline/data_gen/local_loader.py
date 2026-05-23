"""
Local File Loader - For testing with local audio files

Loads audio files from a local directory instead of HuggingFace.
Useful for testing the pipeline with the generated test_samples.
"""

import os
from pathlib import Path
from typing import Iterator, Dict, Any
import numpy as np
import soundfile as sf
from pipeline.config import Config


def load_local_dataset(
    directory: str,
    config: Config
) -> Iterator[Dict[str, Any]]:
    """
    Load audio files from a local directory.
    
    Expects structure:
        directory/
        ├── sample_00.wav
        ├── sample_00.txt  (optional metadata)
        ├── sample_01.wav
        └── ...
    
    Args:
        directory: Path to directory containing audio files
        config: Pipeline configuration
        
    Yields:
        Dict with keys: audio, speaker_id, transcript, source_idx, etc.
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find all audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
    audio_files = sorted([
        f for f in directory.iterdir() 
        if f.suffix.lower() in audio_extensions
    ])
    
    print(f"[LocalLoader] Found {len(audio_files)} audio files in {directory}")
    
    # Limit if max_samples is set
    if config.source.max_samples:
        audio_files = audio_files[:config.source.max_samples]
        print(f"[LocalLoader] Limited to {len(audio_files)} samples")
    
    for idx, audio_path in enumerate(audio_files):
        try:
            # Load audio
            audio, sr = sf.read(str(audio_path))
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != config.audio.target_sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=config.audio.target_sr)
                sr = config.audio.target_sr
            
            # Try to load metadata from companion .txt file
            meta_path = audio_path.with_suffix('.txt')
            metadata = _load_metadata_file(meta_path) if meta_path.exists() else {}
            
            # Build sample dict
            sample = {
                "audio": audio,
                "sample_rate": sr,
                "source_idx": idx,
                "speaker_id": metadata.get("Speaker", f"speaker_{idx:03d}"),
                "transcript": metadata.get("Transcript", f"Sample {idx} from {audio_path.name}"),
                "gender": metadata.get("Gender", "unknown"),
                "accent": metadata.get("Accent", "unknown"),
                "locale": "en",
                "file_path": str(audio_path),
            }
            
            yield sample
            
        except Exception as e:
            print(f"[LocalLoader] Error loading {audio_path}: {e}")
            continue


def _load_metadata_file(path: Path) -> Dict[str, str]:
    """Load metadata from a companion text file."""
    metadata = {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
    except Exception:
        pass
    
    return metadata


if __name__ == "__main__":
    from pipeline.config import load_config
    
    cfg = load_config("../config.yaml")
    
    # Test loading from test_samples
    test_dir = Path(__file__).parent.parent / "test_samples"
    
    if test_dir.exists():
        for sample in load_local_dataset(str(test_dir), cfg):
            print(f"Loaded: {sample['file_path']}")
            print(f"  Speaker: {sample['speaker_id']}")
            print(f"  Transcript: {sample['transcript'][:50]}...")
            print(f"  Duration: {len(sample['audio'])/sample['sample_rate']:.2f}s")
            print()
    else:
        print(f"Test directory not found: {test_dir}")
        print("Run: python test_download.py")
