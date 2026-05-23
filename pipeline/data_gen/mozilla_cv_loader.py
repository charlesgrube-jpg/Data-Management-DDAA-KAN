"""
Mozilla Common Voice Dataset Loader

Loads audio files and metadata from extracted Mozilla Common Voice dataset.
Designed to work with partial downloads.

Usage:
    from pipeline.mozilla_cv_loader import load_mozilla_cv
    
    for sample in load_mozilla_cv("mozilla_cv_data/extracted/cv-corpus-24.0-2025-12-05/en", config):
        print(sample["transcript"], sample["speaker_id"])
"""

import os
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
import pandas as pd
import numpy as np

from pipeline.config import Config


def load_mozilla_cv(
    dataset_path: str,
    config: Config,
    split: str = "train"
) -> Iterator[Dict[str, Any]]:
    """
    Load audio samples from extracted Mozilla Common Voice dataset.
    
    Args:
        dataset_path: Path to extracted CV directory (containing clips/, train.tsv, etc.)
        config: Pipeline configuration
        split: Which split to load ("train", "test", "dev", "validated")
        
    Yields:
        Dict with keys: audio, sample_rate, speaker_id, transcript, gender, accent, etc.
    """
    dataset_path = Path(dataset_path)
    clips_dir = dataset_path / "clips"
    tsv_file = dataset_path / f"{split}.tsv"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Check for combined matched TSV first (best option)
    all_matched_tsv = dataset_path / "all_matched_clips.tsv"
    matched_tsv = dataset_path / f"matched_{split}.tsv"
    
    # Get available clips (support both mp3 and wav) early
    available_clips = set(f.name for f in clips_dir.glob("*.mp3"))
    available_clips.update(f.name for f in clips_dir.glob("*.wav"))
    # Also add with clips/ prefix for matching TSV entries
    available_clips_with_prefix = set(f"clips/{name}" for name in available_clips)
    available_clips.update(available_clips_with_prefix)
    print(f"[Mozilla CV Loader] Available clips in directory: {len(available_clips) // 2}")

    if len(available_clips) == 0:
        print("[Mozilla CV Loader] WARNING: No clips found in clips/ directory")
        return
    
    if all_matched_tsv.exists():
        print(f"[Mozilla CV Loader] Using all_matched_clips.tsv")
        df = pd.read_csv(all_matched_tsv, sep='\t')
    elif matched_tsv.exists():
        print(f"[Mozilla CV Loader] Using matched TSV: {matched_tsv}")
        df = pd.read_csv(matched_tsv, sep='\t')
    else:
        # AGGRESSIVE MODE: Scan ALL standard TSVs to find as many clips as possible
        print(f"[Mozilla CV Loader] Scanning all TSVs to find matches for {len(available_clips)} available clips...")
        
        matches = []
        seen_paths = set()
        
        # Priority order: requested split first, then others
        splits_to_check = [split] + [s for s in ["train", "validated", "test", "dev", "other"] if s != split]
        
        for check_split in splits_to_check:
            check_tsv = dataset_path / f"{check_split}.tsv"
            if not check_tsv.exists():
                continue
                
            print(f"  - Scanning {check_split}.tsv...")
            try:
                for chunk in pd.read_csv(check_tsv, sep='\t', chunksize=100000):
                    # Filter for files we have
                    matched = chunk[chunk['path'].isin(available_clips)]
                    
                    # Remove any we've already seen (duplicates across TSVs)
                    if not matched.empty:
                        original_len = len(matched)
                        matched = matched[~matched['path'].isin(seen_paths)]
                        
                        if not matched.empty:
                            seen_paths.update(matched['path'])
                            matches.append(matched)
                            # print(f"    Found {len(matched)} new matches (unique)")
            except Exception as e:
                print(f"    Error reading {check_split}.tsv: {e}")

        if matches:
            df = pd.concat(matches)
            print(f"[Mozilla CV Loader] AGGREGATED TOTAL: {len(df)} unique entries from all splits")
        else:
            print("[Mozilla CV Loader] ERROR: No matching clips found in any TSV")
            return
    
    print(f"[Mozilla CV Loader] Total entries: {len(df)}")
    
    # Limit samples if configured
    if config.source.max_samples:
        df = df.head(config.source.max_samples)
        print(f"[Mozilla CV Loader] Limited to {len(df)} samples")
    
    # Try to import audio loading libraries
    try:
        import librosa
        use_librosa = True
    except ImportError:
        use_librosa = False
        try:
            from pydub import AudioSegment
            use_pydub = True
        except ImportError:
            use_pydub = False
            print("[Mozilla CV Loader] WARNING: Neither librosa nor pydub available")
            print("                    Install with: pip install librosa")
    
    loaded_count = 0
    skipped_count = 0
    total_bytes_loaded = 0
    max_bytes = config.source.max_size_mb * 1024 * 1024 if config.source.max_size_mb else None
    
    if max_bytes:
        print(f"[Mozilla CV Loader] Size limit enabled: {config.source.max_size_mb} MB")

    for idx, row in df.iterrows():
        # Check size limit before processing
        if max_bytes and total_bytes_loaded >= max_bytes:
            print(f"[Mozilla CV Loader] Reached size limit of {config.source.max_size_mb} MB. Stopping.")
            break
            
        # Get clip path
        clip_filename = row.get('path', '')
        if not clip_filename:
            skipped_count += 1
            continue
        
        # Strip 'clips/' prefix if present (TSV may have paths like 'clips/file.mp3')
        if clip_filename.startswith('clips/'):
            clip_filename = clip_filename[6:]  # Remove 'clips/' prefix
        
        clip_path = clips_dir / clip_filename
        
        # Check if file exists (may not be in partial download)
        if not clip_path.exists():
            skipped_count += 1
            continue
            
        # Accumulate size
        try:
            file_size = clip_path.stat().st_size
            total_bytes_loaded += file_size
        except:
             pass # Failed to get size, ignore
        
        try:
            # Lazy Loading: Do NOT load audio here to save RAM
            # audio, sr = librosa.load(str(clip_path), sr=config.audio.target_sr)
            audio = None
            sr = config.audio.target_sr
            
            # Build sample dict
            sample = {
                "audio": audio, # Lazy loaded later
                "sample_rate": sr,
                "source_idx": idx,
                "speaker_id": str(row.get('client_id', f'speaker_{idx}')),
                "transcript": str(row.get('sentence', '')),
                "gender": str(row.get('gender', 'unknown')),
                "accent": str(row.get('accents', row.get('accent', 'unknown'))),
                "age": str(row.get('age', 'unknown')),
                "locale": str(row.get('locale', 'en')),
                "file_path": str(clip_path),
                "original_size": file_size, # Helpful for tracking
                "up_votes": int(row.get('up_votes', 0)),
                "down_votes": int(row.get('down_votes', 0)),
            }
            
            loaded_count += 1
            yield sample
            
        except Exception as e:
            print(f"[Mozilla CV Loader] Error preparing {clip_filename}: {e}")
            skipped_count += 1
            continue
    
    print(f"[Mozilla CV Loader] Prepared: {loaded_count}, Skipped: {skipped_count}, Total Size: {total_bytes_loaded / 1024 / 1024:.2f} MB")


def get_available_clips(dataset_path: str) -> list:
    """
    Get list of available audio clips in the dataset directory.
    Useful for checking what was extracted from partial download.
    """
    clips_dir = Path(dataset_path) / "clips"
    
    if not clips_dir.exists():
        return []
    
    # Support both mp3 and wav files
    clips = [f.name for f in clips_dir.glob("*.mp3")]
    clips.extend([f.name for f in clips_dir.glob("*.wav")])
    return sorted(clips)



def filter_tsv_by_available_clips(dataset_path: str, split: str = "train") -> pd.DataFrame:
    """
    Filter the TSV metadata to only include rows for clips that exist.
    Useful when working with partial downloads.
    """
    dataset_path = Path(dataset_path)
    tsv_file = dataset_path / f"{split}.tsv"
    clips_dir = dataset_path / "clips"
    
    if not tsv_file.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_file}")
    
    # Load full TSV
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Get available clips
    available = set(get_available_clips(str(dataset_path)))
    
    # Filter
    df_filtered = df[df['path'].isin(available)]
    
    print(f"[Mozilla CV Loader] Filtered {split}.tsv: {len(df_filtered)}/{len(df)} entries have clips")
    
    return df_filtered


if __name__ == "__main__":
    # Test the loader
    from pipeline.config import load_config
    
    cfg = load_config("config.yaml")
    
    cv_path = Path("mozilla_cv_data/extracted/cv-corpus-24.0-2025-12-05/en")
    
    if cv_path.exists():
        # Check what clips we have
        clips = get_available_clips(str(cv_path))
        print(f"Available clips: {len(clips)}")
        
        # Filter TSV
        df = filter_tsv_by_available_clips(str(cv_path), "train")
        print(f"\nSample metadata:")
        print(df[['path', 'sentence', 'client_id', 'gender']].head())
        
        # Test loading
        print("\nLoading samples...")
        for i, sample in enumerate(load_mozilla_cv(str(cv_path), cfg)):
            print(f"  {i}: {sample['transcript'][:50]}... ({len(sample['audio'])/sample['sample_rate']:.2f}s)")
            if i >= 4:
                break
    else:
        print(f"Dataset not found at {cv_path}")
        print("Run download_partial.py first")
