"""
Feature Extraction Pipeline

Reads WAV files from a processed dataset and extracts features (CQT or LFCC).
Outputs PyTorch tensors or NumPy arrays.

Usage:
    python -m pipeline.features.extract_features
    python -m pipeline.features.extract_features --input processed_dataset_20251224 --type lfcc
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.config import load_config
from pipeline.features.cqt_extractor import CQTExtractor
from pipeline.features.lfcc_extractor import LFCCExtractor


def find_latest_dataset(base_pattern: str = "processed_dataset") -> Path:
    """Find the most recent processed dataset folder."""
    cwd = Path(".")
    candidates = sorted(cwd.glob(f"{base_pattern}_*"), reverse=True)
    
    if candidates:
        return candidates[0]
    
    # Fallback: exact name
    if (cwd / base_pattern).exists():
        return cwd / base_pattern
    
    return None


def extract_features(
    input_dir: Path,
    output_dir: Path,
    feature_type: str = "cqt",
    output_format: str = "pt",
    config=None,
    limit: int = None
):
    """
    Extract features from all audio files in input directory.
    
    Args:
        input_dir: Path to processed dataset
        output_dir: Path for output features
        feature_type: "cqt" or "lfcc"
        output_format: "pt" (PyTorch) or "npy" (NumPy)
        config: Pipeline config object
        limit: Max files to process (for testing)
    """
    # Load config if not provided
    if config is None:
        try:
            config = load_config("config.yaml")
        except:
            print("[Features] No config.yaml found, using defaults")
            config = None
    
    # Get sample rate from config or default
    sample_rate = config.audio.target_sr if config else 16000
    
    # Initialize extractor
    print(f"[Features] Initializing {feature_type.upper()} extractor...")
    
    if feature_type == "cqt":
        cqt_config = getattr(config, 'features', None)
        if cqt_config and hasattr(cqt_config, 'cqt'):
            extractor = CQTExtractor(
                sample_rate=sample_rate,
                n_bins=cqt_config.cqt.get('n_bins', 84),
                hop_length=cqt_config.cqt.get('hop_length', 512),
                fmin=cqt_config.cqt.get('fmin', 32.7)
            )
        else:
            extractor = CQTExtractor(sample_rate=sample_rate)
    elif feature_type == "lfcc":
        lfcc_config = getattr(config, 'features', None)
        if lfcc_config and hasattr(lfcc_config, 'lfcc'):
            extractor = LFCCExtractor(
                sample_rate=sample_rate,
                n_lfcc=lfcc_config.lfcc.get('n_lfcc', 60),
                n_filters=lfcc_config.lfcc.get('n_filters', 128)
            )
        else:
            extractor = LFCCExtractor(sample_rate=sample_rate)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all WAV files
    real_dir = input_dir / "real"
    synthetic_dir = input_dir / "synthetic"
    
    all_files = []
    if real_dir.exists():
        all_files.extend(list(real_dir.glob("*.wav")))
    if synthetic_dir.exists():
        all_files.extend(list(synthetic_dir.glob("*.wav")))
    
    # Also check split directories
    for split in ["train", "val", "test"]:
        split_real = input_dir / split / "real"
        split_syn = input_dir / split / "synthetic"
        if split_real.exists():
            all_files.extend(list(split_real.glob("*.wav")))
        if split_syn.exists():
            all_files.extend(list(split_syn.glob("*.wav")))
    
    if not all_files:
        print(f"[Features] No WAV files found in {input_dir}")
        return
    
    print(f"[Features] Found {len(all_files)} WAV files")
    
    if limit:
        all_files = all_files[:limit]
        print(f"[Features] Limited to {limit} files")
    
    # Process files
    metadata_rows = []
    
    for wav_path in tqdm(all_files, desc="Extracting features"):
        try:
            # Extract features
            features = extractor.extract_file(str(wav_path))
            
            # Build output path preserving structure
            rel_path = wav_path.relative_to(input_dir)
            out_path = output_dir / rel_path.with_suffix(f".{output_format}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save
            if output_format == "pt":
                torch.save(torch.from_numpy(features), out_path)
            else:  # npy
                np.save(out_path, features)
            
            # Track metadata
            metadata_rows.append({
                "source_file": str(rel_path),
                "feature_file": str(out_path.relative_to(output_dir)),
                "feature_type": feature_type,
                "shape": str(features.shape)
            })
            
        except Exception as e:
            print(f"\n[Features] Error processing {wav_path.name}: {e}")
            continue
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(output_dir / "features_metadata.csv", index=False)
    
    print(f"\n[Features] Extraction complete!")
    print(f"  Output: {output_dir}")
    print(f"  Files: {len(metadata_rows)}")
    print(f"  Format: {output_format}")
    print(f"  Type: {feature_type}")


def main():
    parser = argparse.ArgumentParser(description="Extract audio features (CQT/LFCC)")
    parser.add_argument("--input", type=str, default=None, help="Input dataset directory")
    parser.add_argument("--output", type=str, default=None, help="Output features directory")
    parser.add_argument("--type", type=str, default="cqt", choices=["cqt", "lfcc"])
    parser.add_argument("--format", type=str, default="pt", choices=["pt", "npy"])
    parser.add_argument("--limit", type=int, default=None, help="Limit files (for testing)")
    
    args = parser.parse_args()
    
    # Find input directory
    if args.input:
        input_dir = Path(args.input)
    else:
        input_dir = find_latest_dataset()
    
    if input_dir is None or not input_dir.exists():
        print("[Features] ERROR: No processed dataset found!")
        print("  Specify with: --input processed_dataset_XXXXXX")
        return 1
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(f"features_{args.type}_{input_dir.name}")
    
    print("=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Type:   {args.type}")
    print(f"Format: {args.format}")
    
    extract_features(
        input_dir=input_dir,
        output_dir=output_dir,
        feature_type=args.type,
        output_format=args.format,
        limit=args.limit
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
