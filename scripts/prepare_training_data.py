#!/usr/bin/env python3
"""
Prepare Training Data with Codec Augmentation

This script creates unified train/val/test CSV files that include:
1. Original audio samples (real + synthetic)
2. Codec-compressed samples for robustness training

Usage:
    python scripts/prepare_training_data.py \
        --datasets processed_dataset_20260108_154819 processed_dataset_20260108_154912 ... \
        --output unified_dataset \
        --include-codec
"""

import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Dict


def collect_samples_from_dataset(dataset_path: Path, include_codec: bool = True) -> Dict[str, List[dict]]:
    """
    Collect all samples from a dataset including codec-compressed if available.
    
    Returns:
        Dict with keys 'train', 'val', 'test' containing lists of sample dicts
    """
    samples = {'train': [], 'val': [], 'test': []}
    
    # Read original samples from CSV files
    for split in ['train', 'val', 'test']:
        csv_path = dataset_path / f"{split}.csv"
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Add full path to filename
                    row['full_path'] = str(dataset_path / row['filename'])
                    row['augmentation'] = 'none'
                    row['source_dataset'] = dataset_path.name
                    samples[split].append(row)
    
    # Add codec-compressed samples (assign to same split as original)
    if include_codec:
        compressed_dir = dataset_path / 'compressed'
        if compressed_dir.exists():
            codec_files = list(compressed_dir.glob('*.wav'))
            print(f"  Found {len(codec_files)} codec-compressed files in {dataset_path.name}")
            
            for codec_file in codec_files:
                # Parse filename to get original sample info
                # Format: cv_XXXXXX_chunkXXX_codec_bitrate.wav
                filename = codec_file.stem
                parts = filename.rsplit('_', 2)  # Split off codec and bitrate
                
                if len(parts) >= 3:
                    original_base = '_'.join(parts[:-2])  # e.g., cv_000001_chunk000
                    codec_type = parts[-2]  # e.g., mp3, aac
                    bitrate = parts[-1]  # e.g., 64k, 128k
                    
                    # Determine which split this belongs to (check each split's files)
                    assigned_split = None
                    for split in ['train', 'val', 'test']:
                        split_dir = dataset_path / split / 'real'
                        if split_dir.exists():
                            # Check if original exists in this split
                            for ext in ['.wav', '']:
                                if (split_dir / f"{original_base}{ext}.wav").exists() or \
                                   (split_dir / f"{original_base}.wav").exists():
                                    assigned_split = split
                                    break
                        if assigned_split:
                            break
                    
                    # Default to train if we can't determine
                    if not assigned_split:
                        assigned_split = 'train'
                    
                    # Determine label based on filename pattern
                    # Codec files are compressed versions of originals
                    label = 'real'  # Codec-compressed real audio is still "real" for training
                    
                    sample = {
                        'filename': str(codec_file.relative_to(dataset_path)),
                        'full_path': str(codec_file),
                        'split': assigned_split,
                        'label': label,
                        'is_synthetic': 'False',
                        'augmentation': f'{codec_type}_{bitrate}',
                        'source_dataset': dataset_path.name,
                        'codec': codec_type,
                        'bitrate': bitrate
                    }
                    samples[assigned_split].append(sample)
    
    return samples


def merge_datasets(dataset_paths: List[Path], include_codec: bool = True) -> Dict[str, List[dict]]:
    """Merge samples from multiple datasets."""
    merged = {'train': [], 'val': [], 'test': []}
    
    for dataset_path in dataset_paths:
        print(f"Processing {dataset_path.name}...")
        samples = collect_samples_from_dataset(dataset_path, include_codec)
        for split in ['train', 'val', 'test']:
            merged[split].extend(samples[split])
    
    # Shuffle each split
    for split in merged:
        random.shuffle(merged[split])
    
    return merged


def write_unified_csv(samples: Dict[str, List[dict]], output_dir: Path):
    """Write unified CSV files for each split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define common fields
    fields = ['full_path', 'split', 'label', 'is_synthetic', 'augmentation', 
              'source_dataset', 'codec', 'bitrate', 'filename']
    
    for split, split_samples in samples.items():
        if not split_samples:
            continue
            
        output_path = output_dir / f"{split}.csv"
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(split_samples)
        
        print(f"Wrote {len(split_samples)} samples to {output_path}")


def create_bark_test_csv(bark_dir: Path, output_path: Path):
    """Create a CSV file for Bark test samples."""
    samples = []
    
    for wav_file in bark_dir.glob('*.wav'):
        samples.append({
            'full_path': str(wav_file),
            'split': 'test_bark',
            'label': 'synthetic',
            'is_synthetic': 'True',
            'augmentation': 'none',
            'source_dataset': 'bark',
            'codec': '',
            'bitrate': '',
            'filename': wav_file.name
        })
    
    with open(output_path, 'w', newline='') as f:
        fields = ['full_path', 'split', 'label', 'is_synthetic', 'augmentation',
                  'source_dataset', 'codec', 'bitrate', 'filename']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)
    
    print(f"Wrote {len(samples)} Bark samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare unified training data")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="List of dataset directories to merge")
    parser.add_argument("--output", type=str, default="unified_dataset",
                        help="Output directory for unified CSVs")
    parser.add_argument("--include-codec", action="store_true", default=True,
                        help="Include codec-compressed samples")
    parser.add_argument("--bark-dir", type=str, default=None,
                        help="Path to Bark test set directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Find dataset directories
    dataset_paths = []
    for ds_name in args.datasets:
        ds_path = Path(ds_name)
        if ds_path.exists():
            dataset_paths.append(ds_path)
        else:
            print(f"Warning: Dataset not found: {ds_name}")
    
    if not dataset_paths:
        print("No datasets found!")
        return
    
    print(f"Found {len(dataset_paths)} datasets")
    print(f"Include codec samples: {args.include_codec}")
    
    # Merge datasets
    merged = merge_datasets(dataset_paths, args.include_codec)
    
    # Write unified CSVs
    output_dir = Path(args.output)
    write_unified_csv(merged, output_dir)
    
    # Handle Bark test set if provided
    if args.bark_dir:
        bark_path = Path(args.bark_dir)
        if bark_path.exists():
            create_bark_test_csv(bark_path, output_dir / "test_bark.csv")
    
    # Print summary
    print("\n=== Summary ===")
    for split, samples in merged.items():
        if samples:
            codec_count = sum(1 for s in samples if s.get('augmentation', 'none') != 'none')
            print(f"{split}: {len(samples)} total ({codec_count} codec-augmented)")


if __name__ == "__main__":
    main()
