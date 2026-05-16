#!/usr/bin/env python3
"""
Pre-extract SSL (Wav2Vec2/WavLM) features for all audio files.

This replaces CQT features with state-of-the-art SSL embeddings.
Features are extracted using GPU and saved as .npy files.

Usage:
    python scripts/preextract_ssl_features.py --model wav2vec2-base
    python scripts/preextract_ssl_features.py --model wavlm-base
"""

import csv
import argparse
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Model name mapping
MODEL_MAPPING = {
    "wav2vec2-base": "facebook/wav2vec2-base-960h",
    "wav2vec2-large": "facebook/wav2vec2-large-960h",
    "wavlm-base": "microsoft/wavlm-base",
    "wavlm-large": "microsoft/wavlm-large",
}


def extract_features_single_gpu(
    csv_path: Path,
    output_dir: Path,
    model_name: str,
    device: str = "cuda"
):
    """Extract SSL features for all samples in a CSV using single GPU."""
    
    # Import here to avoid loading model at import time
    from pipeline.features.ssl_extractor import SSLExtractor
    
    # Load CSV
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Get audio path (not feature path)
            audio_path = row.get('full_path', row.get('filename', ''))
            label = row.get('label', 'real')
            if audio_path and Path(audio_path).exists():
                samples.append((audio_path, label))
    
    if not samples:
        print(f"Warning: No valid audio files found in {csv_path}")
        return 0
    
    print(f"Processing {len(samples)} samples from {csv_path.name}...")
    
    # Initialize extractor
    full_model_name = MODEL_MAPPING.get(model_name, model_name)
    extractor = SSLExtractor(model_name=full_model_name, device=device, pooling="mean")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process samples
    success = 0
    feature_paths = []
    
    for audio_path, label in tqdm(samples, desc=f"Extracting {csv_path.name}"):
        try:
            # Generate output path
            audio_name = Path(audio_path).stem
            output_path = output_dir / f"{audio_name}.npy"
            
            # Skip if already exists
            if output_path.exists():
                feature_paths.append((output_path, label))
                success += 1
                continue
            
            # Extract features
            features = extractor.extract_file(audio_path)
            
            # Save
            np.save(output_path, features)
            feature_paths.append((output_path, label))
            success += 1
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    # Create new CSV with SSL feature paths
    output_csv = csv_path.parent / f"{csv_path.stem}_ssl_features.csv"
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['feature_path', 'label'])
        writer.writeheader()
        for feature_path, label in feature_paths:
            writer.writerow({
                'feature_path': str(feature_path),
                'label': label
            })
    
    print(f"Created: {output_csv} ({success} samples)")
    return success


def main():
    parser = argparse.ArgumentParser(description="Pre-extract SSL features")
    parser.add_argument("--csv_dir", type=str, default="unified_dataset")
    parser.add_argument("--model", type=str, default="wav2vec2-base",
                       choices=list(MODEL_MAPPING.keys()) + list(MODEL_MAPPING.values()))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    csv_dir = Path(args.csv_dir)
    
    # Determine output directory based on model
    model_short = args.model.split("/")[-1].replace("-960h", "")
    output_base = csv_dir / f"ssl_features_{model_short}"
    
    print("="*60)
    print("SSL Feature Extraction")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Output: {output_base}")
    print("="*60)
    
    total = 0
    for split in ['train', 'val', 'test']:
        csv_path = csv_dir / f"{split}.csv"
        if csv_path.exists():
            split_output = output_base / split
            count = extract_features_single_gpu(
                csv_path, split_output, args.model, args.device
            )
            total += count
        else:
            print(f"Warning: {csv_path} not found")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE! Extracted {total} SSL features")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
