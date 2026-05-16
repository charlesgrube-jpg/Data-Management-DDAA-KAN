#!/usr/bin/env python3
"""
Bundle SSL features into single .pt files for instant loading.

Run after preextract_ssl_features.py completes:
    python scripts/bundle_ssl_features.py
"""

import csv
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

TARGET_DIM = 768  # Wav2Vec2-base/WavLM-base hidden size


def bundle_split(csv_path: Path, output_path: Path):
    """Bundle all SSL features from CSV into single tensor file."""
    
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'feature_path' not in row:
                print(f"ERROR: {csv_path} missing feature_path column")
                return
            samples.append((row['feature_path'], row.get('label', 'real')))
    
    print(f"Loading {len(samples)} SSL features from {csv_path.name}...")
    
    features_list = []
    labels_list = []
    
    for feature_path, label_str in tqdm(samples, desc=f"Bundling {csv_path.name}"):
        try:
            features = np.load(feature_path)
            features = torch.from_numpy(features).float()
            
            # Verify shape
            if features.shape[-1] != TARGET_DIM:
                print(f"Warning: {feature_path} has shape {features.shape}, expected (..., {TARGET_DIM})")
                continue
            
            features_list.append(features)
            
            if label_str == 'synthetic' or str(label_str) == '1':
                labels_list.append(1)
            else:
                labels_list.append(0)
                
        except Exception as e:
            print(f"Error loading {feature_path}: {e}")
            continue
    
    # Stack
    features_tensor = torch.stack(features_list)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    
    bundle = {
        'features': features_tensor,
        'labels': labels_tensor,
        'feature_type': 'ssl',
        'feature_dim': TARGET_DIM
    }
    
    torch.save(bundle, output_path)
    size_gb = output_path.stat().st_size / 1e9
    print(f"Saved {output_path} ({size_gb:.2f} GB, {len(labels_tensor)} samples)")


def main():
    csv_dir = Path("unified_dataset")
    
    # Find SSL feature CSVs
    ssl_dirs = list(csv_dir.glob("ssl_features_*"))
    if not ssl_dirs:
        print("ERROR: No ssl_features_* directories found. Run preextract_ssl_features.py first.")
        return
    
    ssl_dir = ssl_dirs[0]  # Use first one found
    print(f"Using SSL features from: {ssl_dir}")
    
    print("="*60)
    print("Bundling SSL Features")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        csv_path = csv_dir / f"{split}_ssl_features.csv"
        output_path = csv_dir / f"{split}_ssl_bundle.pt"
        
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found")
            continue
        
        if output_path.exists():
            print(f"{output_path} already exists, skipping...")
            continue
        
        bundle_split(csv_path, output_path)
    
    print("\n" + "="*60)
    print("DONE! SSL bundles ready for training.")
    print("="*60)


if __name__ == "__main__":
    main()
