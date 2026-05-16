#!/usr/bin/env python3
"""
Bundle all pre-extracted .npy features into single .pt tensor files.
This enables instant loading for training.

Run ONCE after feature extraction:
    python scripts/bundle_features.py

Creates:
    unified_dataset/train_bundle.pt  (~10GB)
    unified_dataset/val_bundle.pt    (~2GB)
    unified_dataset/test_bundle.pt   (~2GB)
"""

import csv
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

TARGET_TIME = 100

def bundle_split(csv_path: Path, output_path: Path):
    """Load all features from CSV and save as single tensor bundle."""
    
    # Read CSV with feature paths
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'feature_path' not in row:
                print(f"ERROR: {csv_path} doesn't have feature_path column!")
                return
            samples.append((row['feature_path'], row.get('label', 'real')))
    
    print(f"Loading {len(samples)} samples from {csv_path.name}...")
    
    features_list = []
    labels_list = []
    
    for feature_path, label_str in tqdm(samples, desc=f"Bundling {csv_path.name}"):
        try:
            # Load .npy
            features = np.load(feature_path)
            features = torch.from_numpy(features).float()
            
            # Ensure consistent shape
            if features.shape[-1] < TARGET_TIME:
                pad_width = TARGET_TIME - features.shape[-1]
                features = torch.nn.functional.pad(features, (0, pad_width))
            elif features.shape[-1] > TARGET_TIME:
                features = features[..., :TARGET_TIME]
            
            features_list.append(features)
            
            # Parse label
            if label_str == 'synthetic' or str(label_str) == '1':
                labels_list.append(1)
            else:
                labels_list.append(0)
                
        except Exception as e:
            print(f"Error loading {feature_path}: {e}")
            continue
    
    # Stack into tensors
    features_tensor = torch.stack(features_list)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    
    # Save bundle
    bundle = {
        'features': features_tensor,
        'labels': labels_tensor
    }
    
    torch.save(bundle, output_path)
    size_gb = output_path.stat().st_size / 1e9
    print(f"Saved {output_path} ({size_gb:.2f} GB, {len(labels_tensor)} samples)")


def main():
    csv_dir = Path("unified_dataset")
    
    print("="*60)
    print("Bundling Features into Single Tensor Files")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        csv_path = csv_dir / f"{split}_with_features.csv"
        output_path = csv_dir / f"{split}_bundle.pt"
        
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping...")
            continue
            
        if output_path.exists():
            print(f"{output_path} already exists, skipping...")
            continue
        
        bundle_split(csv_path, output_path)
    
    print("\n" + "="*60)
    print("DONE! Training will now load instantly.")
    print("="*60)


if __name__ == "__main__":
    main()
