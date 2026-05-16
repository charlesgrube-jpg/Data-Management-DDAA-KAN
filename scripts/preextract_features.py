#!/usr/bin/env python3
"""
Pre-extract CQT features to disk for faster training.
Uses multiprocessing for 16x speedup.

Run this ONCE before training:
    python scripts/preextract_features.py --csv_dir unified_dataset --output_dir unified_dataset/features
"""

import argparse
import csv
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from functools import partial

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Global extractor (initialized per worker)
_extractor = None

def init_worker():
    """Initialize CQT extractor in each worker process."""
    global _extractor
    # Import here to avoid fork issues
    from pipeline.features.cqt_extractor import CQTExtractor
    _extractor = CQTExtractor(device="cpu")
    print(f"[Worker {os.getpid()}] Initialized CQT extractor")

def process_sample(args):
    """Process a single sample: extract features and save to disk."""
    file_path, output_path = args
    global _extractor
    
    try:
        # Skip if already exists
        if Path(output_path).exists():
            return output_path
        
        # Extract features
        features = _extractor.extract_file(file_path)
        
        # Save
        np.save(output_path, features)
        return output_path
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_features_parallel(csv_path: Path, output_dir: Path, num_workers: int = 16):
    """Extract CQT features using multiprocessing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_path = row.get('full_path', row.get('filename', ''))
            label_str = row.get('label', 'real')
            label = 1 if label_str == 'synthetic' else 0
            if file_path and Path(file_path).exists():
                samples.append((file_path, label))
    
    print(f"Processing {len(samples)} samples from {csv_path.name} with {num_workers} workers...")
    
    # Prepare work items: (input_path, output_path)
    work_items = []
    for file_path, label in samples:
        rel_name = Path(file_path).name.replace('.wav', '.npy')
        output_path = str(output_dir / rel_name)
        work_items.append((file_path, output_path))
    
    # Process in parallel
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(process_sample, work_items),
            total=len(work_items),
            desc=f"Extracting {csv_path.name}"
        ))
    
    # Filter successful results
    feature_paths = [r for r in results if r is not None]
    print(f"Successfully extracted {len(feature_paths)}/{len(samples)} features")
    
    # Create new CSV with feature paths
    new_csv_path = csv_path.parent / f"{csv_path.stem}_with_features.csv"
    
    # Build path map
    path_map = {}
    for (file_path, _), result in zip(work_items, results):
        if result:
            path_map[Path(file_path).name] = result
    
    # Write new CSV
    with open(csv_path, 'r') as f_in, open(new_csv_path, 'w', newline='') as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames) + ['feature_path']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            fname = Path(row.get('full_path', row.get('filename', ''))).name
            if fname in path_map:
                row['feature_path'] = path_map[fname]
                writer.writerow(row)
    
    print(f"Created: {new_csv_path}")
    return len(feature_paths)


def main():
    parser = argparse.ArgumentParser(description="Pre-extract CQT features (parallel)")
    parser.add_argument("--csv_dir", type=str, default="unified_dataset")
    parser.add_argument("--output_dir", type=str, default="unified_dataset/features")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    
    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"="*60)
    print(f"Parallel Feature Extraction")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_dir}")
    print(f"="*60)
    
    total = 0
    for split in ['train', 'val', 'test']:
        csv_path = csv_dir / f"{split}.csv"
        if csv_path.exists():
            split_output = output_dir / split
            count = extract_features_parallel(csv_path, split_output, args.workers)
            total += count
        else:
            print(f"Warning: {csv_path} not found")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE! Extracted {total} features")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
