"""
Speaker-Disjoint Splitter Module (Hash-Based)

Implements deterministic, stateless splitting based on speaker ID.
Supports streaming processing by deciding splits per-sample without loading the full dataset.
Respects pre-assigned splits (e.g. for external datasets).
"""

import hashlib
from typing import Dict, List, Any
from pipeline.config import Config


def assign_split(sample: Dict[str, Any], config: Config) -> Dict[str, Any]:
    """
    Assign split label ('train', 'val', 'test') to a sample using hash of speaker_id.
    
    This is strictly deterministic: the same speaker_id will ALWAYS imply the same split.
    This allows us to process infinite-sized datasets in a streaming fashion.
    
    Args:
        sample: Dictionary containing 'speaker_id' or 'source_id'
        config: Pipeline configuration with split ratios
        
    Returns:
        The sample dict with 'split' key added/updated
    """
    # 1. GUARD CLAUSE: Respect Pre-assigned Splits
    # If a sample already has a split (e.g. from 11Labs/RVC ingest), KEEP IT.
    if sample.get('split'):
        return sample
        
    # 2. Extract Key
    # Use speaker_id (Real Source) to ensure content disjointness.
    # Fallback to 'client_id' or 'unknown' if missing.
    speaker_key = sample.get("speaker_id", sample.get("source_id", sample.get("client_id", "unknown")))
    
    # 3. Deterministic Hashing
    # MD5 is fast and uniform enough for this purpose.
    # We use the config seed to salt the hash if available, ensuring 
    # different splits for different runs if desired (but usually we want stability).
    salt = str(config.seed) if config.seed is not None else "42"
    key_string = f"{speaker_key}_{salt}"
    
    # Hex digest -> Integer
    h = int(hashlib.md5(key_string.encode()).hexdigest(), 16)
    
    # Normalize to [0.0, 1.0]
    # Modulo 10000 gives us 4 decimal places of precision
    p = (h % 10000) / 10000.0
    
    # 4. Assign Split
    train_end = config.splits.train
    val_end = train_end + config.splits.val
    
    if p < train_end:
        sample['split'] = 'train'
    elif p < val_end:
        sample['split'] = 'val'
    else:
        sample['split'] = 'test'
        
    return sample


def create_speaker_disjoint_splits(
    samples: List[Dict[str, Any]],
    config: Config
) -> List[Dict[str, Any]]:
    """
    Apply hash-based splitting to a list of samples.
    
    Args:
        samples: List of sample dictionaries
        config: Pipeline configuration
        
    Returns:
        List of samples with 'split' assigned
    """
    print(f"[Splitter] Assigning splits to {len(samples)} samples using Hash Strategy...")
    
    counts = {"train": 0, "val": 0, "test": 0}
    
    for sample in samples:
        assign_split(sample, config)
        counts[sample['split']] += 1
        
    print(f"[Splitter] Result: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    return samples


def verify_no_speaker_leakage(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Verify that no speaker appears in multiple splits.
    
    Returns:
        Dict with verification result and any violations found
    """
    split_speakers: Dict[str, set] = {
        "train": set(),
        "val": set(),
        "test": set()
    }
    
    for sample in samples:
        split = sample.get("split")
        speaker = sample.get("speaker_id", sample.get("source_id"))
        
        if split and speaker:
            split_speakers[split].add(speaker)
    
    # Check for overlaps
    train_val_overlap = split_speakers["train"] & split_speakers["val"]
    train_test_overlap = split_speakers["train"] & split_speakers["test"]
    val_test_overlap = split_speakers["val"] & split_speakers["test"]
    
    all_overlaps = train_val_overlap | train_test_overlap | val_test_overlap
    
    return {
        "passed": len(all_overlaps) == 0,
        "train_test_overlap": list(train_test_overlap),
        "total_violations": len(all_overlaps)
    }


if __name__ == "__main__":
    # Test
    from pipeline.config import load_config
    cfg = load_config("../config.yaml")
    
    mock_samples = [
        {"speaker_id": "spk_1"}, 
        {"speaker_id": "spk_2"}, 
        {"speaker_id": "spk_1"}, # Should follow spk_1
        {"speaker_id": "spk_3", "split": "test"} # Pre-assigned
    ]
    
    processed = create_speaker_disjoint_splits(mock_samples, cfg)
    for p in processed:
        print(f"Spk: {p['speaker_id']} -> {p['split']}")


def get_split_statistics(samples: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """
    Calculate statistics for each split.
    
    Returns:
        Dict with statistics per split
    """
    stats = {
        "train": {"total": 0, "real": 0, "synthetic": 0, "speakers": set()},
        "val": {"total": 0, "real": 0, "synthetic": 0, "speakers": set()},
        "test": {"total": 0, "real": 0, "synthetic": 0, "speakers": set()}
    }
    
    for sample in samples:
        split = sample.get("split")
        if not split:
            continue
        
        # Ensure split key exists in stats (handle custom splits)
        if split not in stats:
             stats[split] = {"total": 0, "real": 0, "synthetic": 0, "speakers": set()}
             
        stats[split]["total"] += 1
        
        if sample.get("is_synthetic", False):
            stats[split]["synthetic"] += 1
        else:
            stats[split]["real"] += 1
        
        speaker = sample.get("speaker_id", sample.get("source_id"))
        if speaker:
            stats[split]["speakers"].add(speaker)
    
    # Convert sets to counts
    for split in stats:
        stats[split]["num_speakers"] = len(stats[split]["speakers"])
        stats[split]["speakers"] = None  # Don't include full set in output
        
        # Calculate balance ratio
        total = stats[split]["total"]
        if total > 0:
            stats[split]["real_ratio"] = stats[split]["real"] / total
            stats[split]["synthetic_ratio"] = stats[split]["synthetic"] / total
    
    return stats
