"""
Speaker-Disjoint Splitter Module

Creates train/val/test splits where NO speaker appears in multiple splits.
This is CRITICAL to prevent data leakage and ensure real-world generalization.
"""

import random
from typing import Dict, List, Set, Any
from collections import defaultdict
from pipeline.config import Config


def create_speaker_disjoint_splits(
    samples: List[Dict[str, Any]],
    config: Config
) -> List[Dict[str, Any]]:
    """
    Assign split labels ensuring no speaker appears in multiple splits.
    
    Args:
        samples: List of sample dictionaries (must have 'speaker_id' key)
        config: Pipeline configuration
        
    Returns:
        Same samples with 'split' key added
    """
    # Group samples by speaker
    speaker_samples = defaultdict(list)
    for sample in samples:
        speaker_id = sample.get("speaker_id", sample.get("source_id"))
        speaker_samples[speaker_id].append(sample)
    
    speakers = list(speaker_samples.keys())
    print(f"[Splitter] Found {len(speakers)} unique speakers")
    
    # Shuffle speakers (using config seed if set)
    random.shuffle(speakers)
    
    # Calculate split boundaries
    n = len(speakers)
    train_end = int(n * config.splits.train)
    val_end = train_end + int(n * config.splits.val)
    
    train_speakers = set(speakers[:train_end])
    val_speakers = set(speakers[train_end:val_end])
    test_speakers = set(speakers[val_end:])
    
    print(f"[Splitter] Split speakers: train={len(train_speakers)}, "
          f"val={len(val_speakers)}, test={len(test_speakers)}")
    
    # Assign split to each sample
    train_count = val_count = test_count = 0
    
    for sample in samples:
        speaker_id = sample.get("speaker_id", sample.get("source_id"))
        
        if speaker_id in train_speakers:
            sample["split"] = "train"
            train_count += 1
        elif speaker_id in val_speakers:
            sample["split"] = "val"
            val_count += 1
        else:
            sample["split"] = "test"
            test_count += 1
    
    print(f"[Splitter] Split samples: train={train_count}, "
          f"val={val_count}, test={test_count}")
    
    return samples


def verify_no_speaker_leakage(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Verify that no speaker appears in multiple splits.
    
    Returns:
        Dict with verification result and any violations found
    """
    split_speakers: Dict[str, Set[str]] = {
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
        "train_val_overlap": list(train_val_overlap),
        "train_test_overlap": list(train_test_overlap),
        "val_test_overlap": list(val_test_overlap),
        "total_violations": len(all_overlaps)
    }


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


def resplit_for_balance(
    samples: List[Dict[str, Any]],
    config: Config,
    max_attempts: int = 10
) -> List[Dict[str, Any]]:
    """
    Attempt to create splits with balanced real/synthetic ratio.
    
    Retries splitting with different random shuffles if balance is poor.
    """
    tolerance = config.validation.balance_tolerance
    target = 0.5  # Target 50% real
    
    for attempt in range(max_attempts):
        samples = create_speaker_disjoint_splits(samples, config)
        stats = get_split_statistics(samples)
        
        # Check balance in each split
        balanced = True
        for split in ["train", "val", "test"]:
            ratio = stats[split].get("real_ratio", 0.5)
            if abs(ratio - target) > tolerance:
                balanced = False
                break
        
        if balanced:
            print(f"[Splitter] Achieved balanced splits on attempt {attempt + 1}")
            return samples
        
        print(f"[Splitter] Attempt {attempt + 1}: splits not balanced, retrying...")
    
    print(f"[Splitter] WARNING: Could not achieve balanced splits after {max_attempts} attempts")
    return samples


if __name__ == "__main__":
    # Test with mock data
    mock_samples = []
    
    # Create 100 speakers, 10 samples each (5 real + 5 synthetic)
    for speaker_idx in range(100):
        speaker_id = f"speaker_{speaker_idx:03d}"
        
        for sample_idx in range(5):
            # Real sample
            mock_samples.append({
                "speaker_id": speaker_id,
                "source_id": f"src_{speaker_idx:03d}",
                "is_synthetic": False,
                "chunk_idx": sample_idx
            })
            # Synthetic sample
            mock_samples.append({
                "speaker_id": speaker_id,
                "source_id": f"src_{speaker_idx:03d}",
                "is_synthetic": True,
                "chunk_idx": sample_idx
            })
    
    from pipeline.config import load_config
    cfg = load_config("../config.yaml")
    
    # Run splitter
    mock_samples = create_speaker_disjoint_splits(mock_samples, cfg)
    
    # Verify no leakage
    result = verify_no_speaker_leakage(mock_samples)
    print(f"No leakage: {result['passed']}")
    
    # Get stats
    stats = get_split_statistics(mock_samples)
    for split, s in stats.items():
        print(f"{split}: {s['total']} samples, {s['num_speakers']} speakers, "
              f"real={s['real_ratio']:.2%}")
