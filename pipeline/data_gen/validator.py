"""
Validation Module

Quality checks for the processed dataset:
- Class balance (real/synthetic ratio)
- Speaker leakage verification
- Duplicate detection
- File integrity checks
"""

import hashlib
import numpy as np
from typing import Dict, List, Any, Set
from collections import defaultdict
from pipeline.config import Config
from .splitter import verify_no_speaker_leakage, get_split_statistics


def validate_dataset(
    samples: List[Dict[str, Any]],
    config: Config
) -> Dict[str, Any]:
    """
    Run all validation checks on the dataset.
    
    Args:
        samples: List of processed samples
        config: Pipeline configuration
        
    Returns:
        Dict with validation results for each check
    """
    results = {}
    
    # Check 1: Speaker leakage
    if config.validation.check_speaker_leakage:
        leakage_result = verify_no_speaker_leakage(samples)
        results["speaker_leakage"] = {
            "passed": leakage_result["passed"],
            "violations": leakage_result["total_violations"]
        }
        if not leakage_result["passed"]:
            print(f"[Validator] FAILED: Speaker leakage detected!")
    
    # Check 2: Class balance
    if config.validation.check_balance:
        balance_result = check_class_balance(samples, config)
        results["class_balance"] = balance_result
        if not balance_result["passed"]:
            print(f"[Validator] WARNING: Class imbalance detected")
    
    # Check 3: Duplicates
    if config.validation.check_duplicates:
        duplicate_result = check_duplicates(samples)
        results["duplicates"] = duplicate_result
        if not duplicate_result["passed"]:
            print(f"[Validator] WARNING: Duplicates detected")
    
    # Check 4: Minimum samples per split
    min_samples_result = check_minimum_samples(samples, config)
    results["minimum_samples"] = min_samples_result
    if not min_samples_result["passed"]:
        print(f"[Validator] WARNING: Insufficient samples in some splits")
    
    # Check 5: Audio integrity
    integrity_result = check_audio_integrity(samples)
    results["audio_integrity"] = integrity_result
    if not integrity_result["passed"]:
        print(f"[Validator] WARNING: Audio integrity issues found")
    
    # Overall pass/fail
    results["all_passed"] = all(
        r.get("passed", True) for r in results.values()
    )
    
    return results


def check_class_balance(
    samples: List[Dict[str, Any]],
    config: Config
) -> Dict[str, Any]:
    """
    Check if real/synthetic ratio is within tolerance.
    """
    tolerance = config.validation.balance_tolerance
    target = 0.5
    
    stats = get_split_statistics(samples)
    issues = []
    
    for split in ["train", "val", "test"]:
        if split not in stats or stats[split]["total"] == 0:
            continue
        
        real_ratio = stats[split].get("real_ratio", 0.5)
        
        if abs(real_ratio - target) > tolerance:
            issues.append({
                "split": split,
                "real_ratio": real_ratio,
                "expected": f"{target-tolerance:.2f} - {target+tolerance:.2f}"
            })
    
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "stats": stats
    }


def check_duplicates(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check for duplicate audio content using hashing.
    """
    hashes = defaultdict(list)
    
    for idx, sample in enumerate(samples):
        audio = sample.get("audio")
        if audio is None:
            continue
        
        # Create hash of audio content
        audio_hash = compute_audio_hash(audio)
        hashes[audio_hash].append(idx)
    
    # Find duplicates (hashes with multiple samples)
    duplicates = {h: indices for h, indices in hashes.items() if len(indices) > 1}
    
    return {
        "passed": len(duplicates) == 0,
        "num_duplicates": sum(len(v) - 1 for v in duplicates.values()),
        "duplicate_groups": len(duplicates)
    }


def compute_audio_hash(audio: np.ndarray) -> str:
    """
    Compute hash of audio array for duplicate detection.
    """
    # Round to reduce floating point noise
    rounded = np.round(audio * 1000).astype(np.int16)
    return hashlib.md5(rounded.tobytes()).hexdigest()


def check_minimum_samples(
    samples: List[Dict[str, Any]],
    config: Config
) -> Dict[str, Any]:
    """
    Check if each split has minimum required samples.
    """
    min_required = config.validation.min_samples_per_split
    split_counts = defaultdict(int)
    
    for sample in samples:
        split = sample.get("split")
        if split:
            split_counts[split] += 1
    
    issues = []
    for split in ["train", "val", "test"]:
        count = split_counts.get(split, 0)
        if count < min_required:
            issues.append({
                "split": split,
                "count": count,
                "required": min_required
            })
    
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "counts": dict(split_counts)
    }


def check_audio_integrity(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check for audio quality issues.
    """
    issues = []
    
    for idx, sample in enumerate(samples):
        audio = sample.get("audio")
        if audio is None:
            issues.append({"idx": idx, "issue": "missing_audio"})
            continue
        
        # Check for NaN/Inf
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            issues.append({"idx": idx, "issue": "nan_or_inf"})
            continue
        
        # Check for silence (all zeros or very low RMS)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-6:
            issues.append({"idx": idx, "issue": "silence"})
            continue
        
        # Check for clipping
        if np.any(np.abs(audio) > 0.999):
            issues.append({"idx": idx, "issue": "clipping"})
    
    return {
        "passed": len(issues) == 0,
        "num_issues": len(issues),
        "issue_types": list(set(i["issue"] for i in issues)),
        "issues": issues[:10] if len(issues) > 10 else issues  # Limit output
    }


def generate_validation_report(
    results: Dict[str, Any],
    output_path: str = None
) -> str:
    """
    Generate human-readable validation report.
    """
    lines = ["=" * 60]
    lines.append("DATASET VALIDATION REPORT")
    lines.append("=" * 60)
    
    overall = "✅ PASSED" if results.get("all_passed") else "❌ FAILED"
    lines.append(f"\nOverall: {overall}\n")
    
    for check_name, check_result in results.items():
        if check_name == "all_passed":
            continue
        
        status = "✅" if check_result.get("passed", True) else "❌"
        lines.append(f"{status} {check_name}")
        
        if not check_result.get("passed", True):
            if "issues" in check_result:
                for issue in check_result["issues"][:5]:
                    lines.append(f"    - {issue}")
            if "violations" in check_result:
                lines.append(f"    - Violations: {check_result['violations']}")
    
    lines.append("\n" + "=" * 60)
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


if __name__ == "__main__":
    # Test with mock data
    mock_samples = []
    
    for i in range(100):
        mock_samples.append({
            "speaker_id": f"speaker_{i % 20}",
            "audio": np.random.randn(16000 * 3) * 0.1,
            "is_synthetic": i % 2 == 0,
            "split": ["train", "val", "test"][i % 3]
        })
    
    from pipeline.config import load_config
    cfg = load_config("../config.yaml")
    
    results = validate_dataset(mock_samples, cfg)
    report = generate_validation_report(results)
    print(report)
