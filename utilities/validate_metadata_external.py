"""
External Metadata Validator (Standalone Module)

Validates `metadata_external.json` to prevent:
1. Leakage: External synthetic data accidentally assigned to 'train' split.
2. Runtime crashes: Missing required fields or wrong types.

Usage:
    python utilities/validate_metadata_external.py external_data/metadata_external.json

Exit Codes:
    0 = Valid
    1 = Validation errors found
    2 = File not found / parse error

Author: Pipeline Automation
Date: 2026-01-04
"""

import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# =============================================================================
# Schema Definition
# =============================================================================

REQUIRED_FIELDS = {
    "path": str,
    "split": str,
    "method": str,
    "is_synthetic": bool,
}

OPTIONAL_FIELDS = {
    "speaker_id": str,
    "source_id": str,
    "duration": (int, float),
    "generator": str,
    "source": str,
}

ALLOWED_SPLITS = {"val", "test"}  # External data CANNOT be 'train'
ALLOWED_METHODS = {"elevenlabs", "asvspoof", "rvc", "wavefake", "external"}


@dataclass
class ValidationError:
    """Represents a single validation failure."""
    index: int
    path: str
    field: str
    reason: str

    def __str__(self):
        return f"[{self.index}] {self.path or 'unknown'}: {self.field} - {self.reason}"


@dataclass
class ValidationResult:
    """Aggregated validation result."""
    total_samples: int = 0
    valid_samples: int = 0
    errors: List[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"Validation Summary:",
            f"  Total Samples: {self.total_samples}",
            f"  Valid Samples: {self.valid_samples}",
            f"  Errors: {len(self.errors)}",
        ]
        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors[:20]:  # Show first 20 errors
                lines.append(f"  {err}")
            if len(self.errors) > 20:
                lines.append(f"  ... and {len(self.errors) - 20} more errors.")
        return "\n".join(lines)


# =============================================================================
# Validation Logic
# =============================================================================

def validate_sample(sample: Dict[str, Any], index: int) -> List[ValidationError]:
    """Validate a single sample against the schema."""
    errors = []
    sample_path = sample.get("path", "unknown_path")

    # 1. Check Required Fields
    for field_name, expected_type in REQUIRED_FIELDS.items():
        if field_name not in sample:
            errors.append(ValidationError(
                index, sample_path, field_name,
                f"Missing required field"
            ))
        elif not isinstance(sample[field_name], expected_type):
            errors.append(ValidationError(
                index, sample_path, field_name,
                f"Expected {expected_type.__name__}, got {type(sample[field_name]).__name__}"
            ))

    # 2. Check Optional Fields (if present, must be correct type)
    for field_name, expected_types in OPTIONAL_FIELDS.items():
        if field_name in sample:
            if isinstance(expected_types, tuple):
                if not isinstance(sample[field_name], expected_types):
                    errors.append(ValidationError(
                        index, sample_path, field_name,
                        f"Expected one of {expected_types}, got {type(sample[field_name]).__name__}"
                    ))
            elif not isinstance(sample[field_name], expected_types):
                errors.append(ValidationError(
                    index, sample_path, field_name,
                    f"Expected {expected_types.__name__}, got {type(sample[field_name]).__name__}"
                ))

    # 3. CRITICAL: Split Leakage Check
    split = sample.get("split", "")
    if split == "train":
        errors.append(ValidationError(
            index, sample_path, "split",
            "SECURITY: External data CANNOT be in 'train' split. This would cause leakage."
        ))
    elif split not in ALLOWED_SPLITS:
        errors.append(ValidationError(
            index, sample_path, "split",
            f"Invalid split '{split}'. Allowed: {ALLOWED_SPLITS}"
        ))

    # 4. is_synthetic Must Be True
    if sample.get("is_synthetic") is False:
        errors.append(ValidationError(
            index, sample_path, "is_synthetic",
            "External samples MUST be marked is_synthetic=True"
        ))

    # 5. Method Validation (Warning only if unknown)
    method = sample.get("method", "")
    if method and method not in ALLOWED_METHODS:
        # Not a hard error, just a warning
        pass  # Could log a warning here

    # 6. Path Existence Check (Optional - can be slow for large datasets)
    # Uncomment if you want to verify files exist:
    # if sample.get("path") and not Path(sample["path"]).exists():
    #     errors.append(ValidationError(
    #         index, sample_path, "path",
    #         f"File does not exist: {sample['path']}"
    #     ))

    return errors


def validate_metadata_file(file_path: Path) -> ValidationResult:
    """Validate an entire metadata_external.json file."""
    result = ValidationResult()

    if not file_path.exists():
        result.errors.append(ValidationError(
            -1, str(file_path), "file",
            f"File not found: {file_path}"
        ))
        return result

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
    except json.JSONDecodeError as e:
        result.errors.append(ValidationError(
            -1, str(file_path), "json",
            f"JSON parse error: {e}"
        ))
        return result

    if not isinstance(samples, list):
        result.errors.append(ValidationError(
            -1, str(file_path), "structure",
            f"Expected a JSON array, got {type(samples).__name__}"
        ))
        return result

    result.total_samples = len(samples)

    for i, sample in enumerate(samples):
        sample_errors = validate_sample(sample, i)
        if sample_errors:
            result.errors.extend(sample_errors)
        else:
            result.valid_samples += 1

    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate external metadata JSON for the deepfake pipeline."
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to metadata_external.json"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    result = validate_metadata_file(file_path)

    print(result.summary())

    if result.is_valid:
        print("\n✅ Validation PASSED")
        sys.exit(0)
    else:
        print("\n❌ Validation FAILED")
        sys.exit(1)


# =============================================================================
# Test Fixtures (For Development/Testing)
# =============================================================================

TEST_FIXTURES = {
    "valid_minimal": {
        "path": "/data/audio/sample_001.wav",
        "split": "test",
        "method": "elevenlabs",
        "is_synthetic": True
    },
    "valid_full": {
        "path": "/data/audio/sample_002.wav",
        "split": "val",
        "method": "rvc",
        "is_synthetic": True,
        "speaker_id": "spk_ext_001",
        "source_id": "src_ext_001",
        "duration": 3.5,
        "generator": "rvc_v2"
    },
    "invalid_train_split": {
        "path": "/data/audio/sample_003.wav",
        "split": "train",  # INVALID: External cannot be train
        "method": "elevenlabs",
        "is_synthetic": True
    },
    "invalid_missing_path": {
        # "path": missing
        "split": "test",
        "method": "elevenlabs",
        "is_synthetic": True
    },
    "invalid_not_synthetic": {
        "path": "/data/audio/sample_005.wav",
        "split": "test",
        "method": "elevenlabs",
        "is_synthetic": False  # INVALID: Must be True
    },
    "invalid_wrong_type": {
        "path": "/data/audio/sample_006.wav",
        "split": 123,  # INVALID: Should be string
        "method": "elevenlabs",
        "is_synthetic": "yes"  # INVALID: Should be bool
    }
}


def run_self_test():
    """Run validation on built-in test fixtures."""
    print("Running self-test with built-in fixtures...\n")
    
    for name, sample in TEST_FIXTURES.items():
        errors = validate_sample(sample, 0)
        status = "✅ PASS" if not errors else "❌ FAIL"
        expected = "PASS" if name.startswith("valid") else "FAIL"
        test_passed = (status.startswith("✅") == name.startswith("valid"))
        
        print(f"{name}: {status}")
        if errors:
            for e in errors:
                print(f"    -> {e.field}: {e.reason}")
        
        if not test_passed:
            print(f"    ⚠️ Unexpected result! Expected {expected}")
    
    print("\nSelf-test complete.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        run_self_test()
    else:
        main()
