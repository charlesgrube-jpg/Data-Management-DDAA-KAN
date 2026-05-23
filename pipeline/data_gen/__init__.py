"""
Data Generation Module

Dataset loading, splitting, validation, and export:
- Multiple data loaders (HuggingFace, Mozilla CV, local files)
- Speaker-disjoint train/val/test splitting
- Quality validation checks
- Structured export with metadata
"""

from .mozilla_cv_loader import load_mozilla_cv
from .local_loader import load_local_dataset
from .splitter import create_speaker_disjoint_splits
from .validator import validate_dataset, generate_validation_report
from .exporter import export_dataset

__all__ = [
    "load_mozilla_cv",
    "load_local_dataset", 
    "create_speaker_disjoint_splits",
    "validate_dataset",
    "generate_validation_report",
    "export_dataset",
]
