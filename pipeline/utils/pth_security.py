"""
Security utilities for loading PyTorch checkpoint files.
"""

import os
from pathlib import Path


def is_safe_to_load(model_path: str) -> bool:
    """
    Perform a basic safety check before loading a .pt/.pth checkpoint.

    Checks:
    - File exists
    - File extension is .pt or .pth
    - File size is non-zero

    Args:
        model_path: Path to the checkpoint file.

    Returns:
        True if file appears safe to load, False otherwise.
    """
    path = Path(model_path)
    if not path.exists():
        return False
    if path.suffix not in (".pt", ".pth"):
        return False
    if path.stat().st_size == 0:
        return False
    return True
