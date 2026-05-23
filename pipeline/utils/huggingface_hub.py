"""
Utilities for working with Hugging Face Hub models.
"""

from pathlib import Path


def get_cache_dir() -> Path:
    """Return the default HuggingFace model cache directory."""
    import os
    cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return Path(cache)


def model_is_cached(model_id: str) -> bool:
    """
    Check if a HuggingFace model is already cached locally.

    Args:
        model_id: HuggingFace model identifier (e.g. "facebook/wav2vec2-base-960h")

    Returns:
        True if model appears to be cached, False otherwise.
    """
    cache_dir = get_cache_dir() / "hub"
    slug = "models--" + model_id.replace("/", "--")
    return (cache_dir / slug).exists()
