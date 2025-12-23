"""
Effects Module

Channel degradation and quality simulation:
- Quality tiers: clean, mobile, noisy
- Noise addition
- Reverb simulation
- Low-pass filtering
- Compression artifacts
"""

from .effects import apply_effects, apply_effects_to_pair, select_quality_tier

__all__ = ["apply_effects", "apply_effects_to_pair", "select_quality_tier"]
