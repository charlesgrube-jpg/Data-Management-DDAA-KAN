"""
DDAA Defense Models

This module contains all detector architectures for deepfake audio detection.
"""

from .base_detector import BaseDetector
from .transformer_detector import TransformerDetector
from .kan_detector import KANDetector
from .neural_ode_detector import NeuralODEDetector
from .pinn_detector import PINNDetector

__all__ = [
    "BaseDetector",
    "TransformerDetector",
    "KANDetector",
    "NeuralODEDetector",
    "PINNDetector",
    "get_detector",
]


def get_detector(name: str, **kwargs):
    """
    Factory function to get a detector by name.

    Args:
        name: One of "transformer", "kan", "neural_ode", "pinn"
        **kwargs: Model-specific keyword arguments

    Returns:
        Detector instance
    """
    name = name.lower()
    if name == "transformer":
        from .transformer_detector import TransformerDetector
        return TransformerDetector(**kwargs)
    elif name == "kan":
        from .kan_detector import KANDetector
        return KANDetector(**kwargs)
    elif name == "neural_ode":
        from .neural_ode_detector import NeuralODEDetector
        return NeuralODEDetector(**kwargs)
    elif name == "pinn":
        from .pinn_detector import PINNDetector
        return PINNDetector(**kwargs)
    else:
        raise ValueError(
            f"Unknown detector: {name}. Choose from: transformer, kan, neural_ode, pinn"
        )
