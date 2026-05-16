"""
Base Detector Interface for DDAA

All defense models inherit from this abstract base class.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseDetector(nn.Module, ABC):
    """
    Abstract base class for deepfake audio detectors.
    
    All detector implementations must inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(self, num_classes: int = 2, **kwargs):
        """
        Initialize detector.
        
        Args:
            num_classes: Number of output classes (default 2: real/fake)
        """
        super().__init__()
        self.num_classes = num_classes
        self._config = kwargs
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the detector.
        
        Args:
            x: Input features tensor of shape (batch, n_bins, time_frames)
               Typically CQT spectrograms with n_bins=84
               
        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions.
        
        Args:
            x: Input features tensor
            
        Returns:
            Predicted class indices (0=real, 1=fake)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.
        
        Args:
            x: Input features tensor
            
        Returns:
            Softmax probabilities of shape (batch, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate feature representations.
        
        Useful for visualization, attack analysis, etc.
        Override in subclasses for model-specific feature extraction.
        
        Args:
            x: Input features tensor
            
        Returns:
            Internal feature representation
        """
        # Default: return input (subclasses should override)
        return x
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "name": self.__class__.__name__,
            "num_classes": self.num_classes,
            "num_parameters": self.count_parameters(),
            **self._config
        }
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> "BaseDetector":
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to saved checkpoint
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract config from checkpoint if available
        config = checkpoint.get("config", {})
        config.update(kwargs)
        
        # Create model and load weights
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model
    
    def save_checkpoint(
        self, 
        path: str, 
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        **extra_info
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Save path
            optimizer: Optional optimizer to save
            epoch: Optional epoch number
            **extra_info: Additional info to save
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.get_config(),
            **extra_info
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
            
        torch.save(checkpoint, path)


if __name__ == "__main__":
    # This is an abstract class, cannot be instantiated directly
    print("BaseDetector is an abstract class.")
    print("Use one of: TransformerDetector, KANDetector, NeuralODEDetector, PINNDetector")
