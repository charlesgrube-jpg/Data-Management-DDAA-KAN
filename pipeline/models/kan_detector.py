"""
KAN (Kolmogorov-Arnold Network) Deepfake Audio Detector

Uses learnable spline-based activation functions for complex non-linear 
decision boundaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .base_detector import BaseDetector


class SplineLinear(nn.Module):
    """
    Linear layer with learnable spline-like activation functions.
    
    Uses a simpler RBF-based approach for numerical stability.
    Implements the KAN layer where each connection has its own 
    learnable activation function.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_knots: int = 8,
        spline_order: int = 3,
        grid_range: tuple = (-1, 1)
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_knots = num_knots
        
        # Base linear transformation (for residual connection)
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * (1.0 / in_features ** 0.5)
        )
        
        # Grid points for RBF interpolation
        grid_min, grid_max = grid_range
        grid = torch.linspace(grid_min, grid_max, num_knots)
        self.register_buffer('grid', grid)
        
        # Width of RBF basis functions
        self.bandwidth = (grid_max - grid_min) / (num_knots - 1)
        
        # Spline coefficients: (out_features, in_features, num_knots)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, num_knots) * 0.1
        )
        
        # Scale factors
        self.scale_base = nn.Parameter(torch.ones(1))
        self.scale_spline = nn.Parameter(torch.ones(1))
    
    def compute_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF basis functions.
        
        Args:
            x: Input tensor of shape (batch, in_features)
            
        Returns:
            Basis values of shape (batch, in_features, num_knots)
        """
        # x: (batch, in_features) -> (batch, in_features, 1)
        x = x.unsqueeze(-1)
        
        # grid: (num_knots,) -> (1, 1, num_knots)
        grid = self.grid.view(1, 1, -1)
        
        # Gaussian RBF: exp(-0.5 * ((x - center) / bandwidth)^2)
        basis = torch.exp(-0.5 * ((x - grid) / self.bandwidth) ** 2)
        
        return basis  # (batch, in_features, num_knots)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN layer.
        
        Args:
            x: Input tensor of shape (batch, in_features)
            
        Returns:
            Output tensor of shape (batch, out_features)
        """
        # Base linear transformation
        base_out = F.linear(x, self.base_weight)
        
        # Spline transformation using RBF basis
        basis = self.compute_basis(x)  # (batch, in_features, num_knots)
        
        # Weighted sum: einsum over num_knots dimension
        # spline_weight: (out_features, in_features, num_knots)
        # basis: (batch, in_features, num_knots)
        # Output: (batch, out_features)
        spline_out = torch.einsum('bik,oik->bo', basis, self.spline_weight)
        
        # Combine base and spline outputs
        output = self.scale_base * base_out + self.scale_spline * spline_out
        
        return output


class KANDetector(BaseDetector):
    """
    KAN-based detector for deepfake audio.
    
    Architecture:
    1. Flatten CQT to 1D (global pooling over time)
    2. Stack of KAN layers with spline activations
    3. Classification head
    
    KANs excel at learning complex non-linear decision boundaries
    with potentially better interpretability.
    """
    
    def __init__(
        self,
        n_bins: int = 84,
        hidden_dims: List[int] = [128, 64, 32],
        num_knots: int = 8,
        spline_order: int = 3,
        dropout: float = 0.1,
        pool_type: str = "mean",
        num_classes: int = 2,
        **kwargs
    ):
        """
        Args:
            n_bins: Number of CQT frequency bins
            hidden_dims: Hidden layer dimensions
            num_knots: Number of B-spline knots per activation
            spline_order: B-spline order (3 = cubic)
            dropout: Dropout probability
            pool_type: Temporal pooling type ("mean", "max", or "both")
            num_classes: Number of output classes
        """
        super().__init__(
            num_classes=num_classes,
            n_bins=n_bins,
            hidden_dims=hidden_dims,
            num_knots=num_knots,
            spline_order=spline_order,
            dropout=dropout,
            pool_type=pool_type
        )
        
        self.n_bins = n_bins
        self.pool_type = pool_type
        
        # Determine input dimension after pooling
        if pool_type == "both":
            input_dim = n_bins * 2
        else:
            input_dim = n_bins
        
        # Build KAN layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(SplineLinear(prev_dim, hidden_dim, num_knots, spline_order))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.kan_layers = nn.ModuleList(layers)
        
        # Classification head
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        
    def _pool_time(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool over time dimension.
        
        Args:
            x: Input of shape (batch, n_bins, time)
            
        Returns:
            Pooled features of shape (batch, pooled_dim)
        """
        if self.pool_type == "mean":
            return x.mean(dim=-1)
        elif self.pool_type == "max":
            return x.max(dim=-1)[0]
        else:  # both
            mean_pool = x.mean(dim=-1)
            max_pool = x.max(dim=-1)[0]
            return torch.cat([mean_pool, max_pool], dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input CQT features of shape (batch, n_bins, time_frames)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Pool over time
        x = self._pool_time(x)  # (batch, n_bins) or (batch, n_bins * 2)
        
        # Pass through KAN layers
        for layer in self.kan_layers:
            x = layer(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        x = self._pool_time(x)
        for layer in self.kan_layers:
            x = layer(x)
        return x


# Alternative: Use efficient-kan library if available
def get_efficient_kan_detector(**kwargs):
    """
    Create KAN detector using efficient-kan library if available.
    Falls back to custom implementation otherwise.
    """
    try:
        from efficient_kan import KAN
        print("[KAN] Using efficient-kan library")
        
        class EfficientKANDetector(BaseDetector):
            def __init__(self, n_bins=84, hidden_dims=[128, 64, 32], 
                        num_classes=2, pool_type="mean", **kwargs):
                super().__init__(num_classes=num_classes, **kwargs)
                
                self.pool_type = pool_type
                input_dim = n_bins * 2 if pool_type == "both" else n_bins
                
                self.kan = KAN(
                    layers_hidden=[input_dim] + hidden_dims + [num_classes]
                )
            
            def forward(self, x):
                if self.pool_type == "mean":
                    x = x.mean(dim=-1)
                elif self.pool_type == "max":
                    x = x.max(dim=-1)[0]
                else:
                    x = torch.cat([x.mean(dim=-1), x.max(dim=-1)[0]], dim=-1)
                return self.kan(x)
        
        return EfficientKANDetector(**kwargs)
        
    except ImportError:
        print("[KAN] efficient-kan not available, using custom implementation")
        return KANDetector(**kwargs)


if __name__ == "__main__":
    # Test the model
    model = KANDetector(
        n_bins=84,
        hidden_dims=[128, 64, 32],
        num_knots=8
    )
    
    # Dummy input (batch=2, n_bins=84, time=100)
    x = torch.randn(2, 84, 100)
    
    # Forward pass
    output = model(x)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Config: {model.get_config()}")
