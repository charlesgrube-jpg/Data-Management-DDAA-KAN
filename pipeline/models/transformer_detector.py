"""
Transformer-based Deepfake Audio Detector

Uses self-attention over CQT time frames for classification.
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from .base_detector import BaseDetector


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerDetector(BaseDetector):
    """
    Transformer-based detector for deepfake audio.
    
    Architecture:
    1. Linear projection of CQT bins to hidden dimension
    2. Positional encoding
    3. Stack of Transformer encoder layers
    4. Global average pooling
    5. Classification head
    """
    
    def __init__(
        self,
        n_bins: int = 84,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 2,
        **kwargs
    ):
        """
        Args:
            n_bins: Number of CQT frequency bins (input dimension)
            hidden_dim: Transformer hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            num_classes: Number of output classes
        """
        super().__init__(
            num_classes=num_classes,
            n_bins=n_bins,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.n_bins = n_bins
        self.hidden_dim = hidden_dim
        
        # Input projection: (batch, n_bins, time) -> (batch, time, hidden_dim)
        self.input_projection = nn.Linear(n_bins, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input CQT features of shape (batch, n_bins, time_frames)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Transpose to (batch, time, n_bins) for linear projection
        x = x.transpose(1, 2)
        
        # Project to hidden dimension
        x = self.input_projection(x)  # (batch, time, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, time, hidden_dim)
        
        # Global average pooling over time
        x = x.mean(dim=1)  # (batch, hidden_dim)
        
        # Classification
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head.
        
        Args:
            x: Input CQT features of shape (batch, n_bins, time_frames)
            
        Returns:
            Features of shape (batch, hidden_dim)
        """
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return x
    
    def get_attention_weights(
        self, 
        x: torch.Tensor,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Get attention weights from a specific layer.
        Useful for visualization and interpretability.
        
        Note: Requires PyTorch >= 1.9 with need_weights support
        """
        # This is a simplified version - full implementation would 
        # require custom transformer layers with attention weight storage
        raise NotImplementedError(
            "Attention weight extraction requires custom implementation. "
            "Consider using a hook-based approach."
        )


if __name__ == "__main__":
    # Test the model
    model = TransformerDetector(
        n_bins=84,
        hidden_dim=256,
        num_heads=8,
        num_layers=4
    )
    
    # Dummy input (batch=2, n_bins=84, time=100)
    x = torch.randn(2, 84, 100)
    
    # Forward pass
    output = model(x)
    features = model.get_features(x)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Feature shape: {features.shape}")
    print(f"Config: {model.get_config()}")
