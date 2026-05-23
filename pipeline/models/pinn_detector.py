"""
Physics-Informed Neural Network (PINN) Deepfake Audio Detector

Incorporates physical constraints from acoustic signal properties
to regularize the network and potentially improve robustness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .base_detector import BaseDetector


class PhysicsConstraints(nn.Module):
    """
    Compute physics-based regularization losses for audio signals.
    
    Incorporates acoustic signal properties as soft constraints:
    - Energy conservation (signal energy should be preserved through transformations)
    - Spectral smoothness (adjacent frequency bins should be correlated)
    - Temporal continuity (adjacent time frames should be related)
    """
    
    def __init__(
        self,
        energy_weight: float = 0.1,
        smoothness_weight: float = 0.1,
        continuity_weight: float = 0.1
    ):
        super().__init__()
        self.energy_weight = energy_weight
        self.smoothness_weight = smoothness_weight
        self.continuity_weight = continuity_weight
    
    def energy_conservation_loss(
        self, 
        x_in: torch.Tensor, 
        x_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize energy change through the network.
        
        For legitimate audio processing, energy should be approximately preserved.
        """
        energy_in = (x_in ** 2).sum(dim=-1).mean()
        energy_out = (x_out ** 2).sum(dim=-1).mean()
        
        # Relative energy change
        energy_ratio = energy_out / (energy_in + 1e-8)
        return (energy_ratio - 1.0) ** 2
    
    def spectral_smoothness_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encourage spectral smoothness in frequency domain.
        
        Real audio typically has correlated adjacent frequency bins.
        """
        if x.dim() == 2:
            # (batch, features)
            diff = x[:, 1:] - x[:, :-1]
        else:
            # (batch, freq, time) - smoothness over frequency
            diff = x[:, 1:, :] - x[:, :-1, :]
        
        return (diff ** 2).mean()
    
    def temporal_continuity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encourage temporal continuity across time frames.
        
        Real audio has temporal correlations.
        """
        if x.dim() < 3:
            return torch.tensor(0.0, device=x.device)
        
        # (batch, freq, time)
        diff = x[:, :, 1:] - x[:, :, :-1]
        return (diff ** 2).mean()
    
    def forward(
        self, 
        x_in: torch.Tensor, 
        x_out: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total physics loss and individual components.
        
        Args:
            x_in: Input features
            x_out: Transformed features
            
        Returns:
            Total physics loss, dictionary of individual losses
        """
        losses = {}
        total = torch.tensor(0.0, device=x_in.device)
        
        if self.energy_weight > 0:
            energy_loss = self.energy_conservation_loss(x_in, x_out)
            losses['energy'] = energy_loss.item()
            total = total + self.energy_weight * energy_loss
        
        if self.smoothness_weight > 0:
            smooth_loss = self.spectral_smoothness_loss(x_out)
            losses['smoothness'] = smooth_loss.item()
            total = total + self.smoothness_weight * smooth_loss
        
        if self.continuity_weight > 0:
            cont_loss = self.temporal_continuity_loss(x_in)
            losses['continuity'] = cont_loss.item()
            total = total + self.continuity_weight * cont_loss
        
        losses['total'] = total.item()
        return total, losses


class PINNBlock(nn.Module):
    """
    Physics-informed neural network block.
    
    Standard MLP with physics-based activation functions and 
    residual connections to preserve physical properties.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "sin"
    ):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        
        # Physics-inspired activations
        if activation == "sin":
            self.activation = torch.sin
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "swish":
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            self.activation = F.gelu
        
        # Residual projection if dimensions don't match
        self.residual = (
            nn.Linear(in_features, out_features) 
            if in_features != out_features 
            else nn.Identity()
        )
        
        # Initialize with small weights for stability
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        return x + residual


class PINNDetector(BaseDetector):
    """
    Physics-Informed Neural Network detector for deepfake audio.
    
    Architecture:
    1. Initial spectral embedding
    2. Stack of PINN blocks with physics-inspired activations
    3. Physics regularization during training
    4. Classification head
    
    The physics constraints encourage the network to learn
    representations that respect acoustic signal properties,
    potentially improving robustness to adversarial attacks.
    """
    
    def __init__(
        self,
        n_bins: int = 84,
        hidden_dims: list = [256, 128, 64],
        activation: str = "sin",
        energy_weight: float = 0.1,
        smoothness_weight: float = 0.1,
        continuity_weight: float = 0.1,
        dropout: float = 0.1,
        pool_type: str = "mean",
        use_spectral_norm: bool = True,
        num_classes: int = 2,
        **kwargs
    ):
        """
        Args:
            n_bins: Number of CQT frequency bins
            hidden_dims: Hidden layer dimensions
            activation: Activation function ("sin", "tanh", "swish", "gelu")
            energy_weight: Weight for energy conservation loss
            smoothness_weight: Weight for spectral smoothness loss
            continuity_weight: Weight for temporal continuity loss
            dropout: Dropout probability
            pool_type: Temporal pooling type
            use_spectral_norm: Whether to use spectral normalization (Lipschitz constraint)
            num_classes: Number of output classes
        """
        super().__init__(
            num_classes=num_classes,
            n_bins=n_bins,
            hidden_dims=hidden_dims,
            activation=activation,
            energy_weight=energy_weight,
            smoothness_weight=smoothness_weight,
            continuity_weight=continuity_weight,
            dropout=dropout,
            pool_type=pool_type,
            use_spectral_norm=use_spectral_norm
        )
        
        self.n_bins = n_bins
        self.pool_type = pool_type
        
        # Physics constraints module
        self.physics = PhysicsConstraints(
            energy_weight, smoothness_weight, continuity_weight
        )
        
        # Input dimension after pooling
        input_dim = n_bins * 2 if pool_type == "both" else n_bins
        
        # Build PINN layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            block = PINNBlock(prev_dim, hidden_dim, activation)
            if use_spectral_norm:
                # Apply spectral norm to linear layer for Lipschitz constraint
                block.linear = nn.utils.spectral_norm(block.linear)
            layers.append(block)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.pinn_layers = nn.Sequential(*layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
        # Store intermediate activations for physics loss
        self._intermediate = None
    
    def _pool_time(self, x: torch.Tensor) -> torch.Tensor:
        """Pool over time dimension."""
        if self.pool_type == "mean":
            return x.mean(dim=-1)
        elif self.pool_type == "max":
            return x.max(dim=-1)[0]
        else:
            return torch.cat([x.mean(dim=-1), x.max(dim=-1)[0]], dim=-1)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_physics_loss: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input CQT features of shape (batch, n_bins, time_frames)
            return_physics_loss: If True, also return physics regularization loss
            
        Returns:
            Logits of shape (batch, num_classes)
            Optionally: (logits, physics_loss, physics_dict)
        """
        # Store input for physics loss
        x_input = x
        
        # Pool over time
        x = self._pool_time(x)
        x_pooled = x
        
        # Pass through PINN layers
        x = self.pinn_layers(x)
        
        # Store intermediate for physics loss computation
        self._intermediate = x
        
        # Classification
        logits = self.classifier(x)
        
        if return_physics_loss:
            physics_loss, physics_dict = self.physics(x_pooled, x)
            return logits, physics_loss, physics_dict
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        x = self._pool_time(x)
        return self.pinn_layers(x)
    
    def training_step(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor,
        criterion: nn.Module
    ) -> Tuple[torch.Tensor, dict]:
        """
        Custom training step that includes physics loss.
        
        Args:
            x: Input batch
            y: Target labels
            criterion: Classification loss function (e.g., CrossEntropyLoss)
            
        Returns:
            Total loss, dictionary of loss components
        """
        logits, physics_loss, physics_dict = self.forward(x, return_physics_loss=True)
        
        # Classification loss
        cls_loss = criterion(logits, y)
        
        # Total loss
        total_loss = cls_loss + physics_loss
        
        return total_loss, {
            'classification': cls_loss.item(),
            'physics': physics_loss.item(),
            **physics_dict
        }


if __name__ == "__main__":
    # Test the model
    model = PINNDetector(
        n_bins=84,
        hidden_dims=[256, 128, 64],
        activation="sin"
    )
    
    # Dummy input (batch=2, n_bins=84, time=100)
    x = torch.randn(2, 84, 100)
    
    # Forward pass
    output = model(x)
    output_with_physics, physics_loss, physics_dict = model(x, return_physics_loss=True)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Physics loss: {physics_loss.item():.6f}")
    print(f"Physics dict: {physics_dict}")
    print(f"Config: {model.get_config()}")
