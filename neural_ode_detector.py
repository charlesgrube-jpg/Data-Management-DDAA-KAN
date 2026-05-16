"""
Neural ODE Deepfake Audio Detector

Uses continuous-depth transformations via ODE solvers for potentially
more robust feature learning.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from .base_detector import BaseDetector


class ODEFunc(nn.Module):
    """
    Neural network defining the ODE dynamics: dh/dt = f(h, t).
    
    The continuous transformation is defined by this function.
    """
    
    def __init__(self, hidden_dim: int, time_dependent: bool = True):
        super().__init__()
        
        self.time_dependent = time_dependent
        input_dim = hidden_dim + 1 if time_dependent else hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Track NFE (number of function evaluations) for debugging
        self.nfe = 0
    
    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute dh/dt.
        
        Args:
            t: Current time (scalar tensor)
            h: Current hidden state (batch, hidden_dim)
            
        Returns:
            Derivative dh/dt of same shape as h
        """
        self.nfe += 1
        
        if self.time_dependent:
            # Concatenate time to the input
            t_expanded = t.expand(h.shape[0], 1)
            h_t = torch.cat([h, t_expanded], dim=-1)
        else:
            h_t = h
        
        return self.net(h_t)
    
    def reset_nfe(self):
        self.nfe = 0


class NeuralODEBlock(nn.Module):
    """
    Neural ODE block that integrates the ODE dynamics.
    
    Uses torchdiffeq if available, otherwise implements a simple
    Euler/RK4 solver as fallback.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        t_span: tuple = (0.0, 1.0),
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        time_dependent: bool = True
    ):
        super().__init__()
        
        self.odefunc = ODEFunc(hidden_dim, time_dependent)
        self.t_span = t_span
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        
        # Check if torchdiffeq is available
        self._use_torchdiffeq = False
        try:
            from torchdiffeq import odeint
            self._odeint = odeint
            self._use_torchdiffeq = True
        except ImportError:
            print("[NeuralODE] torchdiffeq not available, using RK4 fallback")
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Integrate ODE from t0 to t1.
        
        Args:
            h: Initial state (batch, hidden_dim)
            
        Returns:
            Final state (batch, hidden_dim)
        """
        self.odefunc.reset_nfe()
        
        t = torch.tensor(
            [self.t_span[0], self.t_span[1]], 
            dtype=h.dtype, 
            device=h.device
        )
        
        if self._use_torchdiffeq:
            # Use adaptive solver from torchdiffeq
            solution = self._odeint(
                self.odefunc, 
                h, 
                t,
                method=self.solver,
                rtol=self.rtol,
                atol=self.atol
            )
            return solution[-1]  # Return final state
        else:
            # Fallback: simple RK4 solver
            return self._rk4_solve(h, t)
    
    def _rk4_solve(
        self, 
        h: torch.Tensor, 
        t: torch.Tensor, 
        num_steps: int = 10
    ) -> torch.Tensor:
        """Simple RK4 solver as fallback."""
        t0, t1 = t[0], t[1]
        dt = (t1 - t0) / num_steps
        
        for _ in range(num_steps):
            k1 = self.odefunc(t0, h)
            k2 = self.odefunc(t0 + dt/2, h + dt/2 * k1)
            k3 = self.odefunc(t0 + dt/2, h + dt/2 * k2)
            k4 = self.odefunc(t0 + dt, h + dt * k3)
            
            h = h + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            t0 = t0 + dt
        
        return h
    
    @property
    def nfe(self) -> int:
        """Number of function evaluations in last forward pass."""
        return self.odefunc.nfe


class NeuralODEDetector(BaseDetector):
    """
    Neural ODE-based detector for deepfake audio.
    
    Architecture:
    1. Initial embedding layer
    2. Neural ODE block (continuous transformation)
    3. Classification head
    
    Neural ODEs can provide smoother feature transformations and 
    potentially better robustness to adversarial perturbations.
    """
    
    def __init__(
        self,
        n_bins: int = 84,
        hidden_dim: int = 128,
        num_ode_blocks: int = 2,
        t_span: tuple = (0.0, 1.0),
        solver: str = "dopri5",
        dropout: float = 0.1,
        pool_type: str = "mean",
        num_classes: int = 2,
        **kwargs
    ):
        """
        Args:
            n_bins: Number of CQT frequency bins
            hidden_dim: Hidden dimension for ODE dynamics
            num_ode_blocks: Number of ODE blocks to stack
            t_span: Integration time span (t0, t1)
            solver: ODE solver ("dopri5", "rk4", "euler")
            dropout: Dropout probability
            pool_type: Temporal pooling type
            num_classes: Number of output classes
        """
        super().__init__(
            num_classes=num_classes,
            n_bins=n_bins,
            hidden_dim=hidden_dim,
            num_ode_blocks=num_ode_blocks,
            t_span=t_span,
            solver=solver,
            dropout=dropout,
            pool_type=pool_type
        )
        
        self.n_bins = n_bins
        self.pool_type = pool_type
        
        # Input dimension after pooling
        input_dim = n_bins * 2 if pool_type == "both" else n_bins
        
        # Initial embedding
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Stack of Neural ODE blocks
        self.ode_blocks = nn.ModuleList([
            NeuralODEBlock(hidden_dim, t_span, solver)
            for _ in range(num_ode_blocks)
        ])
        
        # Normalization after each ODE block
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_ode_blocks)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def _pool_time(self, x: torch.Tensor) -> torch.Tensor:
        """Pool over time dimension."""
        if self.pool_type == "mean":
            return x.mean(dim=-1)
        elif self.pool_type == "max":
            return x.max(dim=-1)[0]
        else:
            return torch.cat([x.mean(dim=-1), x.max(dim=-1)[0]], dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input CQT features of shape (batch, n_bins, time_frames)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Pool over time
        x = self._pool_time(x)
        
        # Initial embedding
        x = self.input_layer(x)
        
        # Pass through ODE blocks with residual connections
        for ode_block, norm in zip(self.ode_blocks, self.norms):
            x_ode = ode_block(x)
            x = norm(x + x_ode)  # Residual connection
            x = self.dropout(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        x = self._pool_time(x)
        x = self.input_layer(x)
        for ode_block, norm in zip(self.ode_blocks, self.norms):
            x_ode = ode_block(x)
            x = norm(x + x_ode)
        return x
    
    def get_total_nfe(self) -> int:
        """Get total number of function evaluations across all ODE blocks."""
        return sum(block.nfe for block in self.ode_blocks)


if __name__ == "__main__":
    # Test the model
    model = NeuralODEDetector(
        n_bins=84,
        hidden_dim=128,
        num_ode_blocks=2
    )
    
    # Dummy input (batch=2, n_bins=84, time=100)
    x = torch.randn(2, 84, 100)
    
    # Forward pass
    output = model(x)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total NFE: {model.get_total_nfe()}")
    print(f"Config: {model.get_config()}")
