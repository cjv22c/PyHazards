from typing import Optional
import torch
import torch.nn as nn
class ConvLEMWildfire(nn.Module):
    """
    ConvLEM-based wildfire prediction model (minimal version).
    """
    
    def __init__(
        self,
        in_dim: int,
        num_counties: int,
        past_days: int,
        hidden_dim: int = 144,
        num_layers: int = 2,
        dt: float = 1.0,
        activation: str = 'tanh',
        use_reset_gate: bool = False,
        dropout: float = 0.1,
        adjacency: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.num_counties = num_counties
        self.past_days = past_days
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Placeholder: just a simple linear layer for now
        self.placeholder = nn.Linear(in_dim, 1)
    
    def forward(self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Minimal forward pass.
        
        Args:
            x: (batch, past_days, num_counties, in_dim)
            adjacency: Optional adjacency matrix
            
        Returns:
            logits: (batch, num_counties)
        """
        B, T, N, F = x.shape
        
        # Validate shapes
        if T != self.past_days:
            raise ValueError(f"Expected past_days={self.past_days}, got {T}")
        if N != self.num_counties:
            raise ValueError(f"Expected num_counties={self.num_counties}, got {N}")
        if F != self.in_dim:
            raise ValueError(f"Expected in_dim={self.in_dim}, got {F}")
        
        # Placeholder forward: just use last timestep
        last_step = x[:, -1, :, :]  # (B, N, F)
        logits = self.placeholder(last_step).squeeze(-1)  # (B, N)
        
        return logits


def convlem_wildfire_builder(
    task: str,
    in_dim: int,
    num_counties: int,
    past_days: int,
    **kwargs,
) -> ConvLEMWildfire:
    """Builder function for ConvLEM wildfire model."""
    
    if task.lower() not in {"classification", "binary_classification"}:
        raise ValueError(
            f"ConvLEM wildfire model is classification-only, got task='{task}'"
        )
    
    return ConvLEMWildfire(
        in_dim=in_dim,
        num_counties=num_counties,
        past_days=past_days,
        hidden_dim=kwargs.get("hidden_dim", 144),
        num_layers=kwargs.get("num_layers", 2),
        dt=kwargs.get("dt", 1.0),
        activation=kwargs.get("activation", "tanh"),
        use_reset_gate=kwargs.get("use_reset_gate", False),
        dropout=kwargs.get("dropout", 0.1),
        adjacency=kwargs.get("adjacency"),
    )


__all__ = ["ConvLEMWildfire", "convlem_wildfire_builder"]

"""
To test: 
from pyhazards.models import build_model

model = build_model("convlem_wildfire",task="classification",in_dim=12,num_counties=58,past_days=8)
import torch
x = torch.randn(4, 8, 58, 12)
logits = model(x)
print(f"Output shape: {logits.shape}")  # Should be (4, 58)
"""