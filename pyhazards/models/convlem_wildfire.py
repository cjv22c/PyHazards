from typing import Optional
import torch
import torch.nn as nn

def convlem_wildfire_builder(
    task: str,
    in_dim: int,
    num_counties: int,
    past_days: int,
    **kwargs,
) -> nn.Module:

    # Validate task
    if task.lower() not in {"classification", "binary_classification"}:
        raise ValueError(
            f"ConvLEM wildfire model is classification-only, got task='{task}'"
        )
    
    # Import here to avoid circular imports
    from .convlem_wildfire import ConvLEMWildfire
    
    # Build model with defaults merged with kwargs
    model = ConvLEMWildfire(
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
    
    return model
