"""
Custom metrics for ConvLEM wildfire prediction.
"""

from __future__ import annotations

import torch
from pyhazards.metrics import MetricBase


class WildfireAccuracy(MetricBase):
    """
    Binary accuracy for wildfire predictions.
    
    Computes accuracy after applying sigmoid threshold (default: 0.5).
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.name = "accuracy"
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metric with batch predictions.
        
        Args:
            outputs: Model logits (batch, num_counties)
            targets: Ground truth labels (batch, num_counties)
        """
        probs = torch.sigmoid(outputs)
        preds = (probs > self.threshold).float()
        
        correct = (preds == targets).float().sum()
        total = targets.numel()
        
        self.total += correct.item()
        self.count += total
    
    def compute(self) -> float:
        """Compute final accuracy."""
        if self.count == 0:
            return 0.0
        return self.total / self.count
    
    def reset(self) -> None:
        """Reset metric state."""
        self.total = 0.0
        self.count = 0


class WildfirePrecision(MetricBase):
    """
    Precision for wildfire predictions.
    
    Precision = TP / (TP + FP)
    Measures: Of all predicted fires, how many were actual fires?
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.name = "precision"
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        probs = torch.sigmoid(outputs)
        preds = (probs > self.threshold).float()
        
        # True positives: predicted fire AND actual fire
        tp = ((preds == 1) & (targets == 1)).float().sum()
        # False positives: predicted fire BUT no actual fire
        fp = ((preds == 1) & (targets == 0)).float().sum()
        
        self.tp += tp.item()
        self.fp += fp.item()
    
    def compute(self) -> float:
        if (self.tp + self.fp) == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)
    
    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0


class WildfireRecall(MetricBase):
    """
    Recall for wildfire predictions.
    
    Recall = TP / (TP + FN)
    Measures: Of all actual fires, how many did we catch?
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.name = "recall"
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        probs = torch.sigmoid(outputs)
        preds = (probs > self.threshold).float()
        
        # True positives: predicted fire AND actual fire
        tp = ((preds == 1) & (targets == 1)).float().sum()
        # False negatives: predicted no fire BUT actual fire
        fn = ((preds == 0) & (targets == 1)).float().sum()
        
        self.tp += tp.item()
        self.fn += fn.item()
    
    def compute(self) -> float:
        if (self.tp + self.fn) == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)
    
    def reset(self) -> None:
        self.tp = 0.0
        self.fn = 0.0


class WildfireF1Score(MetricBase):
    """
    F1 Score for wildfire predictions.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    Harmonic mean of precision and recall.
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.name = "f1_score"
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        probs = torch.sigmoid(outputs)
        preds = (probs > self.threshold).float()
        
        tp = ((preds == 1) & (targets == 1)).float().sum()
        fp = ((preds == 1) & (targets == 0)).float().sum()
        fn = ((preds == 0) & (targets == 1)).float().sum()
        
        self.tp += tp.item()
        self.fp += fp.item()
        self.fn += fn.item()
    
    def compute(self) -> float:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        
        if (precision + recall) == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0


class CountyRiskCoverage(MetricBase):
    """
    Percentage of at-risk counties correctly identified.
    
    Useful for understanding spatial coverage of predictions.
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.name = "county_coverage"
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        probs = torch.sigmoid(outputs)
        preds = (probs > self.threshold).float()
        
        # Average across batch dimension to get per-county statistics
        pred_risk = preds.mean(dim=0)  # (num_counties,)
        true_risk = targets.mean(dim=0)  # (num_counties,)
        
        # Counties with any risk in ground truth
        at_risk_counties = (true_risk > 0).float()
        # Counties correctly identified as at-risk
        caught_counties = ((pred_risk > 0) & (true_risk > 0)).float()
        
        self.caught += caught_counties.sum().item()
        self.total_at_risk += at_risk_counties.sum().item()
    
    def compute(self) -> float:
        if self.total_at_risk == 0:
            return 0.0
        return self.caught / self.total_at_risk
    
    def reset(self) -> None:
        self.caught = 0.0
        self.total_at_risk = 0.0


__all__ = [
    "WildfireAccuracy",
    "WildfirePrecision",
    "WildfireRecall",
    "WildfireF1Score",
    "CountyRiskCoverage",
]