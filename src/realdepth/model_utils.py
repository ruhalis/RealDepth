"""
Model utility functions for RealDepth training
"""

from .model import DepthEstimationNet
from .losses import CombinedDepthLoss


def get_model(model_name, max_depth=10.0):
    """
    Factory function to create model

    Args:
        model_name: str - Model name ('realdepth_resnet' or 'realdepth')
        max_depth: float - Maximum depth range in meters

    Returns:
        model: DepthEstimationNet instance
    """
    if model_name in ['realdepth_resnet', 'realdepth']:
        return DepthEstimationNet(base_channels=32, max_depth=max_depth)
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'realdepth_resnet' or 'realdepth'")


def count_params(model):
    """
    Count trainable parameters in model

    Args:
        model: PyTorch model

    Returns:
        int - Number of trainable parameters
    """
    return model.count_parameters()


# Alias for backward compatibility with train.py
DepthLoss = CombinedDepthLoss
