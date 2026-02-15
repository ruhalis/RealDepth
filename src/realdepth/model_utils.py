"""
Model utility functions for RealDepth training
"""

from .model import DepthEstimationNet
from .losses import CombinedDepthLoss


def get_model(model_name, max_depth=10.0, **kwargs):
    """
    Factory function to create model

    Args:
        model_name: str - Model name
        max_depth: float - Maximum depth range in meters
        **kwargs: Additional model-specific arguments

    Returns:
        model: Model instance
    """
    if model_name == 'realdepth':
        pretrained = kwargs.get('pretrained_encoder', True)
        return DepthEstimationNet(max_depth=max_depth, pretrained_encoder=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: 'realdepth'")


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
