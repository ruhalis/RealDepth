"""
Model utility functions for RealDepth training
"""

from model import DepthEstimationNet
from losses import CombinedDepthLoss


def get_model(model_name, max_depth=10.0):
    """
    Factory function to create model variants

    Args:
        model_name: str - Model variant ('realdepth_resnet', 'realdepth', 'realdepth_lite')
        max_depth: float - Maximum depth range in meters

    Returns:
        model: DepthEstimationNet instance
    """
    if model_name in ['realdepth_resnet', 'realdepth']:
        return DepthEstimationNet(base_channels=64, max_depth=max_depth)
    elif model_name == 'realdepth_lite':
        return DepthEstimationNet(base_channels=32, max_depth=max_depth)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: realdepth_resnet, realdepth, realdepth_lite")


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
