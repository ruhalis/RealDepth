"""RealDepth: Real-time 2D camera depth estimation."""

from .model import DepthEstimationNet
from .losses import CombinedDepthLoss, DepthMetrics
from .depth_datasets import RealSenseDataset, create_dataloaders
from .model_utils import get_model, count_params, DepthLoss
from .predictor import setup_device, load_checkpoint, preprocess_rgb_image, predict_depth, create_preprocessing_transform
from .visualization import visualize_depth, save_depth_outputs, display_image_result, create_live_display

__version__ = "0.1.0"

__all__ = [
    "DepthEstimationNet",
    "CombinedDepthLoss",
    "DepthMetrics",
    "RealSenseDataset",
    "create_dataloaders",
    "get_model",
    "count_params",
    "DepthLoss",
    "setup_device",
    "load_checkpoint",
    "preprocess_rgb_image",
    "predict_depth",
    "create_preprocessing_transform",
    "visualize_depth",
    "save_depth_outputs",
    "display_image_result",
    "create_live_display",
]
