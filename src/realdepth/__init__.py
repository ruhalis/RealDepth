"""RealDepth: Real-time 2D camera depth estimation."""

from .model import DepthEstimationNet
from .conv_gru import ConvGRU
from .losses import CombinedDepthLoss, DepthMetrics, TemporalConsistencyLoss
from .depth_datasets import RealSenseDataset, create_dataloaders
from .sequence_dataset import SequenceDataset, create_sequence_dataloaders
from .model_utils import get_model, count_params, DepthLoss
from .predictor import setup_device, load_checkpoint, preprocess_rgb_image, predict_depth, create_preprocessing_transform
from .visualization import visualize_depth, save_depth_outputs, display_image_result, create_live_display
from .validation import run_comprehensive_validation

__version__ = "0.4.0"

__all__ = [
    "DepthEstimationNet",
    "ConvGRU",
    "CombinedDepthLoss",
    "DepthMetrics",
    "TemporalConsistencyLoss",
    "RealSenseDataset",
    "SequenceDataset",
    "create_dataloaders",
    "create_sequence_dataloaders",
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
    "run_comprehensive_validation",
]
