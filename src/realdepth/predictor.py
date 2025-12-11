"""
Core functionality for depth prediction inference.
"""
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np

from .model_utils import get_model


def setup_device(device_arg):
    """
    Select device: cuda/cpu/auto
    """
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return device


def load_checkpoint(checkpoint_path, device):
    """
    Load model checkpoint and extract configuration
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'config' not in checkpoint or 'model' not in checkpoint:
        raise ValueError("Invalid checkpoint format. Expected keys: 'model', 'config'")

    config = checkpoint['config']
    model_name = config.get('model', 'realdepth_resnet')
    max_depth = config.get('max_depth', 10.0)

    model = get_model(model_name, max_depth=max_depth)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    print(f"Model: {model_name}, Max Depth: {max_depth}m")

    return model, config


def create_preprocessing_transform(image_size):
    """
    Create preprocessing transform pipeline for RGB images
    """
    return T.Compose([
        T.Resize(tuple(image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def preprocess_rgb_image(image_bgr, image_size):
    """
    Preprocess RGB image for inference
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Convert to PIL for transforms
    image_pil = Image.fromarray(image_rgb)

    # Apply preprocessing transforms
    transform = create_preprocessing_transform(image_size)
    tensor = transform(image_pil)

    # Add batch dimension
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict_depth(model, rgb_tensor, device):
    """
    Run depth prediction inference
    """
    rgb_tensor = rgb_tensor.to(device)
    depth_tensor = model(rgb_tensor)  # (1, 1, H, W)
    depth_np = depth_tensor.squeeze().cpu().numpy()  # (H, W)
    return depth_np
