"""
RealDepth - Single Image Depth Inference

Usage:
    python infer_image.py <image_path> --checkpoint <checkpoint.pth> [options]

Example:
    python infer_image.py test.jpg --checkpoint experiments/exp1/checkpoints/best.pth
    python infer_image.py image.png -c best.pth -o ./results --device cuda
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image

from model_utils import get_model


def setup_device(device_arg):
    """
    Select device: cuda/cpu/auto

    Args:
        device_arg: 'cuda', 'cpu', or 'auto'

    Returns:
        torch.device
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

    Args:
        checkpoint_path: Path to checkpoint file
        device: torch.device

    Returns:
        model: DepthEstimationNet instance
        config: Configuration dict from checkpoint
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

    # Create model
    model = get_model(model_name, max_depth=max_depth)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    print(f"Model: {model_name}, Max Depth: {max_depth}m")

    return model, config


def preprocess_image(image_bgr, target_size):
    """
    Preprocess image following training pipeline

    Args:
        image_bgr: numpy array (H, W, 3) BGR from OpenCV
        target_size: [height, width] from config

    Returns:
        tensor: (1, 3, H, W) normalized tensor
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Convert to PIL for transforms
    image_pil = Image.fromarray(image_rgb)

    # Apply same transforms as training (depth_datasets.py lines 36-40)
    transform = T.Compose([
        T.Resize(tuple(target_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(image_pil)
    return tensor.unsqueeze(0)  # Add batch dimension


@torch.no_grad()
def predict_depth(model, rgb_tensor, device):
    """
    Run inference

    Args:
        model: DepthEstimationNet instance
        rgb_tensor: (1, 3, H, W) preprocessed tensor
        device: torch.device

    Returns:
        depth: numpy array (H, W) in meters
    """
    rgb_tensor = rgb_tensor.to(device)
    depth_tensor = model(rgb_tensor)  # (1, 1, H, W)
    depth_np = depth_tensor.squeeze().cpu().numpy()  # (H, W)
    return depth_np


def visualize_depth(depth, max_depth):
    """
    Convert depth map to colorized visualization

    Args:
        depth: numpy array (H, W) in meters
        max_depth: Maximum depth for normalization

    Returns:
        depth_color: numpy array (H, W, 3) BGR colorized with JET
    """
    # Normalize to 0-255
    depth_norm = np.clip(depth / max_depth, 0, 1) * 255
    depth_uint8 = depth_norm.astype(np.uint8)

    # Apply JET colormap
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

    return depth_color


def save_outputs(rgb_image, depth, output_dir, image_path, max_depth):
    """
    Save depth predictions to disk

    Saves:
        - {image_name}_depth_gray.png: Grayscale depth (16-bit PNG, millimeters)
        - {image_name}_depth_color.png: JET colorized depth
        - {image_name}_comparison.png: Side-by-side RGB | Depth visualization

    Args:
        rgb_image: numpy array (H, W, 3) BGR original image
        depth: numpy array (H, W) depth in meters
        output_dir: Output directory path
        image_path: Original image path (for filename)
        max_depth: Maximum depth for visualization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename prefix from input image
    image_name = Path(image_path).stem

    # 1. Save grayscale depth as 16-bit PNG (in millimeters)
    depth_mm = (depth * 1000).astype(np.uint16)
    gray_path = output_dir / f'{image_name}_depth_gray.png'
    cv2.imwrite(str(gray_path), depth_mm)

    # 2. Save colorized depth
    depth_color = visualize_depth(depth, max_depth)
    color_path = output_dir / f'{image_name}_depth_color.png'
    cv2.imwrite(str(color_path), depth_color)

    # 3. Save comparison (RGB | Depth side-by-side)
    # Resize depth to match RGB
    h, w = rgb_image.shape[:2]
    depth_color_resized = cv2.resize(depth_color, (w, h))
    comparison = np.hstack([rgb_image, depth_color_resized])
    comparison_path = output_dir / f'{image_name}_comparison.png'
    cv2.imwrite(str(comparison_path), comparison)

    print(f"\nOutputs saved to {output_dir}:")
    print(f"  - {image_name}_depth_gray.png (16-bit, millimeters)")
    print(f"  - {image_name}_depth_color.png (colorized)")
    print(f"  - {image_name}_comparison.png (side-by-side)")


def display_results(rgb_image, depth, max_depth):
    """
    Display results in OpenCV window

    Args:
        rgb_image: numpy array (H, W, 3) BGR original image
        depth: numpy array (H, W) depth in meters
        max_depth: Maximum depth for visualization
    """
    depth_color = visualize_depth(depth, max_depth)

    # Resize for consistent display
    h, w = rgb_image.shape[:2]
    depth_color_resized = cv2.resize(depth_color, (w, h))

    # Side-by-side display
    display = np.hstack([rgb_image, depth_color_resized])

    # Add text labels
    cv2.putText(display, 'RGB Input', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(display, 'Depth Prediction', (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display
    cv2.namedWindow('RealDepth - Image Inference', cv2.WINDOW_NORMAL)
    cv2.imshow('RealDepth - Image Inference', display)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='RealDepth - Single Image Depth Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer_image.py test.jpg --checkpoint best.pth
  python infer_image.py image.png -c best.pth -o ./results
  python infer_image.py photo.jpg -c best.pth --no-display --device cpu
        """
    )

    parser.add_argument('image', type=str,
                       help='Path to input RGB image')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--output-dir', '-o', type=str, default='./inference_output',
                       help='Output directory for depth maps (default: ./inference_output)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'], default='auto',
                       help='Device: cuda/cpu/auto (default: auto)')
    parser.add_argument('--no-display', action='store_true',
                       help='Skip interactive display')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Device selection
    device = setup_device(args.device)

    # Load checkpoint and model
    model, config = load_checkpoint(args.checkpoint, device)

    # Load image
    print(f"\nLoading image: {args.image}")
    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {args.image}")

    print(f"Image size: {image_bgr.shape[1]}x{image_bgr.shape[0]}")

    # Preprocess
    print("\nRunning inference...")
    rgb_tensor = preprocess_image(image_bgr, config['image_size'])

    # Inference
    depth = predict_depth(model, rgb_tensor, device)
    print(f"Depth range: {depth.min():.2f}m - {depth.max():.2f}m")

    # Save outputs
    save_outputs(image_bgr, depth, args.output_dir, args.image, config['max_depth'])

    # Display (optional)
    if not args.no_display:
        display_results(image_bgr, depth, config['max_depth'])

    print("\nDone!")


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except RuntimeError as e:
        if 'CUDA' in str(e):
            print(f"\nCUDA Error: {e}")
            print("Try running with --device cpu")
        else:
            print(f"\nRuntime Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
