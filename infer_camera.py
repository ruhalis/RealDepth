"""
RealDepth - Real-time Camera Depth Inference

Runs depth estimation on RealSense D435i camera live feed

Usage:
    python infer_camera.py --checkpoint <checkpoint.pth> [options]

Example:
    python infer_camera.py --checkpoint experiments/exp1/checkpoints/best.pth
    python infer_camera.py -c best.pth --fps 15 --resolution 640x480

Controls:
    ESC/Q: Quit
    S: Save snapshot
    SPACE: Pause/Resume
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import pyrealsense2 as rs

from model_utils import get_model


class RealSenseCamera:
    """
    RealSense D435i camera wrapper for inference
    """
    def __init__(self, resolution=(1280, 720), fps=30, apply_filters=True):
        """
        Initialize RealSense camera

        Args:
            resolution: (width, height) tuple
            fps: Frame rate (6, 15, or 30)
            apply_filters: Apply depth filters (spatial, temporal, hole-filling)
        """
        self.resolution = resolution
        self.fps = fps
        self.apply_filters = apply_filters

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure color stream
        self.config.enable_stream(
            rs.stream.color,
            resolution[0], resolution[1],
            rs.format.bgr8,
            fps
        )

        # Depth filters (optional)
        if self.apply_filters:
            self.spatial_filter = rs.spatial_filter()
            self.temporal_filter = rs.temporal_filter()
            self.hole_filling = rs.hole_filling_filter()

        # Alignment to color frame
        self.align = rs.align(rs.stream.color)

        self.is_running = False
        self.profile = None

    def start(self):
        """Start camera pipeline"""
        print("Starting RealSense camera...")
        self.profile = self.pipeline.start(self.config)

        # Get device info
        device = self.profile.get_device()
        device_name = device.get_info(rs.camera_info.name)
        print(f"Device: {device_name}")

        # Warm up (skip first 30 frames for auto-exposure)
        print("Warming up camera...")
        for _ in range(30):
            self.pipeline.wait_for_frames()

        self.is_running = True
        print("Camera ready!")

    def get_frame(self):
        """
        Get aligned color frame

        Returns:
            color_image: numpy array (H, W, 3) BGR
        """
        if not self.is_running:
            raise RuntimeError("Camera not started")

        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def stop(self):
        """Stop camera pipeline"""
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False
            print("Camera stopped")


class FPSTracker:
    """Track FPS using exponential moving average"""
    def __init__(self, window=30):
        """
        Initialize FPS tracker

        Args:
            window: Number of frames to average over
        """
        self.window = window
        self.timestamps = []
        self.fps = 0.0

    def update(self):
        """Add current timestamp"""
        current_time = time.time()
        self.timestamps.append(current_time)

        # Keep only recent timestamps
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)

        # Calculate FPS
        if len(self.timestamps) > 1:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            self.fps = (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0

    def get_fps(self):
        """Get current FPS"""
        return self.fps


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

    print(f"Loading checkpoint: {checkpoint_path}")
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

    print(f"Model: {model_name}, Max Depth: {max_depth}m\n")

    return model, config


def preprocess_frame(frame, transform):
    """
    Preprocess camera frame for inference

    Args:
        frame: numpy array (H, W, 3) BGR from camera
        transform: torchvision transforms

    Returns:
        tensor: (1, 3, H, W) normalized tensor
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    tensor = transform(frame_pil)
    return tensor.unsqueeze(0)


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


def create_display(color_image, depth, max_depth, fps, paused):
    """
    Create side-by-side display with RGB | Depth

    Args:
        color_image: numpy array (H, W, 3) BGR from camera
        depth: numpy array (H, W) depth in meters
        max_depth: Maximum depth for visualization
        fps: Current FPS
        paused: Whether inference is paused

    Returns:
        display: numpy array (H, W*2, 3) side-by-side visualization
    """
    # Colorize depth
    depth_color = visualize_depth(depth, max_depth)

    # Resize depth to match RGB
    h, w = color_image.shape[:2]
    depth_color_resized = cv2.resize(depth_color, (w, h))

    # Stack horizontally
    display = np.hstack([color_image, depth_color_resized])

    # Add overlays
    # FPS counter
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(display, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Labels
    cv2.putText(display, 'RGB', (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display, 'Depth Prediction', (w + 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Paused indicator
    if paused:
        cv2.putText(display, 'PAUSED', (display.shape[1]//2 - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    return display


def save_snapshot(color_image, depth, output_dir, count, max_depth):
    """
    Save current frame as snapshot

    Args:
        color_image: numpy array (H, W, 3) BGR from camera
        depth: numpy array (H, W) depth in meters
        output_dir: Output directory path
        count: Snapshot counter
        max_depth: Maximum depth for visualization

    Returns:
        Updated snapshot count
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename_prefix = f"snapshot_{count:04d}_{timestamp}"

    # Save RGB
    cv2.imwrite(str(output_dir / f'{filename_prefix}_rgb.png'), color_image)

    # Save depth (16-bit, millimeters)
    depth_mm = (depth * 1000).astype(np.uint16)
    cv2.imwrite(str(output_dir / f'{filename_prefix}_depth_gray.png'), depth_mm)

    # Save colorized depth
    depth_color = visualize_depth(depth, max_depth)
    h, w = color_image.shape[:2]
    depth_color_resized = cv2.resize(depth_color, (w, h))
    cv2.imwrite(str(output_dir / f'{filename_prefix}_depth_color.png'), depth_color_resized)

    # Save comparison
    comparison = np.hstack([color_image, depth_color_resized])
    cv2.imwrite(str(output_dir / f'{filename_prefix}_comparison.png'), comparison)

    print(f"Snapshot {count} saved: {filename_prefix}")
    return count + 1


def run_inference_loop(camera, model, config, device, args):
    """
    Main real-time inference loop

    Args:
        camera: RealSenseCamera instance
        model: DepthEstimationNet instance
        config: Configuration dict from checkpoint
        device: torch.device
        args: Command-line arguments
    """
    max_depth = config['max_depth']
    target_size = tuple(config['image_size'])

    # Prepare preprocessing transform
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # FPS tracking
    fps_tracker = FPSTracker()
    snapshot_count = 0

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create display window
    cv2.namedWindow('RealDepth - Live Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('RealDepth - Live Camera', 1600, 600)

    print("\n" + "="*60)
    print("Real-time Depth Inference")
    print("="*60)
    print("Controls:")
    print("  ESC/Q: Quit")
    print("  S: Save snapshot")
    print("  SPACE: Pause/Resume")
    print("="*60 + "\n")

    paused = False
    color_image = None
    depth = None

    try:
        while True:
            if not paused:
                # Get camera frame
                color_image = camera.get_frame()
                if color_image is None:
                    continue

                # Preprocess
                rgb_tensor = preprocess_frame(color_image, transform)

                # Inference
                depth = predict_depth(model, rgb_tensor, device)

                # Update FPS
                fps_tracker.update()

            # Visualize (even when paused, show last frame)
            if color_image is not None and depth is not None:
                display = create_display(
                    color_image,
                    depth,
                    max_depth,
                    fps_tracker.get_fps(),
                    paused
                )

                cv2.imshow('RealDepth - Live Camera', display)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):  # ESC or Q
                print("\nQuitting...")
                break
            elif key == ord('s'):  # Save snapshot
                if color_image is not None and depth is not None:
                    snapshot_count = save_snapshot(
                        color_image, depth, output_dir,
                        snapshot_count, max_depth
                    )
            elif key == ord(' '):  # Pause/Resume
                paused = not paused
                print("Paused" if paused else "Resumed")

    finally:
        cv2.destroyAllWindows()
        print(f"\nTotal snapshots saved: {snapshot_count}")


def parse_resolution(res_string):
    """
    Parse resolution string like '1280x720' to (width, height)

    Args:
        res_string: Resolution string (e.g., '1280x720')

    Returns:
        (width, height) tuple
    """
    try:
        width, height = map(int, res_string.lower().split('x'))
        return (width, height)
    except:
        raise ValueError(f"Invalid resolution format: {res_string}. Use format: WIDTHxHEIGHT (e.g., 1280x720)")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='RealDepth - Real-time Camera Depth Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer_camera.py --checkpoint best.pth
  python infer_camera.py -c best.pth --fps 15 --resolution 640x480
  python infer_camera.py -c best.pth --device cpu --no-filters

Controls:
  ESC/Q: Quit
  S: Save snapshot
  SPACE: Pause/Resume
        """
    )

    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'], default='auto',
                       help='Device: cuda/cpu/auto (default: auto)')
    parser.add_argument('--fps', type=int, default=30, choices=[6, 15, 30],
                       help='Camera frame rate (default: 30)')
    parser.add_argument('--resolution', type=str, default='1280x720',
                       help='Camera resolution (default: 1280x720)')
    parser.add_argument('--output-dir', '-o', type=str, default='./snapshots',
                       help='Directory for saved snapshots (default: ./snapshots)')
    parser.add_argument('--no-filters', action='store_true',
                       help='Disable RealSense depth filters')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Device selection
    device = setup_device(args.device)

    # Load checkpoint and model
    model, config = load_checkpoint(args.checkpoint, device)

    # Parse resolution
    resolution = parse_resolution(args.resolution)

    # Initialize RealSense camera
    camera = RealSenseCamera(
        resolution=resolution,
        fps=args.fps,
        apply_filters=not args.no_filters
    )

    try:
        camera.start()
        run_inference_loop(camera, model, config, device, args)
    finally:
        camera.stop()

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
        if 'RealSense' in str(e) or 'rs' in str(e):
            print(f"\nCamera Error: {e}")
            print("Make sure RealSense D435i is connected")
        elif 'CUDA' in str(e):
            print(f"\nCUDA Error: {e}")
            print("Try running with --device cpu")
        else:
            print(f"\nRuntime Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
