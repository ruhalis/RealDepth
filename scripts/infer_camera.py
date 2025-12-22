"""
RealDepth - Real-time Camera Depth Inference

Runs depth estimation on RealSense D435i camera live feed

Usage:
    python infer_camera.py --checkpoint <checkpoint.pth> [options]

"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
# import pyrealsense2 as rs

from realdepth.predictor import setup_device, load_checkpoint, preprocess_rgb_image, predict_depth, create_preprocessing_transform
from realdepth.visualization import save_depth_outputs, create_live_display


class WebcamCamera:
    """
    Standard Webcam wrapper for inference
    """
    def __init__(self, camera_id=0, resolution=(1280, 720), fps=30):
        """
        Initialize Webcam
        
        Args:
            camera_id: Camera device definition (default: 0)
            resolution: (width, height) tuple
            fps: Desired FPS (may not be respected by all webcams)
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.is_running = False

    def start(self):
        """Start camera"""
        print(f"Starting Webcam (ID: {self.camera_id})...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        # Set properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not self.cap.isOpened():
             raise RuntimeError(f"Could not open webcam {self.camera_id}")
             
        # Warm up
        print("Warming up camera...")
        for _ in range(5):
            self.cap.read()
        
        self.is_running = True
        print("Webcam ready!")

    def get_frame(self):
        """Get latest frame"""
        if not self.is_running or self.cap is None:
             raise RuntimeError("Webcam not started")
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def stop(self):
        """Stop camera"""
        if self.cap:
            self.cap.release()
        self.is_running = False
        print("Webcam stopped")


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
    image_size = config['image_size']

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
                rgb_tensor = preprocess_rgb_image(color_image, image_size)

                # Inference
                depth = predict_depth(model, rgb_tensor, device)

                # Update FPS
                fps_tracker.update()

            # Visualize (even when paused, show last frame)
            if color_image is not None and depth is not None:
                display = create_live_display(
                    color_image,
                    depth,
                    max_depth,
                    fps=fps_tracker.get_fps(),
                    paused=paused,
                    use_dynamic_range=not args.fixed_range
                )

                cv2.imshow('RealDepth - Live Camera', display)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):  # ESC or Q
                print("\nQuitting...")
                break
            elif key == ord('s'):  # Save snapshot
                if color_image is not None and depth is not None:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename_prefix = f"snapshot_{snapshot_count:04d}_{timestamp}"
                    save_depth_outputs(
                        color_image, depth, output_dir,
                        filename_prefix, max_depth
                    )
                    print(f"Snapshot {snapshot_count} saved: {filename_prefix}")
                    snapshot_count += 1
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
    parser.add_argument('--camera-type', type=str, default='realsense', choices=['realsense', 'webcam'],
                       help='Camera type to use (default: realsense)')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Webcam ID (default: 0)')
    parser.add_argument('--no-filters', action='store_true',
                       help='Disable RealSense depth filters')
    parser.add_argument('--fixed-range', action='store_true',
                       help='Use fixed depth range (0-max_depth) instead of dynamic per-frame scaling')

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

    # Initialize Camera
    if args.camera_type == 'realsense':
        camera = RealSenseCamera(
            resolution=resolution,
            fps=args.fps,
            apply_filters=not args.no_filters
        )
    else:
        camera = WebcamCamera(
            camera_id=args.camera_id,
            resolution=resolution,
            fps=args.fps
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
