"""
Single Image Depth Inference

Usage:
    python infer_image.py <image_path> --checkpoint <checkpoint.pth> [options]
"""

import sys
import argparse
from pathlib import Path
import cv2

from realdepth.predictor import setup_device, load_checkpoint, preprocess_rgb_image, predict_depth
from realdepth.visualization import save_depth_outputs, display_image_result


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
    rgb_tensor = preprocess_rgb_image(image_bgr, config['image_size'])

    # Inference
    depth = predict_depth(model, rgb_tensor, device)
    print(f"Depth range: {depth.min():.2f}m - {depth.max():.2f}m")

    # Save outputs
    image_name = Path(args.image).stem
    save_depth_outputs(image_bgr, depth, args.output_dir, image_name, config['max_depth'])

    # Display (optional)
    if not args.no_display:
        display_image_result(image_bgr, depth, config['max_depth'])

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
