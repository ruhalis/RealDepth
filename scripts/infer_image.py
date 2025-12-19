"""
Single Image Depth Inference

Usage:
    python infer_image.py <image_path> --checkpoint <checkpoint.pth> [options]
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from realdepth.predictor import setup_device, load_checkpoint, preprocess_rgb_image, predict_depth

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='RealDepth - Single Image Depth Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('image', type=str,
                       help='Path to input RGB image')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint (.pth)')

    return parser.parse_args()


def find_linked_depth(rgb_path):
    """
    Find the corresponding depth file from RGB path.
    """
    rgb_path = Path(rgb_path)
    path_str = str(rgb_path)
    
    if '/rgb/' in path_str:
        depth_path = Path(path_str.replace('/rgb/', '/depth/'))
    else:
        depth_path = Path(path_str.replace('\\rgb\\', '\\depth\\'))

    return depth_path


def load_depth(depth_path):
    """
    Load depth in mm
    """
    depth_img = Image.open(depth_path)
    depth_mm = np.array(depth_img, dtype=np.float32)

    return depth_mm

def convert_depth_to_mm(depth_m):
    """
    Convert depth in mm to meters
    """
    depth_mm = depth_m * 1000.0
    return depth_mm


def create_comparison_plot(rgb_image, depth_gt, depth_pred, max_depth, metrics, save_path=None):
    """
    Create a matplotlib plot with RGB | Ground Truth | Prediction.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    depth_pred = convert_depth_to_mm(depth_pred)
    
    # RGB (convert BGR to RGB)
    axes[0].imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('RGB Input', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Ground Truth Depth
    im_gt = axes[1].imshow(depth_gt, cmap='jet', vmin=0, vmax=max_depth)
    axes[1].set_title('Ground Truth Depth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Predicted Depth
    im_pred = axes[2].imshow(depth_pred, cmap='jet', vmin=0, vmax=max_depth)
    axes[2].set_title('Predicted Depth', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(im_pred, ax=axes, orientation='horizontal', 
                        fraction=0.05, pad=0.08, shrink=0.6)
    cbar.set_label('Depth (meters)', fontsize=12)
    
    # Add metrics text
    metrics_text = (f"MAE: {metrics['mae']:.4f}m ({metrics['mae']*1000:.1f}mm)  |  "
                   f"RMSE: {metrics['rmse']:.4f}m ({metrics['rmse']*1000:.1f}mm)  |  "
                   f"AbsRel: {metrics['abs_rel']:.4f}")
    fig.suptitle(metrics_text, fontsize=11, y=0.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def compute_metrics(pred, gt):
    """Compute error metrics between prediction and ground truth."""
    valid_mask = gt > 0
    if valid_mask.sum() == 0:
        return {'mae': 0, 'rmse': 0, 'abs_rel': 0}
    
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    
    mae = np.abs(pred_valid - gt_valid).mean()
    rmse = np.sqrt(((pred_valid - gt_valid) ** 2).mean())
    abs_rel = (np.abs(pred_valid - gt_valid) / gt_valid).mean()
    
    return {'mae': mae, 'rmse': rmse, 'abs_rel': abs_rel}


def main():
    """Main function"""
    args = parse_args()

    # Device selection
    device = setup_device(args.device)

    # Load checkpoint and model
    model, config = load_checkpoint(args.checkpoint, device)
    max_depth = config['max_depth']

    # Load RGB image
    print(f"\nLoading RGB image: {args.image}")
    rgb_path = Path(args.image)
    if not rgb_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    image_bgr = cv2.imread(str(rgb_path))
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {args.image}")
    print(f"Image size: {image_bgr.shape[1]}x{image_bgr.shape[0]}")

    # Auto-detect and load ground truth depth
    depth_path = find_linked_depth(rgb_path)
    print(f"Looking for depth: {depth_path}")
    
    if not depth_path.exists():
        raise FileNotFoundError(f"Ground truth depth not found: {depth_path}\n"
                               f"Expected structure: .../rgb/filename.png -> .../depth/filename.png")
    
    depth_gt, depth_gt_mm = load_depth(depth_path)
    print(f"Ground truth range: {depth_gt.min():.2f}m - {depth_gt.max():.2f}m "
          f"({depth_gt_mm.min():.0f}mm - {depth_gt_mm.max():.0f}mm)")

    # Run inference
    print("\nRunning inference...")
    rgb_tensor = preprocess_rgb_image(image_bgr, config['image_size'])
    depth_pred = predict_depth(model, rgb_tensor, device)
    
    depth_pred_mm = depth_pred * 1000
    print(f"Prediction range:   {depth_pred.min():.2f}m - {depth_pred.max():.2f}m "
          f"({depth_pred_mm.min():.0f}mm - {depth_pred_mm.max():.0f}mm)")

    # Resize GT to match prediction for metrics
    depth_gt_resized = cv2.resize(depth_gt, (depth_pred.shape[1], depth_pred.shape[0]))

    # Compute metrics
    metrics = compute_metrics(depth_pred, depth_gt_resized)
    print(f"\n--- Error Metrics ---")
    print(f"MAE:     {metrics['mae']:.4f}m ({metrics['mae']*1000:.1f}mm)")
    print(f"RMSE:    {metrics['rmse']:.4f}m ({metrics['rmse']*1000:.1f}mm)")
    print(f"AbsRel:  {metrics['abs_rel']:.4f}")

    # Create output directory and save path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{rgb_path.stem}_comparison.png" if args.save else None

    # Create comparison plot
    print("\nCreating comparison plot...")
    create_comparison_plot(image_bgr, depth_gt_resized, depth_pred, max_depth, metrics, save_path)

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
