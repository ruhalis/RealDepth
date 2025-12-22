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
    Convert depth from meters to millimeters
    """
    return depth_m * 1000.0


def convert_depth_to_m(depth_mm):
    """
    Convert depth from millimeters to meters
    """
    # RealSense depth is often in mm, divide by 1000 to get meters
    return depth_mm / 1000.0


def create_comparison_plot(rgb_image, depth_gt, depth_pred, save_path=None):
    """
    Create a detailed plot with RGB | Ground Truth | Prediction.
    Concatenates images to avoid any whitespace.
    """
    depth_gt_mm = convert_depth_to_mm(depth_gt)
    depth_pred_mm = convert_depth_to_mm(depth_pred)
    
    # Calculate dynamic range for heatmap based on both images
    vmin = min(depth_gt_mm.min(), depth_pred_mm.min())
    vmax = max(depth_gt_mm.max(), depth_pred_mm.max())
    
    # Normalize and apply colormap
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('jet')
    
    # Apply colormap - returns (H, W, 4) float array
    # We take :3 to get RGB, ignoring alpha
    gt_colored = cmap(norm(depth_gt_mm))[:, :, :3]
    pred_colored = cmap(norm(depth_pred_mm))[:, :, :3]
    
    # Prepare RGB image (convert to float 0-1)
    rgb_colored = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Concatenate images horizontally
    combined_img = np.hstack((rgb_colored, gt_colored, pred_colored))
    
    # Create figure with tight layout calculation
    h, w, c = combined_img.shape
    fig_height = 5
    fig_width = fig_height * (w / h)
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    # Create an axes that covers the entire figure (0,0,1,1)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    
    ax.imshow(combined_img)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
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

    device = setup_device('auto')

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
    
    depth_gt_mm = load_depth(depth_path)
    depth_gt = convert_depth_to_m(depth_gt_mm)
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

    # Plot save path (currently disabled as --save was removed)
    save_path = None

    # Create comparison plot
    print("\nCreating comparison plot...")
    create_comparison_plot(image_bgr, depth_gt_resized, depth_pred, save_path)

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
