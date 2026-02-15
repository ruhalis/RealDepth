"""Depth comparison plot for two images with RGB | GT | Predicted layout"""

import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from realdepth.predictor import setup_device, load_checkpoint, preprocess_rgb_image, predict_depth

# Hardcoded checkpoint path
CHECKPOINT = "experiments/realsense_v10/checkpoints/best.pth"


def process_image(rgb_path, depth_path, model, config, device):
    """Process a single image pair and return RGB, GT depth colored, and predicted depth colored."""
    # Load RGB image
    image_bgr = cv2.imread(rgb_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Load ground truth depth (mm -> m)
    depth_gt = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0

    # Run inference
    rgb_tensor = preprocess_rgb_image(image_bgr, config['image_size'])
    depth_pred = predict_depth(model, rgb_tensor, device)

    # Resize RGB and GT to match prediction size
    target_size = (depth_pred.shape[1], depth_pred.shape[0])
    image_rgb_resized = cv2.resize(image_rgb, target_size)
    depth_gt_resized = cv2.resize(depth_gt, target_size)

    # Apply colormaps with dynamic range
    norm_gt = plt.Normalize(vmin=depth_gt_resized.min(), vmax=depth_gt_resized.max())
    norm_pred = plt.Normalize(vmin=depth_pred.min(), vmax=depth_pred.max())
    cmap = plt.get_cmap('jet')

    gt_colored = (cmap(norm_gt(depth_gt_resized))[:, :, :3] * 255).astype(np.uint8)
    pred_colored = (cmap(norm_pred(depth_pred))[:, :, :3] * 255).astype(np.uint8)

    return image_rgb_resized, gt_colored, pred_colored


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <rgb_image1> <rgb_image2>")
        sys.exit(1)

    rgb_paths = [sys.argv[1], sys.argv[2]]
    # Derive depth paths from RGB paths (replace /rgb/ with /depth/)
    depth_paths = [p.replace('/rgb/', '/depth/') for p in rgb_paths]

    # Setup
    device = setup_device('auto')
    model, config = load_checkpoint(CHECKPOINT, device)

    # Process both image pairs
    rows = []
    for rgb_path, depth_path in zip(rgb_paths, depth_paths):
        rgb, gt, pred = process_image(rgb_path, depth_path, model, config, device)
        # Concatenate horizontally: RGB | GT | Predicted
        row = np.hstack((rgb, gt, pred))
        rows.append(row)

    # Stack rows vertically
    combined = np.vstack(rows)

    # Show without any decorations
    h, w = combined.shape[:2]
    fig = plt.figure(figsize=(w/100, h/100), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(combined)
    plt.show()


if __name__ == '__main__':
    main()
