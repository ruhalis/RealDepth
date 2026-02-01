"""Simple depth comparison plot"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from realdepth.predictor import setup_device, load_checkpoint, preprocess_rgb_image, predict_depth

# Hardcoded paths
CHECKPOINT = "experiments/realsense_v10/checkpoints/best.pth"
RGB_PATH = "collected_dataset/20251218_150958/rgb/003360.png"
DEPTH_PATH = "collected_dataset/20251218_150958/depth/003360.png"


def main():
    # Setup
    device = setup_device('auto')
    model, config = load_checkpoint(CHECKPOINT, device)

    # Load RGB image
    image_bgr = cv2.imread(RGB_PATH)

    # Load ground truth depth (mm -> m)
    depth_gt = np.array(Image.open(DEPTH_PATH), dtype=np.float32) / 1000.0

    # Run inference
    rgb_tensor = preprocess_rgb_image(image_bgr, config['image_size'])
    depth_pred = predict_depth(model, rgb_tensor, device)

    # Resize GT to match prediction size
    depth_gt_resized = cv2.resize(depth_gt, (depth_pred.shape[1], depth_pred.shape[0]))

    # Apply colormaps with dynamic range
    norm_gt = plt.Normalize(vmin=depth_gt_resized.min(), vmax=depth_gt_resized.max())
    norm_pred = plt.Normalize(vmin=depth_pred.min(), vmax=depth_pred.max())
    cmap = plt.get_cmap('jet')

    gt_colored = (cmap(norm_gt(depth_gt_resized))[:, :, :3] * 255).astype(np.uint8)
    pred_colored = (cmap(norm_pred(depth_pred))[:, :, :3] * 255).astype(np.uint8)

    # Concatenate horizontally
    combined = np.hstack((gt_colored, pred_colored))

    # Show without any decorations
    h, w = combined.shape[:2]
    fig = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(combined)
    ax.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
