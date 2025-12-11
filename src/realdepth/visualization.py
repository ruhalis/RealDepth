"""
Visualization utilities
"""
from pathlib import Path
import numpy as np
import cv2


def visualize_depth(depth, max_depth):
    """
    Convert depth map to colorized visualization
    """
    # Normalize to 0-255
    depth_norm = np.clip(depth / max_depth, 0, 1) * 255
    depth_uint8 = depth_norm.astype(np.uint8)

    # Apply JET colormap
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

    return depth_color


def save_depth_outputs(rgb_image, depth, output_dir, filename_prefix, max_depth):
    """
    Save depth predictions to disk
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save RGB image
    rgb_path = output_dir / f'{filename_prefix}_rgb.png'
    cv2.imwrite(str(rgb_path), rgb_image)

    # Save grayscale depth as 16-bit PNG (in millimeters)
    depth_mm = (depth * 1000).astype(np.uint16)
    gray_path = output_dir / f'{filename_prefix}_depth_gray.png'
    cv2.imwrite(str(gray_path), depth_mm)

    # Save colorized depth
    depth_color = visualize_depth(depth, max_depth)
    color_path = output_dir / f'{filename_prefix}_depth_color.png'
    cv2.imwrite(str(color_path), depth_color)

    # Save comparison 
    h, w = rgb_image.shape[:2]
    # Resize depth to match RGB
    depth_color_resized = cv2.resize(depth_color, (w, h))
    comparison = np.hstack([rgb_image, depth_color_resized])
    comparison_path = output_dir / f'{filename_prefix}_comparison.png'
    cv2.imwrite(str(comparison_path), comparison)

    print(f"\nOutputs saved to {output_dir}:")
    print(f"  - {filename_prefix}_rgb.png (original RGB)")
    print(f"  - {filename_prefix}_depth_gray.png (16-bit, millimeters)")
    print(f"  - {filename_prefix}_depth_color.png (colorized)")
    print(f"  - {filename_prefix}_comparison.png (side-by-side)")


def display_image_result(rgb_image, depth, max_depth):
    """
    Display static result in OpenCV window
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


def create_live_display(color_image, depth, max_depth, fps=None, paused=False):
    """
    Create live display for real-time inference
    """
    # Colorize depth
    depth_color = visualize_depth(depth, max_depth)

    # Resize depth to match RGB
    h, w = color_image.shape[:2]
    depth_color_resized = cv2.resize(depth_color, (w, h))

    # Stack horizontally
    display = np.hstack([color_image, depth_color_resized])

    # FPS counter 
    if fps is not None:
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
