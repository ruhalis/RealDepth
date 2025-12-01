"""
Intel RealSense D435i Data Collection Script
Collects linked RGB and Depth frames

Usage:
    python collect_dataset.py --duration 1200 --fps 30
    Supports only 6, 15, 30 FPS modes
"""
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import pyrealsense2 as rs

class RealSenseRecorder:
    """
    Records linked RGB and Depth images
    """
    def __init__(
        self,
        output_dir,
        rgb_resolution=(1280, 720),
        depth_resolution=(1280, 720),
        fps=30,
        align_depth=True,
        apply_filters=True
    ):
        self.output_dir = Path(output_dir)
        self.rgb_resolution = rgb_resolution
        self.depth_resolution = depth_resolution
        self.fps = fps
        self.align_depth = align_depth
        self.apply_filters = apply_filters
        
        # Create output directories
        self.rgb_dir = self.output_dir / 'rgb'
        self.depth_dir = self.output_dir / 'depth'
        self.depth_color_dir = self.output_dir / 'depth_colorized'
        
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.depth_color_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(
            rs.stream.color,
            rgb_resolution[0], rgb_resolution[1],
            rs.format.bgr8,
            fps
        )
        self.config.enable_stream(
            rs.stream.depth,
            depth_resolution[0], depth_resolution[1],
            rs.format.z16,
            fps
        )
        
        # Depth processing filters
        if self.apply_filters:
            self.spatial_filter = rs.spatial_filter()
            self.temporal_filter = rs.temporal_filter()
            self.hole_filling = rs.hole_filling_filter()
        
        # Alignment
        if self.align_depth:
            self.align = rs.align(rs.stream.color)
        
        # Colorizer for visualization
        self.colorizer = rs.colorizer()
        self.colorizer.set_option(rs.option.color_scheme, 0)  # Jet colormap
        
        self.frame_count = 0
        self.profile = None
    
    def start(self):
        """Start the RealSense pipeline"""
        print("Starting RealSense pipeline...")
        self.profile = self.pipeline.start(self.config)
        
        # Get device info
        device = self.profile.get_device()
        device_name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        firmware = device.get_info(rs.camera_info.firmware_version)
        
        print(f"Device: {device_name}")
        print(f"Serial: {serial}")
        print(f"Firmware: {firmware}")
        
        # Get depth scale
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth Scale: {self.depth_scale} (1 unit = {self.depth_scale * 1000:.2f} mm)")
        
        # Get intrinsics
        color_profile = self.profile.get_stream(rs.stream.color)
        depth_profile = self.profile.get_stream(rs.stream.depth)
        
        self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        
        # Warm up (skip first few frames)
        print("Warming up camera...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        
        print("Ready to record!")
        return self
    
    def save_intrinsics(self):
        """Save camera intrinsics to file"""
        intrinsics_path = self.output_dir / 'intrinsics.txt'
        with open(intrinsics_path, 'w') as f:
            f.write("Color Camera Intrinsics:\n")
            f.write(f"  Width: {self.color_intrinsics.width}\n")
            f.write(f"  Height: {self.color_intrinsics.height}\n")
            f.write(f"  fx: {self.color_intrinsics.fx}\n")
            f.write(f"  fy: {self.color_intrinsics.fy}\n")
            f.write(f"  cx: {self.color_intrinsics.ppx}\n")
            f.write(f"  cy: {self.color_intrinsics.ppy}\n")
            f.write(f"\nDepth Scale: {self.depth_scale}\n")
        print(f"Saved intrinsics to {intrinsics_path}")
    
    def process_frames(self, frames):
        """Process and align frames"""
        # Align depth to color
        if self.align_depth:
            frames = self.align.process(frames)
        
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None, None
        
        # Apply depth filters
        if self.apply_filters:
            depth_frame = self.spatial_filter.process(depth_frame)
            depth_frame = self.temporal_filter.process(depth_frame)
            depth_frame = self.hole_filling.process(depth_frame)
        
        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Colorized depth for visualization
        depth_colorized = np.asanyarray(
            self.colorizer.colorize(depth_frame).get_data()
        )
        
        return color_image, depth_image, depth_colorized
    
    def save_frame(self, color_image, depth_image, depth_colorized):
        """Save frame to disk"""
        filename = f"{self.frame_count:06d}.png"
        
        # Save RGB (stored as BGR by OpenCV convention)
        rgb_path = self.rgb_dir / filename
        cv2.imwrite(str(rgb_path), color_image)
        
        # Save depth as 16-bit PNG (preserves full precision)
        depth_path = self.depth_dir / filename
        cv2.imwrite(str(depth_path), depth_image)
        
        # Save colorized depth for visualization
        depth_color_path = self.depth_color_dir / filename
        cv2.imwrite(str(depth_color_path), depth_colorized)
        
        self.frame_count += 1
    
    def record(self, duration_seconds, save_interval=None):
        """
        Record frames for specified duration
        
        Args:
            duration_seconds: Total recording time in seconds
            save_interval: Save every N-th frame (None = save all)
        """
        print(f"\nRecording for {duration_seconds} seconds...")
        print(f"Output directory: {self.output_dir}")
        print("Press 'q' to stop early\n")
        
        start_time = time.time()
        frame_idx = 0
        
        # Create preview window
        cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Preview', 1280, 480)
        
        try:
            while True:
                elapsed = time.time() - start_time
                
                if elapsed >= duration_seconds:
                    break
                
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                
                # Process frames
                color_image, depth_image, depth_colorized = self.process_frames(frames)
                
                if color_image is None:
                    continue
                
                frame_idx += 1
                
                # Save frame (optionally skip some frames)
                if save_interval is None or frame_idx % save_interval == 0:
                    self.save_frame(color_image, depth_image, depth_colorized)
                
                # Display preview
                preview = np.hstack([
                    cv2.resize(color_image, (640, 480)),
                    cv2.resize(depth_colorized, (640, 480))
                ])
                
                # Add info text
                remaining = duration_seconds - elapsed
                info_text = f"Frames: {self.frame_count} | Time: {elapsed:.1f}s | Remaining: {remaining:.1f}s"
                cv2.putText(preview, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Preview', preview)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nRecording stopped by user")
                    break
        
        finally:
            cv2.destroyAllWindows()
        
        print(f"\nRecording complete!")
        print(f"Total frames saved: {self.frame_count}")
        print(f"Actual duration: {time.time() - start_time:.1f} seconds")
    
    def stop(self):
        """Stop the pipeline"""
        self.pipeline.stop()
        print("Pipeline stopped")


def main():
    parser = argparse.ArgumentParser(
        description='Record RGB-D dataset from Intel RealSense D435i'
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=60,
        help='Recording duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Camera FPS (default: 30)'
    )
    
    args = parser.parse_args()

    # Get script directory and create output path
    script_dir = Path(__file__).parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = script_dir / 'collected_dataset' / timestamp
    
    print("=" * 60)
    print("RealSense D435i Dataset Collection")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Duration: {args.duration} seconds")
    print(f"FPS: {args.fps}")
    print(f"RGB Resolution: 1280x720")
    print(f"Depth Resolution: 1280x720")
    print(f"Align Depth: True")
    print(f"Apply Filters: True")
    print("=" * 60)
    
    # Create recorder
    recorder = RealSenseRecorder(
        output_dir=output_dir,
        rgb_resolution=(1280, 720),
        depth_resolution=(1280, 720),
        fps=args.fps,
        align_depth=True,
        apply_filters=True
    )
    
    try:
        recorder.start()
        recorder.save_intrinsics()
        recorder.record(
            duration_seconds=args.duration,
            save_interval=None
        )
    finally:
        recorder.stop()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"RGB images: {output_dir / 'rgb'}")
    print(f"Depth images: {output_dir / 'depth'}")
    print(f"Depth colorized: {output_dir / 'depth_colorized'}")
    print(f"Intrinsics: {output_dir / 'intrinsics.txt'}")
    print(f"\nTotal frames: {recorder.frame_count}")
    
    if args.duration >= 60:
        effective_fps = recorder.frame_count / args.duration
        print(f"Effective FPS: {effective_fps:.1f}")


if __name__ == "__main__":
    main()