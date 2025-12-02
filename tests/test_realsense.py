#!/usr/bin/env python3
"""
Diagnostic script for RealSense camera detection
"""
import pyrealsense2 as rs

print("Checking for RealSense devices...")
print("-" * 60)

# Create context
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("ERROR: No RealSense devices found!")
    print("\nPossible causes:")
    print("1. Camera not plugged in properly (try replugging)")
    print("2. USB permissions issue (see below)")
    print("3. Wrong USB port (try USB 3.0 port)")
    print("4. Camera firmware issue")
    print("\n" + "=" * 60)
    print("USB PERMISSIONS FIX:")
    print("=" * 60)
    print("Run these commands:")
    print("  sudo apt-get install librealsense2-udev-rules")
    print("  sudo udevadm control --reload-rules")
    print("  sudo udevadm trigger")
    print("Then unplug and replug the camera")
    print("=" * 60)
else:
    print(f"Found {len(devices)} RealSense device(s):\n")

    for i, dev in enumerate(devices):
        print(f"Device {i}:")
        print(f"  Name: {dev.get_info(rs.camera_info.name)}")
        print(f"  Serial: {dev.get_info(rs.camera_info.serial_number)}")
        print(f"  Firmware: {dev.get_info(rs.camera_info.firmware_version)}")
        print(f"  USB Type: {dev.get_info(rs.camera_info.usb_type_descriptor)}")

        # List available sensors
        sensors = dev.query_sensors()
        print(f"  Sensors ({len(sensors)}):")
        for sensor in sensors:
            print(f"    - {sensor.get_info(rs.camera_info.name)}")

        # Check supported stream profiles
        print("  Supported streams:")
        for sensor in sensors:
            profiles = sensor.get_stream_profiles()
            for profile in profiles:
                if profile.stream_type() == rs.stream.color:
                    vp = profile.as_video_stream_profile()
                    print(f"    Color: {vp.width()}x{vp.height()} @ {vp.fps()}fps")
                elif profile.stream_type() == rs.stream.depth:
                    vp = profile.as_video_stream_profile()
                    print(f"    Depth: {vp.width()}x{vp.height()} @ {vp.fps()}fps")
        print()

    print("Camera detected successfully!")
    print("If your script still fails, try:")
    print("1. Lower FPS (e.g., --fps 15 or --fps 6)")
    print("2. Lower resolution in the script")
    print("3. Check if another application is using the camera")
