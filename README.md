# RealDepth

2D camera depth estimation using U-Net architecture. Predict depth maps from RGB images/
Dataset: RealSense D435i camera data(RGB + depth)

## Results
Training visualization showing RGB inputs (left), ground truth depth image (center) and predicted depth maps (right):

![Training Results](assets/epoch99.jpg)

## Installation

Only Python =>3.8 and <3.11  
I used python3.10
because pyrealsense2 requires python=>3.7<3.11
pyrealsense2 doesn't work on mac

```bash
python3.10 -m venv venv
```

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
python setup.py develop
```



## Quick Start

### Collect Your Own Dataset

Calibrate your 3d camera

**1. Collect data with RealSense D435i:**

```bash
python scripts/collect_dataset.py --duration 1200 --fps 30
```
fps only 6, 15 or 30

This creates `collected_dataset/<timestamp>/` with:
- `rgb/` - RGB images (1280x720)
- `depth/` - 16-bit depth PNGs in millimeters
- `intrinsics.txt` - Camera calibration

**For good data:**
- Move slowly to avoid motion blur
- Record different rooms, lighting, and distances

**2. Split dataset into train/val/test:**

```bash
python scripts/split_dataset.py
```
This organizes data into `dataset/` with 80/10/10 split:

**3. Train on your data:**

Edit your config file that controls training

```bash
python scripts/train.py
```

## Architecture

U-Net style encoder-decoder with skip connections:
- **Encoder**: Extracts features at 5 scales (32 → 64 → 128 → 256 → 512 channels)
- **Skip connections**: Preserve spatial details
- **Decoder**: Reconstructs full-resolution depth map

## License

MIT License