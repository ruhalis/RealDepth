# RealDepth

2D camera depth estimation using U-Net architecture. Predict depth maps from RGB images using either NYU Depth V2 dataset or your own RealSense D435i camera data.

## Installation

```bash
pip install -r requirements.txt
```

**Note for Mac users**: If you're not collecting data with RealSense camera, comment out `pyrealsense2` in `requirements.txt` (it requires x86 architecture).

## Quick Start

### Option 1: Train on nyu dataset

```bash
python train.py --config configs/nyu.yaml
```

The dataset will auto-download from HuggingFace on first run.

### Option 2: Collect Your Own Dataset

**1. Calibrate your camera**

**2. Collect data with RealSense D435i:**

```bash
python collect_dataset.py --duration 1200 --fps 30
```

This creates `collected_dataset/<timestamp>/` with:
- `rgb/` - RGB images (1280x720)
- `depth/` - 16-bit depth PNGs in millimeters
- `intrinsics.txt` - Camera calibration

**Tips for good data:**
- Move slowly to avoid motion blur
- Record different rooms, lighting, and distances
- Include objects at various depths (0.3m to 10m)

**3. Split dataset into train/val/test:**

```bash
python split_dataset.py
```

This organizes data into `dataset/` with 80/10/10 split:
```
dataset/
├── train/
│   ├── rgb/
│   ├── depth/
│   └── filenames.txt
├── val/
└── test/
```

**3. Train on your data:**

```bash
python train.py
```

## Configuration

Configs are in `configs/` directory. Key parameters:

There are key variables for training and data collection

**Monitor training:**
```bash
tensorboard --logdir experiments/<exp_name>/logs
```

## Project Structure

```
RealDepth/
├── configs/               # Training configurations
│   ├── nyu.yaml          # NYU Depth V2 config
│   └── realsense.yaml    # RealSense config
├── model.py              # Network architecture
├── losses.py             # Loss functions & metrics
├── datasets.py           # Data loaders
├── train.py              # Training script
├── collect_dataset.py    # RealSense data collection
└── split_dataset.py      # Dataset splitting
```

## Troubleshooting

**Black predictions:** Learning rate too high → reduce to 1e-5

**Blurry edges:** Increase `w_grad` in config

**Out of memory:** Reduce `batch_size` or `image_size`

**Scale issues:** Add more variety to training data

## Architecture

U-Net style encoder-decoder with skip connections:
- **Encoder**: Extracts features at 5 scales (64 → 128 → 256 → 512 → 1024 channels)
- **Skip connections**: Preserve spatial details
- **Decoder**: Reconstructs full-resolution depth map

## License

MIT License