# RealDepth

2D camera depth estimation using U-Net architecture. Predict depth maps from RGB images using RealSense D435i camera data.

## Results

Training visualization showing RGB inputs (left), ground truth depth image (center) and predicted depth maps (right):

![Training Results](assets/epoch99.jpg)

## Installation

### Option 1: Install as package (recommended)
```bash
pip install -e .
```

### Option 2: Install dependencies only
```bash
pip install -r requirements.txt
```

**Note for Mac users**: For RealSense support: `pip install -e .[realsense]`

## Quick Start

### Collect Your Own Dataset

**1. Calibrate your camera**

**2. Collect data with RealSense D435i:**

```bash
python scripts/collect_dataset.py --duration 1200 --fps 30
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
python scripts/split_dataset.py
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

**4. Train on your data:**

```bash
python scripts/train.py
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
├── realdepth/            # Core library package
│   ├── __init__.py      # Package initialization
│   ├── model.py         # Network architecture
│   ├── losses.py        # Loss functions & metrics
│   ├── depth_datasets.py # Data loaders
│   └── model_utils.py   # Model utilities
├── scripts/             # Executable scripts
│   ├── train.py
│   ├── collect_dataset.py
│   ├── split_dataset.py
│   ├── infer_image.py
│   └── infer_camera.py
├── configs/             # Training configurations
│   └── realsense.yaml  # RealSense config
├── tests/              # Unit tests
├── setup.py            # Package installation
└── pyproject.toml      # Modern packaging config
```

## Troubleshooting

**Black predictions:** Learning rate too high → reduce to 1e-5

**Blurry edges:** Increase `w_grad` in config

**Out of memory:** Reduce `batch_size` or `image_size`

**Scale issues:** Add more variety to training data



## Architecture

U-Net style encoder-decoder with skip connections:
- **Encoder**: Extracts features at 5 scales (32 → 64 → 128 → 256 → 512 channels)
- **Skip connections**: Preserve spatial details
- **Decoder**: Reconstructs full-resolution depth map

## License

MIT License