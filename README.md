# RealDepth

Lightweight real-time monocular depth estimation (~1.07M params) with sim-to-real transfer. Train on synthetic data from NVIDIA Isaac Sim digital twins, deploy on real hardware.

## Motivation

Building a general-purpose monocular depth model that works on arbitrary scenes requires massive, diverse datasets and large models (e.g., MiDaS uses 12+ datasets). For practical applications like robotics, warehouse automation, or indoor navigation, we don't need a model that works everywhere — we need one that works reliably in a **specific known environment**.

Our approach: build a digital twin of the target environment in NVIDIA Isaac Sim, generate unlimited synthetic training data with perfect ground truth depth, and train a small, fast model that transfers to the real version of that scene. This makes the problem tractable for a lightweight architecture while still achieving useful real-world performance.

## Goal

Demonstrate that a compact (~1.07M parameter) monocular depth model can be:
1. **Trained entirely (or primarily) on synthetic data** generated from an Isaac Sim digital twin
2. **Transferred to the real world** via domain adaptation strategies (domain randomization, mixed synthetic+real fine-tuning)
3. **Deployed in real-time** on standard hardware for the target environment

The key research question: *How effectively can sim-to-real transfer work for monocular depth estimation at small model scale, and which domain adaptation strategies close the gap most efficiently?*

## Results

Training visualization showing RGB inputs (left), ground truth depth image (center) and predicted depth maps (right):

![Training Results](assets/epoch99.jpg)

## Installation

Requires Python >=3.8, <3.11 (pyrealsense2 constraint). pyrealsense2 doesn't work on macOS.

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py develop
```

## Quick Start

### Collect Your Own Dataset

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
This organizes data into `dataset/` with 80/10/10 split.

**3. Train on your data:**

Edit your config file in `configs/` that controls training.

```bash
python scripts/train.py
```

### Inference

```bash
python scripts/infer_image.py --checkpoint <path> --image <path>   # Single image
python scripts/infer_camera.py --checkpoint <path>                  # Real-time camera feed
python scripts/infer_compare.py --checkpoint <path>                 # Side-by-side comparison
```

### Validation

```bash
python scripts/validate.py --checkpoint <path> --config <config_path> --split <val|test>
```

## Architecture

MobileNetV2 encoder + ConvGRU temporal fusion + lightweight depthwise separable decoder (~1.07M params):

- **Encoder** (543K params): Pretrained MobileNetV2 layers 0-13, extracts features at 5 scales: 16ch@1/1, 16ch@1/2, 24ch@1/4, 32ch@1/8, 96ch@1/16
- **ConvGRU** (498K params): Convolutional GRU at the 96ch bottleneck (1/16 scale). Propagates temporal context across video frames via update/reset gates
- **Decoder** (28K params): 4 lightweight decoder blocks using NN upsample + 5x5 depthwise separable convs (NNConv5 from FastDepth). Channel plan: 96 → 64 → 32 → 16 → 16 → 1
- **Skip connections**: Preserve spatial details from encoder to decoder

### Three-Stage Training
1. **Stage 1**: Freeze encoder + decoder, train only ConvGRU
2. **Stage 2**: Freeze encoder, train ConvGRU + decoder
3. **Stage 3**: Unfreeze all, end-to-end fine-tuning with lower encoder LR

## Sim-to-Real Pipeline

1. **Create digital twin** of the target environment in NVIDIA Isaac Sim
2. **Generate synthetic dataset** with domain randomization (varied lighting, textures, camera noise):
   ```bash
   /home/nurtay/isaacsim/python.sh scripts/collect_isaac_sim.py --headless --usd_path "warehouse.usd" --num_frames 3000 --num_objects 40
   ```
3. **Train on synthetic data** → evaluate on real data → measure sim-to-real gap
4. **Domain adaptation** — apply strategies to close the gap:
   - Domain randomization in Isaac Sim
   - Fine-tuning on small real-world dataset (collected via RealSense D435i)
   - Mixed synthetic + real training

## License

MIT License