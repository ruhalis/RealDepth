# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RealDepth is a real-time 2D camera depth estimation system using a MobileNetV2 encoder, ConvGRU temporal fusion, and lightweight depthwise separable decoder to predict depth maps from RGB video. ~1.07M parameters, enabling real-time depth estimation on standard hardware with temporal consistency across frames.

Requires Python >=3.8, <3.11 (constrained by pyrealsense2). Development uses Python 3.10.

## Common Commands

### Setup
```bash
# Create venv and install
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py develop  # Installs realdepth package in dev mode (no PYTHONPATH needed)
```

If not using `setup.py develop`, prefix commands with `PYTHONPATH=src`.

### Training
```bash
python scripts/train.py                          # Uses configs/realsense.yaml
python scripts/train.py --config <path>          # Custom config
tensorboard --logdir experiments/<exp_name>/logs  # Monitor training
```

### Inference
```bash
python scripts/infer_image.py --checkpoint <path> --image <path>
python scripts/infer_camera.py --checkpoint <path>          # Real-time camera feed
python scripts/infer_compare.py --checkpoint <path>         # Side-by-side comparison
```

### Validation
```bash
python scripts/validate.py --checkpoint <path> --config <config_path> --split <val|test>
```

### Testing
```bash
python -m pytest tests/
python -m pytest tests/test_losses.py  # Single test file
```

### Data Collection
```bash
python scripts/collect_dataset.py --duration 1200 --fps 30  # fps: 6, 15, or 30
python scripts/split_dataset.py                              # 80/10/10 train/val/test split
```

## Architecture

### Model (`src/realdepth/model.py`) — 1.07M params
MobileNetV2 encoder + ConvGRU temporal fusion + lightweight depthwise separable decoder:
- **Encoder** (`encoder.py`, 543K params): Pretrained MobileNetV2 layers 0-13 (stops at 1/16, 96ch). Adds a lightweight 3→16 conv for 1/1 skip connection. 5 feature scales: 16ch@1/1, 16ch@1/2, 24ch@1/4, 32ch@1/8, 96ch@1/16.
- **ConvGRU** (`conv_gru.py`, 498K params): Convolutional GRU at the 96ch bottleneck (1/16 scale). Propagates temporal context across video frames via update/reset gates. Hidden state stored on model.
- **Decoder** (`decoder.py`, 28K params): 4 `LightDecoderBlock`s using NN upsample + 5x5 depthwise separable convs (NNConv5 from FastDepth). Channel plan: 96→64→32→16→16→1.
- **Output**: Sigmoid scaled by `max_depth`, producing `(B, 1, H, W)` depth in meters.
- **Temporal API**: `model.reset_temporal()` clears hidden state at sequence boundaries. `model(frame)` automatically carries state across calls.
- **Model name**: `realdepth` in `model_utils.py`

### Loss System (`src/realdepth/losses.py`)
`CombinedDepthLoss` combines weighted components: L1, ScaleInvariant, Gradient (Sobel), SSIM, BerHu. All losses accept an optional `mask` parameter for valid-pixel-only computation.

`TemporalConsistencyLoss` penalizes L1 difference between consecutive depth predictions (reduces flicker).

`DepthMetrics` provides evaluation (AbsRel, RMSE, MAE, delta thresholds) with `compute_stratified()` for depth-range analysis (3m, 5m, 10m).

### Inference Pipeline (`src/realdepth/predictor.py`)
Shared utilities for all inference scripts: `load_checkpoint()` handles model instantiation from checkpoint config, `predict_depth()` runs inference, `preprocess_rgb_image()` handles BGR->RGB conversion and ImageNet normalization.

### Supporting Modules
- `model_utils.py` — Model factory (`get_model`), supports `'realdepth'`. `DepthLoss` alias for backward compat
- `depth_datasets.py` — `RealSenseDataset` for single-frame loading (`dataset/{train,val,test}/{rgb,depth}/`)
- `sequence_dataset.py` — `SequenceDataset` for consecutive frame sequences (temporal training). `create_sequence_dataloaders(cfg)` factory function. Detects frame gaps via filename numbering.
- `visualization.py` — Depth colorization (JET colormap), side-by-side display, live camera display
- `validation.py` — `run_comprehensive_validation()` shared between `train.py` and `validate.py`
- `plot.py` — Training loss curve and component loss plotting

### Training
`scripts/train.py`: `Trainer` class with two-stage training:
1. **Stage 1** (`freeze_encoder_epochs`): Freeze encoder, train ConvGRU + decoder
2. **Stage 2** (remaining): Unfreeze all, end-to-end fine-tuning with lower encoder LR

Uses sequence-based data loading, separate encoder/decoder/ConvGRU LR groups, temporal consistency loss (`w_temporal`), AdamW + CosineAnnealingLR, gradient clipping (max_norm=1.0), TensorBoard logging every 10 batches, visualization grids every 5 epochs, auto comprehensive validation at end. Best model saved based on val loss. Config: `configs/realsense.yaml`.

## Key Implementation Details

- **Depth format**: RealSense stores 16-bit PNG in millimeters, converted via `depth_scale` (0.001)
- **Valid mask**: `(depth > 0) & (depth <= max_depth)` — all losses/metrics respect this
- **RGB normalization**: ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **Checkpoint format**: `{'model': state_dict, 'config': cfg_dict}`
- **Temporal state**: Hidden state stored on model instance, reset via `model.reset_temporal()` at sequence boundaries (pause/resume, new scene)
- **Config**: YAML in `configs/`, key params: `data_dir`, `max_depth`, `depth_scale`, `image_size`, `sequence_length`, `model`, `freeze_encoder_epochs`, `encoder_lr`, `w_temporal`, loss weights (`w_l1`, `w_si`, `w_grad`, `w_ssim`, `w_berhu`)

## Troubleshooting

- **Black predictions**: Learning rate too high, reduce to 1e-5
- **Blurry edges**: Increase `w_grad` (edge preservation weight)
- **OOM**: Reduce `batch_size` or `image_size` or `sequence_length`
- **Temporal flicker**: Increase `w_temporal` (temporal consistency weight)
- **pyrealsense2**: Not available on macOS; only needed for data collection scripts
