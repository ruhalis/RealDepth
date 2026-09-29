"""RGB | RealSense | RealDepth comparison panels.

Renders one PNG per frame with three panels side by side: the input RGB, the
RealSense depth it was recorded with, and the model's prediction. GT and
prediction share a colour scale so the two are directly comparable — scaling
each by its own min/max (as the previous version of this script did) makes a
wrong prediction look right.

    python scripts/infer_compare.py --checkpoint experiments/simple_ft_v2/checkpoints/best.pth \
        --data dataset_simple --split test --frames 00000000 01000601 03000300 04001200 \
        --out comparison_output
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from realdepth.predictor import (setup_device, load_checkpoint,
                                 preprocess_rgb_image, predict_depth)

LABEL_H = 34


def colorize(depth, lo, hi, invalid=None):
    """JET colormap over a fixed [lo, hi] range; invalid pixels go black."""
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    out = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if invalid is not None:
        out[invalid] = 0
    return out


def label(img, text):
    """Stack a caption bar above an image panel."""
    bar = np.zeros((LABEL_H, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='dataset_simple')
    p.add_argument('--split', default='test')
    p.add_argument('--frames', nargs='+', required=True, help='frame stems')
    p.add_argument('--out', default='comparison_output')
    p.add_argument('--device', default='auto')
    args = p.parse_args()

    device = setup_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    max_depth = cfg['max_depth']
    split_dir = Path(args.data) / args.split

    intr_path = split_dir / 'intrinsics.json'
    intr_all = json.loads(intr_path.read_text()) if intr_path.exists() else {}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'frame':<12} {'AbsRel':>8} {'MAE':>9} {'RMSE':>9} {'valid':>7}")
    print('-' * 50)

    for stem in args.frames:
        bgr = cv2.imread(str(split_dir / 'rgb' / f'{stem}.png'))
        if bgr is None:
            raise SystemExit(f"no RGB for {stem}")
        gt = np.asarray(Image.open(split_dir / 'depth' / f'{stem}.png'),
                        dtype=np.float32) * cfg.get('depth_scale', 0.001)

        # Per-frame intrinsics when the split carries them — the model is
        # camera-aware, so feeding the wrong camera shifts the whole scale.
        intrinsics = None
        if stem in intr_all:
            k = intr_all[stem]
            intrinsics = torch.tensor([[k['fx'] / k['width'], k['fy'] / k['height'],
                                        k['cx'] / k['width'], k['cy'] / k['height']]],
                                      dtype=torch.float32)

        rgb_t = preprocess_rgb_image(bgr, cfg['image_size'])
        pred = predict_depth(model, rgb_t, device, intrinsics=intrinsics)

        h, w = pred.shape
        gt_r = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        bgr_r = cv2.resize(bgr, (w, h))
        valid = (gt_r > 0) & (gt_r <= max_depth)

        d = gt_r[valid]
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)

        err = np.abs(pred[valid] - d)
        abs_rel = float((err / np.clip(d, 1e-3, None)).mean())
        mae, rmse = float(err.mean()), float(np.sqrt((err ** 2).mean()))
        print(f"{stem:<12} {abs_rel:>8.4f} {mae*100:>7.1f}cm {rmse*100:>7.1f}cm "
              f"{100*valid.mean():>6.1f}%")

        panel = np.hstack([
            label(bgr_r, 'RGB (input)'),
            label(colorize(gt_r, lo, hi, ~valid), f'RealSense D435I  {lo:.2f}-{hi:.2f} m'),
            label(colorize(pred, lo, hi), f'RealDepth  AbsRel {abs_rel:.3f}  MAE {mae*100:.1f} cm'),
        ])
        cv2.imwrite(str(out_dir / f'compare_{stem}.png'), panel)

    print(f"\nSaved to {out_dir}/")


if __name__ == '__main__':
    main()
