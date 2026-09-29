"""Capture live RGB-D from a RealSense and render RGB | RealSense | RealDepth panels.

Uses the camera's own intrinsics from the stream profile rather than an assumed
FOV: the model is camera-aware, so feeding it the wrong camera shifts the whole
depth scale. Depth is aligned to colour so all three panels show one viewpoint.

    python3 scripts/capture_compare.py --checkpoint <ckpt> --shots 4 --interval 2.0
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

from realdepth.predictor import (setup_device, load_checkpoint,
                                 preprocess_rgb_image, predict_depth)

LABEL_H = 34


def colorize(depth, lo, hi, invalid=None):
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    out = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if invalid is not None:
        out[invalid] = 0
    return out


def label(img, text):
    bar = np.zeros((LABEL_H, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--shots', type=int, default=4)
    p.add_argument('--interval', type=float, default=2.0)
    p.add_argument('--out', default='live_comparison')
    p.add_argument('--device', default='auto')
    args = p.parse_args()

    device = setup_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    max_depth = cfg['max_depth']
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe, conf = rs.pipeline(), rs.config()
    conf.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    conf.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(conf)
    align = rs.align(rs.stream.color)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()

    ci = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intr = torch.tensor([[ci.fx / ci.width, ci.fy / ci.height,
                          ci.ppx / ci.width, ci.ppy / ci.height]], dtype=torch.float32)
    print(f"\nintrinsics: fx={ci.fx:.1f} fy={ci.fy:.1f} cx={ci.ppx:.1f} cy={ci.ppy:.1f} "
          f"({ci.width}x{ci.height})   depth_scale={scale}")

    try:
        for _ in range(30):            # let auto-exposure settle
            pipe.wait_for_frames()

        print(f"\n{'shot':<6} {'AbsRel':>8} {'MAE':>9} {'RMSE':>9} {'valid':>7}")
        print('-' * 46)
        for i in range(args.shots):
            frames = align.process(pipe.wait_for_frames())
            bgr = np.asanyarray(frames.get_color_frame().get_data())
            gt = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * scale

            if hasattr(model, 'reset_temporal'):
                model.reset_temporal()
            pred = predict_depth(model, preprocess_rgb_image(bgr, cfg['image_size']),
                                 device, intrinsics=intr)

            h, w = pred.shape
            gt_r = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
            bgr_r = cv2.resize(bgr, (w, h))
            valid = (gt_r > 0) & (gt_r <= max_depth)

            if valid.sum() < 1000:
                print(f"{i:<6} нет валидной глубины — пропуск")
                continue
            d = gt_r[valid]
            lo, hi = np.percentile(d, 2), np.percentile(d, 98)
            err = np.abs(pred[valid] - d)
            ar = float((err / np.clip(d, 1e-3, None)).mean())
            mae, rmse = float(err.mean()), float(np.sqrt((err ** 2).mean()))
            print(f"{i:<6} {ar:>8.4f} {mae*100:>7.1f}cm {rmse*100:>7.1f}cm {100*valid.mean():>6.1f}%")

            panel = np.hstack([
                label(bgr_r, 'RGB (input)'),
                label(colorize(gt_r, lo, hi, ~valid), f'RealSense D435I  {lo:.2f}-{hi:.2f} m'),
                label(colorize(pred, lo, hi), f'RealDepth  AbsRel {ar:.3f}  MAE {mae*100:.1f} cm'),
            ])
            cv2.imwrite(str(out_dir / f'live_{i:02d}.png'), panel)
            time.sleep(args.interval)
    finally:
        pipe.stop()
    print(f"\nSaved to {out_dir}/")


if __name__ == '__main__':
    main()
