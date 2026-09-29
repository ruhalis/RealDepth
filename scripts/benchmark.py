"""Measure RealDepth inference speed.

Separates the three costs that make up a live pipeline so it is clear which one
dominates: RGB preprocessing (CPU), the forward pass, and depth colourization.
Camera capture is measured only with --camera, which needs pyrealsense2 and a
connected device.

    python scripts/benchmark.py --checkpoint experiments/simple_ft_v2/checkpoints/best.pth
"""
import argparse
import time

import cv2
import numpy as np
import torch

from realdepth.predictor import (setup_device, load_checkpoint,
                                 preprocess_rgb_image, fov_to_intrinsics)
from realdepth.model_utils import count_params


def timeit(fn, iters, warmup, cuda):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return np.array(ts) * 1000.0  # ms


def report(name, ms):
    print(f"{name:<34} {ms.mean():>7.2f} {np.percentile(ms,50):>7.2f} "
          f"{np.percentile(ms,99):>7.2f} {1000/ms.mean():>8.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--camera', action='store_true', help='also measure live capture')
    args = p.parse_args()

    device = setup_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    H, W = cfg['image_size']
    cuda = device.type == 'cuda'

    print(f"\nParams: {count_params(model):,}   input {W}x{H}   "
          f"iters={args.iters} warmup={args.warmup}")
    print(f"\n{'stage':<34} {'mean':>7} {'p50':>7} {'p99':>7} {'FPS':>8}  (ms)")
    print('-' * 70)

    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    intr = fov_to_intrinsics(69.4, W, H).to(device)   # D435I colour FOV
    rgb_t = preprocess_rgb_image(frame, cfg['image_size']).to(device)

    report('preprocess (CPU: resize+norm)',
           timeit(lambda: preprocess_rgb_image(frame, cfg['image_size']),
                  args.iters, args.warmup, False))

    with torch.no_grad():
        # The ConvGRU hidden state lives on the model and carries dtype with it,
        # so every precision switch needs a reset or the next cat() mismatches.
        model.reset_temporal()
        report('forward fp32',
               timeit(lambda: model(rgb_t, intrinsics=intr), args.iters, args.warmup, cuda))

        if cuda:
            model.reset_temporal()
            model_h = model.half()
            rgb_h = rgb_t.half()
            report('forward fp16',
                   timeit(lambda: model_h(rgb_h, intrinsics=intr.half()),
                          args.iters, args.warmup, cuda))
            model.float()
            model.reset_temporal()

        depth = model(rgb_t, intrinsics=intr)
        report('GPU->CPU copy',
               timeit(lambda: depth.squeeze().cpu().numpy(), args.iters, args.warmup, cuda))

    d = depth.squeeze().detach().cpu().numpy()
    report('colourize (JET)',
           timeit(lambda: cv2.applyColorMap(
               (np.clip(d / cfg['max_depth'], 0, 1) * 255).astype(np.uint8),
               cv2.COLORMAP_JET), args.iters, args.warmup, False))

    def full():
        t = preprocess_rgb_image(frame, cfg['image_size']).to(device)
        with torch.no_grad():
            out = model(t, intrinsics=intr)
        o = out.squeeze().cpu().numpy()
        cv2.applyColorMap((np.clip(o / cfg['max_depth'], 0, 1) * 255).astype(np.uint8),
                          cv2.COLORMAP_JET)
    print('-' * 70)
    report('END-TO-END (no capture)', timeit(full, args.iters, args.warmup, cuda))

    if cuda:
        print(f"\nVRAM peak: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")

    if args.camera:
        import pyrealsense2 as rs
        pipe, conf = rs.pipeline(), rs.config()
        conf.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipe.start(conf)
        try:
            for _ in range(args.warmup):
                pipe.wait_for_frames()
            ts = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                f = pipe.wait_for_frames()
                img = np.asanyarray(f.get_color_frame().get_data())
                t = preprocess_rgb_image(img, cfg['image_size']).to(device)
                with torch.no_grad():
                    model(t, intrinsics=intr).squeeze().cpu().numpy()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            print('-' * 70)
            report('LIVE (capture+infer)', np.array(ts) * 1000)
        finally:
            pipe.stop()


if __name__ == '__main__':
    main()
