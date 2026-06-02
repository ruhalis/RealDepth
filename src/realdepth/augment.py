"""
Per-sequence image augmentation for temporal depth training.

Augmentation parameters are sampled ONCE per sequence (via ``sample()``) and
applied identically to every frame in that sequence, so the temporal stream
still looks like a single coherent camera to the ConvGRU.

- Photometric / sharpness / noise are applied to RGB only (depth/mask untouched).
- Horizontal flip is geometric: applied to RGB + depth + mask together, and the
  caller must mirror the principal point (cx_n -> 1 - cx_n).
"""
import io
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF


class SequenceAugmentor:
    def __init__(self, config=None):
        config = config or {}
        self.enable = config.get('enable', True)
        self.hflip_prob = config.get('hflip_prob', 0.5)

        ph = config.get('photometric', {})
        self.ph_prob = ph.get('prob', 0.8)
        self.ph_brightness = ph.get('brightness', 0.3)
        self.ph_contrast = ph.get('contrast', 0.3)
        self.ph_saturation = ph.get('saturation', 0.3)
        self.ph_hue = ph.get('hue', 0.05)
        self.ph_gamma = ph.get('gamma', 0.2)

        sh = config.get('sharpness', {})
        self.sh_prob = sh.get('prob', 0.5)
        self.sh_min_scale = sh.get('min_scale', 0.4)
        self.sh_blur_max = sh.get('blur_max', 1.5)

        ns = config.get('noise', {})
        self.ns_prob = ns.get('prob', 0.5)
        self.ns_gaussian_std = ns.get('gaussian_std', 12.0)
        self.ns_jpeg_prob = ns.get('jpeg_prob', 0.3)
        self.ns_jpeg_min_q = ns.get('jpeg_min_q', 40)

    def sample(self, rng):
        """Sample one set of augmentation parameters for a whole sequence."""
        if not self.enable:
            return None

        p = {'flip': rng.random() < self.hflip_prob}

        if rng.random() < self.ph_prob:
            p['photometric'] = {
                'brightness': 1.0 + rng.uniform(-self.ph_brightness, self.ph_brightness),
                'contrast': 1.0 + rng.uniform(-self.ph_contrast, self.ph_contrast),
                'saturation': 1.0 + rng.uniform(-self.ph_saturation, self.ph_saturation),
                'hue': rng.uniform(-self.ph_hue, self.ph_hue),
                'gamma': 1.0 + rng.uniform(-self.ph_gamma, self.ph_gamma),
            }

        if rng.random() < self.sh_prob:
            p['sharpness'] = {
                'scale': rng.uniform(self.sh_min_scale, 1.0),
                'blur': rng.uniform(0.0, self.sh_blur_max),
            }

        if rng.random() < self.ns_prob:
            p['noise'] = {
                'std': rng.uniform(0.0, self.ns_gaussian_std),
                'jpeg_q': (int(rng.uniform(self.ns_jpeg_min_q, 95))
                           if rng.random() < self.ns_jpeg_prob else None),
            }

        return p

    def apply(self, rgb_pil, depth_np, params):
        """Apply sampled params to one frame.

        Args:
            rgb_pil: PIL.Image RGB
            depth_np: float32 HxW depth array (meters)
            params: dict from sample(); None disables augmentation.

        Returns:
            (rgb_pil, depth_np) augmented.
        """
        if not params:
            return rgb_pil, depth_np

        # --- Geometric (RGB + depth together) ---
        if params.get('flip'):
            rgb_pil = TF.hflip(rgb_pil)
            depth_np = np.ascontiguousarray(depth_np[:, ::-1])

        # --- Photometric (RGB only) ---
        ph = params.get('photometric')
        if ph:
            rgb_pil = TF.adjust_brightness(rgb_pil, ph['brightness'])
            rgb_pil = TF.adjust_contrast(rgb_pil, ph['contrast'])
            rgb_pil = TF.adjust_saturation(rgb_pil, ph['saturation'])
            rgb_pil = TF.adjust_hue(rgb_pil, ph['hue'])
            rgb_pil = TF.adjust_gamma(rgb_pil, max(ph['gamma'], 1e-3))

        # --- Sharpness / resolution feel (RGB only) ---
        sh = params.get('sharpness')
        if sh:
            w, h = rgb_pil.size
            if sh['scale'] < 0.999:
                small = rgb_pil.resize(
                    (max(1, int(w * sh['scale'])), max(1, int(h * sh['scale']))),
                    Image.BILINEAR,
                )
                rgb_pil = small.resize((w, h), Image.BILINEAR)
            if sh['blur'] > 0.05:
                rgb_pil = rgb_pil.filter(ImageFilter.GaussianBlur(radius=sh['blur']))

        # --- Sensor noise / JPEG (RGB only) ---
        ns = params.get('noise')
        if ns:
            if ns['std'] > 0.1:
                arr = np.asarray(rgb_pil, dtype=np.float32)
                arr += np.random.normal(0.0, ns['std'], arr.shape)
                rgb_pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            if ns['jpeg_q'] is not None:
                buf = io.BytesIO()
                rgb_pil.save(buf, format='JPEG', quality=ns['jpeg_q'])
                buf.seek(0)
                rgb_pil = Image.open(buf).convert('RGB')

        return rgb_pil, depth_np
