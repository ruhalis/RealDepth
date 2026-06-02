"""
Tests for camera-aware conditioning, per-sequence augmentation, and the
intrinsics path through SequenceDataset.
"""
import json
import numpy as np
import torch
from PIL import Image

from realdepth.model import DepthEstimationNet, CANONICAL_INTRINSICS
from realdepth.conv_gru import ConvGRU
from realdepth.augment import SequenceAugmentor
from realdepth.sequence_dataset import SequenceDataset


class TestCameraAwareModel:
    def test_forward_with_intrinsics(self):
        model = DepthEstimationNet(max_depth=10.0, pretrained_encoder=False,
                                   camera_aware=True)
        model.eval()
        x = torch.randn(2, 3, 480, 640)
        intr = torch.tensor([[0.9, 1.2, 0.5, 0.5], [0.6, 0.6, 0.4, 0.5]])
        with torch.no_grad():
            out = model(x, intrinsics=intr)
        assert out.shape == (2, 1, 480, 640)

    def test_none_intrinsics_uses_canonical(self):
        """model(x) without intrinsics still runs (canonical fallback)."""
        model = DepthEstimationNet(max_depth=10.0, pretrained_encoder=False,
                                   camera_aware=True)
        model.eval()
        x = torch.randn(1, 3, 480, 640)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 480, 640)

    def test_intrinsics_change_output(self):
        """Different intrinsics should change the prediction."""
        model = DepthEstimationNet(max_depth=10.0, pretrained_encoder=False,
                                   camera_aware=True)
        model.eval()
        x = torch.randn(1, 3, 480, 640)
        wide = torch.tensor([[0.4, 0.4, 0.5, 0.5]])
        tele = torch.tensor([[1.5, 1.5, 0.5, 0.5]])
        with torch.no_grad():
            model.reset_temporal()
            out_wide = model(x, intrinsics=wide)
            model.reset_temporal()
            out_tele = model(x, intrinsics=tele)
        assert not torch.allclose(out_wide, out_tele)

    def test_camera_aware_gru_input_channels(self):
        model = DepthEstimationNet(max_depth=10.0, pretrained_encoder=False,
                                   camera_aware=True, ray_channels=2)
        assert model.temporal.conv_gates.in_channels == 96 + 2 + 96  # x + ray + hidden

    def test_camera_off_matches_legacy(self):
        model = DepthEstimationNet(max_depth=10.0, pretrained_encoder=False,
                                   camera_aware=False)
        assert model.ray_channels == 0
        assert model.temporal.conv_gates.in_channels == 96 + 96
        model.eval()
        x = torch.randn(1, 3, 256, 320)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 256, 320)

    def test_ray_map_canonical_centered(self):
        """Ray map for canonical centered intrinsics is antisymmetric about center."""
        model = DepthEstimationNet(max_depth=10.0, pretrained_encoder=False,
                                   camera_aware=True)
        ray = model._build_ray_map(None, 4, 4, torch.device('cpu'), torch.float32)
        assert ray.shape == (1, 2, 4, 4)
        # centered principal point -> ray_x mean ~ 0
        assert abs(ray[0, 0].mean().item()) < 1e-5


class TestSequenceAugmentor:
    def _params(self, **over):
        cfg = {
            'enable': True, 'hflip_prob': 0.0,
            'photometric': {'prob': 0.0}, 'sharpness': {'prob': 0.0},
            'noise': {'prob': 0.0},
        }
        cfg.update(over)
        return cfg

    def test_disabled_returns_none(self):
        aug = SequenceAugmentor({'enable': False})
        assert aug.sample(np.random.default_rng(0)) is None

    def test_flip_applied_to_rgb_and_depth(self):
        aug = SequenceAugmentor(self._params(hflip_prob=1.0))
        params = aug.sample(np.random.default_rng(0))
        assert params['flip'] is True
        rgb = Image.fromarray(np.arange(12, dtype=np.uint8).reshape(2, 2, 3) * 10)
        depth = np.arange(4, dtype=np.float32).reshape(2, 2)
        rgb2, depth2 = aug.apply(rgb, depth, params)
        assert np.array_equal(depth2, depth[:, ::-1])
        assert np.array_equal(np.array(rgb2), np.array(rgb)[:, ::-1])

    def test_no_op_when_all_probs_zero(self):
        aug = SequenceAugmentor(self._params())
        params = aug.sample(np.random.default_rng(0))
        assert params['flip'] is False
        assert 'photometric' not in params


class TestSequenceDatasetIntrinsics:
    def _make_dataset(self, root, n=4, cx_frac=0.25, augment=False, aug_cfg=None):
        split = root / 'train'
        rgb_dir = split / 'rgb'
        depth_dir = split / 'depth'
        rgb_dir.mkdir(parents=True)
        depth_dir.mkdir(parents=True)
        w, h = 64, 48
        meta = {}
        for i in range(n):
            stem = f"{i:06d}"
            Image.fromarray(
                (np.random.rand(h, w, 3) * 255).astype(np.uint8)
            ).save(rgb_dir / f"{stem}.png")
            Image.fromarray(
                (np.random.rand(h, w) * 5000).astype(np.uint16)
            ).save(depth_dir / f"{stem}.png")
            meta[stem] = {'fx': 50.0, 'fy': 50.0,
                          'cx': w * cx_frac, 'cy': h / 2.0,
                          'width': w, 'height': h}
        with open(split / 'intrinsics.json', 'w') as f:
            json.dump(meta, f)
        return SequenceDataset(
            data_dir=str(root), sequence_length=2, image_size=(48, 64),
            split='train', augment=augment, augment_config=aug_cfg,
        )

    def test_returns_normalized_intrinsics(self, tmp_path):
        ds = self._make_dataset(tmp_path, cx_frac=0.25)
        item = ds[0]
        assert item['intrinsics'].shape == (4,)
        # cx_n = cx / W = 0.25
        assert item['intrinsics'][2].item() == \
            __import__('pytest').approx(0.25, abs=1e-5)

    def test_flip_mirrors_principal_point(self, tmp_path):
        aug_cfg = {'enable': True, 'hflip_prob': 1.0,
                   'photometric': {'prob': 0.0}, 'sharpness': {'prob': 0.0},
                   'noise': {'prob': 0.0}}
        ds = self._make_dataset(tmp_path, cx_frac=0.25, augment=True, aug_cfg=aug_cfg)
        item = ds[0]
        # flipped: cx_n -> 1 - 0.25 = 0.75
        assert item['intrinsics'][2].item() == \
            __import__('pytest').approx(0.75, abs=1e-5)

    def test_fallback_canonical_without_json(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        # remove the intrinsics file and rebuild
        (tmp_path / 'train' / 'intrinsics.json').unlink()
        ds2 = SequenceDataset(
            data_dir=str(tmp_path), sequence_length=2, image_size=(48, 64),
            split='train',
        )
        item = ds2[0]
        assert tuple(round(v, 3) for v in item['intrinsics'].tolist()) == \
            tuple(round(v, 3) for v in CANONICAL_INTRINSICS)
