"""
RealDepth
"""
import torch
import torch.nn as nn

from .encoder import MobileNetV2Encoder
from .decoder import LightDecoder
from .conv_gru import ConvGRU


# Canonical camera used when no intrinsics are supplied (single-image inference,
# back-compat). Normalized intrinsics [fx/W, fy/H, cx/W, cy/H].
# fx_n = 1 / (2 * tan(fov/2)); ~60 deg horizontal/vertical, centered principal point.
CANONICAL_INTRINSICS = (0.866, 0.866, 0.5, 0.5)


class DepthEstimationNet(nn.Module):
    """RealDepth: MobileNetV2 encoder + ConvGRU temporal fusion + lightweight decoder

    ConvGRU at the 1/16 scale bottleneck (96ch) propagates temporal context
    across consecutive video frames. Hidden state is stored on the model
    so the inference interface is just model(frame).

    When ``camera_aware`` is set, a per-pixel ray-direction map built from the
    camera intrinsics is concatenated to the bottleneck features before the
    ConvGRU. This lets the model reason about FOV/aspect so it generalizes
    across cameras while keeping metric depth well-posed.
    """

    def __init__(self, max_depth=10.0, pretrained_encoder=True,
                 camera_aware=True, ray_channels=2):
        super().__init__()
        self.max_depth = max_depth
        self.camera_aware = camera_aware
        self.ray_channels = ray_channels if camera_aware else 0

        self.encoder = MobileNetV2Encoder(pretrained=pretrained_encoder)
        self.temporal = ConvGRU(
            input_channels=96 + self.ray_channels, hidden_channels=96
        )
        self.decoder = LightDecoder()
        self.hidden_state = None

    def _build_ray_map(self, intrinsics, h, w, device, dtype):
        """Build a (B, 2, h, w) normalized ray-direction map from intrinsics.

        Args:
            intrinsics: (B, 4) normalized [fx/W, fy/H, cx/W, cy/H], or None.
            h, w: spatial size of the feature map to condition.
        """
        if intrinsics is None:
            intrinsics = torch.tensor(
                CANONICAL_INTRINSICS, device=device, dtype=dtype
            ).unsqueeze(0)
        intrinsics = intrinsics.to(device=device, dtype=dtype)
        b = intrinsics.size(0)
        fx, fy, cx, cy = intrinsics[:, 0], intrinsics[:, 1], intrinsics[:, 2], intrinsics[:, 3]

        # Pixel-center coordinates normalized to [0, 1], resolution-independent.
        u = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w  # (w,)
        v = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h  # (h,)
        vv, uu = torch.meshgrid(v, u, indexing='ij')  # (h, w) each
        uu = uu.unsqueeze(0).expand(b, h, w)
        vv = vv.unsqueeze(0).expand(b, h, w)

        ray_x = (uu - cx.view(b, 1, 1)) / fx.view(b, 1, 1)
        ray_y = (vv - cy.view(b, 1, 1)) / fy.view(b, 1, 1)
        return torch.stack([ray_x, ray_y], dim=1)  # (B, 2, h, w)

    def forward(self, x, intrinsics=None, reset_temporal=False):
        if reset_temporal:
            self.hidden_state = None

        features = self.encoder(x)

        # Apply ConvGRU at bottleneck (features[4] = 96ch @ 1/16)
        bottleneck = features[4]
        if self.camera_aware:
            ray_map = self._build_ray_map(
                intrinsics, bottleneck.size(2), bottleneck.size(3),
                bottleneck.device, bottleneck.dtype,
            )
            # Match the (possibly larger) batch of the feature map.
            if ray_map.size(0) != bottleneck.size(0):
                ray_map = ray_map.expand(bottleneck.size(0), -1, -1, -1)
            gru_input = torch.cat([bottleneck, ray_map], dim=1)
        else:
            gru_input = bottleneck

        self.hidden_state = self.temporal(gru_input, self.hidden_state)
        features[4] = self.hidden_state

        depth = self.decoder(features)
        return depth * self.max_depth

    def reset_temporal(self):
        """Reset hidden state (call at sequence boundaries)."""
        self.hidden_state = None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_encoder(self):
        """Freeze pretrained encoder for stage 1 training"""
        self.encoder.freeze()

    def unfreeze_encoder(self):
        """Unfreeze encoder for stage 2 fine-tuning"""
        self.encoder.unfreeze()

    def freeze_decoder(self):
        """Freeze decoder weights."""
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        """Unfreeze decoder weights."""
        for param in self.decoder.parameters():
            param.requires_grad = True
