"""
RealDepth
"""
import torch
import torch.nn as nn

from .encoder import MobileNetV2Encoder
from .decoder import LightDecoder


class DepthEstimationNet(nn.Module):
    """RealDepth: MobileNetV2 encoder + lightweight decoder"""

    def __init__(self, max_depth=10.0, pretrained_encoder=True):
        super().__init__()
        self.max_depth = max_depth
        self.encoder = MobileNetV2Encoder(pretrained=pretrained_encoder)
        self.decoder = LightDecoder()

    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth * self.max_depth

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_encoder(self):
        """Freeze pretrained encoder for stage 1 training"""
        self.encoder.freeze()

    def unfreeze_encoder(self):
        """Unfreeze encoder for stage 2 fine-tuning"""
        self.encoder.unfreeze()
