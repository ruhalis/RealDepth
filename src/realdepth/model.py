"""
RealDepth
"""
import torch
import torch.nn as nn

from .encoder import MobileNetV2Encoder
from .decoder import LightDecoder
from .conv_gru import ConvGRU


class DepthEstimationNet(nn.Module):
    """RealDepth: MobileNetV2 encoder + ConvGRU temporal fusion + lightweight decoder

    ConvGRU at the 1/16 scale bottleneck (96ch) propagates temporal context
    across consecutive video frames. Hidden state is stored on the model
    so the inference interface is just model(frame).
    """

    def __init__(self, max_depth=10.0, pretrained_encoder=True):
        super().__init__()
        self.max_depth = max_depth
        self.encoder = MobileNetV2Encoder(pretrained=pretrained_encoder)
        self.temporal = ConvGRU(input_channels=96, hidden_channels=96)
        self.decoder = LightDecoder()
        self.hidden_state = None

    def forward(self, x, reset_temporal=False):
        if reset_temporal:
            self.hidden_state = None

        features = self.encoder(x)

        # Apply ConvGRU at bottleneck (features[4] = 96ch @ 1/16)
        self.hidden_state = self.temporal(features[4], self.hidden_state)
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
