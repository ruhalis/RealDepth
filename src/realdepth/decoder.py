"""
Lightweight Decoder

Uses depthwise separable convolutions and nearest-neighbor upsampling
(NNConv5)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class LightDecoderBlock(nn.Module):
    """Lightweight decoder block: NN upsample + DWS conv + skip concat"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels + skip_channels, out_channels, 5, 1, 2),
            DepthwiseSeparableConv(out_channels, out_channels)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LightDecoder(nn.Module):
    """Lightweight decoder matching MobileNetV2 encoder channel dims

    Channel flow (from MobileNetV2Encoder):
      features[4]: 1/16, 96ch  (bottleneck)
      features[3]: 1/8,  32ch  (skip)
      features[2]: 1/4,  24ch  (skip)
      features[1]: 1/2,  16ch  (skip)
      features[0]: 1/1,  16ch  (skip)

    Decoder channel plan:
      dec4: 96+32 -> 64   (1/16 -> 1/8)
      dec3: 64+24 -> 32   (1/8 -> 1/4)
      dec2: 32+16 -> 16   (1/4 -> 1/2)
      dec1: 16+16 -> 16   (1/2 -> 1/1)
      final: 16 -> 1      (depth output)
    """

    def __init__(self):
        super().__init__()
        self.dec4 = LightDecoderBlock(96, 32, 64)
        self.dec3 = LightDecoderBlock(64, 24, 32)
        self.dec2 = LightDecoderBlock(32, 16, 16)
        self.dec1 = LightDecoderBlock(16, 16, 16)

        self.final = nn.Sequential(
            DepthwiseSeparableConv(16, 16),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        x = self.dec4(features[4], features[3])
        x = self.dec3(x, features[2])
        x = self.dec2(x, features[1])
        x = self.dec1(x, features[0])
        return self.final(x)
