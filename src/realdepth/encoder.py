"""
MobileNetV2 Encoder

Replaces custom ResidualBlock encoder with pretrained MobileNetV2 backbone
"""
import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV2Encoder(nn.Module):
    """MobileNetV2 encoder with multi-scale feature extraction

    Extracts features at 5 scales matching the V1 encoder interface:
      features[0]: 1/1  scale, 16 channels  (lightweight initial conv)
      features[1]: 1/2  scale, 16 channels  (from layers 0-1)
      features[2]: 1/4  scale, 24 channels  (from layers 2-3)
      features[3]: 1/8  scale, 32 channels  (from layers 4-6)
      features[4]: 1/16 scale, 96 channels  (from layers 7-13, bottleneck)

    We stop at 1/16 (96ch) instead of 1/32 (1280ch) to keep the
    bottleneck lightweight and match the V1 architecture depth

    MobileNetV2's first layer has stride=2, so we add a separate
    lightweight conv for the 1/1 scale skip connection
    """

    def __init__(self, pretrained=True):
        super().__init__()
        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        features = mobilenet.features  # 19 layers (0-18)

        # Full-resolution feature extractor (for skip connection at 1/1)
        self.initial = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )

        # MobileNetV2 layers grouped by output scale
        self.layer0 = features[0:2]   # 3->32->16, output at 1/2
        self.layer1 = features[2:4]   # 16->24, output at 1/4
        self.layer2 = features[4:7]   # 24->32, output at 1/8
        self.layer3 = features[7:14]  # 32->96, output at 1/16

    def forward(self, x):
        features = []

        f0 = self.initial(x)        # 1/1, 16ch
        features.append(f0)

        x = self.layer0(x)          # 1/2, 16ch
        features.append(x)

        x = self.layer1(x)          # 1/4, 24ch
        features.append(x)

        x = self.layer2(x)          # 1/8, 32ch
        features.append(x)

        x = self.layer3(x)          # 1/16, 96ch
        features.append(x)

        return features

    def freeze(self):
        """Freeze all pretrained MobileNetV2 layers (for stage 1 training)"""
        for layer in [self.layer0, self.layer1, self.layer2, self.layer3]:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all layers (for stage 2 fine-tuning)"""
        for param in self.parameters():
            param.requires_grad = True
