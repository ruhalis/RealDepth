"""
2D Camera Depth Estimation
A U-Net style encoder-decoder architecture for predicting depth from RGB images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic convolutional block: Conv -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection with 1x1 conv if dimensions change
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class Encoder(nn.Module):
    """
    Encoder
    1/32 of input resolution
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        self.initial = nn.Sequential(
            ConvBlock(in_channels, base_channels, 7, 1, 3),
            ConvBlock(base_channels, base_channels)
        )
        
        self.enc1 = self._make_layer(base_channels, base_channels * 2, stride=2)      # 1/2
        self.enc2 = self._make_layer(base_channels * 2, base_channels * 4, stride=2)  # 1/4
        self.enc3 = self._make_layer(base_channels * 4, base_channels * 8, stride=2)  # 1/8
        self.enc4 = self._make_layer(base_channels * 8, base_channels * 16, stride=2) # 1/16
        
    def _make_layer(self, in_ch, out_ch, stride):
        return nn.Sequential(
            ResidualBlock(in_ch, out_ch, stride),
            ResidualBlock(out_ch, out_ch)
        )
    
    def forward(self, x):
        features = []
        
        x = self.initial(x)
        features.append(x)  # 1/1 scale
        
        x = self.enc1(x)
        features.append(x)  # 1/2 scale
        
        x = self.enc2(x)
        features.append(x)  # 1/4 scale
        
        x = self.enc3(x)
        features.append(x)  # 1/8 scale
        
        x = self.enc4(x)
        features.append(x)  # 1/16 scale (bottleneck)
        
        return features


class DecoderBlock(nn.Module):
    """
    Decoder block: Upsample + Concatenate skip + Convolve
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        # 2x upsampling
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        
        # concatenation with skip connection
        self.conv = nn.Sequential(
            ConvBlock(in_channels // 2 + skip_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    """
    Decoder: Reconstructs depth map from encoder features using skip connections
    """
    def __init__(self, base_channels=64):
        super().__init__()
        
        # decoder blocks
        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, base_channels)
        
        # final output layer
        self.final = nn.Sequential(
            ConvBlock(base_channels, base_channels // 2),
            nn.Conv2d(base_channels // 2, 1, 3, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range, scale to actual depth later
        )
    
    def forward(self, features):
        # f0: 1/1, f1: 1/2, f2: 1/4, f3: 1/8, f4: 1/16
        x = self.dec4(features[4], features[3])  # 1/16 -> 1/8
        x = self.dec3(x, features[2])             # 1/8 -> 1/4
        x = self.dec2(x, features[1])             # 1/4 -> 1/2
        x = self.dec1(x, features[0])             # 1/2 -> 1/1
        
        return self.final(x)


class DepthEstimationNet(nn.Module):
    """
    RGB Depth Estimation
    """
    def __init__(self, in_channels=3, base_channels=64, max_depth=10.0):
        super().__init__()
        self.max_depth = max_depth
        self.encoder = Encoder(in_channels, base_channels)
        self.decoder = Decoder(base_channels)
    
    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth * self.max_depth  # Scale to actual depth range
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)