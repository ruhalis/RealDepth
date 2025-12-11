"""
Dataset loader for RealDepth training with RealSense D435i camera data
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import torchvision.transforms as T

class RealSenseDataset(Dataset):
    """
    Custom RealSense dataset loader

    Expected directory structure (from split_dataset.py):
        data_dir/
            train/
                rgb/
                    00000.png
                    ...
                depth/
                    00000.png
                    ...
            val/
                rgb/
                depth/
            test/
                rgb/
                depth/
    """
    def __init__(self, data_dir, image_size=(480, 640), max_depth=10.0, depth_scale=0.001, split='train'):
        """
        Args:
            data_dir: Path to dataset directory
            image_size: (height, width) tuple
            max_depth: Maximum valid depth in meters
            depth_scale: Scale factor to convert depth to meters (default: 0.001 for mm to m)
            split: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.max_depth = max_depth
        self.depth_scale = depth_scale

        # Load from split directory structure
        split_dir = self.data_dir / split
        rgb_dir = split_dir / 'rgb'
        depth_dir = split_dir / 'depth'

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                f"Please use split_dataset.py to organize your data into train/val/test splits."
            )
        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
        if not depth_dir.exists():
            raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

        # Load file lists
        self.rgb_files = sorted(list(rgb_dir.glob('*.png')))
        self.depth_files = sorted(list(depth_dir.glob('*.png')))

        if len(self.rgb_files) == 0:
            raise ValueError(f"No RGB images found in {rgb_dir}")
        if len(self.depth_files) == 0:
            raise ValueError(f"No depth images found in {depth_dir}")
        if len(self.rgb_files) != len(self.depth_files):
            raise ValueError(f"Mismatched RGB ({len(self.rgb_files)}) and depth ({len(self.depth_files)}) file counts")

        print(f"RealSense {split}: {len(self.rgb_files)} samples from {rgb_dir}")

        # RGB transforms (ImageNet normalization)
        self.rgb_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Depth transforms
        self.depth_transform = T.Resize(image_size)

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # Load RGB
        rgb = Image.open(self.rgb_files[idx]).convert('RGB')
        rgb = self.rgb_transform(rgb)

        # Load depth (16-bit PNG)
        depth = Image.open(self.depth_files[idx])
        depth = np.array(depth, dtype=np.float32)

        # Apply depth scale (mm to meters)
        depth = depth * self.depth_scale

        # Resize
        depth = Image.fromarray(depth)
        depth = self.depth_transform(depth)
        depth = torch.from_numpy(np.array(depth)).unsqueeze(0).float()

        # Clamp to max_depth
        depth = torch.clamp(depth, 0, self.max_depth)

        # Create valid mask
        mask = ((depth > 0) & (depth <= self.max_depth)).float()

        return {
            'rgb': rgb,
            'depth': depth,
            'mask': mask
        }


def create_dataloaders(cfg):
    """
    Factory function to create train and validation dataloaders for RealSense dataset

    Args:
        cfg: Configuration dictionary with keys:
            - data_dir: Path to RealSense dataset
            - batch_size: Batch size
            - image_size: [height, width]
            - max_depth: Maximum depth in meters
            - num_workers: Number of data loading workers
            - depth_scale: Depth scale factor (optional, default: 0.001)

    Returns:
        (train_loader, val_loader): Tuple of DataLoader objects
    """
    # Validate required config
    if 'data_dir' not in cfg:
        raise ValueError("Config must contain 'data_dir' for RealSense dataset")

    # Extract parameters
    data_dir = cfg['data_dir']
    batch_size = cfg['batch_size']
    image_size = tuple(cfg['image_size'])
    max_depth = cfg['max_depth']
    num_workers = cfg.get('num_workers', 4)
    depth_scale = cfg.get('depth_scale', 0.001)

    # Create datasets
    train_dataset = RealSenseDataset(
        data_dir=data_dir,
        image_size=image_size,
        max_depth=max_depth,
        depth_scale=depth_scale,
        split='train'
    )
    val_dataset = RealSenseDataset(
        data_dir=data_dir,
        image_size=image_size,
        max_depth=max_depth,
        depth_scale=depth_scale,
        split='val'
    )

    # Create dataloaders
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader
