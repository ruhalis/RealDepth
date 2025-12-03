"""
Dataset loaders for RealDepth training
Supports: NYU Depth V2 (HuggingFace), RealSense custom data
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import torchvision.transforms as T
from datasets import load_dataset


class NYUDepthV2Dataset(Dataset):
    """
    NYU Depth V2 dataset from HuggingFace
    """
    def __init__(self, split='train', image_size=(480, 640), max_depth=10.0):
        """
        Args:
            split: 'train' or 'test'
            image_size: (height, width) tuple
            max_depth: Maximum valid depth in meters
        """
        self.split = split
        self.image_size = image_size
        self.max_depth = max_depth

        # Load from HuggingFace
        print(f"Loading NYU Depth V2 {split} split from HuggingFace...")
        self.dataset = load_dataset('sayakpaul/nyu_depth_v2', split=split)
        print(f"Loaded {len(self.dataset)} samples")

        # RGB transforms (ImageNet normalization)
        self.rgb_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Depth transforms
        self.depth_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Load images
        rgb = sample['image'].convert('RGB')
        depth = sample['depth'].convert('L')  # Grayscale

        # Apply transforms
        rgb = self.rgb_transform(rgb)
        depth = self.depth_transform(depth)

        # NYU depth is stored in millimeters, convert to meters
        depth = depth.float() / 1000.0

        # Clamp to max_depth
        depth = torch.clamp(depth, 0, self.max_depth)

        # Create valid mask
        mask = ((depth > 0) & (depth <= self.max_depth)).float()

        return {
            'rgb': rgb,
            'depth': depth,
            'mask': mask
        }


class RealSenseDataset(Dataset):
    """
    Custom RealSense dataset loader

    Supports two directory structures:

    1. Split structure (preferred, from split_dataset.py):
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

    2. Flat structure (legacy, auto-splits by ratio):
        data_dir/
            rgb/
                00000.png
                ...
            depth/
                00000.png
                ...
    """
    def __init__(self, data_dir, image_size=(480, 640), max_depth=10.0, depth_scale=0.001, split='train', train_ratio=0.9):
        """
        Args:
            data_dir: Path to dataset directory
            image_size: (height, width) tuple
            max_depth: Maximum valid depth in meters
            depth_scale: Scale factor to convert depth to meters (default: 0.001 for mm to m)
            split: 'train' or 'val'
            train_ratio: Ratio of training samples (default: 0.9, only used if split dirs don't exist)
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.max_depth = max_depth
        self.depth_scale = depth_scale

        # Check if split directory structure exists (dataset/train/rgb, dataset/val/rgb)
        # If so, use it. Otherwise fall back to flat structure with manual splitting
        split_dir = self.data_dir / split
        if split_dir.exists() and (split_dir / 'rgb').exists():
            # Use split directory structure
            rgb_dir = split_dir / 'rgb'
            depth_dir = split_dir / 'depth'
            use_split_dirs = True
        else:
            # Use flat directory structure
            rgb_dir = self.data_dir / 'rgb'
            depth_dir = self.data_dir / 'depth'
            use_split_dirs = False

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

        # Only apply train/val split if using flat structure
        if not use_split_dirs:
            num_samples = len(self.rgb_files)
            num_train = int(num_samples * train_ratio)

            if split == 'train':
                self.rgb_files = self.rgb_files[:num_train]
                self.depth_files = self.depth_files[:num_train]
            else:  # val
                self.rgb_files = self.rgb_files[num_train:]
                self.depth_files = self.depth_files[num_train:]

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
    Factory function to create train and validation dataloaders

    Args:
        cfg: Configuration dictionary with keys:
            - dataset: 'nyu' or 'realsense'
            - data_dir: Path to data (required for realsense)
            - batch_size: Batch size
            - image_size: [height, width]
            - max_depth: Maximum depth in meters
            - num_workers: Number of data loading workers
            - depth_scale: Depth scale factor (optional, for realsense, default: 0.001)

    Returns:
        (train_loader, val_loader): Tuple of DataLoader objects
    """
    dataset_type = cfg['dataset']
    batch_size = cfg['batch_size']
    image_size = tuple(cfg['image_size'])
    max_depth = cfg['max_depth']
    num_workers = cfg.get('num_workers', 4)

    if dataset_type == 'nyu':
        # NYU Depth V2 from HuggingFace
        train_dataset = NYUDepthV2Dataset(
            split='train',
            image_size=image_size,
            max_depth=max_depth
        )
        val_dataset = NYUDepthV2Dataset(
            split='test',
            image_size=image_size,
            max_depth=max_depth
        )

    elif dataset_type == 'realsense':
        # Custom RealSense data
        if 'data_dir' not in cfg:
            raise ValueError("'data_dir' must be specified in config for realsense dataset")

        data_dir = cfg['data_dir']
        depth_scale = cfg.get('depth_scale', 0.001)

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

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Choose 'nyu' or 'realsense'")

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
