"""
Sequential frame dataset for temporal depth estimation training
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import torchvision.transforms as T


class SequenceDataset(Dataset):
    """Loads consecutive frames as sequences for temporal training

    Expected directory structure (same as RealSenseDataset):
        data_dir/
            {split}/
                rgb/
                    00000.png, 00001.png, ...
                depth/
                    00000.png, 00001.png, ...

    Detects sequence gaps via filename numbering — non-consecutive frame
    numbers indicate a new sequence (e.g., camera restart)

    Returns:
        'rgb':   (T, 3, H, W)  — T consecutive RGB frames
        'depth': (T, 1, H, W)  — T consecutive depth maps
        'mask':  (T, 1, H, W)  — T validity masks
    """

    def __init__(self, data_dir, sequence_length=3, image_size=(480, 640),
                 max_depth=10.0, depth_scale=0.001, split='train'):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.max_depth = max_depth
        self.depth_scale = depth_scale

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

        self.rgb_files = sorted(list(rgb_dir.glob('*.png')))
        self.depth_files = sorted(list(depth_dir.glob('*.png')))

        if len(self.rgb_files) == 0:
            raise ValueError(f"No RGB images found in {rgb_dir}")
        if len(self.rgb_files) != len(self.depth_files):
            raise ValueError(
                f"Mismatched RGB ({len(self.rgb_files)}) and depth ({len(self.depth_files)}) file counts"
            )

        # Extract frame numbers for gap detection
        self.frame_numbers = []
        for f in self.rgb_files:
            try:
                self.frame_numbers.append(int(f.stem))
            except ValueError:
                self.frame_numbers.append(-1)

        # Build valid sequence start indices
        self.sequence_starts = self._find_valid_sequences()
        print(f"SequenceDataset {split}: {len(self.sequence_starts)} sequences "
              f"(T={sequence_length}) from {len(self.rgb_files)} frames")

        # Transforms
        self.rgb_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.depth_transform = T.Resize(image_size)

    def _find_valid_sequences(self):
        """Find indices where a valid sequence of length T starts

        A valid sequence has consecutive frame numbers with no gaps.
        """
        starts = []
        n = len(self.rgb_files)
        T = self.sequence_length

        for i in range(n - T + 1):
            valid = True
            for j in range(1, T):
                # Check if frame numbers are consecutive
                if (self.frame_numbers[i + j] - self.frame_numbers[i + j - 1]) != 1:
                    valid = False
                    break
            if valid:
                starts.append(i)

        return starts

    def __len__(self):
        return len(self.sequence_starts)

    def _load_frame(self, idx):
        """Load a single RGB + depth frame"""
        # RGB
        rgb = Image.open(self.rgb_files[idx]).convert('RGB')
        rgb = self.rgb_transform(rgb)

        # Depth (16-bit PNG in mm)
        depth = Image.open(self.depth_files[idx])
        depth = np.array(depth, dtype=np.float32) * self.depth_scale
        depth = Image.fromarray(depth)
        depth = self.depth_transform(depth)
        depth = torch.from_numpy(np.array(depth)).unsqueeze(0).float()
        depth = torch.clamp(depth, 0, self.max_depth)

        # Validity mask
        mask = ((depth > 0) & (depth <= self.max_depth)).float()

        return rgb, depth, mask

    def __getitem__(self, idx):
        start = self.sequence_starts[idx]
        rgbs, depths, masks = [], [], []

        for t in range(self.sequence_length):
            rgb, depth, mask = self._load_frame(start + t)
            rgbs.append(rgb)
            depths.append(depth)
            masks.append(mask)

        return {
            'rgb': torch.stack(rgbs),      # (T, 3, H, W)
            'depth': torch.stack(depths),   # (T, 1, H, W)
            'mask': torch.stack(masks),     # (T, 1, H, W)
        }


def create_sequence_dataloaders(cfg):
    """Factory function to create sequential train/val dataloaders

    Args:
        cfg: Configuration dict with keys:
            - data_dir, batch_size, image_size, max_depth
            - sequence_length (default: 3)
            - num_workers, depth_scale (optional)

    Returns:
        (train_loader, val_loader)
    """
    data_dir = cfg['data_dir']
    batch_size = cfg['batch_size']
    image_size = tuple(cfg['image_size'])
    max_depth = cfg['max_depth']
    sequence_length = cfg.get('sequence_length', 3)
    num_workers = cfg.get('num_workers', 4)
    depth_scale = cfg.get('depth_scale', 0.001)

    train_dataset = SequenceDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        image_size=image_size,
        max_depth=max_depth,
        depth_scale=depth_scale,
        split='train',
    )
    val_dataset = SequenceDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        image_size=image_size,
        max_depth=max_depth,
        depth_scale=depth_scale,
        split='val',
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader
