#!/usr/bin/env python3
"""
Split collected RealSense dataset into train/val/test sets.
"""
import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm


def find_all_sessions(input_dir: Path) -> list:
    """Find all session folders (timestamps) in the input directory."""
    sessions = []
    for item in input_dir.iterdir():
        if item.is_dir() and (item / "rgb").exists() and (item / "depth").exists():
            sessions.append(item)
    return sorted(sessions)

def get_image_pairs(session_dir: Path) -> list:
    """Get all RGB-depth image pairs from a session."""
    rgb_dir = session_dir / "rgb"
    depth_dir = session_dir / "depth"
    
    pairs = []
    for rgb_file in sorted(rgb_dir.glob("*.png")):
        depth_file = depth_dir / rgb_file.name
        if depth_file.exists():
            pairs.append({
                'session': session_dir.name,
                'filename': rgb_file.name,
                'rgb_path': rgb_file,
                'depth_path': depth_file
            })
    return pairs

def split_data(pairs: list, train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42):
    """Split data into train/val/test sets."""
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return {
        'train': shuffled[:train_end],
        'val': shuffled[train_end:val_end],
        'test': shuffled[val_end:]
    }

def copy_files(splits: dict, output_dir: Path, copy_intrinsics: Path = None):
    """Copy files to the new dataset structure."""
    
    for split_name, pairs in splits.items():
        split_dir = output_dir / split_name
        rgb_dir = split_dir / "rgb"
        depth_dir = split_dir / "depth"
        
        # Create directories
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying {split_name} set ({len(pairs)} samples)...")
        
        for i, pair in enumerate(tqdm(pairs, desc=split_name)):
            # Create new filename with zero-padded index
            new_filename = f"{i:06d}.png"
            
            # Copy RGB
            shutil.copy2(pair['rgb_path'], rgb_dir / new_filename)
            
            # Copy depth
            shutil.copy2(pair['depth_path'], depth_dir / new_filename)
        
        # Create split file listing all samples
        with open(split_dir / "filenames.txt", 'w') as f:
            for i in range(len(pairs)):
                f.write(f"{i:06d}\n")
    
    # Copy intrinsics to output directory
    if copy_intrinsics and copy_intrinsics.exists():
        shutil.copy2(copy_intrinsics, output_dir / "intrinsics.txt")
        print(f"\nCopied intrinsics.txt to {output_dir}")

def main():
    # Determine project root (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    INPUT_DIR = "collected_dataset"  # Input directory containing collected data
    OUTPUT_DIR = "dataset"           # Output directory for split dataset
    TRAIN_RATIO = 0.8                # Train split ratio
    VAL_RATIO = 0.1                  # Validation split ratio
    TEST_RATIO = 0.1                 # Test split ratio
    SEED = 42                        # Random seed for reproducibility

    # Validate ratios
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Warning: Ratios sum to {total_ratio}, normalizing...")
        TRAIN_RATIO /= total_ratio
        VAL_RATIO /= total_ratio
        TEST_RATIO /= total_ratio

    # Use project root for paths
    input_dir = project_root / INPUT_DIR
    output_dir = project_root / OUTPUT_DIR
    
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return
    
    # Find all sessions
    sessions = find_all_sessions(input_dir)
    if not sessions:
        print(f"Error: No valid sessions found in '{input_dir}'!")
        print("Expected structure: input_dir/session_name/rgb/*.png and depth/*.png")
        return
    
    print(f"Found {len(sessions)} session(s):")
    for s in sessions:
        print(f"  - {s.name}")
    
    # Collect all image pairs from all sessions
    all_pairs = []
    intrinsics_path = None
    
    for session in sessions:
        pairs = get_image_pairs(session)
        all_pairs.extend(pairs)
        
        # Get intrinsics from first session that has it
        if intrinsics_path is None:
            potential_intrinsics = session / "intrinsics.txt"
            if potential_intrinsics.exists():
                intrinsics_path = potential_intrinsics
    
    print(f"\nTotal image pairs: {len(all_pairs)}")
    
    if len(all_pairs) == 0:
        print("Error: No image pairs found!")
        return
    
    # Split data
    splits = split_data(all_pairs, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(splits['train'])} ({len(splits['train'])/len(all_pairs)*100:.1f}%)")
    print(f"  Val:   {len(splits['val'])} ({len(splits['val'])/len(all_pairs)*100:.1f}%)")
    print(f"  Test:  {len(splits['test'])} ({len(splits['test'])/len(all_pairs)*100:.1f}%)")
    
    # Create output directory
    if output_dir.exists():
        response = input(f"\nOutput directory '{output_dir}' already exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    copy_files(splits, output_dir, intrinsics_path)
    
    print(f"Dataset created successfully at: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
