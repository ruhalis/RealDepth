#!/usr/bin/env python3
"""
Split collected RealSense dataset into train/val/test sets.

Splits at the sequence level: extracts all valid T-frame sequences,
shuffles them, then assigns to train/val/test. Temporal order is
preserved within each sequence, but sequences are shuffled across splits
for better diversity. Numbering gaps between sequences let
SequenceDataset._find_valid_sequences() detect boundaries.
"""
import random
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm


# Gap inserted between sequences in the output numbering so that
# SequenceDataset._find_valid_sequences() detects the boundary.
SEQUENCE_GAP = 10


def find_all_sessions(input_dir: Path) -> list:
    """Find all session folders (timestamps) in the input directory."""
    sessions = []
    for item in input_dir.iterdir():
        if item.is_dir() and (item / "rgb").exists() and (item / "depth").exists():
            sessions.append(item)
    return sorted(sessions)


def get_image_pairs(session_dir: Path) -> list:
    """Get all RGB-depth image pairs from a session, in sorted order."""
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


def extract_sequences(all_session_pairs, sequence_length):
    """Extract all valid T-frame sequences from all sessions.

    Each sequence is a list of T consecutive frame pairs from
    the same session.

    Returns:
        list of sequences, where each sequence is a list of T pair dicts
    """
    sequences = []

    for session_dir, pairs in all_session_pairs:
        n = len(pairs)
        if n < sequence_length:
            continue

        # Extract frame numbers for gap detection
        frame_numbers = []
        for p in pairs:
            try:
                frame_numbers.append(int(Path(p['filename']).stem))
            except ValueError:
                frame_numbers.append(-1)

        # Find valid consecutive sequences
        for i in range(n - sequence_length + 1):
            valid = True
            for j in range(1, sequence_length):
                if (frame_numbers[i + j] - frame_numbers[i + j - 1]) != 1:
                    valid = False
                    break
            if valid:
                sequences.append(pairs[i:i + sequence_length])

    return sequences


def split_sequences(sequences, train_ratio, val_ratio, seed=42):
    """Shuffle sequences and assign to train/val/test splits."""
    random.seed(seed)

    shuffled = sequences.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        'train': shuffled[:train_end],
        'val': shuffled[train_end:val_end],
        'test': shuffled[val_end:],
    }


def copy_sequences(splits, output_dir, copy_intrinsics=None):
    """Copy sequences to output, with consecutive numbering within each
    sequence and a gap between sequences."""
    for split_name, seq_list in splits.items():
        split_dir = output_dir / split_name
        rgb_dir = split_dir / "rgb"
        depth_dir = split_dir / "depth"

        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        total_frames = sum(len(seq) for seq in seq_list)
        print(f"\nCopying {split_name} set ({len(seq_list)} sequences, "
              f"{total_frames} frames)...")

        current_idx = 0
        all_filenames = []

        for seq_i, seq in enumerate(tqdm(seq_list, desc=split_name)):
            # Add gap between sequences (not before the first one)
            if seq_i > 0:
                current_idx += SEQUENCE_GAP

            for pair in seq:
                new_filename = f"{current_idx:06d}.png"

                shutil.copy2(pair['rgb_path'], rgb_dir / new_filename)
                shutil.copy2(pair['depth_path'], depth_dir / new_filename)

                all_filenames.append(f"{current_idx:06d}")
                current_idx += 1

        # Write filenames index
        with open(split_dir / "filenames.txt", 'w') as f:
            for name in all_filenames:
                f.write(f"{name}\n")

    # Copy intrinsics
    if copy_intrinsics and copy_intrinsics.exists():
        shutil.copy2(copy_intrinsics, output_dir / "intrinsics.txt")
        print(f"\nCopied intrinsics.txt to {output_dir}")


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    INPUT_DIR = "collected_dataset"
    OUTPUT_DIR = "dataset"
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    SEED = 42

    # Load sequence_length from config
    config_path = project_root / 'configs' / 'realsense.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sequence_length = cfg.get('sequence_length', 3)

    # Validate ratios
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Warning: Ratios sum to {total_ratio}, normalizing...")
        TRAIN_RATIO /= total_ratio
        VAL_RATIO /= total_ratio
        TEST_RATIO /= total_ratio

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

    # Collect pairs per session
    all_session_pairs = []
    intrinsics_path = None
    total_frames = 0

    for session in sessions:
        pairs = get_image_pairs(session)
        count = len(pairs)
        all_session_pairs.append((session, pairs))
        total_frames += count
        print(f"  - {session.name}: {count} frames")

        if intrinsics_path is None:
            potential_intrinsics = session / "intrinsics.txt"
            if potential_intrinsics.exists():
                intrinsics_path = potential_intrinsics

    print(f"\nTotal frames: {total_frames}")

    if total_frames == 0:
        print("Error: No image pairs found!")
        return

    # Extract all valid sequences
    sequences = extract_sequences(all_session_pairs, sequence_length)
    print(f"Valid {sequence_length}-frame sequences: {len(sequences)}")

    if len(sequences) == 0:
        print("Error: No valid sequences found! Check that frames are numbered consecutively.")
        return

    # Shuffle and split sequences
    splits = split_sequences(sequences, TRAIN_RATIO, VAL_RATIO, SEED)

    # Print split info
    print(f"\nSplit sizes:")
    for name, seq_list in splits.items():
        n_seq = len(seq_list)
        n_frames = sum(len(s) for s in seq_list)
        pct = n_seq / len(sequences) * 100
        print(f"  {name:5s}: {n_seq:5d} sequences ({n_frames} frames, {pct:.1f}%)")

    # Create output directory
    if output_dir.exists():
        response = input(f"\nOutput directory '{output_dir}' already exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy sequences with gaps
    copy_sequences(splits, output_dir, intrinsics_path)

    print(f"\nDataset created successfully at: {output_dir.absolute()}")
    print(f"Sequence length: {sequence_length}, gap between sequences: {SEQUENCE_GAP}")


if __name__ == "__main__":
    main()
