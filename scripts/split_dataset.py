#!/usr/bin/env python3
"""
Split collected RealSense dataset into train/val/test sets.

Two modes:

--by-session (recommended): whole sessions are assigned to one split each,
so no two splits ever contain frames from the same capture. Sessions are
packed greedily (largest first, into whichever split is furthest below its
target) so the *frame* ratios land near 80/10/10 even when session sizes
differ wildly. Original frame numbers are preserved behind a per-session
offset, so genuine temporal adjacency inside a session survives the split.

default (sequence level): extracts all valid T-frame sequences, shuffles
them, then assigns to train/val/test. Temporal order is preserved within
each sequence, but sequences are shuffled across splits. Note this puts
near-duplicate frames from one capture into different splits, which leaks
between train and val/test.

Numbering gaps between sequences/sessions let
SequenceDataset._find_valid_sequences() detect boundaries.
"""
import argparse
import json
import os
import random
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm


# Gap inserted between sequences in the output numbering so that
# SequenceDataset._find_valid_sequences() detects the boundary.
SEQUENCE_GAP = 10

# Numbering stride between sessions in --by-session output. Must exceed any
# real frame number so session blocks never appear consecutive.
SESSION_STRIDE = 1_000_000


def find_all_sessions(input_dir: Path, exclude=()) -> list:
    """Find all session folders (timestamps) in the input directory.

    `exclude` holds folder names to skip -- e.g. a merged/ dataset that lives
    alongside the raw sessions and would otherwise look like one itself.
    """
    exclude = set(exclude)
    sessions = []
    for item in input_dir.iterdir():
        if item.name in exclude:
            continue
        if item.is_dir() and (item / "rgb").exists() and (item / "depth").exists():
            sessions.append(item)
    return sorted(sessions)


def get_image_pairs(session_dir: Path) -> list:
    """Get all RGB-depth image pairs from a session, in sorted order.

    Attaches per-frame camera intrinsics from the session's intrinsics.json
    (if present) so they can travel with the sequence into the split output.
    """
    rgb_dir = session_dir / "rgb"
    depth_dir = session_dir / "depth"

    intrinsics_meta = {}
    intrinsics_json = session_dir / "intrinsics.json"
    if intrinsics_json.exists():
        with open(intrinsics_json) as f:
            intrinsics_meta = json.load(f)

    pairs = []
    for rgb_file in sorted(rgb_dir.glob("*.png")):
        depth_file = depth_dir / rgb_file.name
        if depth_file.exists():
            pairs.append({
                'session': session_dir.name,
                'filename': rgb_file.name,
                'rgb_path': rgb_file,
                'depth_path': depth_file,
                'intrinsics': intrinsics_meta.get(rgb_file.stem),
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


def split_by_session(all_session_pairs, train_ratio, val_ratio, test_ratio, seed=42):
    """Assign whole sessions to splits, balancing on frame count.

    Largest sessions are placed first, each into whichever split is currently
    furthest below its frame-count target. This keeps the frame ratios close
    to the requested ones even though session sizes vary by >10x, while
    guaranteeing a session never straddles two splits.
    """
    sessions = [(d, p) for d, p in all_session_pairs if p]
    total = sum(len(p) for _, p in sessions)

    targets = {
        'train': total * train_ratio,
        'val': total * val_ratio,
        'test': total * test_ratio,
    }

    # Shuffle first so equal-sized sessions don't always land in the same
    # split, then place largest-first for good packing.
    rng = random.Random(seed)
    rng.shuffle(sessions)
    sessions.sort(key=lambda sp: len(sp[1]), reverse=True)

    splits = {'train': [], 'val': [], 'test': []}
    counts = {'train': 0, 'val': 0, 'test': 0}

    for session_dir, pairs in sessions:
        name = max(targets, key=lambda k: targets[k] - counts[k])
        splits[name].append((session_dir, pairs))
        counts[name] += len(pairs)

    # Emit each split in chronological session order.
    for name in splits:
        splits[name].sort(key=lambda sp: sp[0].name)

    return splits


def link_or_copy(src: Path, dst: Path):
    """Hardlink when possible (same filesystem, no extra disk), else copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_sessions(splits, output_dir):
    """Write a session-level split.

    Frames keep their original numbers behind a per-session offset, so real
    consecutive frames stay consecutive and session boundaries read as a
    huge gap to SequenceDataset._find_valid_sequences().
    """
    manifest = {}

    for split_name, session_list in splits.items():
        split_dir = output_dir / split_name
        rgb_dir = split_dir / "rgb"
        depth_dir = split_dir / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        total_frames = sum(len(p) for _, p in session_list)
        print(f"\nWriting {split_name} set ({len(session_list)} sessions, "
              f"{total_frames} frames)...")

        all_filenames = []
        split_intrinsics = {}
        manifest[split_name] = []

        for sess_i, (session_dir, pairs) in enumerate(
                tqdm(session_list, desc=split_name)):
            base = sess_i * SESSION_STRIDE

            for pair in pairs:
                src_stem = Path(pair['filename']).stem
                try:
                    frame_no = int(src_stem)
                except ValueError:
                    frame_no = len(all_filenames)

                stem = f"{base + frame_no:08d}"
                new_filename = f"{stem}.png"

                dst_rgb = rgb_dir / new_filename
                dst_depth = depth_dir / new_filename
                if dst_rgb.exists():
                    dst_rgb.unlink()
                if dst_depth.exists():
                    dst_depth.unlink()
                link_or_copy(pair['rgb_path'], dst_rgb)
                link_or_copy(pair['depth_path'], dst_depth)

                if pair.get('intrinsics') is not None:
                    split_intrinsics[stem] = pair['intrinsics']

                all_filenames.append(stem)

            manifest[split_name].append({
                'session_id': session_dir.name,
                'session_index': sess_i,
                'frames': len(pairs),
                'stem_offset': base,
            })

        with open(split_dir / "filenames.txt", 'w') as f:
            for name in all_filenames:
                f.write(f"{name}\n")

        if split_intrinsics:
            with open(split_dir / "intrinsics.json", 'w') as f:
                json.dump(split_intrinsics, f)
            print(f"  wrote intrinsics.json ({len(split_intrinsics)} frames)")

    with open(output_dir / "session_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote session_manifest.json (which session went to which split)")


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
        split_intrinsics = {}

        for seq_i, seq in enumerate(tqdm(seq_list, desc=split_name)):
            # Add gap between sequences (not before the first one)
            if seq_i > 0:
                current_idx += SEQUENCE_GAP

            for pair in seq:
                stem = f"{current_idx:06d}"
                new_filename = f"{stem}.png"

                shutil.copy2(pair['rgb_path'], rgb_dir / new_filename)
                shutil.copy2(pair['depth_path'], depth_dir / new_filename)

                if pair.get('intrinsics') is not None:
                    split_intrinsics[stem] = pair['intrinsics']

                all_filenames.append(stem)
                current_idx += 1

        # Write filenames index
        with open(split_dir / "filenames.txt", 'w') as f:
            for name in all_filenames:
                f.write(f"{name}\n")

        # Write per-frame intrinsics for this split (used by SequenceDataset)
        if split_intrinsics:
            with open(split_dir / "intrinsics.json", 'w') as f:
                json.dump(split_intrinsics, f)
            print(f"  wrote intrinsics.json ({len(split_intrinsics)} frames)")

    # Copy intrinsics
    if copy_intrinsics and copy_intrinsics.exists():
        shutil.copy2(copy_intrinsics, output_dir / "intrinsics.txt")
        print(f"\nCopied intrinsics.txt to {output_dir}")


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(project_root / 'collected_dataset'),
                        help='Directory containing session folders (each with rgb/ and depth/)')
    parser.add_argument('--output', default=str(project_root / 'dataset'),
                        help='Output directory for train/val/test splits')
    parser.add_argument('--config', default=str(project_root / 'configs' / 'realsense.yaml'),
                        help='Config YAML providing sequence_length')
    parser.add_argument('--sessions', nargs='+', default=None,
                        help='Only use these session folder names (default: all)')
    parser.add_argument('--exclude', nargs='+', default=['merged'],
                        help='Subfolder names to ignore when scanning for sessions')
    parser.add_argument('--by-session', action='store_true',
                        help='Assign whole sessions to a single split each '
                             '(no frames from one capture leak across splits)')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Overwrite an existing output directory without asking')
    args = parser.parse_args()

    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    SEED = 42

    # Load sequence_length from config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sequence_length = cfg.get('sequence_length', 3)

    # Validate ratios
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Warning: Ratios sum to {total_ratio}, normalizing...")
        TRAIN_RATIO /= total_ratio
        VAL_RATIO /= total_ratio
        TEST_RATIO /= total_ratio

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return

    # Find all sessions
    sessions = find_all_sessions(input_dir, exclude=args.exclude)
    if args.sessions:
        sessions = [s for s in sessions if s.name in args.sessions]
        missing = set(args.sessions) - {s.name for s in sessions}
        if missing:
            print(f"Error: Session(s) not found in '{input_dir}': {sorted(missing)}")
            return
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

    if args.by_session:
        splits = split_by_session(all_session_pairs, TRAIN_RATIO, VAL_RATIO,
                                  TEST_RATIO, SEED)

        print(f"\nSplit sizes (session level):")
        for name in ('train', 'val', 'test'):
            n_sess = len(splits[name])
            n_frames = sum(len(p) for _, p in splits[name])
            pct = n_frames / total_frames * 100
            print(f"  {name:5s}: {n_sess:3d} sessions, {n_frames:5d} frames "
                  f"({pct:.1f}%)")

        if output_dir.exists():
            if not args.yes:
                response = input(f"\nOutput directory '{output_dir}' already "
                                 f"exists. Overwrite? [y/N]: ")
                if response.lower() != 'y':
                    print("Aborted.")
                    return
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        copy_sessions(splits, output_dir)

        print(f"\nDataset created successfully at: {output_dir.absolute()}")
        print(f"Session stride in output numbering: {SESSION_STRIDE}")
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
