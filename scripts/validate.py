"""
Model Validation with Depth-Stratified Metrics
"""
import sys
import argparse
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from realdepth.model_utils import get_model
from realdepth.losses import DepthMetrics, format_stratified_metrics
from realdepth.depth_datasets import create_dataloaders, RealSenseDataset

def setup_device(device_arg):
    """
    Select device: cuda/cpu/auto
    """
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return device

def load_checkpoint(checkpoint_path, config_path, device):
    """
    Load model checkpoint and configuration
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'config' not in checkpoint or 'model' not in checkpoint:
        raise ValueError("Invalid checkpoint format. Expected keys: 'model', 'config'")

    # Load config from checkpoint or external file
    if config_path:
        import yaml
        print(f"Loading config from: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint['config']
        print("Using config from checkpoint")

    model_name = config.get('model', 'realdepth_resnet')
    max_depth = config.get('max_depth', 10.0)

    # Create model
    model = get_model(model_name, max_depth=max_depth)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    print(f"Model: {model_name}, Max Depth: {max_depth}m")

    return model, config


def create_validation_loader(config, split='val'):
    """
    Create dataloader for RealSense validation or test set

    Args:
        config: Configuration dict with 'data_dir', 'image_size', 'max_depth', etc.
        split: Dataset split ('val' or 'test')

    Returns:
        DataLoader for specified split
    """
    batch_size = config.get('batch_size', 4)
    num_workers = config.get('num_workers', 4)

    print(f"\nCreating {split} dataloader for RealSense dataset")

    # Validate required config
    if 'data_dir' not in config:
        raise ValueError("Config must contain 'data_dir' for RealSense dataset")

    # Create RealSense dataset
    dataset = RealSenseDataset(
        data_dir=config['data_dir'],
        image_size=tuple(config['image_size']),
        max_depth=config['max_depth'],
        depth_scale=config.get('depth_scale', 0.001),
        split=split
    )

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batch size: {batch_size}, Batches: {len(loader)}")

    return loader


@torch.no_grad()
def validate_model(model, loader, device, depth_thresholds):
    """
    Run comprehensive validation with depth-stratified metrics
    """
    print("\nRunning comprehensive validation...")
    print(f"Depth thresholds: {depth_thresholds} meters")

    model.eval()

    # Accumulate predictions across all batches
    all_preds = []
    all_targets = []
    all_masks = []

    for batch in tqdm(loader, desc="Computing metrics"):
        pred = model(batch['rgb'].to(device))
        all_preds.append(pred)
        all_targets.append(batch['depth'].to(device))
        all_masks.append(batch['mask'].to(device))

    # Concatenate all batches
    print("Concatenating batches...")
    pred = torch.cat(all_preds, dim=0)
    target = torch.cat(all_targets, dim=0)
    mask = torch.cat(all_masks, dim=0)

    # Compute stratified metrics
    print("Computing depth-stratified metrics...")
    stratified_results = DepthMetrics.compute_stratified(
        pred, target, depth_thresholds, mask
    )

    # Flatten the nested dict for easy access
    flat_metrics = {}

    # Overall metrics (no suffix)
    for k, v in stratified_results['overall'].items():
        flat_metrics[k] = v

    # Stratified metrics (with suffix)
    for threshold in depth_thresholds:
        depth_key = f"{int(threshold)}m"
        if depth_key in stratified_results:
            for metric_name, metric_value in stratified_results[depth_key].items():
                flat_metrics[f"{metric_name}_{depth_key}"] = metric_value

    return flat_metrics, stratified_results


def save_results(flat_metrics, stratified_results, output_dir, checkpoint_path):
    """
    Save validation results to JSON file and print formatted table
    """
    # Print formatted table
    print(format_stratified_metrics(stratified_results))

    # Determine output directory
    if output_dir:
        output_dir = Path(output_dir)
    else:
        # Use checkpoint directory as default
        output_dir = Path(checkpoint_path).parent.parent  # Go up to experiment dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    results_path = output_dir / 'stratified_validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(flat_metrics, f, indent=2)

    print(f"\nResults saved to: {results_path}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='RealDepth - Comprehensive Model Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate.py --checkpoint experiments/exp1/checkpoints/best.pth
  python validate.py -c best.pth --split test
  python validate.py -c best.pth --config configs/realsense.yaml
  python validate.py -c best.pth --thresholds 2.0 4.0 6.0 10.0
  python validate.py -c best.pth -o ./validation_results --device cpu
        """
    )

    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional, can extract from checkpoint)')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='val',
                       help='Dataset split to validate on (default: val)')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[3.0, 5.0, 10.0],
                       help='Depth thresholds in meters (default: 3.0 5.0 10.0)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory for results (default: same as checkpoint)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'], default='auto',
                       help='Device: cuda/cpu/auto (default: auto)')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    print("="*80)
    print("RealDepth - Comprehensive Model Validation")
    print("="*80)

    # Device selection
    device = setup_device(args.device)

    # Load checkpoint and model
    model, config = load_checkpoint(args.checkpoint, args.config, device)

    # Create validation loader
    try:
        loader = create_validation_loader(config, args.split)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"\nNote: For RealSense dataset, ensure your data is in the correct structure:")
        print(f"  {config.get('data_dir', 'data_dir')}/{{train,val,test}}/{{rgb,depth}}/")
        sys.exit(1)

    # Run validation
    flat_metrics, stratified_results = validate_model(
        model, loader, device, args.thresholds
    )

    # Save results
    save_results(flat_metrics, stratified_results, args.output_dir, args.checkpoint)

    print("\nValidation complete!")


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except RuntimeError as e:
        if 'CUDA' in str(e):
            print(f"\nCUDA Error: {e}")
            print("Try running with --device cpu")
        else:
            print(f"\nRuntime Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
