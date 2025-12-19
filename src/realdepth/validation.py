"""
Validation utilities for depth estimation models
"""
import torch
from tqdm import tqdm

from .losses import DepthMetrics, format_stratified_metrics


def run_comprehensive_validation(model, loader, device, depth_thresholds=[3.0, 5.0, 10.0]):
    """
    comprehensive validation function with depth-stratified metrics
    """
    print("\nRunning comprehensive validation...")
    print(f"Depth thresholds: {depth_thresholds} meters")

    model.eval()

    # Accumulate predictions across all batches
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing metrics"):
            pred = model(batch['rgb'].to(device))
            # Move to CPU immediately to avoid GPU memory accumulation
            all_preds.append(pred.cpu())
            all_targets.append(batch['depth'])  # Keep on CPU
            all_masks.append(batch['mask'])  # Keep on CPU

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
