"""
RealDepth Training - python train.py
Config: configs/realsense.yaml (edit to customize settings)
"""
import os, sys, time, yaml, numpy as np
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from realdepth.model_utils import get_model, count_params, DepthLoss
from realdepth.losses import DepthMetrics, format_depth_metric
from realdepth.depth_datasets import create_dataloaders

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dirs
        self.exp_dir = Path(cfg['exp_dir']) / cfg['exp_name']
        self.ckpt_dir = self.exp_dir / 'checkpoints'
        self.vis_dir = self.exp_dir / 'visualizations'
        (self.ckpt_dir).mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Model
        self.model = get_model(cfg['model'], max_depth=cfg['max_depth']).to(self.device)
        print(f"Model: {cfg['model']} ({count_params(self.model):,} params)")
        
        # Loss, optimizer, scheduler
        self.criterion = DepthLoss(
            w_l1=cfg.get('w_l1', 1.0),
            w_si=cfg.get('w_si', 0.5),
            w_grad=cfg.get('w_grad', 0.5),
            w_ssim=cfg.get('w_ssim', 0.1),
            w_berhu=cfg.get('w_berhu', 0.1)
        ).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg['lr'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, cfg['epochs'])
        
        # Data
        self.train_loader, self.val_loader = create_dataloaders(cfg)
        
        # Logging
        self.writer = SummaryWriter(self.exp_dir / 'logs')
        self.epoch, self.best_loss, self.step = 0, float('inf'), 0
    
    def train(self):
        for epoch in range(self.cfg['epochs']):
            self.epoch = epoch
            
            # Train
            self.model.train()
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
                rgb = batch['rgb'].to(self.device)
                depth = batch['depth'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                pred = self.model(rgb)
                loss, _ = self.criterion(pred, depth, mask)
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.step += 1

                # Log training metrics every 10 batches
                if self.step % 10 == 0:
                    self.writer.add_scalar('train/loss', loss.item(), self.step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.step)
            
            # Validate
            val_loss, metrics = self._validate()
            self.scheduler.step()

            # Log validation metrics
            self.writer.add_scalar('val/loss', val_loss, epoch)
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f'val/{metric_name}', metric_value, epoch)

            print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, AbsRel={metrics['abs_rel']:.4f}, δ1={metrics['delta1']:.4f}")
            print(f"  Absolute Errors: MAE={format_depth_metric(metrics['mae'])}, "
                  f"Median={format_depth_metric(metrics['mae_median'])}, "
                  f"RMSE={format_depth_metric(metrics['rmse'])}")

            # Save visualizations every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._save_visualizations()

            # Save
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save({'model': self.model.state_dict(), 'config': self.cfg},
                          self.ckpt_dir / 'best.pth')

        print(f"\nDone! Best model: {self.ckpt_dir / 'best.pth'}")

        # Run comprehensive validation with depth-stratified metrics
        self._comprehensive_validate()
    
    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        losses, all_metrics = [], []
        for batch in self.val_loader:
            pred = self.model(batch['rgb'].to(self.device))
            loss, _ = self.criterion(pred, batch['depth'].to(self.device), batch['mask'].to(self.device))
            losses.append(loss.item())
            all_metrics.append(DepthMetrics.compute(pred, batch['depth'].to(self.device)))
        return np.mean(losses), {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

    @torch.no_grad()
    def _comprehensive_validate(self):
        """
        Comprehensive validation with depth-stratified metrics.
        """
        from realdepth.validation import run_comprehensive_validation
        from realdepth.losses import format_stratified_metrics
        import json

        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE VALIDATION WITH DEPTH-STRATIFIED METRICS")
        print("="*80)

        # Load best checkpoint
        best_checkpoint = self.ckpt_dir / 'best.pth'
        if best_checkpoint.exists():
            print(f"Loading best checkpoint: {best_checkpoint}")
            checkpoint = torch.load(best_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
        else:
            print("Warning: No best checkpoint found. Using current model state.")

        print("Evaluating on VALIDATION set")
        print(f"Total batches: {len(self.val_loader)}")

        # Run comprehensive validation using shared function
        depth_thresholds = [3.0, 5.0, 10.0]
        flat_metrics, stratified_results = run_comprehensive_validation(
            self.model, self.val_loader, self.device, depth_thresholds
        )

        # Print formatted table
        print(format_stratified_metrics(stratified_results))

        # Save to JSON
        results_path = self.exp_dir / 'stratified_validation_results.json'
        with open(results_path, 'w') as f:
            json.dump(flat_metrics, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        # Log to TensorBoard
        for metric_name, metric_value in flat_metrics.items():
            self.writer.add_scalar(f'comprehensive_val/{metric_name}', metric_value, self.epoch)

        return flat_metrics

    @torch.no_grad()
    def _save_visualizations(self, num_samples=4):
        """Save RGB | Ground Truth | Prediction grid"""
        import torchvision.utils as vutils

        self.model.eval()
        batch = next(iter(self.val_loader))
        # Use min to handle cases where batch size < num_samples
        actual_samples = min(num_samples, batch['rgb'].shape[0])
        rgb = batch['rgb'][:actual_samples].to(self.device)
        depth_gt = batch['depth'][:actual_samples].to(self.device)
        depth_pred = self.model(rgb)

        # Normalize to 0-1 for visualization
        depth_gt_vis = depth_gt / self.cfg['max_depth']
        depth_pred_vis = depth_pred / self.cfg['max_depth']

        # Convert grayscale to RGB
        depth_gt_rgb = depth_gt_vis.repeat(1, 3, 1, 1)
        depth_pred_rgb = depth_pred_vis.repeat(1, 3, 1, 1)

        # Create grid: RGB | GT | Pred
        vis_samples = []
        for i in range(actual_samples):
            vis_samples.extend([rgb[i], depth_gt_rgb[i], depth_pred_rgb[i]])

        grid = vutils.make_grid(vis_samples, nrow=3, normalize=True, padding=2)
        vutils.save_image(grid, self.vis_dir / f'epoch_{self.epoch:03d}.png')
        self.writer.add_image('validation/samples', grid, self.epoch)

if __name__ == "__main__":
    config_path = 'configs/realsense.yaml'

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault('exp_name', datetime.now().strftime('%Y%m%d_%H%M%S'))
    cfg.setdefault('exp_dir', './experiments')
    cfg.setdefault('w_l1', 1.0)
    cfg.setdefault('w_si', 0.5)
    cfg.setdefault('w_grad', 0.5)
    cfg.setdefault('w_ssim', 0.1)

    Trainer(cfg).train()