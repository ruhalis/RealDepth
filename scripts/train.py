"""
RealDepth Training - python scripts/train.py
Config: configs/realsense.yaml (edit to customize settings)

Training strategy:
    Stage 1 (freeze_encoder_epochs): Freeze MobileNetV2 encoder, train decoder only
    Stage 2 (remaining epochs): Unfreeze all, train end-to-end with lower encoder LR
"""
import os
import sys
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from realdepth.model_utils import get_model, count_params, DepthLoss
from realdepth.losses import DepthMetrics, format_depth_metric
from realdepth.depth_datasets import create_dataloaders
from realdepth.plot import save_loss_plots, save_component_plots


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Dirs
        self.exp_dir = Path(cfg['exp_dir']) / cfg['exp_name']
        self.ckpt_dir = self.exp_dir / 'checkpoints'
        self.vis_dir = self.exp_dir / 'visualizations'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        # Model
        self.model = get_model(cfg['model'], max_depth=cfg['max_depth']).to(self.device)
        print(f"Model: {cfg['model']} ({count_params(self.model):,} params)")

        # Loss
        self.criterion = DepthLoss(
            w_l1=cfg.get('w_l1', 1.0),
            w_si=cfg.get('w_si', 0.5),
            w_grad=cfg.get('w_grad', 0.5),
            w_ssim=cfg.get('w_ssim', 0.1),
            w_berhu=cfg.get('w_berhu', 0.1)
        ).to(self.device)

        # Optimizer with separate LR for encoder
        self.freeze_encoder_epochs = cfg.get('freeze_encoder_epochs', 20)
        encoder_lr = cfg.get('encoder_lr', cfg['lr'] * 0.1)
        self.optimizer = optim.AdamW([
            {'params': self.model.encoder.parameters(), 'lr': encoder_lr},
            {'params': self.model.decoder.parameters(), 'lr': cfg['lr']},
        ], weight_decay=cfg.get('weight_decay', 0.0001))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, cfg['epochs']
        )

        # Data
        self.train_loader, self.val_loader = create_dataloaders(cfg)

        # Logging
        self.writer = SummaryWriter(self.exp_dir / 'logs')
        self.epoch, self.best_loss, self.step = 0, float('inf'), 0
        self.train_losses = []
        self.val_losses = []
        self.train_loss_components = {
            'l1': [], 'scale_inv': [], 'gradient': [],
            'ssim': [], 'berhu': []
        }

    def train(self):
        # Stage 1: freeze encoder
        if self.freeze_encoder_epochs > 0:
            print(f"\n--- Stage 1: Frozen encoder for {self.freeze_encoder_epochs} epochs ---")
            self.model.freeze_encoder()

        for epoch in range(self.cfg['epochs']):
            self.epoch = epoch

            # Stage 2: unfreeze encoder
            if epoch == self.freeze_encoder_epochs:
                print(f"\n--- Stage 2: Unfreezing encoder at epoch {epoch} ---")
                self.model.unfreeze_encoder()

            # Train
            self.model.train()
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
                rgb = batch['rgb'].to(self.device)
                depth = batch['depth'].to(self.device)
                mask = batch['mask'].to(self.device)

                pred = self.model(rgb)
                loss, loss_dict = self.criterion(pred, depth, mask)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.step += 1

                if self.step % 10 == 0:
                    self.writer.add_scalar('train/loss', loss.item(), self.step)
                    self.writer.add_scalar('train/lr_decoder', self.optimizer.param_groups[1]['lr'], self.step)
                    self.writer.add_scalar('train/lr_encoder', self.optimizer.param_groups[0]['lr'], self.step)
                    self.train_losses.append((self.step, loss.item()))

                    for name, value in loss_dict.items():
                        if name != 'total' and isinstance(value, torch.Tensor):
                            self.writer.add_scalar(f'train/loss_{name}', value.item(), self.step)
                            if name in self.train_loss_components:
                                self.train_loss_components[name].append((self.step, value.item()))

            # Validate
            val_loss, metrics = self._validate()
            self.val_losses.append((epoch, val_loss))
            self.scheduler.step()

            self.writer.add_scalar('val/loss', val_loss, epoch)
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f'val/{metric_name}', metric_value, epoch)

            print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, AbsRel={metrics['abs_rel']:.4f}, "
                  f"d1={metrics['delta1']:.4f}")
            print(f"  MAE={format_depth_metric(metrics['mae'])}, "
                  f"RMSE={format_depth_metric(metrics['rmse'])}")

            # Save visualizations every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._save_visualizations()

            # Save best
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(
                    {'model': self.model.state_dict(), 'config': self.cfg},
                    self.ckpt_dir / 'best.pth'
                )

        print(f"\nDone! Best model: {self.ckpt_dir / 'best.pth'}")

        # Save plots
        save_loss_plots(
            self.train_losses, self.val_losses,
            len(self.train_loader), self.vis_dir / 'training_loss.png'
        )
        save_component_plots(
            self.train_loss_components, self.vis_dir / 'loss_components.png'
        )

        # Comprehensive validation
        self._comprehensive_validate()

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        losses, all_metrics = [], []
        for batch in self.val_loader:
            rgb = batch['rgb'].to(self.device)
            depth_gt = batch['depth'].to(self.device)
            mask = batch['mask'].to(self.device)

            pred = self.model(rgb)
            loss, _ = self.criterion(pred, depth_gt, mask)

            losses.append(loss.item())
            all_metrics.append(DepthMetrics.compute(pred, depth_gt))

        return np.mean(losses), {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

    @torch.no_grad()
    def _comprehensive_validate(self):
        from realdepth.validation import run_comprehensive_validation
        from realdepth.losses import format_stratified_metrics
        import json

        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION WITH DEPTH-STRATIFIED METRICS")
        print("="*80)

        best_checkpoint = self.ckpt_dir / 'best.pth'
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])

        depth_thresholds = [3.0, 5.0, 10.0]
        flat_metrics, stratified_results = run_comprehensive_validation(
            self.model, self.val_loader, self.device, depth_thresholds
        )

        print(format_stratified_metrics(stratified_results))

        results_path = self.exp_dir / 'stratified_validation_results.json'
        with open(results_path, 'w') as f:
            json.dump(flat_metrics, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        for metric_name, metric_value in flat_metrics.items():
            self.writer.add_scalar(f'comprehensive_val/{metric_name}', metric_value, self.epoch)

    @torch.no_grad()
    def _save_visualizations(self, num_samples=4):
        """Save RGB | Ground Truth | Prediction grid"""
        import torchvision.utils as vutils

        self.model.eval()
        batch = next(iter(self.val_loader))
        actual_samples = min(num_samples, batch['rgb'].shape[0])
        rgb = batch['rgb'][:actual_samples].to(self.device)
        depth_gt = batch['depth'][:actual_samples].to(self.device)
        depth_pred = self.model(rgb)

        depth_gt_vis = depth_gt / self.cfg['max_depth']
        depth_pred_vis = depth_pred / self.cfg['max_depth']

        depth_gt_rgb = depth_gt_vis.repeat(1, 3, 1, 1)
        depth_pred_rgb = depth_pred_vis.repeat(1, 3, 1, 1)

        # Grid: RGB | GT | Pred
        vis_samples = []
        for i in range(actual_samples):
            vis_samples.extend([rgb[i], depth_gt_rgb[i], depth_pred_rgb[i]])

        grid = vutils.make_grid(vis_samples, nrow=3, normalize=True, padding=2)
        vutils.save_image(grid, self.vis_dir / f'epoch_{self.epoch:03d}.png')
        self.writer.add_image('validation/samples', grid, self.epoch)


def main():
    parser = argparse.ArgumentParser(description='RealDepth Training')
    parser.add_argument('--config', default='configs/realsense.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault('exp_name', 'realsense_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    cfg.setdefault('exp_dir', './experiments')
    cfg.setdefault('w_l1', 1.0)
    cfg.setdefault('w_si', 0.5)
    cfg.setdefault('w_grad', 0.5)
    cfg.setdefault('w_ssim', 0.1)
    cfg.setdefault('w_berhu', 0.1)

    Trainer(cfg).train()


if __name__ == "__main__":
    main()
