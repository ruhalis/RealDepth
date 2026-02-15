"""
RealDepth Training - python scripts/train.py
Config: configs/realsense.yaml (edit to customize settings)

Three-stage training strategy:
    Stage 1 (freeze_all_epochs): Freeze encoder + decoder, train only ConvGRU
    Stage 2 (freeze_encoder_epochs): Freeze encoder, train ConvGRU + decoder
    Stage 3 (remaining epochs): Unfreeze all, end-to-end fine-tuning with lower encoder LR

Optionally loads a pretrained checkpoint to initialize encoder + decoder weights.
Temporal consistency loss penalizes flicker between consecutive predictions.
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
from realdepth.losses import DepthMetrics, TemporalConsistencyLoss, format_depth_metric
from realdepth.sequence_dataset import create_sequence_dataloaders
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

        # Load pretrained checkpoint if provided (encoder + decoder weights)
        pretrained_path = cfg.get('pretrained_checkpoint')
        if pretrained_path and Path(pretrained_path).exists():
            print(f"Loading pretrained checkpoint: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            pretrained_state = checkpoint['model']

            # Load matching keys, skip any new/changed layers
            model_state = self.model.state_dict()
            loaded_keys = []
            for k, v in pretrained_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
                    loaded_keys.append(k)
            self.model.load_state_dict(model_state)
            print(f"Loaded {len(loaded_keys)}/{len(model_state)} keys from pretrained checkpoint")
        elif pretrained_path:
            print(f"WARNING: Pretrained checkpoint not found: {pretrained_path}")
            print("Training from scratch (encoder still uses ImageNet weights)")

        print(f"Model: {cfg['model']} ({count_params(self.model):,} params)")

        # Losses
        self.criterion = DepthLoss(
            w_l1=cfg.get('w_l1', 1.0),
            w_si=cfg.get('w_si', 0.5),
            w_grad=cfg.get('w_grad', 0.5),
            w_ssim=cfg.get('w_ssim', 0.1),
            w_berhu=cfg.get('w_berhu', 0.1)
        ).to(self.device)
        self.temporal_criterion = TemporalConsistencyLoss().to(self.device)
        self.w_temporal = cfg.get('w_temporal', 0.5)

        # Stage boundaries
        self.freeze_all_epochs = cfg.get('freeze_all_epochs', 10)
        self.freeze_encoder_epochs = cfg.get('freeze_encoder_epochs', 30)

        # Optimizer with separate LR for encoder, decoder, and ConvGRU
        encoder_lr = cfg.get('encoder_lr', cfg['lr'] * 0.1)
        self.optimizer = optim.AdamW([
            {'params': self.model.encoder.parameters(), 'lr': encoder_lr},
            {'params': self.model.decoder.parameters(), 'lr': cfg['lr']},
            {'params': self.model.temporal.parameters(), 'lr': cfg['lr']},
        ], weight_decay=cfg.get('weight_decay', 0.0001))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, cfg['epochs']
        )

        # Data (sequence-based)
        self.train_loader, self.val_loader = create_sequence_dataloaders(cfg)
        self.sequence_length = cfg.get('sequence_length', 3)

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
        # Stage 1: freeze encoder + decoder, train only ConvGRU
        if self.freeze_all_epochs > 0:
            print(f"\n--- Stage 1: Train only ConvGRU for {self.freeze_all_epochs} epochs ---")
            self.model.freeze_encoder()
            self.model.freeze_decoder()

        for epoch in range(self.cfg['epochs']):
            self.epoch = epoch

            # Stage 2: unfreeze decoder
            if epoch == self.freeze_all_epochs:
                print(f"\n--- Stage 2: Unfreezing decoder at epoch {epoch} ---")
                self.model.unfreeze_decoder()

            # Stage 3: unfreeze encoder
            if epoch == self.freeze_encoder_epochs:
                print(f"\n--- Stage 3: Unfreezing encoder at epoch {epoch} ---")
                self.model.unfreeze_encoder()

            # Train
            train_loss = self._train_epoch()
            self.train_losses.append((epoch, train_loss))

            # Validate
            val_loss, metrics = self._validate()
            self.val_losses.append((epoch, val_loss))
            self.scheduler.step()

            self.writer.add_scalar('val/loss', val_loss, epoch)
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f'val/{metric_name}', metric_value, epoch)

            print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
                  f"AbsRel={metrics['abs_rel']:.4f}, d1={metrics['delta1']:.4f}")
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

    def _train_epoch(self):
        self.model.train()
        epoch_losses = []

        for batch in tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}"):
            rgb_seq = batch['rgb'].to(self.device)      # (B, T, 3, H, W)
            depth_seq = batch['depth'].to(self.device)   # (B, T, 1, H, W)
            mask_seq = batch['mask'].to(self.device)     # (B, T, 1, H, W)

            B, T = rgb_seq.shape[:2]

            self.model.reset_temporal()
            total_loss = torch.tensor(0.0, device=self.device)
            predictions = []

            for t in range(T):
                pred = self.model(rgb_seq[:, t])
                loss, loss_dict = self.criterion(pred, depth_seq[:, t], mask_seq[:, t])
                total_loss = total_loss + loss
                predictions.append(pred)

            # Temporal consistency loss between consecutive predictions
            temporal_loss = torch.tensor(0.0, device=self.device)
            for t in range(1, T):
                mask_intersect = mask_seq[:, t] * mask_seq[:, t - 1]
                temporal_loss = temporal_loss + self.temporal_criterion(
                    predictions[t], predictions[t - 1], mask_intersect
                )
            if T > 1:
                temporal_loss = temporal_loss / (T - 1)
            total_loss = total_loss / T + self.w_temporal * temporal_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.step += 1

            epoch_losses.append(total_loss.item())

            if self.step % 10 == 0:
                self.writer.add_scalar('train/loss', total_loss.item(), self.step)
                self.writer.add_scalar('train/temporal_loss', temporal_loss.item(), self.step)
                self.writer.add_scalar('train/lr_decoder', self.optimizer.param_groups[1]['lr'], self.step)
                self.writer.add_scalar('train/lr_encoder', self.optimizer.param_groups[0]['lr'], self.step)
                self.train_losses.append((self.step, total_loss.item()))

                for name, value in loss_dict.items():
                    if name != 'total' and isinstance(value, torch.Tensor):
                        self.writer.add_scalar(f'train/loss_{name}', value.item(), self.step)
                        if name in self.train_loss_components:
                            self.train_loss_components[name].append((self.step, value.item()))

        return np.mean(epoch_losses)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        losses, all_metrics = [], []

        for batch in self.val_loader:
            rgb_seq = batch['rgb'].to(self.device)
            depth_seq = batch['depth'].to(self.device)
            mask_seq = batch['mask'].to(self.device)

            B, T = rgb_seq.shape[:2]

            self.model.reset_temporal()

            # Run through sequence, evaluate on last frame
            for t in range(T - 1):
                self.model(rgb_seq[:, t])

            pred = self.model(rgb_seq[:, T - 1])
            loss, _ = self.criterion(pred, depth_seq[:, T - 1], mask_seq[:, T - 1])

            losses.append(loss.item())
            all_metrics.append(DepthMetrics.compute(pred, depth_seq[:, T - 1]))

        return np.mean(losses), {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

    @torch.no_grad()
    def _comprehensive_validate(self):
        from realdepth.validation import run_comprehensive_validation
        from realdepth.losses import format_stratified_metrics
        from realdepth.depth_datasets import create_dataloaders
        import json

        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION WITH DEPTH-STRATIFIED METRICS")
        print("="*80)

        best_checkpoint = self.ckpt_dir / 'best.pth'
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])

        # Use single-frame val loader for comprehensive metrics
        _, single_val_loader = create_dataloaders(self.cfg)
        self.model.reset_temporal()

        depth_thresholds = [3.0, 5.0, 10.0]
        flat_metrics, stratified_results = run_comprehensive_validation(
            self.model, single_val_loader, self.device, depth_thresholds
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
        """Save RGB | Ground Truth | Prediction grid using last frame of sequences."""
        import torchvision.utils as vutils

        self.model.eval()
        batch = next(iter(self.val_loader))
        rgb_seq = batch['rgb'].to(self.device)
        depth_seq = batch['depth'].to(self.device)

        actual_samples = min(num_samples, rgb_seq.shape[0])
        B, T = rgb_seq.shape[:2]

        self.model.reset_temporal()

        # Run through sequence
        for t in range(T - 1):
            self.model(rgb_seq[:actual_samples, t])
        depth_pred = self.model(rgb_seq[:actual_samples, T - 1])

        # Use last frame for visualization
        rgb_last = rgb_seq[:actual_samples, T - 1]
        depth_gt = depth_seq[:actual_samples, T - 1]

        depth_gt_vis = (depth_gt / self.cfg['max_depth']).repeat(1, 3, 1, 1)
        depth_pred_vis = (depth_pred / self.cfg['max_depth']).repeat(1, 3, 1, 1)

        vis_samples = []
        for i in range(actual_samples):
            vis_samples.extend([rgb_last[i], depth_gt_vis[i], depth_pred_vis[i]])

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
    cfg.setdefault('w_temporal', 0.5)

    Trainer(cfg).train()


if __name__ == "__main__":
    main()
