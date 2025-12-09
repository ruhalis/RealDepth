"""
Loss Functions for 2D Camera Depth Estimator
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    """Absolute Error"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, mask=None):
        diff = torch.abs(pred - target)
        
        if mask is not None:
            diff = diff * mask
            return diff.sum() / (mask.sum() + 1e-8)
        
        return diff.mean()


class ScaleInvariantLoss(nn.Module):
    """
    Scale-Invariant Loss
    Invariant to global scaling of the depth
    """
    def __init__(self, lambda_weight=0.5):
        super().__init__()
        self.lambda_weight = lambda_weight
    
    def forward(self, pred, target, mask=None):
        # Add small epsilon to avoid log(0)
        pred = torch.clamp(pred, min=1e-8)
        target = torch.clamp(target, min=1e-8)
        
        # Compute log difference
        log_diff = torch.log(pred) - torch.log(target)
        
        if mask is not None:
            log_diff = log_diff * mask
            n = mask.sum() + 1e-8
        else:
            n = log_diff.numel()
        
        loss = (log_diff ** 2).sum() / n
        loss -= self.lambda_weight * (log_diff.sum() ** 2) / (n ** 2)
        
        return loss


class GradientLoss(nn.Module):
    """
    Gradient Loss
    Preserves edges and fine details
    """
    def __init__(self):
        super().__init__()
        
        # Sobel kernels
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0))
        
        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]
        ], dtype=torch.float32).unsqueeze(0))
    
    def forward(self, pred, target, mask=None):
        pred_dx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_dy = F.conv2d(pred, self.sobel_y, padding=1)
        
        target_dx = F.conv2d(target, self.sobel_x, padding=1)
        target_dy = F.conv2d(target, self.sobel_y, padding=1)
        
        # difference of gradients
        diff_x = torch.abs(pred_dx - target_dx)
        diff_y = torch.abs(pred_dy - target_dy)
        
        if mask is not None:
            diff_x = diff_x * mask
            diff_y = diff_y * mask
            n = mask.sum() + 1e-8
            return (diff_x.sum() + diff_y.sum()) / (2 * n)
        
        return (diff_x.mean() + diff_y.mean()) / 2


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss
    Captures luminance, contrast, and structure
    """
    def __init__(self, window_size=11, c1=0.01**2, c2=0.03**2):
        super().__init__()
        self.window_size = window_size
        self.c1 = c1
        self.c2 = c2
        
        # Gaussian window
        self.register_buffer('window', self._create_window(window_size))
    
    def _create_window(self, size):
        """Create 2D Gaussian window"""
        sigma = 1.5
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        window = g.unsqueeze(0) * g.unsqueeze(1)
        return window.unsqueeze(0).unsqueeze(0)
    
    def forward(self, pred, target, mask=None):
        # Compute local means
        mu_pred = F.conv2d(pred, self.window, padding=self.window_size // 2)
        mu_target = F.conv2d(target, self.window, padding=self.window_size // 2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        # Compute local variances and covariance
        sigma_pred_sq = F.conv2d(pred ** 2, self.window, padding=self.window_size // 2) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, self.window, padding=self.window_size // 2) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, self.window, padding=self.window_size // 2) - mu_pred_target
        
        # SSIM formula
        ssim = ((2 * mu_pred_target + self.c1) * (2 * sigma_pred_target + self.c2)) / \
               ((mu_pred_sq + mu_target_sq + self.c1) * (sigma_pred_sq + sigma_target_sq + self.c2))
        
        # Return 1 - SSIM as loss (we want to maximize SSIM)
        if mask is not None:
            ssim = ssim * mask
            return 1 - ssim.sum() / (mask.sum() + 1e-8)
        
        return 1 - ssim.mean()


class BerHuLoss(nn.Module):
    """
    Reverse Huber (BerHu) Loss
    Behaves like L1 for small errors and L2 for large errors
    """
    def __init__(self, threshold_ratio=0.2):
        super().__init__()
        self.threshold_ratio = threshold_ratio
    
    def forward(self, pred, target, mask=None):
        diff = pred - target
        abs_diff = torch.abs(diff)
        
        # Dynamic threshold based on max error
        c = self.threshold_ratio * abs_diff.max().detach()
        
        # BerHu loss
        berhu = torch.where(
            abs_diff <= c,
            abs_diff,
            (diff ** 2 + c ** 2) / (2 * c + 1e-8)
        )
        
        if mask is not None:
            berhu = berhu * mask
            return berhu.sum() / (mask.sum() + 1e-8)
        
        return berhu.mean()


class CombinedDepthLoss(nn.Module):
    """
    Combined loss function with weights
    """
    def __init__(
        self,
        w_l1=1.0,
        w_si=0.5,
        w_grad=0.5,
        w_ssim=0.1,
        use_berhu=False
    ):
        super().__init__()
        
        self.w_l1 = w_l1
        self.w_si = w_si
        self.w_grad = w_grad
        self.w_ssim = w_ssim
        
        if use_berhu:
            self.l1_loss = BerHuLoss()
        else:
            self.l1_loss = L1Loss()
        
        self.si_loss = ScaleInvariantLoss()
        self.grad_loss = GradientLoss()
        self.ssim_loss = SSIMLoss()
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: Predicted depth (B, 1, H, W)
            target: Ground truth depth (B, 1, H, W)
            mask: Valid depth mask (B, 1, H, W)
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        losses = {}
        total = 0
        
        if self.w_l1 > 0:
            losses['l1'] = self.l1_loss(pred, target, mask)
            total += self.w_l1 * losses['l1']
        
        if self.w_si > 0:
            losses['scale_inv'] = self.si_loss(pred, target, mask)
            total += self.w_si * losses['scale_inv']
        
        if self.w_grad > 0:
            losses['gradient'] = self.grad_loss(pred, target, mask)
            total += self.w_grad * losses['gradient']
        
        if self.w_ssim > 0:
            losses['ssim'] = self.ssim_loss(pred, target, mask)
            total += self.w_ssim * losses['ssim']
        
        losses['total'] = total
        
        return total, losses



# Evaluation Metrics
class DepthMetrics:
    """
    Standard depth estimation evaluation metrics
    """
    
    @staticmethod
    def compute(pred, target, mask=None):
        """
        Compute all metrics
        
        Args:
            pred: Predicted depth (B, 1, H, W) or (H, W)
            target: Ground truth depth, same shape as pred
            mask: Valid pixels mask, optional
        
        Returns:
            Dictionary with all metrics
        """
        pred = pred.flatten()
        target = target.flatten()
        
        if mask is not None:
            mask = mask.flatten().bool()
            pred = pred[mask]
            target = target[mask]
        
        # Remove invalid depths
        valid = (target > 0) & (pred > 0)
        pred = pred[valid]
        target = target[valid]
        
        if len(pred) == 0:
            return {
                'abs_rel': 0, 'sq_rel': 0, 'rmse': 0, 'rmse_log': 0,
                'mae': 0, 'mae_median': 0,
                'delta1': 0, 'delta2': 0, 'delta3': 0
            }
        
        # Absolute Relative Error
        abs_rel = torch.mean(torch.abs(pred - target) / target)
        
        # Squared Relative Error
        sq_rel = torch.mean(((pred - target) ** 2) / target)
        
        # RMSE
        rmse = torch.sqrt(torch.mean((pred - target) ** 2))
        
        # RMSE log
        rmse_log = torch.sqrt(torch.mean((torch.log(pred) - torch.log(target)) ** 2))

        # Mean Absolute Error
        mae = torch.mean(torch.abs(pred - target))

        # Median Absolute Error
        mae_median = torch.median(torch.abs(pred - target))

        # Threshold accuracies
        thresh = torch.max(pred / target, target / pred)
        delta1 = (thresh < 1.25).float().mean()
        delta2 = (thresh < 1.25 ** 2).float().mean()
        delta3 = (thresh < 1.25 ** 3).float().mean()
        
        return {
            'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(),
            'rmse': rmse.item(),
            'rmse_log': rmse_log.item(),
            'mae': mae.item(),
            'mae_median': mae_median.item(),
            'delta1': delta1.item(),
            'delta2': delta2.item(),
            'delta3': delta3.item()
        }

    @staticmethod
    def compute_stratified(pred, target, depth_thresholds=[3.0, 5.0, 10.0], mask=None):
        """
        Compute depth-stratified metrics for cumulative depth ranges.

        For each threshold T, computes metrics for all pixels with depth <= T.

        Args:
            pred: Predicted depth (B, 1, H, W) or (H, W)
            target: Ground truth depth, same shape as pred
            depth_thresholds: List of depth thresholds in meters (default: [3.0, 5.0, 10.0])
            mask: Valid pixels mask, optional

        Returns:
            Dictionary with structure:
            {
                'overall': {...},  # All valid pixels
                '3m': {...},       # All pixels with depth <= 3m
                '5m': {...},       # All pixels with depth <= 5m
                '10m': {...}       # All pixels with depth <= 10m
            }
        """
        results = {}

        # Compute overall metrics (all valid pixels)
        results['overall'] = DepthMetrics.compute(pred, target, mask)

        # Flatten for easier mask operations
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        # Compute stratified metrics for each depth threshold
        for threshold in depth_thresholds:
            # Create depth-stratified mask
            depth_mask = (target_flat <= threshold) & (target_flat > 0)

            if mask is not None:
                depth_mask = depth_mask & mask.flatten().bool()

            # Compute metrics for this depth range
            if depth_mask.sum() > 0:
                # Get masked values
                pred_masked = pred_flat[depth_mask]
                target_masked = target_flat[depth_mask]

                # Compute metrics
                metrics = DepthMetrics.compute(pred_masked, target_masked)
                metrics['num_pixels'] = depth_mask.sum().item()
            else:
                # No pixels in this range - return zeros
                metrics = {
                    'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse': 0.0, 'rmse_log': 0.0,
                    'mae': 0.0, 'mae_median': 0.0,
                    'delta1': 0.0, 'delta2': 0.0, 'delta3': 0.0,
                    'num_pixels': 0
                }

            # Store with threshold key
            key = f"{int(threshold)}m"
            results[key] = metrics

        # Add num_pixels to overall metrics
        valid_mask = (target_flat > 0) & (pred_flat > 0)
        if mask is not None:
            valid_mask = valid_mask & mask.flatten().bool()
        results['overall']['num_pixels'] = valid_mask.sum().item()

        return results


def format_depth_metric(value_meters):
    """Convert depth metric to human-readable format (cm or m)"""
    if value_meters < 1.0:
        return f"{value_meters * 100:.1f} cm"
    else:
        return f"{value_meters:.2f} m"


def format_stratified_metrics(metrics_dict):
    """
    Format stratified metrics as a readable table.

    Args:
        metrics_dict: Dictionary from compute_stratified()

    Returns:
        Formatted string with table
    """
    output = []
    output.append("\n" + "="*104)
    output.append("DEPTH-STRATIFIED VALIDATION METRICS")
    output.append("="*104)

    # Header
    header = f"{'Range':<12} {'#Pixels':<12} {'AbsRel':<10} {'RMSE':<12} {'MAE':<12} {'δ1':<8} {'δ2':<8} {'δ3':<8}"
    output.append(header)
    output.append("-"*104)

    # Overall row
    overall = metrics_dict['overall']
    num_pixels = overall.get('num_pixels', 'N/A')
    if num_pixels != 'N/A':
        num_pixels = f"{num_pixels:,}"
    row = (f"{'Overall':<12} {num_pixels:<12} {overall['abs_rel']:<10.4f} "
           f"{format_depth_metric(overall['rmse']):<12} {format_depth_metric(overall['mae']):<12} "
           f"{overall['delta1']:<8.4f} {overall['delta2']:<8.4f} {overall['delta3']:<8.4f}")
    output.append(row)

    # Stratified rows
    for depth_range in ['3m', '5m', '10m']:
        if depth_range in metrics_dict:
            m = metrics_dict[depth_range]
            num_pixels_str = f"{m['num_pixels']:,}"
            row = (f"{'≤ ' + depth_range:<12} {num_pixels_str:<12} {m['abs_rel']:<10.4f} "
                   f"{format_depth_metric(m['rmse']):<12} {format_depth_metric(m['mae']):<12} "
                   f"{m['delta1']:<8.4f} {m['delta2']:<8.4f} {m['delta3']:<8.4f}")

            # Add warning if no pixels in range
            if m['num_pixels'] == 0:
                row += " [WARNING: No pixels in this range]"

            output.append(row)

    output.append("="*104)
    return "\n".join(output)
