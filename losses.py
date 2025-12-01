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
            'delta1': delta1.item(),
            'delta2': delta2.item(),
            'delta3': delta3.item()
        }
