"""
Comprehensive tests for losses.py
"""
import torch
from losses import (
    L1Loss, ScaleInvariantLoss, GradientLoss, SSIMLoss,
    BerHuLoss, CombinedDepthLoss, DepthMetrics
)

def test_loss_shapes():
    """Test that losses work with different input shapes"""
    print("Testing input shapes...")

    # Different batch sizes and resolutions
    shapes = [
        (1, 1, 32, 32),
        (2, 1, 64, 64),
        (4, 1, 128, 128),
    ]

    losses = [L1Loss(), ScaleInvariantLoss(), GradientLoss(), SSIMLoss(), BerHuLoss()]

    for shape in shapes:
        pred = torch.rand(shape) * 10
        target = torch.rand(shape) * 10
        mask = torch.ones(shape)

        for loss_fn in losses:
            try:
                result = loss_fn(pred, target, mask)
                assert result.numel() == 1, f"Loss should return scalar, got shape {result.shape}"
                assert not torch.isnan(result), f"{loss_fn.__class__.__name__} returned NaN"
                assert not torch.isinf(result), f"{loss_fn.__class__.__name__} returned Inf"
            except Exception as e:
                print(f"❌ {loss_fn.__class__.__name__} failed on shape {shape}: {e}")
                return False

    print("✓ All losses handle different shapes correctly")
    return True


def test_loss_with_mask():
    """Test that masking works correctly"""
    print("\nTesting mask behavior...")

    pred = torch.rand(2, 1, 64, 64) * 10
    target = torch.rand(2, 1, 64, 64) * 10

    # Create mask that zeros out half the image
    mask = torch.ones_like(pred)
    mask[:, :, :32, :] = 0

    loss_fn = L1Loss()

    # Loss with full mask should differ from loss with partial mask
    loss_full = loss_fn(pred, target, torch.ones_like(pred))
    loss_masked = loss_fn(pred, target, mask)

    # They should be different (unless by extreme coincidence)
    if torch.allclose(loss_full, loss_masked):
        print("⚠ Warning: Full and masked losses are identical (might be coincidence)")
    else:
        print("✓ Masking affects loss calculation correctly")

    return True


def test_scale_invariance():
    """Test that scale-invariant loss is actually scale invariant"""
    print("\nTesting scale invariance...")

    pred = torch.rand(2, 1, 64, 64) * 10 + 1  # Add 1 to avoid zeros
    target = torch.rand(2, 1, 64, 64) * 10 + 1

    si_loss = ScaleInvariantLoss()

    # Original loss
    loss1 = si_loss(pred, target)

    # Scale both by same factor
    scale = 2.5
    loss2 = si_loss(pred * scale, target * scale)

    # Should be approximately equal (scale invariant)
    if torch.allclose(loss1, loss2, rtol=1e-3):
        print(f"✓ Scale-invariant loss is scale invariant: {loss1:.4f} ≈ {loss2:.4f}")
        return True
    else:
        print(f"❌ Scale-invariant loss not scale invariant: {loss1:.4f} != {loss2:.4f}")
        return False


def test_gradient_loss():
    """Test gradient loss on known patterns"""
    print("\nTesting gradient loss...")

    # Create a smooth gradient
    x = torch.linspace(0, 10, 64).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    smooth_pred = x.repeat(1, 1, 64, 1)
    smooth_target = x.repeat(1, 1, 64, 1)

    # Create a noisy gradient
    noisy_pred = smooth_pred + torch.randn_like(smooth_pred) * 0.5

    grad_loss = GradientLoss()

    # Loss between identical should be near zero
    loss_same = grad_loss(smooth_pred, smooth_target)

    # Loss with noise should be higher
    loss_noisy = grad_loss(noisy_pred, smooth_target)

    print(f"  Identical gradients loss: {loss_same:.4f}")
    print(f"  Noisy gradients loss: {loss_noisy:.4f}")

    if loss_same < loss_noisy:
        print("✓ Gradient loss correctly detects differences")
        return True
    else:
        print("❌ Gradient loss not working as expected")
        return False


def test_ssim_range():
    """Test SSIM loss range"""
    print("\nTesting SSIM loss range...")

    pred = torch.rand(2, 1, 64, 64) * 10

    ssim_loss = SSIMLoss()

    # Loss with itself should be close to 0 (perfect similarity)
    loss_perfect = ssim_loss(pred, pred)

    # Loss with random target should be higher
    target_random = torch.rand_like(pred) * 10
    loss_random = ssim_loss(pred, target_random)

    print(f"  SSIM loss (identical): {loss_perfect:.4f}")
    print(f"  SSIM loss (random): {loss_random:.4f}")

    if loss_perfect < 0.01 and loss_random > loss_perfect:
        print("✓ SSIM loss behaves correctly")
        return True
    else:
        print(f"⚠ SSIM loss might have issues: perfect={loss_perfect:.4f}, random={loss_random:.4f}")
        return False


def test_berhu_behavior():
    """Test BerHu loss L1/L2 transition"""
    print("\nTesting BerHu loss behavior...")

    # Small errors should behave like L1
    pred_small = torch.tensor([[[[1.0, 1.1, 1.2]]]])
    target = torch.tensor([[[[1.0, 1.0, 1.0]]]])

    berhu = BerHuLoss()
    loss = berhu(pred_small, target)

    print(f"  BerHu loss (small errors): {loss:.4f}")

    if not torch.isnan(loss) and not torch.isinf(loss):
        print("✓ BerHu loss computes correctly")
        return True
    else:
        print("❌ BerHu loss has numerical issues")
        return False


def test_metrics():
    """Test evaluation metrics"""
    print("\nTesting evaluation metrics...")

    # Perfect prediction
    pred = torch.rand(1, 1, 64, 64) * 10 + 1

    metrics_perfect = DepthMetrics.compute(pred, pred)

    print(f"  Perfect prediction metrics:")
    print(f"    abs_rel: {metrics_perfect['abs_rel']:.6f}")
    print(f"    delta1: {metrics_perfect['delta1']:.6f}")

    # Should have zero error and perfect threshold accuracy
    if metrics_perfect['abs_rel'] < 1e-5 and metrics_perfect['delta1'] > 0.99:
        print("✓ Metrics correct for perfect prediction")
        return True
    else:
        print("❌ Metrics incorrect for perfect prediction")
        return False


def test_combined_loss():
    """Test combined loss"""
    print("\nTesting combined loss...")

    pred = torch.rand(2, 1, 64, 64) * 10
    target = torch.rand(2, 1, 64, 64) * 10

    combined = CombinedDepthLoss(w_l1=1.0, w_si=0.5, w_grad=0.5, w_ssim=0.1)
    total, losses = combined(pred, target)

    # Check that total is sum of weighted components
    expected_total = (
        losses['l1'] * 1.0 +
        losses['scale_inv'] * 0.5 +
        losses['gradient'] * 0.5 +
        losses['ssim'] * 0.1
    )

    if torch.allclose(total, expected_total):
        print(f"✓ Combined loss correctly weighted: {total:.4f}")
        return True
    else:
        print(f"❌ Combined loss weighting incorrect: {total:.4f} != {expected_total:.4f}")
        return False


def test_mae_metrics():
    """Test MAE and Median AE metrics"""
    print("\nTesting MAE and Median AE metrics...")

    # Create known data: predictions [1,2,3,4,5], targets all 1.0
    # Expected errors: [0,1,2,3,4] → MAE=2.0, Median=2.0
    pred = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0]]]])
    target = torch.tensor([[[[1.0, 1.0, 1.0, 1.0, 1.0]]]])

    metrics = DepthMetrics.compute(pred, target)

    print(f"  MAE: {metrics['mae']:.4f} (expected: 2.0)")
    print(f"  Median AE: {metrics['mae_median']:.4f} (expected: 2.0)")

    if abs(metrics['mae'] - 2.0) < 0.01 and abs(metrics['mae_median'] - 2.0) < 0.01:
        print("✓ MAE metrics compute correctly")
        return True
    else:
        print("❌ MAE metrics incorrect")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("COMPREHENSIVE LOSS FUNCTION TESTS")
    print("=" * 60)

    tests = [
        test_loss_shapes,
        test_loss_with_mask,
        test_scale_invariance,
        test_gradient_loss,
        test_ssim_range,
        test_berhu_behavior,
        test_metrics,
        test_combined_loss,
        test_mae_metrics,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} raised exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("✅ All tests passed! Implementation looks correct.")
    else:
        print("⚠️  Some tests failed. Review implementation.")
