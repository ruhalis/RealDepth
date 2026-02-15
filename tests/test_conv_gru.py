"""
Tests for ConvGRU module and model integration.
"""
import torch
import pytest

from realdepth.conv_gru import ConvGRU
from realdepth.model import DepthEstimationNet
from realdepth.losses import TemporalConsistencyLoss
from realdepth.model_utils import get_model


class TestConvGRU:
    """Unit tests for ConvGRU cell."""

    def test_output_shape(self):
        """ConvGRU output matches expected spatial and channel dims."""
        gru = ConvGRU(input_channels=96, hidden_channels=96)
        x = torch.randn(1, 96, 30, 40)
        h = gru(x)
        assert h.shape == (1, 96, 30, 40)

    def test_output_shape_with_prev_hidden(self):
        """ConvGRU works with explicit previous hidden state."""
        gru = ConvGRU(input_channels=96, hidden_channels=96)
        x = torch.randn(2, 96, 30, 40)
        h_prev = torch.randn(2, 96, 30, 40)
        h = gru(x, h_prev)
        assert h.shape == (2, 96, 30, 40)

    def test_hidden_state_updates(self):
        """Hidden state changes across consecutive calls."""
        gru = ConvGRU(input_channels=96, hidden_channels=96)
        x1 = torch.randn(1, 96, 30, 40)
        x2 = torch.randn(1, 96, 30, 40)

        h1 = gru(x1)
        h2 = gru(x2, h1)
        assert not torch.allclose(h1, h2)

    def test_none_hidden_initializes_zeros(self):
        """Passing h_prev=None is equivalent to zeros."""
        gru = ConvGRU(input_channels=96, hidden_channels=96)
        x = torch.randn(1, 96, 30, 40)

        h_none = gru(x, None)
        h_zeros = gru(x, torch.zeros(1, 96, 30, 40))
        assert torch.allclose(h_none, h_zeros)

    def test_gradient_flow(self):
        """Gradients flow through ConvGRU."""
        gru = ConvGRU(input_channels=96, hidden_channels=96)
        x = torch.randn(1, 96, 30, 40, requires_grad=True)

        h = gru(x)
        loss = h.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_param_count(self):
        """ConvGRU(96, 96) should have ~498K params."""
        gru = ConvGRU(input_channels=96, hidden_channels=96)
        param_count = sum(p.numel() for p in gru.parameters())
        # conv_gates: (192, 192, 3, 3) + bias(192) = 331,968 + 192
        # conv_candidate: (192, 96, 3, 3) + bias(96) = 165,888 + 96
        # Total = 497,952
        assert 400_000 < param_count < 600_000, f"Unexpected param count: {param_count}"

    def test_batch_size_independence(self):
        """Different batch sizes produce same results for same input."""
        gru = ConvGRU(input_channels=96, hidden_channels=96)
        gru.eval()
        x = torch.randn(1, 96, 30, 40)

        with torch.no_grad():
            h_single = gru(x)
            h_batch = gru(x.repeat(3, 1, 1, 1))

        assert torch.allclose(h_single, h_batch[0:1], atol=1e-6)


class TestDepthEstimationNet:
    """Integration tests for the model with ConvGRU."""

    def test_forward_shape(self):
        """Model produces correct output shape."""
        model = DepthEstimationNet(max_depth=10.0, pretrained_encoder=False)
        model.eval()
        x = torch.randn(1, 3, 480, 640)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 480, 640)

    def test_sequential_frames(self):
        """Model processes 3 sequential frames with temporal state."""
        model = DepthEstimationNet(max_depth=10.0, pretrained_encoder=False)
        model.eval()

        model.reset_temporal()
        frames = [torch.randn(1, 3, 480, 640) for _ in range(3)]

        with torch.no_grad():
            outputs = [model(f) for f in frames]

        for out in outputs:
            assert out.shape == (1, 1, 480, 640)
            assert out.min() >= 0
            assert out.max() <= 10.0

    def test_reset_temporal(self):
        """reset_temporal clears hidden state."""
        model = DepthEstimationNet(max_depth=10.0, pretrained_encoder=False)
        model.eval()
        x = torch.randn(1, 3, 480, 640)

        with torch.no_grad():
            model(x)
            assert model.hidden_state is not None
            model.reset_temporal()
            assert model.hidden_state is None

    def test_reset_via_forward_flag(self):
        """reset_temporal=True in forward clears hidden state."""
        model = DepthEstimationNet(max_depth=10.0, pretrained_encoder=False)
        model.eval()
        x = torch.randn(1, 3, 480, 640)

        with torch.no_grad():
            model(x)
            assert model.hidden_state is not None
            model(x, reset_temporal=True)
            # hidden_state is set to the new state after reset+forward
            assert model.hidden_state is not None

    def test_gradient_flow_temporal(self):
        """Gradients flow through ConvGRU in the full model."""
        model = DepthEstimationNet(max_depth=10.0, pretrained_encoder=False)
        model.train()
        model.reset_temporal()

        x1 = torch.randn(1, 3, 256, 320)
        x2 = torch.randn(1, 3, 256, 320)

        out1 = model(x1)
        out2 = model(x2)
        loss = out2.mean()
        loss.backward()

        # ConvGRU params should have gradients
        for name, param in model.temporal.named_parameters():
            assert param.grad is not None, f"No gradient for temporal.{name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for temporal.{name}"

    def test_param_count(self):
        """Model should have encoder + ConvGRU + decoder params."""
        model = DepthEstimationNet(max_depth=10.0, pretrained_encoder=False)
        total = sum(p.numel() for p in model.parameters())
        gru_params = sum(p.numel() for p in model.temporal.parameters())
        base_params = total - gru_params

        # ConvGRU should be ~498K params
        assert 400_000 < gru_params < 600_000, f"Unexpected ConvGRU params: {gru_params}"
        # Total should be base + ConvGRU
        assert total == base_params + gru_params

    def test_freeze_unfreeze(self):
        """Freeze/unfreeze encoder and decoder work correctly."""
        model = DepthEstimationNet(max_depth=10.0, pretrained_encoder=False)

        model.freeze_encoder()
        model.freeze_decoder()
        # Only ConvGRU + encoder.initial (not frozen by freeze_encoder) should be trainable
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        gru_params = sum(p.numel() for p in model.temporal.parameters())
        initial_params = sum(p.numel() for p in model.encoder.initial.parameters())
        assert trainable == gru_params + initial_params

        model.unfreeze_decoder()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        assert trainable == gru_params + initial_params + decoder_params

        model.unfreeze_encoder()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable == total

    def test_model_factory(self):
        """get_model('realdepth') returns DepthEstimationNet with ConvGRU."""
        model = get_model('realdepth', max_depth=10.0, pretrained_encoder=False)
        assert isinstance(model, DepthEstimationNet)
        assert hasattr(model, 'temporal')
        assert hasattr(model, 'reset_temporal')


class TestTemporalConsistencyLoss:
    """Tests for the temporal consistency loss."""

    def test_identical_predictions(self):
        """Loss is zero for identical predictions."""
        loss_fn = TemporalConsistencyLoss()
        pred = torch.randn(2, 1, 480, 640)
        loss = loss_fn(pred, pred)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_with_mask(self):
        """Loss respects validity mask."""
        loss_fn = TemporalConsistencyLoss()
        pred1 = torch.ones(1, 1, 4, 4)
        pred2 = torch.zeros(1, 1, 4, 4)
        mask = torch.zeros(1, 1, 4, 4)
        mask[0, 0, :2, :2] = 1.0  # only top-left 2x2

        loss = loss_fn(pred1, pred2, mask)
        # All masked pixels have diff=1.0, so loss = 4 / 4 = 1.0
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_gradient_flow(self):
        """Gradients flow through temporal consistency loss."""
        loss_fn = TemporalConsistencyLoss()
        pred1 = torch.randn(1, 1, 4, 4, requires_grad=True)
        pred2 = torch.randn(1, 1, 4, 4)
        loss = loss_fn(pred1, pred2)
        loss.backward()
        assert pred1.grad is not None
