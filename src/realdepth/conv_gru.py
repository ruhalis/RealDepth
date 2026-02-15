"""
ConvGRU — Convolutional Gated Recurrent Unit for temporal fusion
"""
import torch
import torch.nn as nn


class ConvGRU(nn.Module):
    """Single-layer Convolutional GRU cell

    Args:
        input_channels: Number of input feature channels (default: 96)
        hidden_channels: Number of hidden state channels (default: 96)
        kernel_size: Convolution kernel size (default: 3)
    """

    def __init__(self, input_channels=96, hidden_channels=96, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv_gates = nn.Conv2d(
            input_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size,
            padding=padding,
            bias=True,
        )

        # Candidate hidden state
        self.conv_candidate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x, h_prev=None):
        """
        Args:
            x: Input features (B, C_in, H, W)
            h_prev: Previous hidden state (B, C_hidden, H, W) or None

        Returns:
            h: Updated hidden state (B, C_hidden, H, W)
        """
        if h_prev is None:
            h_prev = torch.zeros(
                x.size(0), self.hidden_channels, x.size(2), x.size(3),
                device=x.device, dtype=x.dtype,
            )

        # Compute update and reset gates
        combined = torch.cat([h_prev, x], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        z = gates[:, :self.hidden_channels]   # update gate
        r = gates[:, self.hidden_channels:]   # reset gate

        # Candidate hidden state
        combined_candidate = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_candidate(combined_candidate))

        # Final hidden state
        h = (1 - z) * h_prev + z * h_tilde
        return h
