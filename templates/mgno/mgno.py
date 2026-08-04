"""MGNO1d — Multi-Grid Neural Operator for 1D Burgers.

A multi-scale operator that processes the input at several spatial
resolutions and fuses the results. Combines spectral (Fourier) and
spatial (convolutional) processing at each scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Fourier spectral convolution – same as FNO baseline."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        eff_modes = min(self.modes, N // 2 + 1)
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, self.out_channels, N // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :eff_modes] = torch.einsum(
            'bix,iox->box', x_ft[:, :, :eff_modes],
            self.weights1[:, :, :eff_modes]
        )
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class MGScaleBlock(nn.Module):
    """One scale of multi-grid processing: spectral + spatial path."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.spatial = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(min(8, width), width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.spatial(x)))


class MGNO1d(nn.Module):
    """Multi-Grid Neural Operator with multi-scale spectral processing.

    Projects input to multiple resolution levels via strided pooling,
    processes each scale independently with spectral conv blocks,
    then fuses results via upsampling and summation.

    Args:
        modes: Fourier modes per scale block.
        width: Hidden channel width.
        in_channels: Number of input time steps (T_in).
        out_channels: Ignored — always outputs scalar field.
        n_layers: Number of blocks per scale (total layers = n_layers × n_scales).
        n_scales: Number of resolution levels (default 3).
    """

    def __init__(self, modes: int = 12, width: int = 32, in_channels: int = 10,
                 out_channels: int = 1, n_layers: int = 4, n_scales: int = 3, **kwargs):
        super().__init__()
        blocks_per_scale = max(1, n_layers // n_scales)

        # ── Input projection ──
        self.fc_in = nn.Linear(in_channels + 1, width)

        # ── Per-scale processing chains ──
        self.scale_blocks = nn.ModuleList()
        for s in range(n_scales):
            scale_chain = nn.ModuleList([
                MGScaleBlock(width, modes)
                for _ in range(blocks_per_scale)
            ])
            self.scale_blocks.append(scale_chain)

        # ── Scale fusion weights (learnable) ──
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

        # ── Output head ──
        self.head = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def get_grid(self, B, N, device):
        return torch.linspace(0, 1, N, device=device).view(1, N, 1).repeat(B, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T_in)  →  output: (B, N)
        B, N, T_in = x.shape

        # Input projection
        grid = self.get_grid(B, N, x.device)
        x = torch.cat([x, grid], dim=-1)        # (B, N, T_in+1)
        x = self.fc_in(x)                       # (B, N, width)
        x = x.permute(0, 2, 1)                  # (B, width, N)

        # Multi-scale processing
        outputs = []
        n_scales = len(self.scale_blocks)

        for s, blocks in enumerate(self.scale_blocks):
            # Downsample to current scale
            scale_factor = 2 ** s
            if scale_factor > 1:
                x_s = F.avg_pool1d(x, kernel_size=scale_factor, stride=scale_factor)
            else:
                x_s = x

            # Process at this scale
            for blk in blocks:
                x_s = blk(x_s)

            # Upsample back to original resolution
            if x_s.shape[-1] != N:
                x_s = F.interpolate(x_s, size=N, mode='linear', align_corners=True)

            outputs.append(x_s)

        # Weighted fusion of scales
        w = F.softmax(self.scale_weights, dim=0)
        fused = sum(w[i] * outputs[i] for i in range(n_scales))

        # Output head
        fused = fused.permute(0, 2, 1)           # (B, N, width)
        out = self.head(fused)                   # (B, N, 1)
        return out.squeeze(-1)                   # (B, N)


Model = MGNO1d
