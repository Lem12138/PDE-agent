"""DeepONet1d — Deep Operator Network for 1D Burgers.

Reference: Lu et al., "Learning nonlinear operators via DeepONet",
           Nature Machine Intelligence, 2021.

Architecture: Branch net encodes the input function (initial condition window)
              into coefficients; Trunk net maps spatial query points to basis
              functions. Output = Σ coeff_i · basis_i(y).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepONet1d(nn.Module):
    """DeepONet for 1D time-dependent PDEs.

    Args:
        modes: Ignored (accepted for CLI compatibility).
        width: Hidden width for both branch and trunk networks.
        in_channels: Number of input time steps (T_in).
        out_channels: Ignored — always outputs scalar field.
        n_layers: Ignored (accepted for CLI compatibility).
        n_basis: Number of basis functions.
        nx: Number of spatial grid points.
    """

    def __init__(self, modes: int = 12, width: int = 64, in_channels: int = 10,
                 out_channels: int = 1, n_layers: int = 4, n_basis: int = 128,
                 nx: int = 256, **kwargs):
        super().__init__()
        branch_hidden = width
        trunk_hidden = width

        # ── Branch: function → global coefficients ──
        self.branch = nn.Sequential(
            nn.Conv1d(in_channels, branch_hidden, 1),
            nn.GELU(),
            nn.Conv1d(branch_hidden, branch_hidden, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.branch_fc = nn.Sequential(
            nn.Linear(branch_hidden, branch_hidden),
            nn.GELU(),
            nn.Linear(branch_hidden, n_basis),
        )

        # ── Trunk: location → basis functions ──
        self.trunk = nn.Sequential(
            nn.Linear(1, trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, n_basis),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T_in)  →  output: (B, N)
        B, N, _ = x.shape

        # Branch: encode input function globally
        x_enc = x.permute(0, 2, 1)              # (B, T_in, N)
        x_enc = self.branch(x_enc)               # (B, branch_hidden, 1)
        coeff = self.branch_fc(x_enc.squeeze(-1))  # (B, n_basis)

        # Trunk: per-point basis functions
        grid = torch.linspace(0, 1, N, device=x.device).view(1, N, 1).repeat(B, 1, 1)
        basis = self.trunk(grid)                 # (B, N, n_basis)

        # Dot product: Σ coeff_i · basis_i(y)
        out = torch.einsum('bp,bnp->bn', coeff, basis)  # (B, N)
        return out


Model = DeepONet1d
