"""Curvature Modulator — d^2theta/ds^2 semantic boundary detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CurvatureModulator(nn.Module):
    """d^2theta/ds^2 semantic boundary detection from curvature.py."""
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, z, si):
        B, S, D = z.shape
        ratio = z[:, 1:] / (z[:, :-1] + 1e-8)
        dtheta = torch.angle(ratio)
        d2theta = dtheta[:, 1:] - dtheta[:, :-1]
        curv = d2theta.abs()
        curv_full = F.pad(curv, (0, 0, 1, 1))
        z_dir = z / (z.abs() + 1e-8)
        return z + self.weight * curv_full * z_dir
