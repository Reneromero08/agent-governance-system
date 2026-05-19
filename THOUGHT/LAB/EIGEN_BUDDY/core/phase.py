"""Phase Accumulator — layer-wise learned phase shift."""
import torch
import torch.nn as nn


class PhaseAccumulator(nn.Module):
    """Layer-wise learned phase shift. Init 0.1 avoids dead gradient."""
    def __init__(self, d):
        super().__init__()
        self.ang = nn.Parameter(torch.ones(d) * 0.1)

    def forward(self, z):
        c, s = torch.cos(self.ang), torch.sin(self.ang)
        return torch.complex(z.real * c - z.imag * s,
                            z.real * s + z.imag * c)
