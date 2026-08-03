"""Complex Position Encoding — fixed sinusoidal position encoding as complex phase rotation."""
import torch
import torch.nn as nn
import math


class ComplexPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding as complex phase rotation."""
    def __init__(self, d_model, max_seq=512):
        super().__init__()
        pos = torch.arange(0, max_seq).unsqueeze(1).float()
        dim = torch.arange(0, d_model).float()
        wavelength = 10000.0 ** (2.0 * dim / d_model)
        phase = pos / wavelength
        self.register_buffer('phase', phase)

    def forward(self, z, offset=0):
        seq_len = z.shape[1]
        theta = self.phase[offset:offset + seq_len, :]
        c = torch.cos(theta).unsqueeze(0)
        s = torch.sin(theta).unsqueeze(0)
        zr = z.real * c - z.imag * s
        zi = z.real * s + z.imag * c
        return torch.complex(zr, zi)
