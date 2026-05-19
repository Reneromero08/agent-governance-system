"""Native Eigen Core — standalone pure physics engine. Zero text dependencies.

Drop-in module for Feral Resident, Phase 4b Lattice, or any system needing
complex-plane attention processing. Vocabulary-agnostic, token-free.

Input: (B, S, D) complex continuous vectors.
Output: (B, S, D) complex continuous vectors + phase coherence.

Usage:
    from native_eigen_core import NativeEigenCore
    core = NativeEigenCore(d=16, heads=4, layers=6)
    z = torch.complex(torch.randn(2, 32, 16), torch.randn(2, 32, 16))
    z_out, phase_coh = core(z)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# Attention
# ============================================================================

class MultiHeadComplexAttention(nn.Module):
    """Semiotic gravity attention. Per-head Q/K/V, returns (z, si) tuple.

    Two merge modes:
      concat — classical concatenation + linear (proven on WikiText-2)
      born   — Q56 coherent head sum + alignment, O(H^2) cross-terms
    """
    def __init__(self, d_model=16, n_heads=4, merge='concat', geo_init=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.H = n_heads
        self.dh = d_model // n_heads
        hd = d_model
        self.merge_mode = merge

        self.qr = nn.Linear(d_model, hd, bias=False)
        self.qi = nn.Linear(d_model, hd, bias=False)
        self.kr = nn.Linear(d_model, hd, bias=False)
        self.ki = nn.Linear(d_model, hd, bias=False)
        self.vr = nn.Linear(d_model, hd, bias=False)
        self.vi = nn.Linear(d_model, hd, bias=False)

        if merge == 'born':
            self.align_r = nn.Parameter(torch.randn(self.dh, d_model) * 0.02)
            self.align_i = nn.Parameter(torch.randn(self.dh, d_model) * 0.02)
        else:
            self.or_ = nn.Linear(hd, d_model, bias=False)
            self.oi = nn.Linear(hd, d_model, bias=False)

        self.scale = 1.0 / math.sqrt(self.dh)

        if geo_init and merge == 'concat':
            self._geometric_init()
        else:
            init_w = [self.qr, self.qi, self.kr, self.ki, self.vr, self.vi]
            if merge == 'concat': init_w += [self.or_, self.oi]
            for w in init_w: nn.init.normal_(w.weight, std=0.02)

    def _geometric_init(self):
        """Q56 D8: biological 2pi/H spacing, 45deg Q-K offset."""
        angle = 2.0 * math.pi / self.H
        head_angles = torch.arange(self.H, dtype=torch.float32) * angle
        qk_offset = math.pi / 4.0
        noise_std = 0.01
        for name, w in [('qr', self.qr), ('qi', self.qi), ('kr', self.kr),
                         ('ki', self.ki), ('vr', self.vi), ('vi', self.vi)]:
            base = torch.randn(w.weight.shape) * 0.02
            for h in range(self.H):
                start, end = h * self.dh, (h + 1) * self.dh
                c, s = math.cos(head_angles[h]), math.sin(head_angles[h])
                template = base[start:end].clone()
                w.weight.data[start:end] = template * c
                if name.startswith('q'):
                    w.weight.data[start:end] *= math.cos(qk_offset)
                elif name.startswith('k'):
                    w.weight.data[start:end] *= math.cos(-qk_offset)
                w.weight.data[start:end] += torch.randn_like(w.weight.data[start:end]) * noise_std
        nn.init.normal_(self.or_.weight, std=0.02)
        nn.init.normal_(self.oi.weight, std=0.02)

    def forward(self, x):
        B, S, D = x.shape
        qr = self.qr(x.real) - self.qi(x.imag)
        qi = self.qr(x.imag) + self.qi(x.real)
        kr = self.kr(x.real) - self.ki(x.imag)
        ki = self.kr(x.imag) + self.ki(x.real)
        vr = self.vr(x.real) - self.vi(x.imag)
        vi = self.vr(x.imag) + self.vi(x.real)

        qr = qr.view(B, S, self.H, self.dh).transpose(1, 2)
        qi = qi.view(B, S, self.H, self.dh).transpose(1, 2)
        kr = kr.view(B, S, self.H, self.dh).transpose(1, 2)
        ki = ki.view(B, S, self.H, self.dh).transpose(1, 2)
        vr = vr.view(B, S, self.H, self.dh).transpose(1, 2)
        vi = vi.view(B, S, self.H, self.dh).transpose(1, 2)

        sr = (qr @ kr.transpose(-2, -1) + qi @ ki.transpose(-2, -1)) * self.scale
        si = (qi @ kr.transpose(-2, -1) - qr @ ki.transpose(-2, -1)) * self.scale

        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        sr = sr.masked_fill(mask, float('-inf'))
        si = si.masked_fill(mask, 0.0)

        attn = F.softmax(sr, dim=-1)
        out_r = attn @ vr
        out_i = attn @ vi

        if self.merge_mode == 'born':
            psi_r = out_r.sum(dim=1) / math.sqrt(self.H)
            psi_i = out_i.sum(dim=1) / math.sqrt(self.H)
            or_ = psi_r @ self.align_r + psi_i @ self.align_i
            oi_ = psi_r @ self.align_i - psi_i @ self.align_r
        else:
            out_r = out_r.transpose(1, 2).contiguous().view(B, S, -1)
            out_i = out_i.transpose(1, 2).contiguous().view(B, S, -1)
            or_ = self.or_(out_r) - self.oi(out_i)
            oi_ = self.or_(out_i) + self.oi(out_r)

        return torch.complex(or_, oi_), si


# ============================================================================
# Curvature + Phase
# ============================================================================

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


class PhaseAccumulator(nn.Module):
    """Layer-wise learned phase shift. Init 0.1 avoids dead gradient."""
    def __init__(self, d):
        super().__init__()
        self.ang = nn.Parameter(torch.ones(d) * 0.1)

    def forward(self, z):
        c, s = torch.cos(self.ang), torch.sin(self.ang)
        return torch.complex(z.real * c - z.imag * s,
                            z.real * s + z.imag * c)


# ============================================================================
# Core Engine
# ============================================================================

class NativeEigenCore(nn.Module):
    """Pure physics engine. Complex vectors in, complex vectors out.

    Zero knowledge of tokens, embeddings, or vocabularies.
    Ready to drop into Feral Resident, Phase 4b Lattice, or any pipeline.
    """
    def __init__(self, d=16, heads=4, layers=2, merge='concat', geo_init=True):
        super().__init__()
        self.d = d
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': MultiHeadComplexAttention(d, heads, merge, geo_init),
                'curve': CurvatureModulator(d),
                'phase': PhaseAccumulator(d),
            }) for _ in range(layers)
        ])

    def forward(self, z):
        total_si = 0
        for layer in self.layers:
            z, si = layer['attn'](z)
            z = layer['curve'](z, si)
            z = layer['phase'](z)
            total_si = total_si + si

        per_sample_phase = total_si.mean(dim=(1, 2, 3))
        cos_mean = torch.cos(per_sample_phase).mean()
        sin_mean = torch.sin(per_sample_phase).mean()
        phase_coh = (cos_mean**2 + sin_mean**2).sqrt()
        return z, phase_coh

    def head_metrics(self, z):
        """Q56: per-head diagnostics for pruning/reinit."""
        B = z.shape[0]
        H = self.layers[0]['attn'].H
        head_coh = torch.zeros(B, H)
        for layer in self.layers:
            _, si = layer['attn'](z)
            z, _ = layer['attn'](z)
            z = layer['curve'](z, si)
            z = layer['phase'](z)
            per_head_si = si.abs().mean(dim=(-2, -1))
            head_coh += 1.0 / (1.0 + per_head_si)
        head_coh /= len(self.layers)
        mean_coh = head_coh.mean(dim=0)
        return {
            'head_coh': mean_coh,
            'leaders': mean_coh > 0.5,
            'dead': mean_coh < 0.2,
            'laggard': (mean_coh >= 0.2) & (mean_coh < 0.4),
        }

    def reinit_heads(self, dead_mask):
        """Q56 D6: re-init dead heads (beats pruning)."""
        count = 0
        for layer in self.layers:
            attn = layer['attn']
            H, dh = attn.H, attn.dh
            for h in range(H):
                if dead_mask[h]:
                    start, end = h * dh, (h + 1) * dh
                    for w in [attn.qr, attn.qi, attn.kr, attn.ki, attn.vr, attn.vi]:
                        nn.init.normal_(w.weight.data[start:end], std=0.02)
                    count += 1
        return count

    def prune_heads(self, dead_mask):
        """Q56 D1: zero out dead/laggard heads."""
        count = 0
        for layer in self.layers:
            attn = layer['attn']
            H, dh = attn.H, attn.dh
            for h in range(H):
                if dead_mask[h]:
                    start, end = h * dh, (h + 1) * dh
                    for w in [attn.qr, attn.qi, attn.kr, attn.ki, attn.vr, attn.vi]:
                        w.weight.data[start:end] = 0
                    count += 1
        return count


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    core = NativeEigenCore(d=16, heads=4, layers=6)
    z = torch.complex(torch.randn(2, 32, 16), torch.randn(2, 32, 16))
    z_out, pc = core(z)
    params = sum(p.numel() for p in core.parameters())
    print(f"NativeEigenCore: d=16 heads=4 layers=6 params={params:,}")
    print(f"  Input:  {z.shape}  Output: {z_out.shape}  phase_coh={pc.item():.4f}")
    m = core.head_metrics(z)
    print(f"  Heads: leaders={m['leaders'].sum().item()} dead={m['dead'].sum().item()}")
    print("  Standalone OK — zero text dependencies")
