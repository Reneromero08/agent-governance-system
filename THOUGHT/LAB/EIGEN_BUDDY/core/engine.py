"""Native Eigen Core — standalone pure physics engine. Zero text dependencies."""
import torch
import torch.nn as nn

try:
    from .attention import MultiHeadComplexAttention
    from .curvature import CurvatureModulator
    from .phase import PhaseAccumulator
except ImportError:
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from attention import MultiHeadComplexAttention
    from curvature import CurvatureModulator
    from phase import PhaseAccumulator


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
