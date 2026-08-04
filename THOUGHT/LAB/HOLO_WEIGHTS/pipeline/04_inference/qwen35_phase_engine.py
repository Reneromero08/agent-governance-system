"""
qwen35_phase_engine.py — Phase-Native Non-Collapse Forward (B frontier)

B1 decisive experiment: the measured corruption source is the MLP product
(silu(gate)*up compounds two truncated maps to ~0.18 cosine even on exact
inputs). This module rebuilds the MLP in the phase domain:

    gate/up rails -> S^1 phase embedding (e^{i*pi*tanh(v)})
    product       -> phase ADDITION (z_g * z_u)   [HRR / phase_mul core]
    phase-lock    -> unit(2z + conj(z)^2)          [audio-lane drift suppressor]
    down-project  -> complex linear through the .holo factors
    boundary      -> .real projection (declared CollapseBoundary)

Data-free. No training. No fitting. The question: does the phase-domain
product preserve the exact MLP output direction where the real-domain
product destroys it?

Doctrine: THE ALGORITHM IS DEAD. Catalysis is the hologram. Phase is state.
Measurement only at the boundary.
"""

import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline" / "04_inference"))

from qwen35_holo_engine import (  # noqa: E402
    Qwen35HoloEngine,
    load_holo,
    load_original,
    _configure_from_dir,
    _apply_config,
)


def phase_embed(x: torch.Tensor) -> torch.Tensor:
    """Map real values to unit phases on S^1. Data-free fixed map."""
    theta = math.pi * torch.tanh(x.float())
    return torch.exp(1j * theta)


def phase_lock(z: torch.Tensor) -> torch.Tensor:
    """Three-well phase lock: unit(2z + conj(z)^2). Drift suppressor."""
    z = z / (z.abs() + 1e-8)
    w = 2 * z + torch.conj(z) ** 2
    return w / (w.abs() + 1e-8)


class PhaseMLP:
    """Twin-rail phase-domain MLP over .holo factors. No training."""

    def __init__(self, holo_weights, layer: int, device="cuda"):
        self.device = torch.device(device)
        prefix = f"model.language_model.layers.{layer}.mlp"
        self.wg_u, self.wg_s = holo_weights.factors(f"{prefix}.gate_proj.weight")
        self.wu_u, self.wu_s = holo_weights.factors(f"{prefix}.up_proj.weight")
        self.wd_u, self.wd_s = holo_weights.factors(f"{prefix}.down_proj.weight")
        self.wg_u = self.wg_u.to(self.device)
        self.wg_s = self.wg_s.to(self.device)
        self.wu_u = self.wu_u.to(self.device)
        self.wu_s = self.wu_s.to(self.device)
        self.wd_u = self.wd_u.to(self.device)
        self.wd_s = self.wd_s.to(self.device)

    def _lin(self, x: torch.Tensor, u: torch.Tensor, svh: torch.Tensor) -> torch.Tensor:
        if x.is_complex():
            uf = u.to(torch.complex64)
            svhf = svh.to(torch.complex64)
            return (x @ svhf.transpose(0, 1)) @ uf.transpose(0, 1)
        return (x.float() @ svh.transpose(0, 1).float()) @ u.transpose(0, 1).float()

    def forward_real(self, x: torch.Tensor) -> torch.Tensor:
        gate = self._lin(x, self.wg_u, self.wg_s)
        up = self._lin(x, self.wu_u, self.wu_s)
        activated = torch.nn.functional.silu(gate) * up
        return self._lin(activated, self.wd_u, self.wd_s)

    def forward_phase(self, x: torch.Tensor, lock: bool = False) -> torch.Tensor:
        gate = self._lin(x, self.wg_u, self.wg_s)
        up = self._lin(x, self.wu_u, self.wu_s)
        zg = phase_embed(gate)
        zu = phase_embed(up)
        if lock:
            zg = phase_lock(zg)
            zu = phase_lock(zu)
        zout = zg * zu  # phase addition = the product in the phase domain
        if lock:
            zout = phase_lock(zout)
        # complex linear through the .holo down factors
        out = self._lin(zout, self.wd_u, self.wd_s)
        return out.real  # declared CollapseBoundary: real projection

    def forward_phase_rail(self, x: torch.Tensor, lock: bool = False) -> torch.Tensor:
        """Twin-rail: common-mode phase from the input, product in relative phase."""
        gate = self._lin(x, self.wg_u, self.wg_s)
        up = self._lin(x, self.wu_u, self.wu_s)
        zg = phase_embed(gate)
        zu = phase_embed(up)
        zc = phase_embed(x.mean(dim=-1, keepdim=True))  # common-mode rail
        zg_rel = zg * torch.conj(zc)
        zu_rel = zu * torch.conj(zc)
        if lock:
            zg_rel = phase_lock(zg_rel)
            zu_rel = phase_lock(zu_rel)
        zout = zg_rel * zu_rel
        out = self._lin(zout, self.wd_u, self.wd_s)
        return out.real

    def forward_complex_si(self, x: torch.Tensor, si: torch.Tensor | None, lock: bool = False) -> tuple:
        """EIGEN_BUDDY mechanism: complex state with persistent si phase channel.

        z = x_r + i*si. Complex linear maps through the .holo factors. modReLU
        magnitude gate (phase_mul pattern). Complex product (phase addition).
        si_new is the accumulated phase curvature - borrowed, computed with,
        passed forward unconsumed (the catalytic substrate).
        """
        xi = si if si is not None else torch.zeros_like(x.float())
        z = x.float() + 1j * xi
        zg = self._lin(z, self.wg_u, self.wg_s)
        zu = self._lin(z, self.wu_u, self.wu_s)
        if lock:
            zg = phase_lock(zg)
            zu = phase_lock(zu)
        # modReLU on gate (phase_mul pattern): magnitude gate
        mag = zg.abs()
        gate = torch.relu(mag - 0.0)
        zg_gated = zg / (mag + 1e-8) * gate
        # complex product = phase addition + cross terms
        zact = zg_gated * zu
        if lock:
            zact = phase_lock(zact)
        out = self._lin(zact, self.wd_u, self.wd_s)
        si_new = out.imag.detach()  # persistent phase curvature channel
        return out.real, si_new

    def forward_complex_si_chain(self, x: torch.Tensor, layers_si: list | None = None, lock: bool = False) -> tuple:
        """B1v2: chain the persistent si channel through the MLP."""
        raise NotImplementedError("chain handled by the runner")


def main() -> None:
    import argparse
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--holo", default=str(REPO / "output" / "qwen4b_k256.holo"))
    ap.add_argument("--model-dir", default="/run/media/reneshizzle/860_1/Reneshizzle/Apps/LM Studio/Qwen/Qwen3.5-4B")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--layers", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    MD = Path(args.model_dir)
    _configure_from_dir(MD)
    orig = load_original(MD)
    holo = load_holo(args.holo)
    tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
    exact = Qwen35HoloEngine(None, orig, exact=True, device="cuda", verbose=False)

    lines = [l.strip() for l in (REPO / "config" / "corpus.txt").read_text().splitlines() if l.strip()]
    lines = lines[: args.prompts]
    max_layers = 32 if args.layers == 0 else min(args.layers, 32)

    # capture exact per-layer (normed mixer input, mlp input, exact mlp output)
    caps = {l: {"x_mlp": [], "y_exact": []} for l in range(max_layers)}
    with torch.no_grad():
        for ln in lines:
            ids = tok(ln, return_tensors="pt")["input_ids"][0]
            exact.prefill(ids, capture_hidden=True)
            # run exact layers one at a time to capture the mlp input on the exact trajectory
            hidden = exact._embed(ids.unsqueeze(0))
            for l in range(max_layers):
                prefix = f"model.language_model.layers.{l}"
                norm = exact._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda")
                n_in = exact._rms_offset(hidden, norm)
                mixed = exact._prefill_layer_mixer(l, n_in, None)
                hidden = hidden + mixed.to(hidden.dtype)
                post = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda")
                mlp_in = exact._rms_offset(hidden, post)
                y_ex = exact._mlp(mlp_in, l, None)
                caps[l]["x_mlp"].append(mlp_in[0].cpu())
                caps[l]["y_exact"].append(y_ex[0].cpu())
                hidden = hidden + y_ex.to(hidden.dtype)

    print("=" * 78)
    print("B1: PHASE-DOMAIN MLP vs REAL-DOMAIN MLP (exact inputs, per-layer cosine)")
    print("=" * 78)
    c_real, c_phase, c_phase_lock, c_rail = [], [], [], []
    c_si, c_si_lock = [], []
    si = None
    si_lock = None
    for l in range(max_layers):
        x = torch.cat(caps[l]["x_mlp"]).to("cuda")
        y = torch.cat(caps[l]["y_exact"]).to("cuda")
        m = PhaseMLP(holo, l, "cuda")
        with torch.no_grad():
            yr = m.forward_real(x)
            yp = m.forward_phase(x)
            ypl = m.forward_phase(x, lock=True)
            yra = m.forward_phase_rail(x)
            ysi, si = m.forward_complex_si(x, si)
            ysil, si_lock = m.forward_complex_si(x, si_lock, lock=True)
        cr = torch.nn.functional.cosine_similarity(y.view(1, -1), yr.view(1, -1)).item()
        cp = torch.nn.functional.cosine_similarity(y.view(1, -1), yp.view(1, -1)).item()
        cpl = torch.nn.functional.cosine_similarity(y.view(1, -1), ypl.view(1, -1)).item()
        cra = torch.nn.functional.cosine_similarity(y.view(1, -1), yra.view(1, -1)).item()
        csi = torch.nn.functional.cosine_similarity(y.view(1, -1), ysi.view(1, -1)).item()
        csil = torch.nn.functional.cosine_similarity(y.view(1, -1), ysil.view(1, -1)).item()
        c_real.append(cr); c_phase.append(cp); c_phase_lock.append(cpl); c_rail.append(cra)
        c_si.append(csi); c_si_lock.append(csil)
        print(f"L{l:02d}: real={cr:.4f}  phase={cp:.4f}  +lock={cpl:.4f}  rail={cra:.4f}  "
              f"si={csi:.4f}  si+lock={csil:.4f}")
    print("-" * 78)
    n = max_layers
    print(f"MEAN: real={sum(c_real)/n:.4f}  phase={sum(c_phase)/n:.4f}  +lock={sum(c_phase_lock)/n:.4f}  "
          f"rail={sum(c_rail)/n:.4f}  si={sum(c_si)/n:.4f}  si+lock={sum(c_si_lock)/n:.4f}")


if __name__ == "__main__":
    main()
