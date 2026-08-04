"""B7: the complex-fold probe - is the phase channel non-degenerate when
the states are genuinely complex?

Construction (EIGEN_BUDDY twin-rail): complex state z = x + i*c where x is
the real trajectory and c is the evolution of a fixed complex phase carrier
(the unconsumed exact source: the exact branch evolves the carrier exactly,
the holo branch scrambles it through truncation).

Fold pair in COMPLEX space at each layer:
    a = (yE + yH)/sqrt2   (fold-even carrier)
    b = (yE - yH)/sqrt2   (fold-odd residue, complex -> NON-degenerate phases)

Equal scalar bandwidth B per layer per stage:
    si      : B/2 PHASES of b, magnitudes borrowed twin-rail from |a|
    direct  : B/4 exact complex components of b (same scalar count)
    corrupt : B/2 random phases + twin-rail magnitudes (control)
    full-exact, uncorrected (controls)

Metrics:
    complex cosine of the corrected final complex state vs exact complex
    real-boundary readout: Re(z) -> norm -> lm_head vs exact real logits
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline" / "04_inference"))

from qwen35_holo_engine import Qwen35HoloEngine, load_holo, load_original, _configure_from_dir  # noqa: E402

EPS = 1e-5


def rms_norm_complex(x: torch.Tensor, c: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """RMS norm over the complex modulus |z| = sqrt(x^2 + c^2)."""
    r = torch.sqrt((x.float() ** 2 + c.float() ** 2).mean(dim=-1, keepdim=True) + EPS)
    wf = w.float()
    return x.float() * (1.0 + wf) / r, c.float() * (1.0 + wf) / r


def carrier(seq: int, d: int, device: torch.device) -> torch.Tensor:
    """Deterministic complex phase carrier, position-modulated."""
    t = torch.arange(seq, device=device).float()
    j = torch.arange(d, device=device).float()
    phase = 2 * torch.pi * (t[:, None] + 1) * (j[None, :] + 1) / (seq * d)
    return torch.exp(1j * phase)


def fold_correct(exact, student, ids, mode: str, B: int, corrupt_seed: int | None = None):
    """Dual-rail complex forward with per-layer borrow; returns corrected
    final complex state (z = x + i*c)."""
    x = student._embed(ids.unsqueeze(0)).to("cuda").float()
    seq, d = x.shape[1], x.shape[2]
    car = carrier(seq, d, x.device)
    c = car.unsqueeze(0).imag.clone()
    ci = car.unsqueeze(0).imag.clone()
    xE = exact._embed(ids.unsqueeze(0)).to("cuda").float()
    for l in range(32):
        prefix = f"model.language_model.layers.{l}"
        nw = exact._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda").float()
        x_n, c_n = rms_norm_complex(x, c, nw)
        x_n = x_n.to(x.dtype); c_n = c_n.to(c.dtype)
        yH = (student._prefill_layer_mixer(l, x_n, None).float()
              + 1j * student._prefill_layer_mixer(l, c_n, None).float())
        yE = (exact._prefill_layer_mixer(l, x_n, None).float()
              + 1j * exact._prefill_layer_mixer(l, c_n, None).float())
        if mode == "full-exact":
            corr = yE
        elif mode == "uncorrected":
            corr = yH
        else:
            a = (yE + yH) / (2**0.5)
            b = (yE - yH) / (2**0.5)
            bflat = b.reshape(-1, d)
            aflat = a.reshape(-1, d)
            nsel = B // 2 if mode == "si" else B // 4
            _, idx = bflat.abs().topk(nsel, dim=-1)
            bhat = torch.zeros_like(bflat)
            if mode == "si":
                ph = torch.angle(bflat.gather(-1, idx))
                bhat.scatter_(-1, idx, (aflat.abs().gather(-1, idx) * torch.exp(1j * ph)).to(torch.complex64))
            elif mode == "direct":
                bhat.scatter_(-1, idx, bflat.gather(-1, idx))
            else:  # corrupt: random phases
                torch.manual_seed(corrupt_seed or 0)
                ph = (torch.rand_like(bflat.gather(-1, idx)) * 2 * torch.pi).to(torch.float64)
                bhat.scatter_(-1, idx, (aflat.abs().gather(-1, idx) * torch.exp(1j * ph)).to(torch.complex64))
            corr = (a + bhat.reshape_as(b)) / (2**0.5)
        x = x + corr.real.float()
        c = c + corr.imag.float()
        # mlp stage
        pw = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda").float()
        m_n, m_c = rms_norm_complex(x, c, pw)
        m_n = m_n.to(x.dtype); m_c = m_c.to(c.dtype)
        mH = (student._mlp(m_n, l, None).float()
              + 1j * student._mlp(m_c, l, None).float())
        mE = (exact._mlp(m_n, l, None).float()
              + 1j * exact._mlp(m_c, l, None).float())
        if mode == "full-exact":
            corr = mE
        elif mode == "uncorrected":
            corr = mH
        else:
            a = (mE + mH) / (2**0.5)
            b = (mE - mH) / (2**0.5)
            bflat = b.reshape(-1, d)
            aflat = a.reshape(-1, d)
            nsel = B // 2 if mode == "si" else B // 4
            _, idx = bflat.abs().topk(nsel, dim=-1)
            bhat = torch.zeros_like(bflat)
            if mode == "si":
                ph = torch.angle(bflat.gather(-1, idx))
                bhat.scatter_(-1, idx, (aflat.abs().gather(-1, idx) * torch.exp(1j * ph)).to(torch.complex64))
            elif mode == "direct":
                bhat.scatter_(-1, idx, bflat.gather(-1, idx))
            else:
                torch.manual_seed(corrupt_seed or 0)
                ph = (torch.rand_like(bflat.gather(-1, idx)) * 2 * torch.pi).to(torch.float64)
                bhat.scatter_(-1, idx, (aflat.abs().gather(-1, idx) * torch.exp(1j * ph)).to(torch.complex64))
            corr = (a + bhat.reshape_as(b)) / (2**0.5)
        x = x + corr.real.float()
        c = c + corr.imag.float()
    # exact complex final state (for the complex metric)
    xE_f = xE.clone(); cE = carrier(seq, d, x.device).unsqueeze(0).imag.clone().float()
    for l in range(32):
        prefix = f"model.language_model.layers.{l}"
        nw = exact._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda").float()
        xE_n, cE_n = rms_norm_complex(xE_f, cE, nw)
        xE_f = xE_f + exact._prefill_layer_mixer(l, xE_n.to(xE_f.dtype), None).float()
        cE = cE + exact._prefill_layer_mixer(l, cE_n.to(cE.dtype), None).float()
        pw = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda").float()
        m_n, m_c = rms_norm_complex(xE_f, cE, pw)
        xE_f = xE_f + exact._mlp(m_n.to(xE_f.dtype), l, None).float()
        cE = cE + exact._mlp(m_c.to(cE.dtype), l, None).float()
    return x, c, xE_f, cE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holo", default=str(REPO / "output" / "qwen4b_k256.holo"))
    ap.add_argument("--model-dir", default="/run/media/reneshizzle/860_1/Reneshizzle/Apps/LM Studio/Qwen/Qwen3.5-4B")
    ap.add_argument("--bandwidth", type=int, default=64)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    MD = Path(args.model_dir)
    _configure_from_dir(MD)
    orig = load_original(MD)
    holo = load_holo(args.holo)
    tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
    exact = Qwen35HoloEngine(None, orig, exact=True, device="cuda", verbose=False)
    student = Qwen35HoloEngine(holo, orig, device="cuda", verbose=False)

    lines = [l.strip() for l in (REPO / "config" / "heldout.txt").read_text().splitlines() if l.strip()]
    lines += ["The quantum mechanical measurement problem asks", "Mathematics is the language in which nature"]
    B = args.bandwidth
    print("B7: complex-fold probe - is the phase channel real in complex space?")
    print(f"    bandwidth = {B} scalars per layer per stage (si: {B//2} phases, direct: {B//4} complex components)")
    print("=" * 78)
    acc = {m: {"cx": [], "cb": [], "rx": []} for m in ["uncorrected", "si", "direct", "corrupt", "full-exact"]}
    with torch.no_grad():
        for ln in lines:
            ids = tok(ln, return_tensors="pt")["input_ids"][0]
            xE_f, cE = None, None
            for m in ["si", "direct", "corrupt", "uncorrected", "full-exact"]:
                x, c, xE_f, cE = fold_correct(exact, student, ids, m, B, corrupt_seed=7)
                zc = x + 1j * c
                zE = xE_f + 1j * cE
                # complex metric: normalized complex states
                rc = (zc.abs() ** 2).mean(-1, keepdim=True).sqrt() + EPS
                rE = (zE.abs() ** 2).mean(-1, keepdim=True).sqrt() + EPS
                u, v = zc / rc, zE / rE
                ccos = (torch.conj(u) * v).sum().real / ((u.abs() ** 2).sum().sqrt() * (v.abs() ** 2).sum().sqrt())
                acc[m]["cx"].append(ccos.item())
                # per-channel complex cosine (last token)
                acc[m]["cb"].append((torch.conj(u[0, -1]) * v[0, -1]).real
                                    .div((u[0, -1].abs() * v[0, -1].abs())).mean().item())
                # real-boundary readout
                nw = exact._exact_weight("model.language_model.norm.weight").to("cuda").float()
                r = torch.sqrt((x[0, -1].float() ** 2).mean(-1, keepdim=True) + EPS)
                xr = student._lm_head(x[0, -1].float() * (1.0 + nw) / r).float()
                xE_n = exact._rms_offset(xE_f[0, -1].float().unsqueeze(0), nw.unsqueeze(0))[0]
                xE_l = exact._lm_head(xE_n.unsqueeze(0))[0].float()
                acc[m]["rx"].append(torch.nn.functional.cosine_similarity(xE_l.view(1, -1), xr.view(1, -1)).item())
    for m in acc:
        a = acc[m]
        print(f"{m:12s}: complex-state cos = {sum(a['cx'])/len(a['cx']):.4f}   "
              f"per-ch cos = {sum(a['cb'])/len(a['cb']):.4f}   "
              f"real-boundary logit = {sum(a['rx'])/len(a['rx']):.4f}   "
              f"per-prompt boundary: {[f'{v:.3f}' for v in a['rx']]}")


if __name__ == "__main__":
    main()
