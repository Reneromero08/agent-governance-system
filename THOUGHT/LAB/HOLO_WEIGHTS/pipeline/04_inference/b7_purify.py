"""B7-purify: Sol's factorial source-separation gate for the B7 phase result.

At each layer/stage (complex twin-rail, z = x + i*c as in B7), extract the
fold pair (a, b) and reconstruct with the FOLD-EVEN BRANCH ZEROED
(correction = yH + b_hat only - no fold-even carrier leakage).

Factored sources for b_hat:
  phase_src   : exact (angle of odd residue b) | random | prompt-swapped
                | layer-swapped
  mag_src     : exact (|b|, PAID) | holo (|yH|, FREE decoder-side)
                | unit
  support_src : exact top-k of |b| (PAID bits) | holo top-k of |yH|
                | fixed predeclared strided (FREE)

Sol's decisive comparison:
  A = exact phase + holo mag + fixed support
  B = random phase + holo mag + fixed support   (identical else)
  A > B  ->  the phase packet itself carries the invariant.
  C = prompt-swapped phase + holo mag + fixed support
  A > C  ->  phases are input-conditioned, not a global prior.

Leakage checks: exactmag (exact mag, free phase), exactsupport (exact
support, random phase), baseline (B7 twin-rail replicant), controls.

Metrics: boundary-logit cosine, top-1 agreement, relative L2, norm ratio.
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
    r = torch.sqrt((x.float() ** 2 + c.float() ** 2).mean(dim=-1, keepdim=True) + EPS)
    wf = w.float()
    return x.float() * (1.0 + wf) / r, c.float() * (1.0 + wf) / r


def carrier(seq: int, d: int, device: torch.device) -> torch.Tensor:
    t = torch.arange(seq, device=device).float()
    j = torch.arange(d, device=device).float()
    phase = 2 * torch.pi * (t[:, None] + 1) * (j[None, :] + 1) / (seq * d)
    return torch.exp(1j * phase)


def fixed_support(d: int, nsel: int) -> torch.Tensor:
    """Predeclared strided support, identical for every layer and prompt."""
    return torch.arange(0, d, d // nsel)[:nsel]


def run_path(exact, student, ids, mode: str, B: int, swap_phases: dict | None = None,
             swap_layer_phases: dict | None = None, corrupt_seed: int = 7,
             return_phases: bool = False):
    """Dual-rail complex forward with factorized correction. Returns logits
    (or per-layer fixed-support phase dict if return_phases).
    swap_phases: {layer: phase tensor} from another prompt."""
    x = student._embed(ids.unsqueeze(0)).to("cuda").float()
    seq, d = x.shape[1], x.shape[2]
    car = carrier(seq, d, x.device)
    c = car.unsqueeze(0).imag.clone()
    prev_phases = {}  # for layer-swap mode
    collected: dict[int, torch.Tensor] = {}
    nsel = B // 2
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
            x = x + yE.real; c = c + yE.imag
        elif mode == "uncorrected":
            x = x + yH.real; c = c + yH.imag
        else:
            b = (yE - yH) / (2**0.5)
            bflat = b.reshape(-1, d)
            # support source
            if mode == "exactsupport":
                _, idx = bflat.abs().topk(nsel, dim=-1)
            elif mode in ("exactmag", "baseline"):
                _, idx = bflat.abs().topk(nsel, dim=-1)
            else:
                hflat = yH.reshape(-1, d)
                if mode == "holosupport":
                    _, idx = hflat.abs().topk(nsel, dim=-1)
                else:
                    idx = fixed_support(d, nsel).unsqueeze(0).repeat(bflat.shape[0], 1).to(bflat.device)
            # phase source
            if mode == "baseline":
                ph = torch.angle(bflat.gather(-1, idx))
            elif mode == "pure-random":
                torch.manual_seed(corrupt_seed)
                ph = torch.rand_like(bflat.gather(-1, idx)) * 2 * torch.pi
            elif mode == "pure-swap":
                sp = swap_phases[l].to(bflat.device)
                rows = torch.arange(bflat.shape[0], device=sp.device).clamp(max=sp.shape[0] - 1)
                ph = sp[rows]
            elif mode == "pure-swapL":
                ph = prev_phases.get(l - 1, torch.angle(bflat.gather(-1, idx)))
            else:  # pure-exact, exactmag, exactsupport
                ph = torch.angle(bflat.gather(-1, idx))
            prev_phases[l] = torch.angle(bflat.gather(-1, idx))
            if mode == "pure-exact":
                collected[l] = torch.angle(bflat.gather(-1, idx)).cpu()
            # magnitude source
            if mode == "exactmag":
                mg = bflat.abs().gather(-1, idx)
            elif mode in ("baseline",):
                mg = ((a := (yE + yH) / (2**0.5)).reshape(-1, d).abs().gather(-1, idx))
            elif mode == "unit":
                mg = torch.ones_like(bflat.gather(-1, idx))
            else:  # holo magnitude (free decoder-side)
                mg = yH.reshape(-1, d).abs().gather(-1, idx)
            bhat = torch.zeros_like(bflat)
            bhat.scatter_(-1, idx, (mg * torch.exp(1j * ph)).to(torch.complex64))
            if mode == "baseline":
                a_ = (yE + yH) / (2**0.5)
                corr = (a_ + bhat.reshape_as(b)) / (2**0.5)
            else:
                corr = bhat.reshape_as(b)
            x = x + corr.real.float(); c = c + corr.imag.float()
        # mlp stage - same factorization
        pw = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda").float()
        m_n, m_c = rms_norm_complex(x, c, pw)
        m_n = m_n.to(x.dtype); m_c = m_c.to(c.dtype)
        mH = (student._mlp(m_n, l, None).float()
              + 1j * student._mlp(m_c, l, None).float())
        mE = (exact._mlp(m_n, l, None).float()
              + 1j * exact._mlp(m_c, l, None).float())
        if mode == "full-exact":
            x = x + mE.real; c = c + mE.imag
        elif mode == "uncorrected":
            x = x + mH.real; c = c + mH.imag
        else:
            b = (mE - mH) / (2**0.5)
            bflat = b.reshape(-1, d)
            if mode == "exactsupport":
                _, idx = bflat.abs().topk(nsel, dim=-1)
            elif mode in ("exactmag", "baseline"):
                _, idx = bflat.abs().topk(nsel, dim=-1)
            else:
                hflat = mH.reshape(-1, d)
                if mode == "holosupport":
                    _, idx = hflat.abs().topk(nsel, dim=-1)
                else:
                    idx = fixed_support(d, nsel).unsqueeze(0).repeat(bflat.shape[0], 1).to(bflat.device)
            if mode == "baseline":
                ph = torch.angle(bflat.gather(-1, idx))
            elif mode == "pure-random":
                torch.manual_seed(corrupt_seed + 1)
                ph = torch.rand_like(bflat.gather(-1, idx)) * 2 * torch.pi
            elif mode == "pure-swap":
                sp = swap_phases[l].to(bflat.device)
                rows = torch.arange(bflat.shape[0], device=sp.device).clamp(max=sp.shape[0] - 1)
                ph = sp[rows]
            elif mode == "pure-swapL":
                ph = prev_phases.get(l - 1, torch.angle(bflat.gather(-1, idx)))
            else:
                ph = torch.angle(bflat.gather(-1, idx))
            prev_phases[l] = torch.angle(bflat.gather(-1, idx))
            if mode == "pure-exact":
                collected[l] = torch.angle(bflat.gather(-1, idx)).cpu()
            if mode == "exactmag":
                mg = bflat.abs().gather(-1, idx)
            elif mode == "baseline":
                mg = ((mE + mH) / (2**0.5)).reshape(-1, d).abs().gather(-1, idx)
            elif mode == "unit":
                mg = torch.ones_like(bflat.gather(-1, idx))
            else:
                mg = mH.reshape(-1, d).abs().gather(-1, idx)
            bhat = torch.zeros_like(bflat)
            bhat.scatter_(-1, idx, (mg * torch.exp(1j * ph)).to(torch.complex64))
            if mode == "baseline":
                a_ = (mE + mH) / (2**0.5)
                corr = (a_ + bhat.reshape_as(b)) / (2**0.5)
            else:
                corr = bhat.reshape_as(b)
            x = x + corr.real.float(); c = c + corr.imag.float()
    nw = exact._exact_weight("model.language_model.norm.weight").to("cuda").float()
    r = torch.sqrt((x[0, -1].float() ** 2).mean(-1, keepdim=True) + EPS)
    logits = student._lm_head(x[0, -1].float() * (1.0 + nw) / r).float()
    if return_phases:
        return collected
    return logits


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
    modes = ["baseline", "pure-exact", "pure-random", "pure-swap", "pure-swapL",
             "exactmag", "exactsupport", "uncorrected", "full-exact"]
    print("B7-purify: Sol's factorial source-separation gate (B=%d, fold-even zeroed except baseline)" % B)
    print("=" * 92)
    acc = {m: {"cos": [], "top1": [], "l2": [], "norm": []} for m in modes}
    cached_phases = None
    with torch.no_grad():
        for pi, ln in enumerate(lines):
            ids = tok(ln, return_tensors="pt")["input_ids"][0]
            el = run_path(exact, student, ids, "full-exact", B, corrupt_seed=7)
            el_n = el / el.norm()
            if cached_phases is None:
                cached_phases = run_path(exact, student, ids, "pure-exact", B,
                                         corrupt_seed=7, return_phases=True)
            for m in modes:
                out = run_path(exact, student, ids, m, B, swap_phases=cached_phases,
                               swap_layer_phases=None, corrupt_seed=7)
                out_n = out / out.norm()
                acc[m]["cos"].append((out_n * el_n).sum().item())
                acc[m]["top1"].append((out.argmax() == el.argmax()).item())
                acc[m]["l2"].append(((out - el).norm().item() / el.norm().item()))
                acc[m]["norm"].append((out.norm().item() / el.norm().item()))
            print(f"[{pi}] {ln[:38]:40s} done")
    print("-" * 92)
    for m in modes:
        if not acc[m]["cos"]:
            continue
        n = len(acc[m]["cos"])
        print(f"{m:14s}: cos={sum(acc[m]['cos'])/n:.4f}  top1={sum(acc[m]['top1'])}/{n}  "
              f"relL2={sum(acc[m]['l2'])/n:.4f}  norm={sum(acc[m]['norm'])/n:.4f}")


if __name__ == "__main__":
    main()
