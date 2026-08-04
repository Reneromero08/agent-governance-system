"""B8: Sol's minimal covariant Wilson transport probe (fixed revision).

Residual stream order: stage t = 2*l (mix) or 2*l+1 (mlp).
Frames F[t] = top-k output-space frame (u[:, :k]) of the stage's output
projection. Connection Q[t] = polar(F[t].T @ F[t+1]) for t+1 > t.
Borrow through L0=3 (exact), extract packet at the LAST borrow stage
(t0 = 2*L0+1 = L3mlp). Propagate 8 stages (L4mix .. L7mlp) with holo-only
magnitudes, then continue layers 8-31 uncorrected holo to the real
boundary. Packet never collapsed.
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline" / "04_inference"))

from qwen35_holo_engine import Qwen35HoloEngine, load_holo, load_original, _configure_from_dir  # noqa: E402

EPS = 1e-5
L0 = 3
PROP = 4
T0 = 2 * L0 + 1          # extraction stage index (L3mlp)
TMAX = 2 * (L0 + PROP)   # last propagation stage index (exclusive = 2*7+2)


def rms_norm_complex(x: torch.Tensor, c: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r = torch.sqrt((x.float() ** 2 + c.float() ** 2).mean(dim=-1, keepdim=True) + EPS)
    wf = w.float()
    return x.float() * (1.0 + wf) / r, c.float() * (1.0 + wf) / r


def carrier(seq: int, d: int, device: torch.device) -> torch.Tensor:
    t = torch.arange(seq, device=device).float()
    j = torch.arange(d, device=device).float()
    return torch.exp(1j * 2 * torch.pi * (t[:, None] + 1) * (j[None, :] + 1) / (seq * d))


def polar(Q: torch.Tensor) -> torch.Tensor:
    a, _, b = torch.linalg.svd(Q)
    return a @ b


def stage_io(l: int, t: int):
    return ("mix" if t % 2 == 0 else "mlp")


def build_frames(orig, k: int, max_l: int) -> list:
    """Flat frames over residual stages; F[t] = d x k output frame."""
    frames = []
    for l in range(max_l + 1):
        for stage in ("mix", "mlp"):
            if stage == "mix":
                la = f"model.language_model.layers.{l}.linear_attn.out_proj.weight"
                path = la if la in orig.weight_map else \
                    f"model.language_model.layers.{l}.self_attn.o_proj.weight"
            else:
                path = f"model.language_model.layers.{l}.mlp.down_proj.weight"
            w = orig.get(path).float()
            u, _, _ = torch.linalg.svd(w, full_matrices=False)
            frames.append(u[:, :k].contiguous())
    return frames


def dual_rail_forward(exact, student, x, c, l, stage):
    prefix = f"model.language_model.layers.{l}"
    nw = exact._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda").float()
    pw = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda").float()
    norm_w = nw if stage == "mix" else pw
    x_n, c_n = rms_norm_complex(x, c, norm_w)
    x_n = x_n.to(x.dtype); c_n = c_n.to(c.dtype)
    if stage == "mix":
        yH = (student._prefill_layer_mixer(l, x_n, None).float()
              + 1j * student._prefill_layer_mixer(l, c_n, None).float())
        yE = (exact._prefill_layer_mixer(l, x_n, None).float()
              + 1j * exact._prefill_layer_mixer(l, c_n, None).float())
    else:
        yH = (student._mlp(x_n, l, None).float()
              + 1j * student._mlp(c_n, l, None).float())
        yE = (exact._mlp(x_n, l, None).float()
              + 1j * exact._mlp(c_n, l, None).float())
    return yH, yE


def exact_trajectory(exact, ids):
    x = exact._embed(ids.unsqueeze(0)).to("cuda").float()
    seq, d = x.shape[1], x.shape[2]
    c = carrier(seq, d, x.device).unsqueeze(0).imag.clone()
    states, labels = [], []
    for l in range(32):
        for stage in ("mix", "mlp"):
            _, yE = dual_rail_forward(exact, exact, x, c, l, stage)
            x = x + yE.real; c = c + yE.imag
            if l < L0 + PROP:
                states.append(x[0, -1].clone())
                labels.append(f"L{l}{stage}")
    nw = exact._exact_weight("model.language_model.norm.weight").to("cuda").float()
    r = torch.sqrt((x[0, -1].float() ** 2).mean(-1, keepdim=True) + EPS)
    logits = exact._lm_head(x[0, -1].float() * (1.0 + nw) / r).float()
    return states, labels, logits


def run_probe(exact, student, ids, frames, k: int, variant: str, s0_from, seed: int):
    x = student._embed(ids.unsqueeze(0)).to("cuda").float()
    seq, d = x.shape[1], x.shape[2]
    c = carrier(seq, d, x.device).unsqueeze(0).imag.clone()
    s = None
    s0 = None
    pkt_norms, inv_errs = [], []
    for l in range(32):
        for stage in ("mix", "mlp"):
            t = 2 * l + (1 if stage == "mlp" else 0)
            F = frames[t].to("cuda")
            yH, yE = dual_rail_forward(exact, student, x, c, l, stage)
            if t <= T0:
                x = x + yE.real; c = c + yE.imag
                if t == T0:
                    b = (yE - yH) / (2**0.5)
                    s_raw = F.T.to(torch.complex64) @ b.reshape(-1, d).T
                    if variant == "correct":
                        s = s_raw / s_raw.abs().clamp_min(EPS)
                    elif variant == "direct":
                        s = torch.zeros(k, seq, device=b.device, dtype=torch.complex64)
                        s[: k // 2] = s_raw[: k // 2]
                    elif variant == "random":
                        g = torch.randn(k, seq, device=b.device)
                        s = (g / g.abs().clamp_min(EPS)).to(torch.complex64)
                    elif variant == "carrier":
                        s = carrier(seq, k, b.device).T.clone()
                    elif variant == "deranged":
                        s = s0_from if s0_from is not None else carrier(seq, k, b.device).T.clone()
                    elif variant in ("identity", "haar"):
                        s = s_raw / s_raw.abs().clamp_min(EPS)
                    s0 = s.clone()
            elif t < TMAX and t > T0:
                if s is not None:
                    Q = polar(frames[t - 1].to("cuda").T @ frames[t].to("cuda"))
                    if variant == "identity":
                        Q = torch.eye(k, device=Q.device)
                    elif variant == "haar":
                        g = torch.randn(k, k, device=Q.device)
                        qq, rr = torch.linalg.qr(g)
                        Q = qq @ torch.diag(torch.sign(torch.diag(rr).clamp_min(1e-8)))
                    s = Q.to(torch.complex64) @ s
                    s_chk = s.clone()
                    for tt in range(t, T0, -1):
                        s_chk = polar(frames[tt - 1].to("cuda").T @ frames[tt].to("cuda")) \
                            .T.to(torch.complex64) @ s_chk
                    inv_errs.append((s_chk - s0).abs().max().item())
                    pkt_norms.append(s.abs().mean().item())
                    hflat = yH.reshape(-1, d)
                    coeff = F.T.to(torch.complex64) @ hflat.T
                    mags = coeff.abs()
                    ph = s / s.abs().clamp_min(EPS)
                    delta = F.to(torch.complex64) @ (mags * ph)
                    yH = yH + delta.T.reshape(-1, d).reshape_as(yH)
                x = x + yH.real; c = c + yH.imag
            else:
                x = x + yH.real; c = c + yH.imag
    nw = exact._exact_weight("model.language_model.norm.weight").to("cuda").float()
    r = torch.sqrt((x[0, -1].float() ** 2).mean(-1, keepdim=True) + EPS)
    logits = student._lm_head(x[0, -1].float() * (1.0 + nw) / r).float()
    return logits, pkt_norms, inv_errs, s0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holo", default=str(REPO / "output" / "qwen4b_k256.holo"))
    ap.add_argument("--model-dir", default="/run/media/reneshizzle/860_1/Reneshizzle/Apps/LM Studio/Qwen/Qwen3.5-4B")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--prompts", type=int, default=8)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    MD = Path(args.model_dir)
    _configure_from_dir(MD)
    orig = load_original(MD)
    holo = load_holo(args.holo)
    tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
    exact = Qwen35HoloEngine(None, orig, exact=True, device="cuda", verbose=False)
    student = Qwen35HoloEngine(holo, orig, device="cuda", verbose=False)

    lines = [l.strip() for l in (REPO / "config" / "corpus.txt").read_text().splitlines() if l.strip()]
    lines = lines[: args.prompts]
    k = args.k
    frames = build_frames(orig, k, 31)
    variants = ["correct", "deranged", "random", "identity", "haar", "direct", "carrier"]
    print(f"B8: minimal covariant transport probe - k={k}, L0={L0}, {TMAX-T0-1} transport stages, {len(lines)} prompts")
    print("=" * 92)
    acc = {v: {"cos": [], "top1": [], "norm": [], "inv": []} for v in variants}
    prev_s0 = None
    with torch.no_grad():
        for pi, ln in enumerate(lines):
            ids = tok(ln, return_tensors="pt")["input_ids"][0]
            states, labels, el = exact_trajectory(exact, ids)
            el_n = el / el.norm()
            for v in variants:
                s0_use = prev_s0 if v == "deranged" else None
                out, pn, inv, s0 = run_probe(exact, student, ids, frames, k, v, s0_use, seed=11 + pi)
                if v == "correct":
                    prev_s0 = s0
                out_n = out / out.norm()
                acc[v]["cos"].append((out_n * el_n).sum().item())
                acc[v]["top1"].append((out.argmax() == el.argmax()).item())
                acc[v]["norm"].append(out.norm().item() / el.norm().item())
                acc[v]["inv"].append(inv[-1] if inv else -1.0)
            print(f"[{pi}] {ln[:34]:36s} done", flush=True)
    print("-" * 92)
    for v in variants:
        n = len(acc[v]["cos"])
        print(f"{v:10s}: cos={sum(acc[v]['cos'])/n:.4f}  top1={sum(acc[v]['top1'])}/{n}  "
              f"norm={sum(acc[v]['norm'])/n:.4f}  inverr={sum(acc[v]['inv'])/n:.2e}")


if __name__ == "__main__":
    main()
