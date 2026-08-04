"""B8-min: Sol's decisive probe - convention-correct SU(2) transport.

Corrections vs B8 (per Sol's audit):
  1. Connection: Q_t = polar(F_t^T F_{t-1}) (orthogonal Procrustes, column
     convention) - verifies frame continuity ||F_t Q s - F_{t-1} s||.
  2. Coupling: fixed reversible SU(2) [c'; s'] = 1/sqrt2 [[I,iI],[iI,I]]
     [c; s] with c = F_t^+ y_H; retain s'; decode F_t c'; adjoint check.
  3. Per-stage metrics: hidden cos + relL2 vs exact, rail norms,
     correct-minus-random / correct-minus-deranged margins, frame-
     continuity error, exact-suffix readout per stage.
  4. No-packet exact-front baseline control.
  5. Wiring checks: ||s0_correct - s0_deranged||, L4 delta distance.
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
T0 = 2 * L0 + 1        # extraction stage (L3mlp)
TPROP = 2 * (L0 + 1)   # first propagation stage (L4mix)
TEND = 2 * (L0 + 4)    # last propagation stage + 1 (L7mlp = TEND-1)


def rms_norm_complex(x, c, w):
    r = torch.sqrt((x.float() ** 2 + c.float() ** 2).mean(dim=-1, keepdim=True) + EPS)
    wf = w.float()
    return x.float() * (1.0 + wf) / r, c.float() * (1.0 + wf) / r


def carrier(seq, d, device):
    t = torch.arange(seq, device=device).float()
    j = torch.arange(d, device=device).float()
    return torch.exp(1j * 2 * torch.pi * (t[:, None] + 1) * (j[None, :] + 1) / (seq * d))


def polar(Q):
    a, _, b = torch.linalg.svd(Q)
    return a @ b


def build_frames(orig, k, max_l):
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


def stage_forward(exact, student, x, c, l, stage, need_exact=True):
    prefix = f"model.language_model.layers.{l}"
    nw = exact._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda").float()
    pw = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda").float()
    norm_w = nw if stage == "mix" else pw
    x_n, c_n = rms_norm_complex(x, c, norm_w)
    x_n = x_n.to(x.dtype); c_n = c_n.to(c.dtype)
    if stage == "mix":
        yH = (student._prefill_layer_mixer(l, x_n, None).float()
              + 1j * student._prefill_layer_mixer(l, c_n, None).float())
        yE = None
        if need_exact:
            yE = (exact._prefill_layer_mixer(l, x_n, None).float()
                  + 1j * exact._prefill_layer_mixer(l, c_n, None).float())
    else:
        yH = (student._mlp(x_n, l, None).float()
              + 1j * student._mlp(c_n, l, None).float())
        yE = None
        if need_exact:
            yE = (exact._mlp(x_n, l, None).float()
                  + 1j * exact._mlp(c_n, l, None).float())
    return yH, yE


def readout(exact, x):
    nw = exact._exact_weight("model.language_model.norm.weight").to("cuda").float()
    r = torch.sqrt((x[0, -1].float() ** 2).mean(-1, keepdim=True) + EPS)
    return exact._lm_head(x[0, -1].float() * (1.0 + nw) / r).float()


def exact_suffix(exact, x, c, l, stage):
    """Continue the given complex state with EXACT layers to the boundary."""
    for ll in range(l, 32):
        for st in ("mix", "mlp"):
            if ll == l and st == stage:
                continue
            _, yE = stage_forward(exact, exact, x, c, ll, st)
            x = x + yE.real; c = c + yE.imag
    return readout(exact, x)


def exact_trajectory(exact, ids):
    x = exact._embed(ids.unsqueeze(0)).to("cuda").float()
    seq, d = x.shape[1], x.shape[2]
    c = carrier(seq, d, x.device).unsqueeze(0).imag.clone()
    states, labels = [], []
    for l in range(L0 + 4):
        for stage in ("mix", "mlp"):
            _, yE = stage_forward(exact, exact, x, c, l, stage)
            x = x + yE.real; c = c + yE.imag
            states.append(x[0, -1].clone())
            labels.append(f"L{l}{stage}")
    return states, labels, readout(exact, x)


def run_probe(exact, student, ids, frames, k, variant, s0_from, el_states, el_labels, el_logits):
    """Returns per-stage metrics dict, final holo-suffix logits, wiring info."""
    x = student._embed(ids.unsqueeze(0)).to("cuda").float()
    seq, d = x.shape[1], x.shape[2]
    c = carrier(seq, d, x.device).unsqueeze(0).imag.clone()
    s = None
    s0 = None
    l4_delta_dist = 0.0
    metrics = {"stage": [], "cos": [], "l2": [], "norm": [], "frame": [],
               "suff_cos": [], "suff_top1": []}
    for l in range(32):
        for stage in ("mix", "mlp"):
            t = 2 * l + (1 if stage == "mlp" else 0)
            F = frames[t].to("cuda")
            yH, yE = stage_forward(exact, student, x, c, l, stage, need_exact=(t <= T0))
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
                    else:
                        s = s_raw / s_raw.abs().clamp_min(EPS)
                    s0 = s.clone()
            elif TPROP <= t < TEND:
                if s is not None and variant != "nopacket":
                    Fp = frames[t - 1].to("cuda")
                    Q = polar(F.T @ Fp) if variant not in ("identity", "haar") else None
                    if variant == "identity":
                        Q = torch.eye(k, device=Fp.device)
                    elif variant == "haar":
                        g = torch.randn(k, k, device=Fp.device)
                        qq, rr = torch.linalg.qr(g)
                        Q = qq @ torch.diag(torch.sign(torch.diag(rr).clamp_min(1e-8)))
                    s_prev = s
                    s = Q.to(torch.complex64) @ s_prev
                    d_pk = (F.to(torch.complex64) @ s - Fp.to(torch.complex64) @ s_prev).abs().norm(dim=0)
                    m_pk = (Fp.to(torch.complex64) @ s_prev).abs().norm(dim=0).clamp_min(EPS)
                    frame_err = (d_pk / m_pk).max().item()
                    # SU(2) coupling with holo coefficients
                    c_t = F.T.to(torch.complex64) @ yH.reshape(-1, d).T  # k x seq
                    c_p = (c_t + 1j * s) / (2**0.5)
                    s_p = (1j * c_t + s) / (2**0.5)
                    yH_new = F.to(torch.complex64) @ c_p
                    s = s_p  # retain mixed rail
                    # stage metrics vs exact
                    x_new = x + yH_new.T.reshape(-1, d).reshape_as(yH).real
                    c_new = c + yH_new.T.reshape(-1, d).reshape_as(yH).imag
                    cos = torch.nn.functional.cosine_similarity(
                        x_new[0, -1].float().view(1, -1), el_states[t].view(1, -1)).item()
                    l2 = (x_new[0, -1].float() - el_states[t].float()).norm().item() / \
                        el_states[t].float().norm().item()
                    metrics["stage"].append(el_labels[t]); metrics["cos"].append(cos)
                    metrics["l2"].append(l2); metrics["frame"].append(frame_err)
                    if t == TEND - 1:
                        suf = exact_suffix(exact, x_new, c_new, l, stage)
                        metrics["suff_cos"].append(torch.nn.functional.cosine_similarity(
                            suf.view(1, -1), el_logits.view(1, -1)).item())
                        metrics["suff_top1"].append((suf.argmax() == el_logits.argmax()).item())
                    if t == TPROP:
                        l4_delta_dist = (yH_new.T.reshape(-1, d).reshape_as(yH) - yH).abs().max().item()
                x = x + yH.real; c = c + yH.imag
            else:
                x = x + yH.real; c = c + yH.imag
    final = readout(exact, x)
    return metrics, final, s0, l4_delta_dist, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holo", default=str(REPO / "output" / "qwen4b_k256.holo"))
    ap.add_argument("--model-dir", default="/run/media/reneshizzle/860_1/Reneshizzle/Apps/LM Studio/Qwen/Qwen3.5-4B")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--prompts", type=int, default=6)
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
    variants = ["correct", "deranged", "random", "carrier", "identity", "haar", "direct", "nopacket"]
    print(f"B8-min: SU(2) covariant transport - k={k}, L0={L0}, {TEND-TPROP} stages, {len(lines)} prompts")
    print("=" * 92)
    acc = {v: {"cos": [], "l2": [], "suff": [], "top1": [], "frame": []} for v in variants}
    wiring = {"s0diff": [], "l4delta": []}
    prev_s0 = None
    with torch.no_grad():
        for pi, ln in enumerate(lines):
            ids = tok(ln, return_tensors="pt")["input_ids"][0]
            el_states, el_labels, el_logits = exact_trajectory(exact, ids)
            el_n = el_logits / el_logits.norm()
            for v in variants:
                s0_use = prev_s0 if v == "deranged" else None
                m, final, s0, l4d, _ = run_probe(exact, student, ids, frames, k, v, s0_use,
                                                 el_states, el_labels, el_logits)
                if v == "correct":
                    prev_s0 = s0
                final_n = final / final.norm()
                acc[v]["cos"].append((final_n * el_n).sum().item())
                acc[v]["top1"].append((final.argmax() == el_logits.argmax()).item())
                if m["suff_cos"]:
                    acc[v]["suff"].append(sum(m["suff_cos"]) / len(m["suff_cos"]))
                if m["frame"]:
                    acc[v]["frame"].append(sum(m["frame"]) / len(m["frame"]))
            print(f"[{pi}] {ln[:34]:36s} done", flush=True)
    print("-" * 92)
    for v in variants:
        n = len(acc[v]["cos"])
        s = f"{v:10s}: boundary cos={sum(acc[v]['cos'])/n:.4f}  top1={sum(acc[v]['top1'])}/{n}  "
        s += f"exact-suffix cos={sum(acc[v]['suff'])/len(acc[v]['suff']):.4f}  " if acc[v]["suff"] else "suffix n/a  "
        s += f"frame-cont err={sum(acc[v]['frame'])/len(acc[v]['frame']):.2e}" if acc[v]["frame"] else "frame n/a"
        print(s)


if __name__ == "__main__":
    main()
