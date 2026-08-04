"""B8-fresh: Sol's FRESHNESS TRIANGLE - one stage at a time, corrected decode.

From the SAME holo state at each stage t in L4mix..L7mlp (8 stages):
  1. fresh:  matched phase from THAT stage's exact odd residue
  2. stale:  L3 phase transported to that stage (polar chain)
  3. deranged: fresh phases from a DIFFERENT prompt (prompt-0 cache)
  4. random:  uniform phases
  5. nopacket: no coupling (baseline)

CORRECTED decode (per Sol):
  c    = F^+ y_H                 (holo coefficients)
  y_perp = y_H - F c             (orthogonal complement - PRESERVED)
  s    = |c| (.) unit(F^+ b)     (holo-side magnitudes)
  c'   = (c + i s)/sqrt2         (SU(2) coupling)
  s'   = (i c + s)/sqrt2         (retained rail)
  y'   = y_perp + F c'           (complement + coupled frame)

Metrics per stage per variant: local hidden cos vs exact state at t,
exact-suffix boundary logit cos + top1. Interpretation rules per Sol:
  fresh > deranged/random but stale fails  -> local info, transport stales
  fresh fails too                         -> phase packet not boundary-useful
  fresh and stale both work               -> corrected multistage rerun
  all == nopacket                         -> close this construction
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
T0 = 2 * L0 + 1       # extraction stage (L3mlp)
TSTART = 2 * (L0 + 1)  # first freshness stage (L4mix) = 8
TEND = 2 * (L0 + 4)    # last + 1 (L7mlp = 15) = 16


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


def trajectory(exact, student, ids, student_ok=True):
    """Dual-rail forward through all 32 layers, capturing per-stage
    outputs and post-stage real states for both branches."""
    x = student._embed(ids.unsqueeze(0)).to("cuda").float() if student_ok else \
        exact._embed(ids.unsqueeze(0)).to("cuda").float()
    seq, d = x.shape[1], x.shape[2]
    c = carrier(seq, d, x.device).unsqueeze(0).imag.clone()
    xE = exact._embed(ids.unsqueeze(0)).to("cuda").float()
    cE = carrier(seq, d, xE.device).unsqueeze(0).imag.clone()
    outH, outE, statesH, statesE = [], [], [], []
    for l in range(32):
        for stage in ("mix", "mlp"):
            prefix = f"model.language_model.layers.{l}"
            nw = exact._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda").float()
            pw = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda").float()
            norm_w = nw if stage == "mix" else pw
            x_n, c_n = rms_norm_complex(x, c, norm_w)
            xE_n, cE_n = rms_norm_complex(xE, cE, norm_w)
            if stage == "mix":
                yH = (student._prefill_layer_mixer(l, x_n.to(x.dtype), None).float()
                      + 1j * student._prefill_layer_mixer(l, c_n.to(c.dtype), None).float())
                yE = (exact._prefill_layer_mixer(l, xE_n.to(xE.dtype), None).float()
                      + 1j * exact._prefill_layer_mixer(l, cE_n.to(cE.dtype), None).float())
            else:
                yH = (student._mlp(x_n, l, None).float()
                      + 1j * student._mlp(c_n, l, None).float())
                yE = (exact._mlp(xE_n, l, None).float()
                      + 1j * exact._mlp(cE_n, l, None).float())
            x = x + yH.real; c = c + yH.imag
            xE = xE + yE.real; cE = cE + yE.imag
            outH.append(yH); outE.append(yE)
            statesH.append(x.clone()); statesE.append(xE.clone())
    return outH, outE, statesH, statesE, x, c


def readout(exact, x, c=None):
    nw = exact._exact_weight("model.language_model.norm.weight").to("cuda").float()
    if c is None:
        r = torch.sqrt((x[0, -1].float() ** 2).mean(-1, keepdim=True) + EPS)
        return exact._lm_head(x[0, -1].float() * (1.0 + nw) / r).float()
    r = torch.sqrt(((x[0, -1].float() ** 2) + (c[0, -1].float() ** 2)).mean(-1, keepdim=True) + EPS)
    return exact._lm_head(x[0, -1].float() * (1.0 + nw) / r).float()


def exact_suffix_real(exact, x, t):
    """Continue the full-sequence REAL state with exact layers after
    stage t to the boundary."""
    start = t + 1
    for s in range(start, 64):
        l, stage = s // 2, ("mix" if s % 2 == 0 else "mlp")
        prefix = f"model.language_model.layers.{l}"
        nw = exact._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda").float()
        pw = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda").float()
        norm_w = nw if stage == "mix" else pw
        x_n = x * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + EPS).to(x.dtype) * (1.0 + norm_w).to(x.dtype)
        if stage == "mix":
            y = exact._prefill_layer_mixer(l, x_n, None)
        else:
            y = exact._mlp(x_n, l, None)
        x = x + y.to(x.dtype)
    return readout(exact, x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holo", default=str(REPO / "output" / "qwen4b_k256.holo"))
    ap.add_argument("--model-dir", default="/run/media/reneshizzle/860_1/Reneshizzle/Apps/LM Studio/Qwen/Qwen3.5-4B")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--prompts", type=int, default=4)
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
    variants = ["fresh", "stale", "deranged", "random", "nopacket"]
    print(f"B8-fresh: freshness triangle - k={k}, stages {TSTART}..{TEND-1}, {len(lines)} prompts")
    print("=" * 92)
    # per-stage aggregation: {stage: {variant: {local, suff, top1}}}
    agg = {}
    fresh_cache = None  # prompt-0 per-stage fresh phases (deranged source)
    with torch.no_grad():
        for pi, ln in enumerate(lines):
            ids = tok(ln, return_tensors="pt")["input_ids"][0]
            outH, outE, stH, stE, xF, cF = trajectory(exact, student, ids)
            el_log = readout(exact, xF, cF)
            # nopacket suffix reference per stage: continue holo state (real)
            my_cache = {}
            for t in range(TSTART, TEND):
                F = frames[t].to("cuda")
                yH, yE = outH[t], outE[t]
                b = (yE - yH) / (2**0.5)
                c_t = F.T.to(torch.complex64) @ yH.reshape(-1, yH.shape[-1]).T
                s_fresh = c_t.abs() * torch.exp(1j * torch.angle(F.T.to(torch.complex64) @ b.reshape(-1, b.shape[-1]).T))
                my_cache[t] = s_fresh.clone()
                if t not in agg:
                    agg[t] = {v: {"local": [], "suff": [], "top1": []} for v in variants}
                for v in variants:
                    if v == "nopacket":
                        y_corr = yH
                    else:
                        if v == "fresh":
                            s = s_fresh
                        elif v == "stale":
                            # L3 packet transported via polar chain T0+1..t
                            F0 = frames[T0].to("cuda")
                            s0 = (F0.T.to(torch.complex64) @ (outE[T0] - outH[T0]).reshape(-1, outH[T0].shape[-1]).T)
                            s0 = s0 / s0.abs().clamp_min(EPS)
                            sp = s0
                            for tt in range(T0 + 1, t + 1):
                                Q = polar(frames[tt].to("cuda").T @ frames[tt - 1].to("cuda"))
                                sp = Q.to(torch.complex64) @ sp
                            s = c_t.abs() * torch.exp(1j * torch.angle(sp))
                        elif v == "deranged":
                            if fresh_cache is not None:
                                sp_ = fresh_cache[t].to(yH.device)
                                rows = torch.arange(c_t.shape[1], device=sp_.device).clamp(max=sp_.shape[1] - 1)
                                s = sp_[:, rows]
                            else:
                                s = s_fresh  # prompt 0: self-deranged fallback
                        else:  # random
                            g = torch.randn(k, yH.shape[1], device=yH.device)
                            s = c_t.abs() * torch.exp(1j * torch.angle(g))
                        y_perp = yH - (F.to(torch.complex64) @ c_t).T.reshape(-1, yH.shape[-1]).reshape_as(yH)
                        c_p = (c_t + 1j * s) / (2**0.5)
                        y_corr = y_perp + (F.to(torch.complex64) @ c_p).T.reshape(-1, yH.shape[-1]).reshape_as(yH)
                    # local hidden cos vs exact state at t (last token)
                    x_new = (stH[t] - yH).cpu() + y_corr.cpu().real
                    xE_t = stE[t].cpu()
                    cos_l = torch.nn.functional.cosine_similarity(
                        x_new[0, -1].float().view(1, -1), xE_t[0, -1].float().view(1, -1)).item()
                    agg[t][v]["local"].append(cos_l)
                    # exact-suffix boundary readout (full-sequence real suffix)
                    suf = exact_suffix_real(exact, x_new.to("cuda"), t)
                    agg[t][v]["suff"].append(torch.nn.functional.cosine_similarity(
                        suf.view(1, -1), el_log.view(1, -1)).item())
                    agg[t][v]["top1"].append((suf.argmax() == el_log.argmax()).item())
            if fresh_cache is None:
                fresh_cache = {t: my_cache[t].clone() for t in my_cache}
            print(f"[{pi}] {ln[:34]:36s} done", flush=True)
    print("-" * 92)
    print(f"{'stage':8s}" + "".join(f"{v:>12s}" for v in variants))
    for t in range(TSTART, TEND):
        row = f"t{t:02d}  "
        for v in variants:
            n = len(agg[t][v]["local"])
            row += f"{sum(agg[t][v]['local'])/n:10.4f}  "
        print(row)
    print("exact-suffix cos:")
    for t in range(TSTART, TEND):
        row = f"t{t:02d}  "
        for v in variants:
            n = len(agg[t][v]["suff"])
            row += f"{sum(agg[t][v]['suff'])/n:10.4f}  "
        print(row)


if __name__ == "__main__":
    main()
