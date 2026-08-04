"""B8-fresh4: FIXED gauge-aligned freshness - proper random-phase null.

Bugs fixed from b8_freshness3:
  1. RANDOM CONTROL: was angle(torch.randn) = +-1 signs (real) -> Re-
     collapse-degenerate (||Re(i F p)|| = 1.5e-8). FIXED: uniform
     complex phases ph = torch.rand * 2*pi -> p = |c| (.) exp(i ph).
     Sanity: print ||Re(i F p_random)|| - must be NONZERO.
  2. DERANGEMENT: prompt 0 self-matched (its deranged source prompt 3
     did not exist yet). FIXED: precompute all prompts' fresh packets,
     strict cyclic p <- (p-1) mod N.
  3. SUFFIX DROPPED: per Sol's terminal action, no more suffix tests on
     the broken dual->real bridge. Local output metrics are PRIMARY and
     the only valid instrument now.

Construction (gauge-aligned, per Sol): d = c_E - c;
s_neutral = -i(sqrt2-1)c; p = |c| (.) unit(-i d); s = s_neutral + p;
c' = c + i p/sqrt2; y' = y_perp + F c' (dual world), applied as
h' = h_real - yH_real + Re(y') on the real trajectory.

Variants: correct | deranged | random(proper) | neutral | exactframed(dual).

Metrics (PRIMARY): output cos + relL2 of Re(y') vs exact real output.
Decision per Sol: correct > deranged AND > random -> phase carries local
info (gauge-aligned); correct == controls -> close the phase-only
construction.
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline" / "04_inference"))

from qwen35_holo_engine import Qwen35HoloEngine, load_holo, load_original, _configure_from_dir  # noqa: E402

EPS = 1e-5
EXEC = list(range(8, 16))


def rms_norm_real(x, w):
    return x * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + EPS).to(x.dtype) * (1.0 + w).to(x.dtype)


def carrier(seq, d, device):
    t = torch.arange(seq, device=device).float()
    j = torch.arange(d, device=device).float()
    return torch.exp(1j * 2 * torch.pi * (t[:, None] + 1) * (j[None, :] + 1) / (seq * d))


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


def real_forward(eng, exact, ids):
    x = eng._embed(ids.unsqueeze(0)).to("cuda").float()
    outs, states = [], []
    for l in range(32):
        prefix = f"model.language_model.layers.{l}"
        nw = exact._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda").float()
        pw = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda").float()
        for stage in ("mix", "mlp"):
            norm_w = nw if stage == "mix" else pw
            x_n = rms_norm_real(x, norm_w)
            y = (eng._prefill_layer_mixer(l, x_n, None) if stage == "mix"
                 else eng._mlp(x_n, l, None)).float()
            x = x + y.to(x.dtype)
            outs.append(y)
            states.append(x.clone())
    return outs, states


def dual_stage(exact, student, x, c, l, stage):
    prefix = f"model.language_model.layers.{l}"
    nw = exact._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda").float()
    pw = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda").float()
    norm_w = nw if stage == "mix" else pw
    r = torch.sqrt((x.float() ** 2 + c.float() ** 2).mean(-1, keepdim=True) + EPS)
    x_n = x.float() * (1.0 + norm_w) / r
    c_n = c.float() * (1.0 + norm_w) / r
    if stage == "mix":
        yH = (student._prefill_layer_mixer(l, x_n.to(x.dtype), None).float()
              + 1j * student._prefill_layer_mixer(l, c_n.to(c.dtype), None).float())
        yE = (exact._prefill_layer_mixer(l, x_n.to(x.dtype), None).float()
              + 1j * exact._prefill_layer_mixer(l, c_n.to(c.dtype), None).float())
    else:
        yH = (student._mlp(x_n, l, None).float()
              + 1j * student._mlp(c_n, l, None).float())
        yE = (exact._mlp(x_n, l, None).float()
              + 1j * exact._mlp(c_n, l, None).float())
    return yH, yE


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
    N = len(lines)
    variants = ["correct", "deranged", "random", "neutral", "exactframed"]
    print(f"B8-fresh4: FIXED gauge-aligned freshness - k={k}, stages {EXEC}, {N} prompts")
    print("  random control: uniform phases (was degenerate +-1 signs); derangement: strict cyclic")
    print("=" * 92)

    # pass 1: collect fresh packets + dual/real data for ALL prompts
    pkts_all, h_all, e_all = {}, {}, {}
    for pi, ln in enumerate(lines):
        ids = tok(ln, return_tensors="pt")["input_ids"][0]
        h_out, h_states = real_forward(student, exact, ids)
        e_out, e_states = real_forward(exact, exact, ids)
        x = student._embed(ids.unsqueeze(0)).to("cuda").float()
        seq, d = x.shape[1], x.shape[2]
        c = carrier(seq, d, x.device).unsqueeze(0).imag.clone()
        dual_outH, dual_outE = {}, {}
        for l in range(8):
            for stage in ("mix", "mlp"):
                t = 2 * l + (1 if stage == "mlp" else 0)
                yH, yE = dual_stage(exact, student, x, c, l, stage)
                dual_outH[t] = yH; dual_outE[t] = yE
                x = x + yH.real; c = c + yH.imag
        pkts = {}
        for t in EXEC:
            F = frames[t].to("cuda")
            yH, yE = dual_outH[t], dual_outE[t]
            c_t = F.T.to(torch.complex64) @ yH.reshape(-1, d).T
            c_E = F.T.to(torch.complex64) @ yE.reshape(-1, d).T
            pkts[t] = {"c": c_t, "cE": c_E, "d": c_E - c_t,
                       "yH": yH, "yE": yE, "seq": yH.shape[1], "d_": d}
        pkts_all[pi] = pkts
        h_all[pi] = (h_out, h_states)
        e_all[pi] = (e_out, e_states)
        print(f"pass1 [{pi}] {ln[:34]:36s}", flush=True)

    out_cos = {v: {t: [] for t in EXEC} for v in variants}
    out_l2 = {v: {t: [] for t in EXEC} for v in variants}
    h_l2 = {v: {t: [] for t in EXEC} for v in variants}
    rand_null, der_diff = [], []
    for pi in range(N):
        h_out, h_states = h_all[pi]
        e_out, e_states = e_all[pi]
        pkts = pkts_all[pi]
        der_pkts = pkts_all[(pi - 1) % N]  # strict cyclic: 0<-3, 1<-0, 2<-1, 3<-2
        for t in EXEC:
            F = frames[t].to("cuda")
            yH, yE = pkts[t]["yH"], pkts[t]["yE"]
            c_t, c_E, d_ = pkts[t]["c"], pkts[t]["cE"], pkts[t]["d"]
            seq, dim = pkts[t]["seq"], pkts[t]["d_"]
            yH_real, yE_real = h_out[t], e_out[t]
            s_neutral = (-1j * (2**0.5 - 1) * c_t).to(torch.complex64)
            for v in variants:
                if v == "neutral":
                    y_corr = yH
                elif v == "exactframed":
                    y_corr = yH - (F.to(torch.complex64) @ c_t).T.reshape(-1, dim).reshape_as(yH) \
                        + (F.to(torch.complex64) @ c_E).T.reshape(-1, dim).reshape_as(yH)
                else:
                    if v == "correct":
                        p = c_t.abs() * torch.exp(1j * torch.angle(-1j * d_))
                    elif v == "deranged":
                        pd_ = der_pkts[t]["d"].to(yH.device)
                        rows = torch.arange(seq, device=pd_.device).clamp(max=pd_.shape[1] - 1)
                        p = c_t.abs() * torch.exp(1j * torch.angle(-1j * pd_[:, rows]))
                    else:  # random: PROPER uniform complex phases
                        ph = torch.rand(k, seq, device=yH.device) * 2 * torch.pi
                        p = c_t.abs() * torch.exp(1j * ph)
                        if v == "random" and t == EXEC[0] and pi == 0:
                            rand_null.append((F.to(torch.complex64) @ (1j * p)).T
                                             .reshape(-1, dim).reshape_as(yH).real.norm().item())
                    s = s_neutral + p
                    c_p = (c_t + 1j * s) / (2**0.5)
                    y_corr = yH - (F.to(torch.complex64) @ c_t).T.reshape(-1, dim).reshape_as(yH) \
                        + (F.to(torch.complex64) @ c_p).T.reshape(-1, dim).reshape_as(yH)
                cos_y = torch.nn.functional.cosine_similarity(
                    y_corr[0, -1].real.float().view(1, -1), yE_real[0, -1].float().view(1, -1)).item()
                l2_y = (y_corr[0, -1].real.float() - yE_real[0, -1].float()).norm().item() / \
                    yE_real[0, -1].float().norm().item()
                h_corr = (h_states[t] - yH_real) + y_corr.real
                hl2 = (h_corr[0, -1].float() - e_states[t][0, -1].float()).norm().item() / \
                    e_states[t][0, -1].float().norm().item()
                out_cos[v][t].append(cos_y); out_l2[v][t].append(l2_y); h_l2[v][t].append(hl2)
                if v == "correct" and t == EXEC[0]:
                    pc = c_t.abs() * torch.exp(1j * torch.angle(-1j * d_))
                    pdc = der_pkts[t]["d"].to(yH.device)
                    rows = torch.arange(seq, device=pdc.device).clamp(max=pdc.shape[1] - 1)
                    pd_ = c_t.abs() * torch.exp(1j * torch.angle(-1j * pdc[:, rows]))
                    der_diff.append((pc - pd_).abs().max().item())
        print(f"pass2 [{pi}] done", flush=True)
    print("-" * 92)
    print("random-null sanity ||Re(i F p_random)||: %.4f (must be NONZERO; was 1.5e-08 when broken)"
          % (rand_null[0] if rand_null else -1))
    print("correct-vs-deranged packet max diff: %.4f (must be nonzero - derangement routing works)"
          % (max(der_diff) if der_diff else -1))
    for name, tbl in (("OUTPUT cos (PRIMARY)", out_cos), ("OUTPUT relL2 (PRIMARY)", out_l2),
                      ("HIDDEN relL2", h_l2)):
        print(name)
        hdr = "t    " + "".join(f"{v:>11s}" for v in variants)
        print(hdr)
        for t in EXEC:
            row = f"t{t:02d} "
            for v in variants:
                row += f"{sum(tbl[v][t])/len(tbl[v][t]):11.6f}"
            print(row)


if __name__ == "__main__":
    main()
