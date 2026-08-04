"""B8-fresh3: Sol's final gauge-aligned freshness probe.

Gauge fix: decoder c' = (c + i s)/sqrt2 rotates the rail by +pi/2.
  d = c_E - c                        (odd-residue frame coefficients)
  s_neutral = -i(sqrt2 - 1) c        (neutral rail: (c + i s_neutral)/sqrt2 == c)
  p = |c| (.) unit(-i d)             (phase-only packet, gauge-aligned: i p aligned with d)
  s = s_neutral + p
  c' = c + i p / sqrt2               (exact source contributes only the odd-residue PHASE)
  y' = y_perp + F c' = y_H + F(i p)/sqrt2

Built-in wiring check: p=0 (neutral) must reproduce nopacket EXACTLY.
Wiring identity: h'_oracle - h'_nopacket == Re(F(c_E - c_H)) - printed.

Variants: correct (unit(-i d)) | deranged (strict cyclic) | random |
neutral (= nopacket check) | exactframed (y_perp + F c_E).
PRIMARY metric: immediate output cos + relL2 (Re(y') vs exact real
output). Secondary: real exact suffix logits.

Decision per Sol: correct > deranged/random locally -> phase carries
local info (prior decoder was wrong gauge; transport still separately
falsified). correct == controls -> close the phase-only construction.
exactframed ineffective -> close this k=32 frame regardless.
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline" / "04_inference"))

from qwen35_holo_engine import Qwen35HoloEngine, load_holo, load_original, _configure_from_dir  # noqa: E402

EPS = 1e-5
TSTART = 8
TEND = 16
EXEC = list(range(TSTART, TEND))


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


def real_forward(eng, exact, ids, layers):
    x = eng._embed(ids.unsqueeze(0)).to("cuda").float()
    outs, states = [], []
    for l in layers:
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
    return outs, states, x


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
    variants = ["correct", "deranged", "random", "neutral", "exactframed"]
    print(f"B8-fresh3: gauge-aligned freshness - k={k}, stages {EXEC}, {len(lines)} prompts")
    print("=" * 92)
    out_cos = {v: {t: [] for t in EXEC} for v in variants}
    out_l2 = {v: {t: [] for t in EXEC} for v in variants}
    h_l2 = {v: {t: [] for t in EXEC} for v in variants}
    suf_res = {v: {t: [] for t in EXEC} for v in variants}
    neutral_checks, wiring_ids, all_pkts = [], [], {}
    with torch.no_grad():
        for pi, ln in enumerate(lines):
            ids = tok(ln, return_tensors="pt")["input_ids"][0]
            e_out, e_states, e_final = real_forward(exact, exact, ids, range(32))
            nw = exact._exact_weight("model.language_model.norm.weight").to("cuda").float()
            el_real = exact._lm_head(rms_norm_real(e_final, nw))[0, -1].float()
            el_n = el_real / el_real.norm()
            h_out, h_states, _ = real_forward(student, exact, ids, range(32))
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
                d_ = c_E - c_t
                pkts[t] = {"c": c_t, "cE": c_E, "d": d_}
            all_pkts[pi] = pkts
            der_pkts = all_pkts.get((pi - 1) % args.prompts)
            for t in EXEC:
                F = frames[t].to("cuda")
                yH, yE = dual_outH[t], dual_outE[t]
                c_t, c_E, d_ = pkts[t]["c"], pkts[t]["cE"], pkts[t]["d"]
                yH_real, yE_real = h_out[t], e_out[t]
                s_neutral = (-1j * (2**0.5 - 1) * c_t).to(torch.complex64)
                for v in variants:
                    if v == "neutral":
                        s = s_neutral
                    elif v == "exactframed":
                        y_corr = yH - (F.to(torch.complex64) @ c_t).T.reshape(-1, d).reshape_as(yH) \
                            + (F.to(torch.complex64) @ c_E).T.reshape(-1, d).reshape_as(yH)
                        s = None
                    else:
                        if v == "correct":
                            p = c_t.abs() * torch.exp(1j * torch.angle(-1j * d_))
                        elif v == "deranged":
                            if der_pkts is not None:
                                pd_ = der_pkts[t]["d"].to(yH.device)
                                rows = torch.arange(c_t.shape[1], device=pd_.device).clamp(max=pd_.shape[1] - 1)
                                p = c_t.abs() * torch.exp(1j * torch.angle(-1j * pd_[:, rows]))
                            else:
                                p = c_t.abs() * torch.exp(1j * torch.angle(-1j * d_))
                        else:  # random
                            g = torch.randn(k, c_t.shape[1], device=yH.device)
                            p = c_t.abs() * torch.exp(1j * torch.angle(g))
                        s = s_neutral + p
                        c_p = (c_t + 1j * s) / (2**0.5)
                        y_corr = yH - (F.to(torch.complex64) @ c_t).T.reshape(-1, d).reshape_as(yH) \
                            + (F.to(torch.complex64) @ c_p).T.reshape(-1, d).reshape_as(yH)
                    if v != "exactframed":
                        # wiring check: neutral must reproduce nopacket exactly
                        if v == "neutral":
                            c_pn = (c_t + 1j * s) / (2**0.5)
                            y_n = yH - (F.to(torch.complex64) @ c_t).T.reshape(-1, d).reshape_as(yH) \
                                + (F.to(torch.complex64) @ c_pn).T.reshape(-1, d).reshape_as(yH)
                            neutral_checks.append((y_n - yH).abs().max().item())
                    # PRIMARY: output cos + relL2 (Re(y') vs exact real output)
                    cos_y = torch.nn.functional.cosine_similarity(
                        y_corr[0, -1].real.float().view(1, -1), yE_real[0, -1].float().view(1, -1)).item()
                    l2_y = (y_corr[0, -1].real.float() - yE_real[0, -1].float()).norm().item() / \
                        yE_real[0, -1].float().norm().item()
                    out_cos[v][t].append(cos_y)
                    out_l2[v][t].append(l2_y)
                    h_corr = (h_states[t] - yH_real) + y_corr.real
                    l2_h = (h_corr[0, -1].float() - e_states[t][0, -1].float()).norm().item() / \
                        e_states[t][0, -1].float().norm().item()
                    h_l2[v][t].append(l2_h)
                    # wiring identity: h'_oracle - h'_nopacket == Re(F(c_E - c_H))
                    if v == "exactframed":
                        lhs = h_corr[0, -1].float() - h_states[t][0, -1].float()
                        rhs = (F.to(torch.complex64) @ (c_E - c_t))[..., 0].real
                        wiring_ids.append((lhs.cpu() - rhs.cpu()).abs().max().item())
                    # SECONDARY: real exact suffix logits
                    suf_x = h_corr
                    for s in range(t + 1, 64):
                        ll, stg = s // 2, ("mix" if s % 2 == 0 else "mlp")
                        prefix = f"model.language_model.layers.{ll}"
                        nw_ = exact._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda").float()
                        pw_ = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda").float()
                        norm_w = nw_ if stg == "mix" else pw_
                        xn = rms_norm_real(suf_x, norm_w)
                        yy = (exact._prefill_layer_mixer(ll, xn, None) if stg == "mix"
                              else exact._mlp(xn, ll, None)).float()
                        suf_x = suf_x + yy.to(suf_x.dtype)
                    suf = exact._lm_head(rms_norm_real(suf_x, nw))[0, -1].float()
                    suf_res[v][t].append(torch.nn.functional.cosine_similarity(
                        (suf / suf.norm()).view(1, -1), el_n.view(1, -1)).item())
            print(f"[{pi}] {ln[:34]:36s} done", flush=True)
    print("-" * 92)
    print("neutral-rail wiring check (max |y_neutral - yH|): %.3e" % max(neutral_checks))
    print("wiring identity max dev (h'_framed - h'_nopacket vs Re(F(cE-cH))): %.3e" % max(wiring_ids))
    for name, tbl in (("OUTPUT cos (PRIMARY)", out_cos), ("OUTPUT relL2 (PRIMARY)", out_l2),
                      ("HIDDEN relL2", h_l2), ("SUFFIX cos (secondary)", suf_res)):
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
