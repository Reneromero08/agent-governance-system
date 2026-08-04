"""B8-fresh2: Sol's decisive rerun - oracle-verified freshness triangle.

WORLD = REAL boundary (consistent per Sol): reference = full exact REAL
trajectory logits; local states = real holo trajectory; suffix = real
exact continuation (sanity check: suffix from exact real state at t must
reproduce the reference to tolerance). The COMPLEX dual world (carrier
rails) is used ONLY to measure the odd-residue phase channel and build
the packet; the correction is collapsed to Re at the point of injection.

Correction per stage t (Sol's algebra):
  c     = F^+ yH          (dual-world holo coefficients)
  y_perp = yH - F c       (orthogonal complement preserved)
  s     = |c| (.) unit(F^+ b)  (holo magnitudes)
  c'    = (c + i s)/sqrt2 (SU(2))
  y'    = y_perp + F c'
  h'    = h_real - yH_real + Re(y')

ORACLE control (mechanical identity): s_oracle = -i(sqrt2 c_E - c) makes
c' == c_E exactly; y' == y_perp + F c_E == the exact framed output. If the
identity error is not ~0 the decoder algebra is broken. (Thus oracle and
exactframed are the SAME control - printed once.)

Variants: fresh (that stage's exact residue) / stale (L3 transported) /
deranged (strict cyclic: prompt p <- prompt (p-1) mod 4) / random /
nopacket / oracle(=exactframed).

Metrics per stage per variant: local hidden cos (Re(h'), exact real
state), output cos (Re(y'), exact real output), exact-REAL-suffix
boundary logit cos. Residual energy in/out of F printed.

Interpretation (Sol): oracle identity fails -> broken; oracle ok and
fresh > deranged/random -> local phase info exists; oracle ok, fresh
fails but oracle/exactframed works -> phase without amplitude is
insufficient; fresh == deranged == random locally under valid reference
-> close the phase-only construction; fresh works locally but stale
does not -> extraction works, transport fails.
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
T0 = 2 * L0 + 1          # 7 (L3mlp, extraction)
TSTART = 2 * (L0 + 1)    # 8 (L4mix)
TEND = 2 * (L0 + 4) + 2  # 16 (exclusive, t15 = L7mlp included)
EXEC = list(range(TSTART, TEND))


def rms_norm_real(x, w):
    return x * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + EPS).to(x.dtype) * (1.0 + w).to(x.dtype)


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


def real_forward(eng, exact, ids, layers):
    """REAL forward through `layers` (engine eng) on the sequence.
    Returns (outputs per stage, post-stage states, final logits if full)."""
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
    variants = ["fresh", "stale", "deranged", "random", "nopacket", "oracle"]
    print(f"B8-fresh2: oracle-verified freshness - k={k}, stages {EXEC}, {len(lines)} prompts, real boundary")
    print("=" * 92)
    results = {v: {t: [] for t in EXEC} for v in variants}
    out_cos = {v: {t: [] for t in EXEC} for v in variants}
    suf_res = {v: {t: [] for t in EXEC} for v in variants}
    oracle_errs, resid_in, resid_out, bound_errs = [], [], [], []
    all_fresh = {}
    with torch.no_grad():
        for pi, ln in enumerate(lines):
            ids = tok(ln, return_tensors="pt")["input_ids"][0]
            # real exact trajectory (reference + exact states/outputs)
            e_out, e_states, e_final = real_forward(exact, exact, ids, range(32))
            nw = exact._exact_weight("model.language_model.norm.weight").to("cuda").float()
            el_real = exact._lm_head(rms_norm_real(e_final, nw))[0, -1].float()
            el_n = el_real / el_real.norm()
            # real holo trajectory
            h_out, h_states, _ = real_forward(student, exact, ids, range(32))
            # dual world through L7 for the phase channel
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
            # boundary sanity: exact real state at t, real suffix -> el_real
            for t in EXEC:
                suf_x = e_states[t]
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
                bound_errs.append((suf / suf.norm() - el_n).abs().max().item())
            # fresh packets per stage for this prompt
            fresh = {}
            for t in EXEC:
                F = frames[t].to("cuda")
                yH, yE = dual_outH[t], dual_outE[t]
                b = (yE - yH) / (2**0.5)
                c_t = F.T.to(torch.complex64) @ yH.reshape(-1, d).T
                ph = torch.exp(1j * torch.angle(F.T.to(torch.complex64) @ b.reshape(-1, d).T))
                fresh[t] = (c_t.abs() * ph).clone()
                flat = (yE - yH).reshape(-1, d).T
                Fproj = F.to(torch.complex64) @ (F.T.to(torch.complex64) @ flat)
                resid_in.append((Fproj.abs() ** 2).sum().sqrt().item() / flat.abs().norm().item())
                resid_out.append(((flat - Fproj).abs() ** 2).sum().sqrt().item() / flat.abs().norm().item())
            all_fresh[pi] = fresh
            der_src = all_fresh.get((pi - 1) % args.prompts)
            for t in EXEC:
                F = frames[t].to("cuda")
                yH, yE = dual_outH[t], dual_outE[t]
                b = (yE - yH) / (2**0.5)
                c_t = F.T.to(torch.complex64) @ yH.reshape(-1, d).T
                c_E = F.T.to(torch.complex64) @ yE.reshape(-1, d).T
                y_perp = yH - (F.to(torch.complex64) @ c_t).T.reshape(-1, d).reshape_as(yH)
                yH_real, yE_real = h_out[t], e_out[t]
                for v in variants:
                    if v == "nopacket":
                        y_corr = yH
                    elif v == "oracle":
                        s = (-1j * ((2**0.5) * c_E - c_t)).to(torch.complex64)
                        c_p = (c_t + 1j * s) / (2**0.5)
                        oracle_errs.append((c_p - c_E).abs().max().item())
                        y_corr = y_perp + (F.to(torch.complex64) @ c_p).T.reshape(-1, d).reshape_as(yH)
                    else:
                        if v == "fresh":
                            s = fresh[t]
                        elif v == "stale":
                            F0 = frames[T0].to("cuda")
                            s0 = (F0.T.to(torch.complex64) @ (dual_outE[T0] - dual_outH[T0]).reshape(-1, d).T)
                            s0 = s0 / s0.abs().clamp_min(EPS)
                            sp = s0
                            for tt in range(T0 + 1, t + 1):
                                Q = polar(frames[tt].to("cuda").T @ frames[tt - 1].to("cuda"))
                                sp = Q.to(torch.complex64) @ sp
                            s = c_t.abs() * torch.exp(1j * torch.angle(sp))
                        elif v == "deranged":
                            if der_src is not None:
                                sd = der_src[t].to(yH.device)
                                rows = torch.arange(c_t.shape[1], device=sd.device).clamp(max=sd.shape[1] - 1)
                                s = sd[:, rows]
                            else:
                                s = fresh[t]
                        else:  # random
                            g = torch.randn(k, c_t.shape[1], device=yH.device)
                            s = c_t.abs() * torch.exp(1j * torch.angle(g))
                        c_p = (c_t + 1j * s) / (2**0.5)
                        y_corr = y_perp + (F.to(torch.complex64) @ c_p).T.reshape(-1, d).reshape_as(yH)
                    # local metrics on the REAL boundary
                    h_corr = (h_states[t] - yH_real) + y_corr.real
                    cos_h = torch.nn.functional.cosine_similarity(
                        h_corr[0, -1].float().view(1, -1), e_states[t][0, -1].float().view(1, -1)).item()
                    cos_y = torch.nn.functional.cosine_similarity(
                        y_corr[0, -1].real.float().view(1, -1), yE_real[0, -1].float().view(1, -1)).item()
                    results[v][t].append(cos_h)
                    out_cos[v][t].append(cos_y)
                    # real exact suffix from the corrected state
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
    print("oracle identity err (max |c'_oracle - c_E|): %.3e" % max(oracle_errs))
    print("boundary sanity (exact-real-state suffix vs reference, max): %.3e" % max(bound_errs))
    print("residual energy in/out of F: in=%.4f out=%.4f" % (sum(resid_in) / len(resid_in),
                                                             sum(resid_out) / len(resid_out)))
    for name, tbl in (("LOCAL hidden cos", results), ("OUTPUT cos", out_cos), ("SUFFIX cos", suf_res)):
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
