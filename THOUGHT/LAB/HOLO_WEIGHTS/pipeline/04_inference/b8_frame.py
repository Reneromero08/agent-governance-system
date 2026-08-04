"""B8-frame: REAL-world exact framed replacement control (no dual world).

The dual-world correction cannot map cleanly onto the real trajectory
(wiring identity dev 0.39). Sol's decision tree needs the clean real-
world control: does replacing the k=32 frame component of the REAL holo
output with the EXACT one move the real boundary at all?

  c_real  = F^+ yH_real          (real frame coefficients)
  c_Ereal = F^+ yE_real
  y_corr  = yH_real + F (c_Ereal - c_real)
  h'      = h_real - yH_real + y_corr

Variants: framed (exact frame replacement) | nopacket.
Metrics: OUTPUT cos + relL2 vs exact output (PRIMARY), hidden relL2,
real exact suffix logit cos. If framed == nopacket -> the k=32 frame is
not a boundary-control surface, closing this frame regardless of any
phase packet content.
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
    nw = exact._exact_weight("model.language_model.norm.weight").to("cuda").float()
    logits = exact._lm_head(rms_norm_real(x, nw))[0, -1].float()
    return outs, states, logits


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
    print(f"B8-frame: real-world exact framed replacement - k={k}, stages {EXEC}, {len(lines)} prompts")
    print("=" * 92)
    res = {v: {t: {"cos": [], "l2": [], "hl2": [], "suf": []} for t in EXEC}
           for v in ["framed", "nopacket"]}
    resid_in, resid_out = [], []
    with torch.no_grad():
        for pi, ln in enumerate(lines):
            ids = tok(ln, return_tensors="pt")["input_ids"][0]
            e_out, e_states, el = real_forward(exact, exact, ids)
            h_out, h_states, _ = real_forward(student, exact, ids)
            el_n = el / el.norm()
            nw = exact._exact_weight("model.language_model.norm.weight").to("cuda").float()
            for t in EXEC:
                F = frames[t].to("cuda")
                yH, yE = h_out[t], e_out[t]
                d_ = yH.shape[-1]
                cH = F.T @ yH.reshape(-1, d_).T  # k x seq (real)
                cE = F.T @ yE.reshape(-1, d_).T
                # residual energy split on the REAL residue
                flat = (yE - yH).reshape(-1, d_).T  # d x seq
                Fproj = F @ (F.T @ flat)
                resid_in.append((Fproj ** 2).sum().sqrt().item() / flat.norm().item())
                resid_out.append(((flat - Fproj) ** 2).sum().sqrt().item() / flat.norm().item())
                corr = (F @ (cE - cH)).T.reshape_as(yH)
                for v in ["framed", "nopacket"]:
                    y_corr = yH + corr if v == "framed" else yH
                    cos_y = torch.nn.functional.cosine_similarity(
                        y_corr[0, -1].float().view(1, -1), yE[0, -1].float().view(1, -1)).item()
                    l2_y = (y_corr[0, -1].float() - yE[0, -1].float()).norm().item() / \
                        yE[0, -1].float().norm().item()
                    h_corr = (h_states[t] - yH) + y_corr
                    hl2 = (h_corr[0, -1].float() - e_states[t][0, -1].float()).norm().item() / \
                        e_states[t][0, -1].float().norm().item()
                    res[v][t]["cos"].append(cos_y); res[v][t]["l2"].append(l2_y)
                    res[v][t]["hl2"].append(hl2)
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
                    res[v][t]["suf"].append(torch.nn.functional.cosine_similarity(
                        (suf / suf.norm()).view(1, -1), el_n.view(1, -1)).item())
            print(f"[{pi}] {ln[:34]:36s} done", flush=True)
    print("-" * 92)
    print("REAL-residue energy in/out of F (norm fractions): in=%.4f out=%.4f" %
          (sum(resid_in) / len(resid_in), sum(resid_out) / len(resid_out)))
    for name, key in (("OUTPUT cos", "cos"), ("OUTPUT relL2", "l2"),
                      ("HIDDEN relL2", "hl2"), ("SUFFIX cos", "suf")):
        print(name)
        hdr = "t    " + "".join(f"{v:>11s}" for v in ["framed", "nopacket"])
        print(hdr)
        for t in EXEC:
            row = f"t{t:02d} "
            for v in ["framed", "nopacket"]:
                row += f"{sum(res[v][t][key])/len(res[v][t][key]):11.6f}"
            print(row)


if __name__ == "__main__":
    main()
