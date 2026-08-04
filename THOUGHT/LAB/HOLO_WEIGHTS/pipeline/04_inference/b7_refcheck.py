"""B7-ref: Sol's reference validation for the manual complex loop.

1. Carrier-zero equivalence: manual complex loop with c=0 must reproduce
   the exact engine's real forward, layer by layer (hidden cosine per
   layer + final real-boundary logit agreement).
2. Conjugation equivariance: F(z_bar) = conj(F(z)) for real weights.
3. Radial null: does the pure-exact correction's L2 gain come from norm
   repair alone? y_radial = yH * (||yE||/||yH||) leaves cosine unchanged;
   compare its relL2/norm against pure-exact's.
"""
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


def manual_loop(exact, ids, with_carrier: bool = False, hidden_trace: dict | None = None):
    """Manual complex-modulus-norm forward through the EXACT weights.
    c=0 unless with_carrier. Returns (x_final, per-layer hidden)."""
    x = exact._embed(ids.unsqueeze(0)).to("cuda").float()
    seq, d = x.shape[1], x.shape[2]
    if with_carrier:
        t = torch.arange(seq, device=x.device).float()
        j = torch.arange(d, device=x.device).float()
        car = torch.exp(1j * 2 * torch.pi * (t[:, None] + 1) * (j[None, :] + 1) / (seq * d))
        c = car.unsqueeze(0).imag.clone()
    else:
        c = torch.zeros_like(x)
    for l in range(32):
        prefix = f"model.language_model.layers.{l}"
        nw = exact._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda").float()
        x_n, c_n = rms_norm_complex(x, c, nw)
        x_n = x_n.to(x.dtype); c_n = c_n.to(c.dtype)
        yE = (exact._prefill_layer_mixer(l, x_n, None).float()
              + 1j * exact._prefill_layer_mixer(l, c_n, None).float())
        x = x + yE.real; c = c + yE.imag
        if hidden_trace is not None:
            hidden_trace[l] = x[0, -1].cpu()
        pw = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda").float()
        m_n, m_c = rms_norm_complex(x, c, pw)
        m_n = m_n.to(x.dtype); m_c = m_c.to(c.dtype)
        mE = (exact._mlp(m_n, l, None).float()
              + 1j * exact._mlp(m_c, l, None).float())
        x = x + mE.real; c = c + mE.imag
    return x, c


def main() -> None:
    import argparse
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/run/media/reneshizzle/860_1/Reneshizzle/Apps/LM Studio/Qwen/Qwen3.5-4B")
    args = ap.parse_args()
    MD = Path(args.model_dir)
    _configure_from_dir(MD)
    orig = load_original(MD)
    tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
    exact = Qwen35HoloEngine(None, orig, exact=True, device="cuda", verbose=False)

    lines = [l.strip() for l in (REPO / "config" / "heldout.txt").read_text().splitlines() if l.strip()]
    lines += ["The quantum mechanical measurement problem asks", "Mathematics is the language in which nature"]
    print("B7-ref: Sol's reference validation")
    print("=" * 78)

    # 1. carrier-zero equivalence, layer by layer
    layer_cos = []
    for ln in lines:
        ids = tok(ln, return_tensors="pt")["input_ids"][0]
        with torch.no_grad():
            trace = {}
            x_manual, _ = manual_loop(exact, ids, with_carrier=False, hidden_trace=trace)
            # engine real forward, capture hidden per layer
            h = exact._embed(ids.unsqueeze(0))
            for l in range(32):
                prefix = f"model.language_model.layers.{l}"
                nw = exact._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda")
                n_in = exact._rms_offset(h, nw)
                mixed = exact._prefill_layer_mixer(l, n_in, None)
                h = h + mixed.to(h.dtype)
                post = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda")
                m_in = exact._rms_offset(h, post)
                h = h + exact._mlp(m_in, l, None).to(h.dtype)
                if l in trace:
                    c_ = torch.nn.functional.cosine_similarity(trace[l].view(1, -1),
                                                               h[0, -1].cpu().float().view(1, -1)).item()
                    layer_cos.append(c_)
            # real-boundary logit agreement
            nw = exact._exact_weight("model.language_model.norm.weight").to("cuda").float()
            r = torch.sqrt((x_manual[0, -1].float() ** 2).mean(-1, keepdim=True) + EPS)
            log_man = exact._lm_head(x_manual[0, -1].float() * (1.0 + nw) / r).float()
            log_eng = exact.prefill(ids)[0, -1].float()
            c_ = torch.nn.functional.cosine_similarity(log_man.view(1, -1), log_eng.view(1, -1)).item()
            l2_ = (log_man - log_eng).norm().item() / log_eng.norm().item()
            print(f"[{ln[:38]:40s}] logit cos={c_:.4f} relL2={l2_:.4f}")
    print(f"hidden cos across layers: min={min(layer_cos):.4f} mean={sum(layer_cos)/len(layer_cos):.4f}")

    # 2. conjugation equivariance on the first line
    ln = lines[0]
    ids = tok(ln, return_tensors="pt")["input_ids"][0]
    with torch.no_grad():
        # z and z-bar: embed real; conjugate the carrier
        x1, c1 = manual_loop(exact, ids, with_carrier=True)
        x2, c2 = manual_loop(exact, ids, with_carrier=True)  # need -carrier variant
    # manual conjugated run: flip carrier sign
    x = exact._embed(ids.unsqueeze(0)).to("cuda").float()
    seq, d = x.shape[1], x.shape[2]
    t = torch.arange(seq, device=x.device).float()
    j = torch.arange(d, device=x.device).float()
    car = torch.exp(1j * 2 * torch.pi * (t[:, None] + 1) * (j[None, :] + 1) / (seq * d))
    c = (-car.unsqueeze(0).imag.clone())
    with torch.no_grad():
        for l in range(32):
            prefix = f"model.language_model.layers.{l}"
            nw = exact._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda").float()
            x_n, c_n = rms_norm_complex(x, c, nw)
            yE = (exact._prefill_layer_mixer(l, x_n.to(x.dtype), None).float()
                  + 1j * exact._prefill_layer_mixer(l, c_n.to(c.dtype), None).float())
            x = x + yE.real; c = c + yE.imag
            pw = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda").float()
            m_n, m_c = rms_norm_complex(x, c, pw)
            mE = (exact._mlp(m_n.to(x.dtype), l, None).float()
                  + 1j * exact._mlp(m_c.to(c.dtype), l, None).float())
            x = x + mE.real; c = c + mE.imag
    # F(zbar) should be conj(F(z)): compare x2+i*c2 with conj(x1+i*c1)
    z1 = x1 + 1j * c1
    z2 = x + 1j * c
    eq = (torch.conj(z1) - z2).abs().max().item() / z1.abs().max().item()
    print(f"conjugation equivariance: max rel dev = {eq:.2e}")

    # 3. radial null on the first line
    student = Qwen35HoloEngine(load_holo(str(REPO / "output" / "qwen4b_k256.holo")), orig,
                               device="cuda", verbose=False)
    with torch.no_grad():
        h = student._embed(ids.unsqueeze(0))
        for l in range(32):
            prefix = f"model.language_model.layers.{l}"
            nw = exact._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda")
            n_in = student._rms_offset(h, nw)
            h = h + student._prefill_layer_mixer(l, n_in, None).to(h.dtype)
            post = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda")
            m_in = student._rms_offset(h, post)
            h = h + student._mlp(m_in, l, None).to(h.dtype)
        nw = exact._exact_weight("model.language_model.norm.weight").to("cuda").float()
        r = torch.sqrt((h[0, -1].float() ** 2).mean(-1, keepdim=True) + EPS)
        log_h = student._lm_head(h[0, -1].float() * (1.0 + nw) / r).float()
        log_e = exact.prefill(ids)[0, -1].float()
        # radial repair: scale holo logits to exact norm
        log_r = log_h * (log_e.norm() / log_h.norm())
        c_rad = torch.nn.functional.cosine_similarity(log_r.view(1, -1), log_e.view(1, -1)).item()
        c_raw = torch.nn.functional.cosine_similarity(log_h.view(1, -1), log_e.view(1, -1)).item()
        l2_rad = (log_r - log_e).norm().item() / log_e.norm().item()
        l2_raw = (log_h - log_e).norm().item() / log_e.norm().item()
        print(f"radial null (prompt 1): cos raw={c_raw:.4f} -> cos radial={c_rad:.4f} "
              f"(unchanged by construction), relL2 {l2_raw:.4f} -> {l2_rad:.4f}")


if __name__ == "__main__":
    main()
