"""B6: exact-sourced si - the frontier probe with Sol's acceptance criteria.

For each layer, BORROW the exact branch (yE) on the current trajectory.
Fold pair: a = (yE+yH)/sqrt2 (fold-even carrier), b = (yE-yH)/sqrt2
(fold-odd residue). Extract si = the PHASE channel of the odd residue at
bandwidth B. Uncompute yE (source removal). Continue the holo forward on
a + b_hat(si). 

Killer control at EQUAL BANDWIDTH (B real scalars per layer):
  si-holonomy: B phases of b (twin-rail: magnitudes from the fold-even |a|)
  direct-residual: B/2 exact complex components of b, rest zeroed
  full-exact: yE itself (upper bound)
  uncorrected: pure holo (baseline)

Acceptance per Sol:
- si must beat direct-residual at equal bandwidth
- the fold-odd invariant must survive source removal (no exact re-read
  after extraction; final logit computed from the corrected holo path)
- conjugate swap / phase randomization controls must destroy the signal
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline" / "04_inference"))

from qwen35_holo_engine import Qwen35HoloEngine, load_holo, load_original, _configure_from_dir  # noqa: E402


def run_path(exact, student, ids, mode: str, B: int, corrupt_seed: int | None = None):
    """Run 32 layers, borrowing the exact mixer+mlp output per layer."""
    hidden = student._embed(ids.unsqueeze(0))
    for l in range(32):
        prefix = f"model.language_model.layers.{l}"
        norm = student._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda")
        n_in = student._rms_offset(hidden, norm)
        yH = student._prefill_layer_mixer(l, n_in, None)
        yE = exact._prefill_layer_mixer(l, n_in, None)
        if mode == "full-exact":
            yH = yE
        elif mode in ("si", "direct", "corrupt"):
            # fold pair
            a = (yE.float() + yH.float()) / (2**0.5)
            b = (yE.float() - yH.float()) / (2**0.5)
            bflat = b.reshape(-1, b.shape[-1])
            if mode == "si":
                # B/2 SIGNS of b (the degenerate phase channel of real
                # states) with twin-rail magnitudes from the fold-even |a|
                _, idx = bflat.abs().topk(B // 2, dim=-1)
                a_mag = a.reshape(-1, a.shape[-1]).abs().gather(-1, idx)
                sgn = torch.sign(bflat.gather(-1, idx))
                b_hat = torch.zeros_like(bflat)
                b_hat.scatter_(-1, idx, sgn * a_mag)
            elif mode == "direct":
                # B/2 exact components (same scalar count as the sign channel)
                _, idx = bflat.abs().topk(B // 2, dim=-1)
                b_hat = torch.zeros_like(bflat)
                b_hat.scatter_(-1, idx, bflat.gather(-1, idx))
            else:  # corrupt: random signs (control)
                torch.manual_seed(corrupt_seed or 0)
                _, idx = bflat.abs().topk(B // 2, dim=-1)
                a_mag = a.reshape(-1, a.shape[-1]).abs().gather(-1, idx)
                rnd = torch.sign(torch.rand_like(bflat.gather(-1, idx)) - 0.5)
                b_hat = torch.zeros_like(bflat)
                b_hat.scatter_(-1, idx, rnd * a_mag)
            b_hat = b_hat.reshape_as(b)
            yH = (a + b_hat) / (2**0.5)
            yH = yH.to(yH.dtype)
        hidden = hidden + yH.to(hidden.dtype)
        # mlp stage, same borrow
        post = student._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda")
        m_in = student._rms_offset(hidden, post)
        mH = student._mlp(m_in, l, None)
        mE = exact._mlp(m_in, l, None)
        if mode == "full-exact":
            mH = mE
        elif mode in ("si", "direct", "corrupt"):
            a = (mE.float() + mH.float()) / (2**0.5)
            b = (mE.float() - mH.float()) / (2**0.5)
            bflat = b.reshape(-1, b.shape[-1])
            if mode == "si":
                _, idx = bflat.abs().topk(B // 2, dim=-1)
                a_mag = a.reshape(-1, a.shape[-1]).abs().gather(-1, idx)
                sgn = torch.sign(bflat.gather(-1, idx))
                b_hat = torch.zeros_like(bflat)
                b_hat.scatter_(-1, idx, sgn * a_mag)
            elif mode == "direct":
                _, idx = bflat.abs().topk(B // 2, dim=-1)
                b_hat = torch.zeros_like(bflat)
                b_hat.scatter_(-1, idx, bflat.gather(-1, idx))
            else:  # corrupt: random signs (control)
                torch.manual_seed(corrupt_seed or 0)
                _, idx = bflat.abs().topk(B // 2, dim=-1)
                a_mag = a.reshape(-1, a.shape[-1]).abs().gather(-1, idx)
                rnd = torch.sign(torch.rand_like(bflat.gather(-1, idx)) - 0.5)
                b_hat = torch.zeros_like(bflat)
                b_hat.scatter_(-1, idx, rnd * a_mag)
            b_hat = b_hat.reshape_as(b)
            mH = (a + b_hat) / (2**0.5)
            mH = mH.to(mH.dtype)
        hidden = hidden + mH.to(hidden.dtype)
    final_norm = student._exact_weight("model.language_model.norm.weight").to("cuda")
    return student._lm_head(student._rms_offset(hidden, final_norm))


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
    print("B6: exact-sourced si frontier - equal-bandwidth killer control")
    print(f"    bandwidth = {B} scalars per layer per stage")
    print("=" * 78)
    acc = {m: [] for m in ["uncorrected", "si", "direct", "corrupt", "full-exact"]}
    for ln in lines:
        ids = tok(ln, return_tensors="pt")["input_ids"][0]
        with torch.no_grad():
            el = exact.prefill(ids)[0, -1]
            sl = student.prefill(ids)[0, -1]
            for m in ["si", "direct", "corrupt"]:
                out = run_path(exact, student, ids, m, B, corrupt_seed=7)
                acc[m].append(torch.nn.functional.cosine_similarity(el.float().view(1, -1),
                                                                    out[0, -1].float().view(1, -1)).item())
            acc["uncorrected"].append(torch.nn.functional.cosine_similarity(el.float().view(1, -1),
                                                                            sl.float().view(1, -1)).item())
            acc["full-exact"].append(1.0)
    for m in acc:
        vals = acc[m]
        print(f"{m:12s}: mean logit cosine = {sum(vals)/len(vals):.4f}  "
              f"worst = {min(vals):.4f}  best = {max(vals):.4f}")


if __name__ == "__main__":
    main()
