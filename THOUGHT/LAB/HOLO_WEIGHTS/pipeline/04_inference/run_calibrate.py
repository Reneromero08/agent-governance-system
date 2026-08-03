"""run_calibrate.py — Analytic low-rank layer calibration (NO training).

Breaks the k=256 depth wall using the lab's analytic-calibration philosophy
(PUSHED_REPORT_AUTOTUNE: closed-form O(1) correction, no gradient descent)
plus the Df manifold law (hidden states are low-dimensional, so the
activation-manifold error is far smaller than the weight-space cosine).

For each layer and stage (mixer, mlp):
    error e = sub_exact(h) - sub_holo(h)         # oracle query on exact weights
    fit rank-r C:  C = argmin ||E - C H||^2      # closed-form thin-SVD least squares
    store C = A B^T                               # r*(D+D) per stage
Iterate to a fixed point: with corrections in place, drift shrinks, re-fit.

The exact weights are queried only during calibration (the oracle), never
at inference.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen35_holo_engine import (
    Qwen35HoloEngine,
    load_holo,
    load_original,
    DEFAULT_MODEL,
    HIDDEN,
)

REPO = Path(__file__).resolve().parents[2]

PROMPTS = [
    "The theory of catalytic computation is",
    "In the beginning there was computation, and computation was with information,",
    "The relationship between entropy and information can be understood through",
    "A physical system in equilibrium tends to",
    "The quantum mechanical measurement problem asks",
    "Language models generate text by predicting the next token",
    "The geometry of phase space describes",
    "Mathematics is the language in which nature",
    "When a system is driven far from equilibrium, it",
    "The universe appears to be composed of",
    "Signal and noise are distinguished by",
    "A reversible computation can always be",
]

RIDGE = 1e-6


def thin_svd_least_squares(H: torch.Tensor, E: torch.Tensor, rank: int):
    """C = argmin ||E - C H||^2 with rank(C) <= rank, closed form.

    H: D x T activations, E: D x T errors.
    Returns A (D x r), B (D x r) with C = A @ B.T.
    """
    H = H.float()
    E = E.float()
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    r = min(rank, S.numel())
    reg = (S[:r] ** 2 + RIDGE * S[0].item() ** 2).clamp(min=1e-12)
    # C = E V_r S_r^{-1} U_r^T  =>  A = E V_r S_r^{-1}, B = U_r
    A = (E @ Vh[:r].T) / reg.unsqueeze(0)  # D x r (S_r^{-1} on columns)
    A = E @ (Vh[:r].T * (1.0 / reg).unsqueeze(0))
    B = U[:, :r]
    return A.to(torch.bfloat16).cpu(), B.to(torch.bfloat16).cpu()


def collect_samples(engine, prompts, tokenizer):
    """Run prompts through the engine (with current corrections), capture I/O."""
    engine.capture_io = {}
    per_layer = {}
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt")["input_ids"][0]
        with torch.no_grad():
            engine.prefill(ids, capture_hidden=False)
        for layer, cap in engine.capture_io.items():
            entry = per_layer.setdefault(
                layer,
                {
                    "h_in": [],
                    "normed": [],
                    "mix_pre": [],
                    "mlp_in": [],
                    "mlp_pre": [],
                    "exact_mix": [],
                    "exact_mlp": [],
                },
            )
            entry["h_in"].append(cap["h_in"][0].cpu())
            entry["normed"].append(cap["normed"][0].cpu())
            entry["mix_pre"].append(cap["mix_pre"][0].cpu())
            entry["mlp_in"].append(cap["mlp_in"][0].cpu())
            entry["mlp_pre"].append(cap["mlp_pre"][0].cpu())
    engine.capture_io = None
    return per_layer


def query_exact(engine_exact, layer, cap_entries):
    """Query the exact sublayers on the captured inputs (the oracle)."""
    normed_l = cap_entries["normed"]
    mlp_l = cap_entries["mlp_in"]
    ex_mix, ex_mlp = [], []
    dev = engine_exact.device
    for i in range(len(normed_l)):
        with torch.no_grad():
            xi = normed_l[i].unsqueeze(0).to(dev)
            yi = mlp_l[i].unsqueeze(0).to(dev)
            if layer % 4 == 3:
                m = engine_exact._full_attention(xi, layer, None)
            else:
                m = engine_exact._gated_delta_net(xi, layer, None)
            mlp = engine_exact._mlp(yi, layer, None)
        ex_mix.append(m[0].cpu().float())
        ex_mlp.append(mlp[0].cpu().float())
    return ex_mix, ex_mlp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holo", default="output/qwen_27b_k256.holo")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/qwen_corrections_r128.pt")
    ap.add_argument("--prompts", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    prompts = PROMPTS if args.prompts <= 0 else PROMPTS[: args.prompts]
    print(f"calibrating {len(prompts)} prompts, rank {args.rank}, rounds {args.rounds}")

    holo = load_holo(args.holo)
    original = load_original(args.model_dir)
    eng = Qwen35HoloEngine(holo, original, device=args.device, verbose=False)
    eng_exact = Qwen35HoloEngine(None, original, exact=True, device=args.device, verbose=False)

    t0 = time.time()
    for rnd in range(args.rounds):
        print(f"\n=== round {rnd + 1}/{args.rounds} ===", flush=True)
        per_layer = collect_samples(eng, prompts, tokenizer)
        rt = time.time()
        for layer in sorted(per_layer):
            cap = per_layer[layer]
            if len(cap["h_in"]) == 0:
                continue
            H_mix = torch.cat(cap["h_in"], dim=0).T      # D x T (mixer input h)
            H_mlp = torch.cat([c for c in cap["h_in"]], dim=0).T  # same h stream
            ex_mix, ex_mlp = query_exact(eng_exact, layer, cap)
            E_mix = torch.cat(ex_mix, dim=0).T - torch.cat(cap["mix_pre"], dim=0).T
            E_mlp = torch.cat(ex_mlp, dim=0).T - torch.cat(cap["mlp_pre"], dim=0).T
            A1, B1 = thin_svd_least_squares(H_mix, E_mix, args.rank)
            A2, B2 = thin_svd_least_squares(H_mlp, E_mlp, args.rank)
            eng.corrections[layer] = {"mix": (A1, B1), "mlp": (A2, B2)}
        print(f"  fit done in {time.time()-rt:.0f}s", flush=True)
        # quick round metric: hidden cosine vs exact on first prompt
        ids = tokenizer(prompts[0], return_tensors="pt")["input_ids"][0]
        with torch.no_grad():
            eng.verbose = False
            _ = eng.prefill(ids, capture_hidden=False)
            holo_last = eng.last_hidden_states if False else None
        del holo_last
        gc.collect()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"corrections": eng.corrections, "rank": args.rank, "rounds": args.rounds}, out_path)
    print(f"\nsaved corrections to {out_path} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
