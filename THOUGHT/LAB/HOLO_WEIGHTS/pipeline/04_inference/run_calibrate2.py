"""Activation-aware, per-projection analytic LoRA calibration (no training).

The student trajectory supplies each projection's actual input X.  On exactly
that X, the safetensors and factorized projections supply an output error E.
Ridge least squares fits E ~= X C^T, then a thin SVD converts C to the existing
engine adapter convention.  No gradients or optimizer are used.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen35_holo_engine import (  # noqa: E402
    DEFAULT_HOLO,
    DEFAULT_MODEL,
    LAYERS,
    Qwen35HoloEngine,
    _human_bytes,
    load_holo,
    load_original,
)


# Deterministic eight-token calibration sequences.  Keeping equal lengths lets
# all training prompts run as one batch while preserving the prompt dimension.
PROMPT_IDS = [
    [9707, 374, 264, 1296, 315, 16831, 323, 1146],
    [1519, 374, 279, 4226, 315, 469, 323, 845],
    [785, 3301, 315, 17834, 323, 1988, 646, 387],
    [40, 2646, 304, 264, 11230, 1691, 311, 264],
    [785, 15900, 10550, 9846, 1672, 430, 279, 1128],
    [4537, 4119, 19255, 1467, 553, 31269, 279, 1828],
    [785, 24610, 315, 9057, 2924, 27597, 264, 1691],
    [13388, 374, 279, 4128, 304, 892, 6850, 279],
    [1687, 264, 1691, 374, 13173, 2201, 504, 11230],
    [785, 15637, 8834, 311, 387, 17289, 315, 264],
    [20830, 323, 14522, 525, 34779, 553, 279, 1622],
    [32, 29344, 19255, 646, 2744, 387, 264, 1528],
]


# Real English prompts, tokenized at runtime by the Qwen tokenizer.
REAL_PROMPTS = [
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

Adapter = dict[str, Any]


def _low_rank_product(
    left: torch.Tensor, right: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate left@right to B@A without materializing the large product."""
    left = left.float()
    right = right.float()
    q_left, r_left = torch.linalg.qr(left, mode="reduced")
    q_right, r_right = torch.linalg.qr(right.transpose(0, 1), mode="reduced")
    middle = r_left @ r_right.transpose(0, 1)
    u, singular, vh = torch.linalg.svd(middle, full_matrices=False)
    keep = min(rank, singular.numel())
    root = singular[:keep].clamp_min(0).sqrt()
    b = (q_left @ u[:, :keep]) * root.unsqueeze(0)
    a = ((q_right @ vh[:keep].transpose(0, 1)) * root.unsqueeze(0)).transpose(0, 1)
    return b, a


def fit_projection(
    x: torch.Tensor,
    exact_out: torch.Tensor,
    holo_out: torch.Tensor,
    rank: int,
    ridge: float,
) -> tuple[Adapter, float, float]:
    """Solve ridge LS in the exact dual form and return an engine LoRA entry."""
    x2 = x.reshape(-1, x.shape[-1]).float()
    exact2 = exact_out.reshape(-1, exact_out.shape[-1]).float()
    holo2 = holo_out.reshape_as(exact2).float()
    error = exact2 - holo2

    # C^T=(X^T X+lI)^-1 X^T E = X^T(XX^T+lI)^-1 E.
    # Therefore C=Z^T X where (XX^T+lI)Z=E.  The largest solve dimension is
    # the token count, not the model width.
    # NORMALIZED ridge (Sol): lambda = alpha * mean(||x_i||^2); the kernel
    # diagonal is ~d ~ 5000, so an absolute lambda like 1e-3 is ~zero.
    kernel = x2 @ x2.transpose(0, 1)
    lam = ridge * kernel.diagonal().mean().item()
    kernel.diagonal().add_(lam)
    try:
        z = torch.linalg.solve(kernel, error)
    except torch.linalg.LinAlgError:
        z = torch.linalg.pinv(kernel) @ error
    b, a = _low_rank_product(z.transpose(0, 1), x2, rank)

    correction = (x2 @ a.transpose(0, 1)) @ b.transpose(0, 1)
    base_cos = float(F.cosine_similarity(exact2.reshape(-1), holo2.reshape(-1), dim=0))
    fit_cos = float(
        F.cosine_similarity(exact2.reshape(-1), (holo2 + correction).reshape(-1), dim=0)
    )
    entry = {
        "A": a.to(torch.bfloat16).cpu(),
        "B": b.to(torch.bfloat16).cpu(),
        "alpha": int(a.shape[0]),
    }
    return entry, base_cos, fit_cos


def damp_adapter(old: Adapter | None, new: Adapter, damp: float, rank: int) -> Adapter:
    if old is None:
        return new
    old_a, old_b = old["A"].float(), old["B"].float()
    new_a, new_b = new["A"].float(), new["B"].float()
    left = torch.cat(((1.0 - damp) * old_b, damp * new_b), dim=1)
    right = torch.cat((old_a, new_a), dim=0)
    b, a = _low_rank_product(left, right, rank)
    return {
        "A": a.to(torch.bfloat16).cpu(),
        "B": b.to(torch.bfloat16).cpu(),
        "alpha": int(a.shape[0]),
    }


def load_adapters(path: str | None) -> dict[str, Adapter]:
    if not path:
        return {}
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict) and "adapters" in loaded:
        loaded = loaded["adapters"]
    if not isinstance(loaded, dict):
        raise TypeError(f"adapter file is not a dictionary: {path}")
    return loaded


def save_adapters(adapters: dict[str, Adapter], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(adapters, temporary)
    temporary.replace(path)


def final_logit_cosines(
    reference: torch.Tensor, candidate: torch.Tensor, lengths: list[int] | None = None
) -> list[float]:
    """Last-REAL-token logit cosine per sequence (mask out padding)."""
    out = []
    for i in range(reference.shape[0]):
        pos = (lengths[i] - 1) if lengths else (reference.shape[1] - 1)
        out.append(
            float(
                F.cosine_similarity(
                    reference[i, pos].float(), candidate[i, pos].float(), dim=0
                )
            )
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holo", default=str(DEFAULT_HOLO))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--prompts", type=int, default=12)
    parser.add_argument("--corpus", type=str, default="", help="text corpus file (one sentence per line)")
    parser.add_argument("--heldout", type=str, default="", help="held-out prompt file")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--damp", type=float, default=0.5)
    parser.add_argument("--out", default=str(DEFAULT_HOLO.parent / "qwen_adapters_proj.pt"))
    parser.add_argument("--adapters", help="optional initial adapter dictionary")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-layers", type=int, default=LAYERS, help="smoke-test aid")
    parser.add_argument("--lm-head-chunk", type=int, default=32768)
    args = parser.parse_args()
    if not 4 <= args.prompts <= len(PROMPT_IDS):
        parser.error(f"--prompts must be in [4, {len(PROMPT_IDS)}]")
    if args.rank < 1 or args.ridge <= 0 or args.rounds < 1:
        parser.error("rank, ridge, and rounds must be positive")
    if not 0.0 <= args.damp <= 1.0:
        parser.error("--damp must be in [0, 1]")
    if not 1 <= args.num_layers <= LAYERS:
        parser.error(f"--num-layers must be in [1, {LAYERS}]")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.out)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if args.corpus:
        raw = Path(args.corpus).read_text().splitlines()
        prompt_strs = [ln.strip() for ln in raw if ln.strip()]
        print(f"using corpus: {len(prompt_strs)} sentences", flush=True)
    else:
        prompt_strs = REAL_PROMPTS[: args.prompts]
    heldout_strs = None
    if args.heldout:
        hr = Path(args.heldout).read_text().splitlines()
        heldout_strs = [ln.strip() for ln in hr if ln.strip()]
    ids_batch = [tokenizer(p, return_tensors="pt")["input_ids"][0].tolist() for p in prompt_strs]
    token_lengths = [len(ids) for ids in ids_batch]
    max_len = max(token_lengths)
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0)
    ids_padded = [ids + [pad_id] * (max_len - len(ids)) for ids in ids_batch]
    prompts = torch.tensor(ids_padded, dtype=torch.long)
    if heldout_strs is not None:
        ho_batch = [tokenizer(p, return_tensors="pt")["input_ids"][0].tolist() for p in heldout_strs]
        ho_lengths = [len(ids) for ids in ho_batch]
        ho_max = max(ho_lengths)
        ho_padded = [ids + [pad_id] * (ho_max - len(ids)) for ids in ho_batch]
        heldout_ids = torch.tensor(ho_padded, dtype=torch.long)
        heldout_lengths = ho_lengths
        train_ids = prompts
        prompt_strs_full = prompt_strs + heldout_strs
        print(f"held-out (file): {heldout_strs}", flush=True)
    else:
        train_ids, heldout_ids = prompts[:-2], prompts[-2:]
        heldout_lengths = token_lengths[-2:]
        prompt_strs_full = prompt_strs
        print("held-out prompts:", [prompt_strs[-2], prompt_strs[-1]], flush=True)

    original = load_original(args.model_dir)
    holo = load_holo(args.holo)
    exact = Qwen35HoloEngine(
        None, original, exact=True, device=args.device, num_layers=args.num_layers,
        lm_head_chunk=args.lm_head_chunk, verbose=False,
    )
    student = Qwen35HoloEngine(
        holo, original, exact=False, device=args.device, num_layers=args.num_layers,
        lm_head_chunk=args.lm_head_chunk, verbose=False,
    )
    adapters = load_adapters(args.adapters)
    print(f"loaded {len(adapters)} initial adapters", flush=True)

    with torch.inference_mode():
        print("computing held-out exact reference ...", flush=True)
        reference_logits = exact.prefill(heldout_ids)

        started = time.perf_counter()
        for round_index in range(1, args.rounds + 1):
            print(f"\n=== round {round_index}/{args.rounds} ===", flush=True)
            student.capture_io = {}
            student.prefill(train_ids, adapters=adapters, compute_logits=False)
            captures = student.capture_io
            student.capture_io = None
            if not captures:
                raise RuntimeError("engine produced no projection captures")

            fitted = dict(adapters)
            projection_count = 0
            for layer in sorted(captures):
                projections = captures[layer].get("proj", {})
                layer_base, layer_fit = [], []
                for name, x_cpu in projections.items():
                    if not holo.has_factor(name):
                        print(f"  L{layer:02d} skip missing factor: {name}", flush=True)
                        continue
                    x = x_cpu.to(args.device)
                    exact_out = exact._linear(
                        x, name, factorized=False, adapters=None
                    )
                    holo_out = student._linear(
                        x, name, factorized=True, adapters=None
                    )
                    new_entry, base_cos, fit_cos = fit_projection(
                        x, exact_out, holo_out, args.rank, args.ridge
                    )
                    fitted[name] = damp_adapter(
                        adapters.get(name), new_entry, args.damp, args.rank
                    )
                    layer_base.append(base_cos)
                    layer_fit.append(fit_cos)
                    projection_count += 1
                    del x, exact_out, holo_out, new_entry
                print(
                    f"  L{layer:02d}: {len(layer_base)} projections "
                    f"train cosine base={sum(layer_base)/len(layer_base):.5f} "
                    f"rank-{args.rank}={sum(layer_fit)/len(layer_fit):.5f}",
                    flush=True,
                )
                gc.collect()

            adapters = fitted
            del captures
            gc.collect()
            corrected_logits = student.prefill(heldout_ids, adapters=adapters)
            cosines = final_logit_cosines(reference_logits, corrected_logits, heldout_lengths)
            mean_cosine = sum(cosines) / len(cosines)
            print(
                f"ROUND {round_index} HELD-OUT FINAL-LOGIT COSINE: "
                f"prompt0={cosines[0]:.6f} prompt1={cosines[1]:.6f} "
                f"mean={mean_cosine:.6f}",
                flush=True,
            )
            save_adapters(adapters, output)
            print(
                f"saved {projection_count} fitted / {len(adapters)} total adapters "
                f"to {output} ({_human_bytes(output.stat().st_size)})",
                flush=True,
            )

    print(f"calibration complete in {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
