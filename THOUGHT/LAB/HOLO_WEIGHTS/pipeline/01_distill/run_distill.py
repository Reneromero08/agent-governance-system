"""
run_distill.py — Out-of-Core Holographic Distillation for Qwen3.6-27B
=====================================================================
Reads the 27B safetensors shard-by-shard, compresses every 2D weight matrix
via randomized SVD (torch.svd_lowrank, k=256) on GPU when possible, and
saves a flat v1 `.holo` (key.U: m x k, key.SVh: k x n, 1D weights kept full).

Output format is compatible with lib/load_holo_v2.py (non-int8_dedup path).
Model path and output come from config/paths.json.
"""

import argparse
import json
import os
import time
import torch
from safetensors import safe_open
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = json.loads((REPO / "config" / "paths.json").read_text())

MODEL_DIR = Path(CONFIG["model"]["qwen3_6_27b"])
DEFAULT_MODEL_DIR = MODEL_DIR
INDEX_PATH = MODEL_DIR / "model.safetensors.index.json"
DEFAULT_OUT = REPO / "output" / "qwen_27b_k256.holo"

SKIP_PREFIXES = ("model.visual.", "mtp.")
KEEP_1D = True
DEFAULT_K = 256


def weight_type(key: str) -> str:
    """Cross-depth basis alignment key: weight kind without layer index."""
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 2 < len(parts):
            return ".".join(parts[i + 2:])
    return "global." + parts[-2] if len(parts) >= 2 else key


def compress_2d(tensor: torch.Tensor, k: int, device: str = "cuda", M=None, oversample: int = 8, niter: int = 2):
    """Randomized SVD to rank k with warm-start basis alignment.

    Returns (U_k, SVh_k) in bf16. M is the previous layer's U of the same
    weight type (m x k); it seeds the subspace so consecutive layers share
    a basis and a shared SVh stays valid (the MERA 0.84-0.89 fidelity path).
    """
    t = tensor.to(torch.float32)
    use_gpu = device == "cuda" and torch.cuda.is_available()
    if use_gpu and t.numel() * 4 > 9 * 1024**3:
        use_gpu = False  # keep 3GB headroom on the 12GB card
    dev = "cuda" if use_gpu else "cpu"
    m, n = t.shape
    q = min(k + oversample, m, n)
    with torch.no_grad():
        a = t.to(dev)
        if M is None:
            Q = torch.randn(m, q, device=dev)
        else:
            Mf = M.to(dev)
            take = min(q, Mf.shape[1])
            pad = q - take
            Q = torch.cat([Mf[:, :take], torch.randn(m, pad, device=dev)], dim=1) if pad > 0 else Mf[:, :take]
        it = niter if M is None else max(1, niter - 1)
        for _ in range(it):
            Q, _ = torch.linalg.qr(a @ (a.T @ Q))
        Q, _ = torch.linalg.qr(Q)
        B = Q.T @ a  # (q, n)
        Uhat, S, Vh = torch.linalg.svd(B, full_matrices=False)
        k_eff = min(k, S.numel())
        U = (Q @ Uhat[:, :k_eff]).to(torch.bfloat16).cpu()
        SVh = (S[:k_eff].unsqueeze(1) * Vh[:k_eff, :]).to(torch.bfloat16).cpu()
    return U, SVh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR), help="override model dir")
    ap.add_argument("--limit-shards", type=int, default=0, help="smoke test: only first N shards")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--no-skip", action="store_true", help="include vision/mtp tensors")
    args = ap.parse_args()
    global MODEL_DIR, INDEX_PATH
    MODEL_DIR = Path(args.model_dir)
    INDEX_PATH = MODEL_DIR / "model.safetensors.index.json"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Model dir: {MODEL_DIR}")
    print(f"Output:    {out_path}")
    print(f"Rank k:    {args.k}  device: {args.device}")

    with open(INDEX_PATH) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    unique_shards = sorted(set(weight_map.values()))
    if args.limit_shards:
        unique_shards = unique_shards[: args.limit_shards]
    print(f"Processing {len(unique_shards)} shards")

    holo = {}
    basis_cache = {}
    stats = {"compressed": 0, "kept": 0, "skipped": 0, "bytes_before": 0, "bytes_after": 0}
    t0 = time.time()

    for i, shard in enumerate(unique_shards):
        sp = MODEL_DIR / shard
        print(f"[{i+1}/{len(unique_shards)}] {shard}", flush=True)
        with safe_open(sp, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not args.no_skip and key.startswith(SKIP_PREFIXES):
                    stats["skipped"] += 1
                    continue
                tensor = f.get_tensor(key)
                stats["bytes_before"] += tensor.numel() * tensor.element_size()

                if tensor.ndim == 2:
                    wt = weight_type(key)
                    M = basis_cache.get(wt)
                    U, SVh = compress_2d(tensor, args.k, args.device, M=M)
                    basis_cache[wt] = U.float()  # keep fp32 for svd_lowrank init
                    holo[key + ".U"] = U
                    holo[key + ".SVh"] = SVh
                    stats["bytes_after"] += (U.numel() + SVh.numel()) * U.element_size()
                    stats["compressed"] += 1
                    if stats["compressed"] % 25 == 0:
                        mb = sum(v.numel() * v.element_size() for v in holo.values()) / 1024**2
                        print(f"    ... {stats['compressed']} compressed, holo so far {mb:.0f} MB", flush=True)
                elif tensor.ndim == 1 and KEEP_1D:
                    holo[key] = tensor.to(torch.bfloat16)
                    stats["bytes_after"] += tensor.numel() * tensor.element_size()
                    stats["kept"] += 1
                else:
                    stats["kept"] += 1
                    holo[key] = tensor.to(torch.bfloat16)
                    stats["bytes_after"] += tensor.numel() * tensor.element_size()

    holo["_config"] = {
        "format": "holo_weights_v1",
        "k": args.k,
        "model": "Qwen3.6-27B",
        "source": str(MODEL_DIR),
        "compressed": stats["compressed"],
        "kept_1d": stats["kept"],
        "skipped": stats["skipped"],
    }

    print(f"Saving {out_path} ...", flush=True)
    torch.save(holo, out_path)
    elapsed = time.time() - t0
    final_gb = out_path.stat().st_size / 1024**3
    print(f"\nDONE in {elapsed/60:.1f} min")
    print(f"compressed={stats['compressed']} kept={stats['kept']} skipped={stats['skipped']}")
    print(f"bytes before={stats['bytes_before']/1024**3:.2f} GB after={stats['bytes_after']/1024**3:.2f} GB")
    print(f"file size={final_gb:.2f} GB")


if __name__ == "__main__":
    main()
