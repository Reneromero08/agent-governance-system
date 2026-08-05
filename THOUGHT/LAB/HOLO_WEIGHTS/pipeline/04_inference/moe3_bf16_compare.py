"""MOE-3: selective BF16 retrieval + Q8-vs-F16 comparison for L0-down.

Sol's cheapest-precision-confirmation: fetch only the needed tensors from
the 27.5GB safetensors via byte-range requests, compare Q8 vs BF16:
  - L0 down: all 90 experts (anomaly target)
  - L7 down: 10 experts (flat control)
  - L0 gate: 10 experts (orientation/dequant control)
Compare: singular-value curve, stable/effective rank, leading-vector
cosine, and the DC test (cos of u0 vs all-ones / row-mean).
"""
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline" / "04_inference"))
from moe1_geometry import dequant_q8_0, load_fused  # noqa: E402

URL = "https://huggingface.co/tvall43/Qwen3.6-14B-A3B-FableVibes/resolve/main/model.safetensors"
AMB, EXP_DIM = 2048, 512


def fetch_range(start: int, end: int) -> bytes:
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", default="/tmp/opencode/moe_bf16.npz")
    args = ap.parse_args()

    # header
    hdr_raw = fetch_range(0, 8 + 1_600_000)
    n = struct.unpack("<Q", hdr_raw[:8])[0]
    meta = json.loads(hdr_raw[8 : 8 + n])
    data_start = 8 + n
    print(f"header {n} bytes, data starts at {data_start}")

    picks = []
    for L in (0, 7):
        for e in range(90 if L == 0 else 10):
            picks.append(f"model.language_model.layers.{L}.mlp.experts.{e}.down_proj.weight")
    for e in range(10):
        picks.append(f"model.language_model.layers.0.mlp.experts.{e}.gate_proj.weight")

    # merge overlapping ranges
    ranges = []
    for name in picks:
        off = meta[name]["data_offsets"]
        ranges.append((data_start + off[0], data_start + off[1] - 1, name))
    ranges.sort()
    merged = []
    for s, e_, name in ranges:
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e_))
        else:
            merged.append((s, e_))
    total = sum(e_ - s + 1 for s, e_ in merged)
    print(f"{len(picks)} tensors, {len(merged)} merged ranges, {total/1e6:.1f} MB")

    blobs = {}
    for s, e_ in merged:
        data = fetch_range(s, e_)
        blobs[s] = data
    tensors = {}
    for name in picks:
        off = meta[name]["data_offsets"]
        s, e_ = data_start + off[0], data_start + off[1]
        # find the blob containing s
        for bs, data in blobs.items():
            if bs <= s < bs + len(data):
                chunk = data[s - bs : e_ - bs]
                tensors[name] = np.frombuffer(chunk, dtype=np.float16).astype(np.float32).reshape(
                    meta[name]["shape"])
                break
    np.savez(args.out, **{k.replace("model.language_model.layers.", "L").replace(".mlp.experts.", ".e").replace(".down_proj.weight", ".down").replace(".gate_proj.weight", ".gate"): v for k, v in tensors.items()})
    print(f"saved {len(tensors)} tensors to {args.out}")

    # comparisons vs Q8
    from gguf import GGUFReader
    reader = GGUFReader(args.gguf)
    print("=" * 78)
    print("Q8 vs BF16: L0 down experts")
    for e in (0, 1, 44, 89):
        q8 = load_fused(reader, "blk.0.ffn_down_exps.weight")[e]
        bf = tensors[f"model.language_model.layers.0.mlp.experts.{e}.down_proj.weight"]
        sq = np.linalg.svd(q8, full_matrices=False)[1]
        sb = np.linalg.svd(bf, full_matrices=False)[1]
        rel = np.abs(sq - sb) / (np.abs(sb) + 1e-30)
        uq = np.linalg.svd(q8, full_matrices=False)[0][:, 0]
        ub = np.linalg.svd(bf, full_matrices=False)[0][:, 0]
        print(f"  e{e}: Q8 s0={sq[0]:.3f} s1={sq[1]:.3f} | BF16 s0={sb[0]:.3f} s1={sb[1]:.3f} | "
              f"max rel dev {rel.max():.3f} | cos(u0_q8, u0_bf16)={abs(uq @ ub):.4f}")
        if e == 0:
            ones = np.ones(AMB)
            print(f"    BF16 cos(u0, all-ones)={abs(ub @ ones / np.linalg.norm(ub) / np.sqrt(AMB)):.4f}")
            rowmean = bf.mean(axis=1)
            print(f"    BF16 cos(u0, row-mean)={abs(ub @ rowmean / (np.linalg.norm(ub)*np.linalg.norm(rowmean)+1e-30)):.4f}")
    print("L7 down controls (e0):")
    e = 0
    q8 = load_fused(reader, "blk.7.ffn_down_exps.weight")[e]
    bf = tensors["model.language_model.layers.7.mlp.experts.0.down_proj.weight"]
    sq, sb = np.linalg.svd(q8, full_matrices=False)[1], np.linalg.svd(bf, full_matrices=False)[1]
    print(f"  Q8 s0={sq[0]:.3f} s1={sq[1]:.3f} | BF16 s0={sb[0]:.3f} s1={sb[1]:.3f}")
    print("L0 gate control (e0):")
    e = 0
    q8 = load_fused(reader, "blk.0.ffn_gate_exps.weight")[e]
    bf = tensors["model.language_model.layers.0.mlp.experts.0.gate_proj.weight"]
    sq, sb = np.linalg.svd(q8, full_matrices=False)[1], np.linalg.svd(bf, full_matrices=False)[1]
    print(f"  Q8 s0={sq[0]:.3f} s1={sq[1]:.3f} | BF16 s0={sb[0]:.3f} s1={sb[1]:.3f}")


if __name__ == "__main__":
    main()
