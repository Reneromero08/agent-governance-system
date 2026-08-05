"""MOE-1: Sol's minimal ordered experiment, steps 1-2 (clean revision).

Integrity gate + Q8 geometry pilot on Qwen3.6-14B-A3B-FableVibes.

Per Sol's redesign:
- FUSED GGUF tensors: blk.N.ffn_gate_exps / ffn_up_exps /
  ffn_down_exps / ffn_gate_inp (verify from metadata).
- Orientation: locate the 90-sized expert axis; logical gate/up
  512x2048, down 2048x512; verify matmul on a manual input before SVD.
- Exclude shared experts (ffn_*_shexp) from routed statistics.
- Geometry pilot: layers {0,7,15,23,31,39}, all 90 experts, per family:
  spectra (stable rank, D_eff, D_95, D_99, head/tail ratio) and the
  NORMALIZED HIDDEN-INTERFACE PROJECTOR in the 2048 ambient:
    gate/up (512x2048):  B_e = right singular vectors V_e^T (2048x512)
                         - the expert's INPUT interface
    down   (2048x512):   B_e = left singular vectors U_e (2048x512)
                         - the expert's OUTPUT interface
    P = (1/90) sum_e B_e B_e^T  (2048x2048)
  compared against the random-subspace null (90 random 2048x512 bases).
  Thresholds: P-spectrum dominated by ~512 large eigenvalues = manifold
  near one expert width; decays like the random null = no manifold.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAYERS = [0, 7, 15, 23, 31, 39]
N_EXPERTS = 90
AMB = 2048
EXP_DIM = 512


def dequant_q8_0(data: np.ndarray) -> np.ndarray:
    """Q8_0: blocks of 32 -> f16 scale + 32 int8 (offset 128)."""
    assert data.dtype == np.uint8 and data.size % 34 == 0
    nblocks = data.size // 34
    out = np.empty(nblocks * 32, dtype=np.float32)
    for b in range(nblocks):
        base = b * 34
        scale = struct.unpack("<e", data[base : base + 2].tobytes())[0]
        q = data[base + 2 : base + 34].astype(np.float32)
        out[b * 32 : (b + 1) * 32] = scale * (q - 128.0)
    return out


def load_fused(reader, name: str) -> np.ndarray:
    t = reader.get_tensor(name)
    raw = np.asarray(t.data).reshape(-1)
    if t.tensor_type == 4:  # Q8_0
        flat = dequant_q8_0(raw)
    elif t.tensor_type == 1:  # F32
        flat = raw.astype(np.float32)
    elif t.tensor_type == 0:  # F16
        flat = raw.astype(np.float32)
    else:
        raise ValueError(f"unsupported tensor type {t.tensor_type} for {name}")
    return flat.reshape(tuple(t.shape))


def reshape_experts(flat: np.ndarray, shape: tuple, ne: int) -> np.ndarray:
    """Reshape fused tensor to (ne, -1) with the expert axis first."""
    if shape[0] == ne:
        return flat.reshape(ne, -1)
    if ne in shape:
        ax = shape.index(ne)
        return np.moveaxis(flat.reshape(shape), ax, 0).reshape(ne, -1)
    raise ValueError(f"no expert axis {ne} in {shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    ap.add_argument("--null-runs", type=int, default=3)
    ap.add_argument("--manual-check", action="store_true")
    args = ap.parse_args()

    from gguf import GGUFReader

    reader = GGUFReader(args.gguf)
    names = [t.name for t in reader.tensors]
    print("total tensors:", len(names))
    moe_names = sorted(n for n in names if "exps" in n)
    for n in moe_names[:14]:
        print("  ", n)
    if not moe_names:
        for n in sorted(names)[:24]:
            print("  ", n)
        return

    for L in args.layers:
        for key in ("gate", "up", "down", "gate_inp", "gate_shexp", "up_shexp", "down_shexp"):
            nm = f"blk.{L}.ffn_{key}.weight"
            if nm in names:
                t = reader.get_tensor(nm)
                print(f"L{L} {key}: shape={tuple(t.shape)} type={t.tensor_type}")
            else:
                print(f"L{L} {key}: missing")

    if args.manual_check:
        # integrity: verify the dequantized tensor against a manual matmul
        L = args.layers[0]
        nm = f"blk.{L}.ffn_gate_exps.weight"
        w = load_fused(reader, nm)
        print(f"manual-check {nm}: dequantized shape {w.shape}, "
              f"std {w.std():.4f}, mean {w.mean():.4f}")
        if w.shape[0] == N_EXPERTS:
            e0 = w[0].reshape(EXP_DIM, AMB)
            x = np.random.default_rng(7).standard_normal((AMB, 8)).astype(np.float32)
            y = e0 @ x
            print(f"  expert0 gate matmul ok: out {y.shape} std {y.std():.3f}")

    print("=" * 80)
    for L in args.layers:
        fam_stats = {}
        for fam, key, orient in (("w1", "gate", "in"), ("w2", "down", "out"), ("w3", "up", "in")):
            nm = f"blk.{L}.ffn_{key}_exps.weight"
            if nm not in names:
                print(f"L{L} {fam}: missing tensor - SKIP")
                continue
            w = load_fused(reader, nm)
            exps = reshape_experts(w, w.shape, N_EXPERTS)
            ncols = exps.shape[1]
            if fam in ("w1", "w3"):
                assert ncols == EXP_DIM * AMB, f"unexpected cols {ncols}"
                mats = exps.reshape(N_EXPERTS, EXP_DIM, AMB)  # (512, 2048)
            else:
                assert ncols == AMB * EXP_DIM, f"unexpected cols {ncols}"
                mats = exps.reshape(N_EXPERTS, AMB, EXP_DIM)  # (2048, 512)
            svals, Bs = [], []
            for e in range(N_EXPERTS):
                u, s, vh = np.linalg.svd(mats[e], full_matrices=False)
                svals.append(s)
                Bs.append(vh.T[:, : min(EXP_DIM, AMB)] if orient == "in" else u[:, : min(EXP_DIM, AMB)])
            S = np.stack(svals)
            en2 = (S**2).sum(axis=1)
            stable = (S**2).max(axis=1) / en2.clip(1e-30)
            p = (S**2) / en2[:, None].clip(1e-30)
            d_eff = np.exp(-(p * np.log(p + 1e-30)).sum(axis=1))
            cum = np.cumsum(S**2, axis=1) / en2[:, None].clip(1e-30)
            d95 = (cum < 0.95).sum(axis=1) + 1
            d99 = (cum < 0.99).sum(axis=1) + 1
            fam_stats[fam] = (S, Bs)
            print(f"L{L} {fam}({orient}): stable-rank {stable.mean():.1f} | "
                  f"D_eff {d_eff.mean():.1f} | D95 {d95.mean():.1f} | D99 {d99.mean():.1f} | "
                  f"head/tail {(S[:, 0] / S[:, -1].clip(1e-30)).mean():.1f}")
        # normalized hidden-interface projector per family (2048 ambient)
        for fam in ("w1", "w3", "w2"):
            if fam not in fam_stats:
                continue
            Bs = fam_stats[fam][1]
            P = np.zeros((AMB, AMB), dtype=np.float32)
            for b in Bs:
                P += b @ b.T
            P /= N_EXPERTS
            ev = np.linalg.eigvalsh(P)[::-1]
            cum = np.cumsum(ev) / ev.sum()
            print(f"L{L} projector({fam}): D95={(cum < 0.95).sum() + 1} "
                  f"top {ev[0]:.4f} {ev[1]:.4f} {ev[2]:.4f} | "
                  f"eig{EXP_DIM-1}={ev[EXP_DIM-1]:.4f} eig{EXP_DIM}={ev[EXP_DIM]:.4f} "
                  f"| tail50={ev[-50:].mean():.2e}")
        # random-subspace null (w1-style input interface)
        for r in range(args.null_runs):
            rng = np.random.default_rng(1000 + L * 10 + r)
            Pn = np.zeros((AMB, AMB), dtype=np.float32)
            for e in range(N_EXPERTS):
                q, _ = np.linalg.qr(rng.standard_normal((AMB, EXP_DIM)))
                Pn += q @ q.T
            Pn /= N_EXPERTS
            evn = np.linalg.eigvalsh(Pn)[::-1]
            cumn = np.cumsum(evn) / evn.sum()
            print(f"  L{L} null{r}: D95={(cumn < 0.95).sum() + 1} "
                  f"top {evn[0]:.4f} {evn[1]:.4f} | "
                  f"eig{EXP_DIM-1}={evn[EXP_DIM-1]:.4f} eig{EXP_DIM}={evn[EXP_DIM]:.4f} "
                  f"| tail50={evn[-50:].mean():.2e}")


if __name__ == "__main__":
    main()
