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


T_Q8 = 8
T_F32 = 0
T_F16 = 1


def load_fused(reader, name: str) -> np.ndarray:
    """Load a tensor, dequantize, return LOGICAL-shape array.

    Conventions verified on this file:
    - gguf header shape is the REVERSED logical shape (shared gate header
      (2048,512) is logical (512,2048)).
    - Data is stored with the LAST header dim CONTIGUOUS (fastest). For
      the 3D fused expert tensors (header (2048,512,90)) this means the
      expert axis is element-interleaved: data[i, o, e] with e fastest,
      so the per-expert logical matrix is transpose(data[:, :, e]).
      The 2D case is the same rule: reshape(header) is already row-major
      logical when header == reversed logical.
    """
    by = {t.name: t for t in reader.tensors}
    t = by[name]
    raw = np.asarray(t.data).reshape(-1)
    if t.tensor_type == T_Q8:
        flat = dequant_q8_0(raw)
    elif t.tensor_type == T_F32:
        flat = raw.astype(np.float32)
    elif t.tensor_type == T_F16:
        flat = raw.astype(np.float32)
    else:
        raise ValueError(f"unsupported tensor type {t.tensor_type} for {name}")
    hdr = tuple(int(x) for x in t.shape)
    data = flat.reshape(hdr)
    if len(hdr) == 3:
        # fused experts: expert axis is the contiguous (last) header dim
        return np.transpose(data, (2, 1, 0))  # (e, dim1, dim0) logical
    return data  # 2D: header (in, out) -> logical (out, in)


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

    by = {t.name: t for t in reader.tensors}
    for L in args.layers:
        for key in ("gate_exps", "up_exps", "down_exps", "gate_inp", "gate_shexp", "up_shexp", "down_shexp"):
            nm = f"blk.{L}.ffn_{key}.weight"
            if nm in by:
                t = by[nm]
                print(f"L{L} {key}: header={tuple(int(x) for x in t.shape)} type={t.tensor_type}")
            else:
                print(f"L{L} {key}: missing")

    if args.manual_check:
        # integrity: verify the dequantized tensor against a manual matmul
        L = args.layers[0]
        nm = f"blk.{L}.ffn_gate_exps.weight"
        w = load_fused(reader, nm)
        print(f"manual-check {nm}: logical shape {w.shape}, std {w.std():.4f}")
        sh = f"blk.{L}.ffn_gate_shexp.weight"
        wsh = load_fused(reader, sh)
        print(f"  shared gate logical {wsh.shape}, std {wsh.std():.4f}")
        if w.shape == (N_EXPERTS, EXP_DIM, AMB):
            e0 = w[0]
            x = np.random.default_rng(7).standard_normal((AMB, 8)).astype(np.float32)
            y0 = e0 @ x
            ysh = wsh @ x
            print(f"  expert0 gate out {y0.shape} std {y0.std():.3f} | shared gate out std {ysh.std():.3f} "
                  f"(same scale expected - orientation sanity)")

    print("=" * 80)
    for L in args.layers:
        fam_stats = {}
        for fam, key, orient in (("w1", "gate", "in"), ("w2", "down", "out"), ("w3", "up", "in")):
            nm = f"blk.{L}.ffn_{key}_exps.weight"
            if nm not in names:
                print(f"L{L} {fam}: missing tensor - SKIP")
                continue
            w = load_fused(reader, nm)
            if fam in ("w1", "w3"):
                assert w.shape == (N_EXPERTS, EXP_DIM, AMB), w.shape
                mats = w  # (90, 512, 2048) gate/up logical
            else:
                assert w.shape == (N_EXPERTS, AMB, EXP_DIM), w.shape
                mats = w  # (90, 2048, 512) down logical
            import torch
            tm = torch.from_numpy(mats).cuda()
            if orient == "in":
                u, s, vh = torch.linalg.svd(tm, full_matrices=False)
                Bs = vh.transpose(1, 2)[:, :, : min(EXP_DIM, AMB)]
            else:
                u, s, vh = torch.linalg.svd(tm, full_matrices=False)
                Bs = u[:, :, : min(EXP_DIM, AMB)]
            S = s.cpu().numpy()
            en2 = (S**2).sum(axis=1)
            stable = en2 / (S**2).max(axis=1).clip(1e-30)  # proper stable rank
            pr = (S**2) / en2[:, None].clip(1e-30)
            d_eff = np.exp(-(pr * np.log(pr + 1e-30)).sum(axis=1))
            cum = np.cumsum(S**2, axis=1) / en2[:, None].clip(1e-30)
            d95 = (cum < 0.95).sum(axis=1) + 1
            d99 = (cum < 0.99).sum(axis=1) + 1
            fam_stats[fam] = (S, [b.cpu().numpy() for b in Bs])
            print(f"L{L} {fam}({orient}): stable-rank {stable.mean():.1f} | "
                  f"D_eff {d_eff.mean():.1f} | D95 {d95.mean():.1f} | D99 {d99.mean():.1f} | "
                  f"head/tail {(S[:, 0] / S[:, -1].clip(1e-30)).mean():.1f}")
        # normalized hidden-interface projector per family (2048 ambient)
        for fam in ("w1", "w3", "w2"):
            if fam not in fam_stats:
                continue
            Bs = fam_stats[fam][1]
            import torch
            B = torch.from_numpy(np.stack(Bs)).cuda()  # (90, 2048, 512)
            P = (B @ B.transpose(1, 2)).mean(0)
            ev = torch.flip(torch.linalg.eigvalsh(P), dims=[0]).cpu().numpy()
            cum = np.cumsum(ev) / ev.sum()
            print(f"L{L} projector({fam}): D95={(cum < 0.95).sum() + 1} "
                  f"top {ev[0]:.4f} {ev[1]:.4f} {ev[2]:.4f} | "
                  f"eig{EXP_DIM-1}={ev[EXP_DIM-1]:.4f} eig{EXP_DIM}={ev[EXP_DIM]:.4f} "
                  f"| tail50={ev[-50:].mean():.2e}")
        # random-subspace null (w1-style input interface)
        for r in range(args.null_runs):
            import torch
            g = torch.randn(N_EXPERTS, AMB, EXP_DIM, device="cuda")
            q, _ = torch.linalg.qr(g)
            Pn = (q @ q.transpose(1, 2)).mean(0)
            evn = torch.flip(torch.linalg.eigvalsh(Pn), dims=[0]).cpu().numpy()
            cumn = np.cumsum(evn) / evn.sum()
            print(f"  L{L} null{r}: D95={(cumn < 0.95).sum() + 1} "
                  f"top {evn[0]:.4f} {evn[1]:.4f} | "
                  f"eig{EXP_DIM-1}={evn[EXP_DIM-1]:.4f} eig{EXP_DIM}={evn[EXP_DIM]:.4f} "
                  f"| tail50={evn[-50:].mean():.2e}")


if __name__ == "__main__":
    main()
