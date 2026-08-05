"""MOE-2: Sol's L0-down anomaly audits (all on the Q8 file, no download).

1. Expert-axis audit: raw element offsets for experts 0,1,44,89 - verify
   changing expert ID changes the intended interleaved elements.
2. Quantization diagnostics: L0-down vs L7-down - block-scale histogram,
   saturation fraction, zero fraction, per-row/col means.
3. DC/scale-vector test: cosine of the leading singular vectors vs
   all-ones, row/col mean vectors, block-scale pattern; recompute the
   spectrum after subtracting row and column means.
4. Shared-spike removal: subtract each expert's leading rank-1 component,
   recompute the projector spectrum - if residual returns to the random
   null: 'one shared carrier direction plus broad expert-specific bulk'.
5. Projector mass quantification: D_eff(P), D_50/90/95/99, mass in top
   {1,2,8,16,32,64} directions, median pairwise affinity
   ||B_e^T B_f||_F^2 / 512 vs random expectation 0.25.
"""
import sys
import struct
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline" / "04_inference"))
from moe1_geometry import dequant_q8_0, load_fused  # noqa: E402

AMB = 2048
EXP_DIM = 512
N_EXPERTS = 90


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    args = ap.parse_args()

    from gguf import GGUFReader

    reader = GGUFReader(args.gguf)
    by = {t.name: t for t in reader.tensors}

    # 1. expert-axis audit on L0 down
    t = by["blk.0.ffn_down_exps.weight"]
    raw = np.asarray(t.data).reshape(-1)
    nq = raw.size // 34  # Q8_0 blocks
    # element (i, o, e) -> block b = (i*512 + o)*90 + e, byte base = b*34
    print("== 1. expert-axis audit (L0 down, interleaved hypothesis) ==")
    for e in (0, 1, 44, 89):
        base_byte = ((0 * 512 + 0) * 90 + e) * 34
        b = base_byte // 34
        scale = struct.unpack("<e", raw[b * 34 : b * 34 + 2].tobytes())[0]
        q = raw[b * 34 + 2 : b * 34 + 34]
        print(f"  e{e}: first block idx {b}, scale {scale:.5f}, int8[0:4] {q[:4].tolist()}")
    # the SAME (i,o) lane across experts must give different values
    b0 = ((0 * 512 + 0) * 90 + 0) * 34
    lane = [struct.unpack("<e", raw[(b0 // 34 + e) * 34 : (b0 // 34 + e) * 34 + 2].tobytes())[0]
            for e in (0, 1, 44, 89)]
    print(f"  lane(i=0,o=0) scales across experts: {[f'{s:.5f}' for s in lane]} "
          f"(must differ - interleaved, not repeated lane)")

    # 2. quantization diagnostics L0-down vs L7-down
    print("== 2. quantization diagnostics ==")
    for L in (0, 7):
        t = by[f"blk.{L}.ffn_down_exps.weight"]
        raw = np.asarray(t.data).reshape(-1)
        nblocks = raw.size // 34
        scales = np.empty(nblocks, dtype=np.float32)
        for b in range(nblocks):
            scales[b] = struct.unpack("<e", raw[b * 34 : b * 34 + 2].tobytes())[0]
        q = raw[2::34].astype(np.float32) if False else None
        int8s = raw[2::34]  # all int8 values (stride 34)
        print(f"  L{L}: type={t.tensor_type} nblocks={nblocks} "
              f"scale min={scales.min():.2e} max={scales.max():.2e} "
              f"median={np.median(scales):.4f} mean={scales.mean():.4f}")
        print(f"       zero-scale blocks: {(scales == 0).sum()} "
              f"| int8 at extrema (0/255): {(int8s == 0).sum()} / {(int8s == 255).sum()} "
              f"| int8==128: {(int8s == 128).sum()}")
        print(f"       int8 std: {int8s.std():.1f} | scales periodic-by-90 check: "
              f"corr(scale[::90], scale[1::90]) = "
              f"{np.corrcoef(scales[::90], scales[1::90])[0, 1]:.3f}")

    # 3. DC/scale-vector test on L0 down experts
    print("== 3. DC/scale-vector test (L0 down) ==")
    w = load_fused(reader, "blk.0.ffn_down_exps.weight")  # (90, 2048, 512)
    e0 = w[0]
    u, s, vh = np.linalg.svd(e0, full_matrices=False)
    ones = np.ones(AMB)
    u0 = u[:, 0]
    print(f"  e0 L0down: s0={s[0]:.3f} s1={s[1]:.3f} s2={s[2]:.3f} s100={s[100]:.3f}")
    print(f"  cos(u0, all-ones) = {abs(u0 @ ones / np.linalg.norm(u0) / np.sqrt(AMB)):.4f}")
    rowmean = e0.mean(axis=1)
    print(f"  cos(u0, row-mean-vector) = {abs(u0 @ rowmean / (np.linalg.norm(u0) * np.linalg.norm(rowmean) + 1e-30)):.4f}")
    # subtract row and col means, recompute spectrum
    e0c = e0 - e0.mean(axis=1, keepdims=True) - e0.mean(axis=0, keepdims=True) + e0.mean()
    _, sc, _ = np.linalg.svd(e0c, full_matrices=False)
    print(f"  after mean subtraction: s0={sc[0]:.3f} s1={sc[1]:.3f} s100={sc[100]:.3f} "
          f"stable-rank={ (sc**2).sum()/sc[0]**2:.1f}")

    # 4. shared-spike removal + projector mass (L0 down, all experts)
    print("== 4. shared-spike removal (L0 down) ==")
    Bs = []
    for e in range(N_EXPERTS):
        m = w[e]
        uu, ss, _ = np.linalg.svd(m, full_matrices=False)
        res = m - ss[0] * np.outer(uu[:, 0], np.linalg.svd(m, full_matrices=False)[2][0])
        Bs.append(uu[:, :512])
    # projector of FULL subspaces
    import torch
    B = torch.from_numpy(np.stack(Bs)).cuda()
    P = (B @ B.transpose(1, 2)).mean(0)
    ev = torch.flip(torch.linalg.eigvalsh(P), dims=[0]).cpu().numpy()
    trace = ev.sum()
    cum = np.cumsum(ev) / trace
    print(f"  P(L0-down): D_eff={np.exp(-((ev/trace) * np.log(ev/trace + 1e-30)).sum()):.1f} "
          f"D50={(cum<0.5).sum()+1} D90={(cum<0.9).sum()+1} D95={(cum<0.95).sum()+1} D99={(cum<0.99).sum()+1}")
    for k in (1, 2, 8, 16, 32, 64):
        print(f"  mass top {k}: {ev[:k].sum()/trace:.4f}")
    # residual projector after removing each expert's leading rank-1
    resBs = []
    for e in range(N_EXPERTS):
        m = w[e]
        uu, ss, vvh = np.linalg.svd(m, full_matrices=False)
        r1 = ss[0] * np.outer(uu[:, 0], vvh[0])
        res = m - r1
        ur, sr, vr = np.linalg.svd(res, full_matrices=False)
        resBs.append(ur[:, :512])
    Br = torch.from_numpy(np.stack(resBs)).cuda()
    Pr = (Br @ Br.transpose(1, 2)).mean(0)
    evr = torch.flip(torch.linalg.eigvalsh(Pr), dims=[0]).cpu().numpy()
    cumr = np.cumsum(evr) / evr.sum()
    print(f"  residual P (after rank-1 removal): D95={(cumr<0.95).sum()+1} "
          f"top {evr[0]:.4f} {evr[1]:.4f} (null top ~0.345, D95 ~1903)")

    # 5. pairwise affinity
    print("== 5. median pairwise affinity (L0 down, 60 pairs) ==")
    rng = np.random.default_rng(3)
    affs = []
    for _ in range(60):
        e1, e2 = rng.choice(N_EXPERTS, 2, replace=False)
        b1, b2 = Bs[e1], Bs[e2]
        affs.append(((b1.T @ b2) ** 2).sum() / 512)
    print(f"  median affinity: {np.median(affs):.4f} (random expectation 0.25) | mean {np.mean(affs):.4f}")


if __name__ == "__main__":
    main()
