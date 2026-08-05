"""MOE-4: Sol's expert-axis tensor-rank test (raw + centered Gram).

X = [vec(W_1); ...; vec(W_90)] in R^{90 x N}.
90x90 Gram G = X X^T computed WITHOUT materializing NxN (use the
flattened 90 x N on GPU). Raw AND centered:
  W_bar = (1/90) sum_e W_e (centroid)
  delta W_e = W_e - W_bar
Report per family (gate/up/down) and the CONCATENATED triplet
(vec(gate); vec(up); vec(down) - intermediate-aligned):
  - centroid energy share: ||W_bar||_F^2 / (1/90) sum_e ||W_e||_F^2
  - centered Gram eigenvalues -> D_eff, D_50/90/95/99
  - pairwise Frobenius correlation mean/median (raw and centered)
  - PREDECLARED: strong D95<=16, partial 16<D95<=32, none D95>=72
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline" / "04_inference"))
from moe1_geometry import load_fused  # noqa: E402

N_EXPERTS = 90


def gram_metrics(X: torch.Tensor, tag: str):
    """X: (90, N) torch float32. Prints Gram eigenvalue metrics."""
    G = X @ X.T  # 90x90
    ev = torch.flip(torch.linalg.eigvalsh(G), dims=[0]).cpu().numpy()
    trace = ev.sum()
    cum = np.cumsum(ev) / trace
    p = ev / trace
    d_eff = np.exp(-(p * np.log(p + 1e-30)).sum())
    d50, d90, d95, d99 = ((cum < q).sum() + 1 for q in (0.5, 0.9, 0.95, 0.99))
    diag = torch.diag(G)
    corr = (G / torch.sqrt(diag[:, None] * diag[None, :]).clamp_min(1e-30)).cpu().numpy()
    off = corr[np.triu_indices(90, 1)]
    print(f"{tag}: D_eff={d_eff:.1f} D50={d50} D90={d90} D95={d95} D99={d99} "
          f"| pair-corr mean={off.mean():.4f} median={np.median(off):.4f}")
    return ev


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 7, 15, 23, 31, 39])
    args = ap.parse_args()

    from gguf import GGUFReader
    reader = GGUFReader(args.gguf)
    print("MOE-4: expert-axis tensor-rank test (predeclared: strong D95<=16, partial 16<D95<=32, none D95>=72)")
    print("=" * 90)
    for L in args.layers:
        mats = {}
        for fam, key, shape in (("w1", "gate", (90, 512, 2048)),
                                ("w3", "up", (90, 512, 2048)),
                                ("w2", "down", (90, 2048, 512))):
            w = load_fused(reader, f"blk.{L}.ffn_{key}_exps.weight")
            assert w.shape == shape, (w.shape, shape)
            mats[fam] = w
        print(f"L{L}:")
        cent = {}
        for fam in ("w1", "w3", "w2"):
            W = torch.from_numpy(mats[fam]).cuda().float()  # (90, r, c)
            X = W.reshape(N_EXPERTS, -1)
            gram_metrics(X, f"  {fam} raw     ")
            Wc = W.mean(0, keepdim=True)  # centroid (1, r, c)
            cent[fam] = Wc
            Xc = (W - Wc).reshape(N_EXPERTS, -1)
            gram_metrics(Xc, f"  {fam} centered")
            ce = (Wc.reshape(-1) ** 2).sum().item() / (W.reshape(-1) ** 2).sum().item()
            print(f"    centroid energy share: {ce:.4f}")
        # triplet: concatenate per expert (intermediate-aligned order)
        tri = torch.cat([torch.from_numpy(mats["w1"]).cuda().float().reshape(N_EXPERTS, -1),
                         torch.from_numpy(mats["w3"]).cuda().float().reshape(N_EXPERTS, -1),
                         torch.from_numpy(mats["w2"]).cuda().float().reshape(N_EXPERTS, -1)], dim=1)
        gram_metrics(tri, "  triplet raw    ")
        W1 = torch.from_numpy(mats["w1"]).cuda().float()
        W3 = torch.from_numpy(mats["w3"]).cuda().float()
        W2 = torch.from_numpy(mats["w2"]).cuda().float()
        tric = torch.cat([(W1 - W1.mean(0, keepdim=True)).reshape(N_EXPERTS, -1),
                          (W3 - W3.mean(0, keepdim=True)).reshape(N_EXPERTS, -1),
                          (W2 - W2.mean(0, keepdim=True)).reshape(N_EXPERTS, -1)], dim=1)
        gram_metrics(tric, "  triplet cent. ")
        ce3 = ((W1.mean(0) ** 2).sum() + (W3.mean(0) ** 2).sum() + (W2.mean(0) ** 2).sum()).item() / \
              ((W1 ** 2).sum() + (W3 ** 2).sum() + (W2 ** 2).sum()).item()
        print(f"    triplet centroid energy share: {ce3:.4f}")


if __name__ == "__main__":
    main()
