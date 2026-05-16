"""Phase 3f: MPS compression vs SVD on synthetic images."""
import numpy as np
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

def mps_decompose(vec, chi):
    N = len(vec)
    L = int(np.ceil(np.log2(N)))
    padded = np.zeros(2**L)
    padded[:N] = vec
    n = 2**L
    
    tensors = []
    remaining = padded.reshape(2, n//2)
    
    for level in range(L - 1):
        U, s, Vt = np.linalg.svd(remaining, full_matrices=False)
        ce = min(chi, len(s))
        cL = 1 if level == 0 else tensors[-1].shape[2]
        tensors.append(U[:, :ce].reshape(cL, remaining.shape[0], ce))
        remaining = np.diag(s[:ce]) @ Vt[:ce, :]
        nd = 2
        remaining = remaining.reshape(ce * nd, remaining.size // (ce * nd))
    
    cL = tensors[-1].shape[2] if tensors else 1
    tensors.append(remaining.reshape(cL, remaining.shape[0], 1))
    return tensors

def mps_reconstruct(tensors, n):
    if not tensors: return np.zeros(n)
    r = tensors[0]
    for t in tensors[1:]:
        r = np.tensordot(r, t, axes=([-1], [0]))
    return r.flatten()[:n]

rng = np.random.RandomState(42)
images = []
for i in range(10):
    x = np.linspace(-1, 1, 32); X, Y = np.meshgrid(x, x)
    if i % 3 == 0: img = np.sin(5*X)*np.cos(5*Y)
    elif i % 3 == 1: img = np.exp(-(X**2+Y**2)/0.3)
    else: img = np.sign(np.sin(8*X))*np.sign(np.cos(8*Y))
    images.append((img - img.min())/(img.max() - img.min() + 1e-12))
images = np.stack(images)
N_img, H, W = images.shape
n_pixels = H * W

chi_vals = [2, 4, 8, 16, 32]

print("=== SVD (per-image k-rank approximation) ===")
svd_psnrs = []
for k in chi_vals:
    ps = []
    for img in images:
        U, s, Vt = np.linalg.svd(img, full_matrices=False)
        st = np.zeros_like(s); st[:k] = s[:k]
        recon = U @ np.diag(st) @ Vt
        mse = np.mean((img - recon)**2)
        ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    svd_psnrs.append((k, float(np.mean(ps))))
    print(f"  k={k:3d}: PSNR={svd_psnrs[-1][1]:.1f} dB")

print("\n=== MPS (Tensor Train with bond dimension chi) ===")
mps_psnrs = []
for chi in chi_vals:
    ps = []
    for img in images:
        vec = img.flatten()
        tensors = mps_decompose(vec, chi)
        recon = mps_reconstruct(tensors, n_pixels)
        mse = np.mean((vec - recon)**2)
        ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    mps_psnrs.append((chi, float(np.mean(ps))))
    print(f"  chi={chi:3d}: PSNR={mps_psnrs[-1][1]:.1f} dB")

print(f"\n=== COMPARISON ===")
print(f"  {'chi':>5s}  {'SVD':>8s}  {'MPS':>8s}  {'Delta':>8s}")
for (k, s), (_, m) in zip(svd_psnrs, mps_psnrs):
    d = m - s
    print(f"  {k:5d}  {s:8.1f}  {m:8.1f}  {d:+8.1f}")

best = max(m - s for (_,s),(_,m) in zip(svd_psnrs,mps_psnrs))
if best > 0.5: print(f"\nMPS wins by up to {best:.1f} dB")
elif best > -0.5: print(f"\nMPS and SVD comparable")
else: print(f"\nSVD wins by up to {-best:.1f} dB")

(RESULTS/"mps_vs_pca.json").write_text(json.dumps({"svd":svd_psnrs,"mps":mps_psnrs}, indent=2))
print("Done")
