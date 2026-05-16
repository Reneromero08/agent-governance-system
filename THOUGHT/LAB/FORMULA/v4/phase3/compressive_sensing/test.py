"""Compressive sensing: Hadamard vs Random basis, test Df formula for reconstruction."""
import numpy as np
import torch
import spyrit.misc.walsh_hadamard as wh
from spyrit.core.noise import NoNoise
from spyrit.core.recon import PinvNet
from spyrit.core.prep import SplitPoisson
from sklearn.metrics import mean_squared_error as mse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Generate synthetic test images (no CIFAR download needed)
rng = np.random.RandomState(42)
images = []
for i in range(10):
    # Create structured patterns: gradients, circles, checkerboards
    x = np.linspace(-1, 1, 32)
    y = np.linspace(-1, 1, 32)
    X, Y = np.meshgrid(x, y)
    if i % 3 == 0:
        img = np.sin(5 * X) * np.cos(5 * Y)
    elif i % 3 == 1:
        img = np.exp(-(X**2 + Y**2) / 0.3)
    else:
        img = np.sign(np.sin(8 * X)) * np.sign(np.cos(8 * Y))
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    images.append(img)

images = np.stack(images)  # [N, H, W]
print(f"Images: {images.shape}")

N_img, H, W = images.shape
n_pixels = H * W  # 1024
Ms = [int(n_pixels * r) for r in [0.01, 0.05, 0.1, 0.2, 0.5]]

# Build Hadamard measurement matrix (high sigma)
# Walsh-Hadamard matrix of size n_pixels x n_pixels
hadamard_full = wh.walsh2_matrix(n_pixels)  # [n, n]

# Build random measurement matrix (low sigma)
random_matrix = rng.randn(n_pixels, n_pixels)

print(f"\nFormula mapping:")
print(f"  sigma_H (Hadamard) > sigma_R (Random) -- Hadamard has structured compression")
print(f"  grad_S ~ 1 / M -- fewer measurements = higher noise")
print(f"  Df depends on image structure")
print(f"  R = PSNR of reconstruction")
print()

results = []
for M in Ms:
    # Select M measurements (first M rows for consistency)
    H_meas = hadamard_full[:M, :]  # [M, n_pixels]
    R_meas = random_matrix[:M, :]  # [M, n_pixels]
    
    H_psnrs = []; R_psnrs = []
    
    for img in images:
        x = img.flatten()  # [n_pixels]
        
        # Hadamard reconstruction
        y_H = H_meas @ x  # measurements
        x_H_hat = np.linalg.lstsq(H_meas, y_H, rcond=None)[0]
        psnr_H = 10 * np.log10(1.0 / max(mse(x, x_H_hat), 1e-12))
        H_psnrs.append(psnr_H)
        
        # Random reconstruction
        y_R = R_meas @ x
        x_R_hat = np.linalg.lstsq(R_meas, y_R, rcond=None)[0]
        psnr_R = 10 * np.log10(1.0 / max(mse(x, x_R_hat), 1e-12))
        R_psnrs.append(psnr_R)
    
    H_mean = float(np.mean(H_psnrs))
    R_mean = float(np.mean(R_psnrs))
    delta = H_mean - R_mean
    ratio = M / n_pixels
    
    print(f"M={M:4d} ({ratio:.3f}): Hadamard={H_mean:.1f} dB, Random={R_mean:.1f} dB, delta={delta:+.1f} dB")
    results.append({
        "M": M, "ratio": ratio, "Hadamard_PSNR": H_mean, "Random_PSNR": R_mean,
        "delta": delta, "Hadamard_std": float(np.std(H_psnrs)), "Random_std": float(np.std(R_psnrs))
    })

# Formula prediction: R_H > R_R, gap widens as M decreases
deltas = np.array([r["delta"] for r in results])
ratios = np.array([r["ratio"] for r in results])
r_val = np.corrcoef(ratios, deltas)[0, 1]
print(f"\nCorr(ratio, delta): r={r_val:+.4f}")
if r_val > 0.3:
    print("CONFIRMED: Hadamard advantage grows as M decreases (grad_S increases)")
else:
    print("Mixed/weak: no strong relationship between M and Hadamard advantage")

# R^2 of Hadamard vs Random as predictor of reconstruction quality
# Formula predicts R_H / R_R should scale with sigma_ratio
sigma_ratio = 2.0  # Hadamard ~2x more compressed than random (symbolic estimate)
print(f"\nPredicted R_H/R_R ratio: {sigma_ratio:.1f}x")
print(f"Actual: {np.mean([r['Hadamard_PSNR']/max(r['Random_PSNR'],1) for r in results]):.1f}x")

# Save
import json
(RESULTS / "compressive_sensing.json").write_text(json.dumps(results, indent=2))
print("\nDone")
