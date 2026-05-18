"""Compute MT and centriole sigma using Celardo's actual dipole data."""
import json, numpy as np
from scipy.linalg import eigvals

GAMMA = 0.00273; COUPLING_REF = 60.0; COHERENT_SCALE = 100.0
LAMBDA_CM = 280e-7; K = 2*np.pi/LAMBDA_CM; A_TO_CM = 1e-8

with open("celardo_dipoles.json") as f:
    celardo = json.load(f)

positions = np.array([[d[0], d[1], d[2]] for d in celardo])  # Angstroms
dipoles = np.array([[d[3], d[4], d[5]] for d in celardo])

def kappa(pi, ui, pj, uj):
    rv = pj - pi; r = np.linalg.norm(rv)
    if r < 1e-10: return 0.0
    rh = rv / r
    return ui@uj - 3.0*(ui@rh)*(uj@rh)

def V_coupling(r_cm, kap):
    kr = K * r_cm
    if kr < 1e-3:
        d = -min(3*GAMMA/(4*kr**3), COUPLING_REF); gc = GAMMA
    else:
        sk, ck = np.sin(kr), np.cos(kr)
        d = (3*GAMMA/4)*(ck/kr - sk/(kr**2) - ck/(kr**3))
        gc = (3*GAMMA/2)*(sk/kr + ck/(kr**2) - sk/(kr**3))
    return complex(np.clip(d, -COUPLING_REF, COUPLING_REF)*kap, -gc*abs(kap)/2)

def build_H(pos_a, dip_a):
    coords_cm = pos_a * A_TO_CM; N = len(pos_a)
    H = np.zeros((N, N), dtype=complex)
    for i in range(N):
        H[i, i] = complex(np.random.default_rng(i).uniform(-100, 100), -GAMMA/2)
    for i in range(N):
        for j in range(i+1, N):
            rc = np.linalg.norm(coords_cm[j] - coords_cm[i])
            if rc > LAMBDA_CM * 5: continue
            k = kappa(pos_a[i], dip_a[i], pos_a[j], dip_a[j])
            if abs(k) < 1e-10: continue
            v = V_coupling(rc, k)
            H[i, j] = complex(v.real*COHERENT_SCALE, v.imag)
            H[j, i] = H[i, j]
    return H

def analyze(pos, dip):
    H = build_H(pos, dip)
    gj = np.maximum(-2*np.imag(eigvals(H)), 0)
    return np.max(gj)/GAMMA, np.mean(gj)/GAMMA

# Build 1-spiral MT (Celardo data IS the first spiral)
print("Celardo 1-spiral MT (N=104):")
sigma_1, mean_1 = analyze(positions, dipoles)
print(f"  sigma = {sigma_1:.2f}, mean = {mean_1:.4f}")

# Build multi-spiral MT by translating along x
for ns in [2, 3, 5]:
    all_pos, all_dip = [], []
    for s in range(ns):
        x_shift = s * 80.0  # 8 nm = 80 Angstroms
        for p, d in zip(positions, dipoles):
            all_pos.append(np.array([p[0] + x_shift, p[1], p[2]]))
            all_dip.append(d)
    all_pos = np.array(all_pos); all_dip = np.array(all_dip)
    if len(all_pos) <= 1000:
        sigma_n, mean_n = analyze(all_pos, all_dip)
        print(f"  {ns}-spiral MT (N={len(all_pos)}): sigma = {sigma_n:.2f}, mean = {mean_n:.4f}")

# Compare to our single-MT calibration target: sigma=35
print(f"\nSingle MT (5 spirals) with Celardo dipoles: sigma={sigma_n:.2f}")
print(f"Our calibration target from paper: sigma=35")
print(f"Ratio: {sigma_n/35:.2f}x")
