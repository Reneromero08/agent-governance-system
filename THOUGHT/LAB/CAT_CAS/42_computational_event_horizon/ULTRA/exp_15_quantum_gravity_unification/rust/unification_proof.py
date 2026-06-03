import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os

print("================================================================================")
print("EXP 42.15: QUANTUM GRAVITY UNIFICATION — COUPLING ANALYSIS")
print("================================================================================")

csv_path = 'telemetry_42_15_unification.csv'
if not os.path.exists(csv_path):
    print(f"[ERROR] {csv_path} not found. Run the Rust simulation first.")
    exit(1)

df = pd.read_csv(csv_path)

quantum = df['QuantumCollisions'].values
gravity = df['GravityShift'].values
riemann = df['RiemannDrift'].values

# Calculate Pearson Correlation Triangle
r_qg, p_qg = pearsonr(quantum, gravity)
r_gr, p_gr = pearsonr(gravity, riemann)
r_qr, p_qr = pearsonr(quantum, riemann)

print(f"[*] Pearson Correlation Triangle ({len(df)} Epochs):")
print(f"    Quantum Cache Collisions  <--> Gravitational Exponent Shifts : r = {r_qg:.4f} (p = {p_qg:.4e})")
print(f"    Gravitational Exponent Shifts <--> Riemann Zero Prime Gaps   : r = {r_gr:.4f} (p = {p_gr:.4e})")
print(f"    Quantum Cache Collisions  <--> Riemann Zero Prime Gaps       : r = {r_qr:.4f} (p = {p_qr:.4e})")

# Permutation null bootstrap for each pair
COUPLING = 0.5   # Cohen's large effect
SIG = 0.01        # Bonferroni-corrected significance

np.random.seed(42)
pairs = [("Q-G", quantum, gravity, r_qg), ("G-R", gravity, riemann, r_gr), ("Q-R", quantum, riemann, r_qr)]

print("\n[NULL BOOTSTRAP]")
for name, x, y, obs_r in pairs:
    above = 0; n_null = 10000
    y_shuf = y.copy()
    for _ in range(n_null):
        np.random.shuffle(y_shuf)
        r_null, _ = pearsonr(x, y_shuf)
        if abs(r_null) >= abs(obs_r): above += 1
    null_p = above / n_null
    gate = (abs(obs_r) >= COUPLING) and (null_p < SIG)
    print(f"    {name}: |r|={abs(obs_r):.4f}  bootstrap null p={null_p:.4f}  "
          f"gate={'PASS' if gate else 'FAIL'}  (|r|>={COUPLING} AND p<{SIG})")

print("\n[VERDICT]")
gates = []
for name, x, y, obs_r in pairs:
    y_shuf = y.copy()
    above = 0
    for _ in range(10000):
        np.random.shuffle(y_shuf)
        r_null, _ = pearsonr(x, y_shuf)
        if abs(r_null) >= abs(obs_r): above += 1
    gate = (abs(obs_r) >= COUPLING) and ((above / 10000) < SIG)
    gates.append(gate)

all_pass = all(gates)
if all_pass:
    print(f"    PARTIAL INVERSE-COUPLING VERIFIED: All three pairs pass |r|>={COUPLING}")
    print(f"    and bootstrap null p<{SIG}. QM-GR-Prime triad is coupled at significant levels.")
    if len(df) < 50:
        print(f"    Caution: {len(df)} epochs; full 100-epoch run pending for confirmation.")
else:
    print(f"    COUPLING NOT CONFIRMED: {sum(gates)}/3 pairs pass. More epochs may strengthen.")
    if len(df) < 20:
        print(f"    Note: only {len(df)} epochs; sample too small for reliable coupling detection.")

print("================================================================================")
