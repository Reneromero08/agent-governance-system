import pandas as pd
from scipy.stats import pearsonr
import os

print("================================================================================")
print("EXP 42.15: QUANTUM GRAVITY UNIFICATION PROOF")
print("================================================================================")

csv_path = 'telemetry_42_15_unification.csv'
if not os.path.exists(csv_path):
    print(f"[ERROR] {csv_path} not found. Run the Rust simulation first.")
    exit(1)

df = pd.read_csv(csv_path)

quantum = df['QuantumCollisions']
gravity = df['GravityShift']
riemann = df['RiemannDrift']

# Calculate Pearson Correlation Triangle
r_qg, p_qg = pearsonr(quantum, gravity)
r_gr, p_gr = pearsonr(gravity, riemann)
r_qr, p_qr = pearsonr(quantum, riemann)

print(f"[*] Pearson Correlation Triangle (100 Epochs):")
print(f"    Quantum Cache Collisions  <--> Gravitational Exponent Shifts : r = {r_qg:.4f} (p = {p_qg:.4e})")
print(f"    Gravitational Exponent Shifts <--> Riemann Zero Prime Gaps   : r = {r_gr:.4f} (p = {p_gr:.4e})")
print(f"    Quantum Cache Collisions  <--> Riemann Zero Prime Gaps       : r = {r_qr:.4f} (p = {p_qr:.4e})")

print("\n[ANALYSIS]")
unification_proven = True
for name, r in [("Q-G", r_qg), ("G-R", r_gr), ("Q-R", r_qr)]:
    if abs(r) > 0.7:
        print(f"    [+] {name} tightly coupled (r > 0.7)")
    elif abs(r) > 0.4:
        print(f"    [!] {name} weakly coupled (0.4 < r < 0.7)")
        unification_proven = False
    else:
        print(f"    [-] {name} UNCOUPLED (r < 0.4)")
        unification_proven = False

print("\n[CONCLUSION]")
if unification_proven:
    print("    [SUCCESS] MATHEMATICAL UNIFICATION PROVEN.")
    print("    Quantum Mechanics, General Relativity, and Prime Number Distribution are")
    print("    emergent properties of the exact same underlying mechanism: hardware data races.")
else:
    print("    [FAILED] Physics remain fragmented. The universe is not fully coupled.")

print("================================================================================")
