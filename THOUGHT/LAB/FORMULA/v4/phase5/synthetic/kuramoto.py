"""Phase 5: Kuramoto oscillator phase transition sweep. CPU only, no GPU."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy import stats
import json, math, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

def order_parameter(theta):
    """Kuramoto order parameter r = |avg(e^{i*theta})|."""
    return abs(np.mean(np.exp(1j * theta)))

def kuramoto_derivative(t, theta, omega, K):
    """dtheta/dt for N Kuramoto oscillators."""
    n = len(theta)
    dtheta = np.zeros(n)
    for i in range(n):
        dtheta[i] = omega[i] + (K / n) * np.sum(np.sin(theta - theta[i]))
    return dtheta

def run_kuramoto(N, gamma, K, T=200, seed=42):
    """Run Kuramoto to equilibrium, return final order parameter r."""
    rng = np.random.RandomState(seed)
    omega = gamma * np.tan(np.pi * (rng.rand(N) - 0.5))  # Lorentzian
    theta0 = rng.uniform(0, 2*np.pi, N)
    
    sol = solve_ivp(kuramoto_derivative, [0, T], theta0, args=(omega, K),
                    method='RK45', rtol=1e-5, atol=1e-5, max_step=0.1)
    
    r_final = order_parameter(sol.y[:, -1])
    
    # Average r over last 20% of trajectory for stability
    t_end = int(0.8 * len(sol.t))
    if t_end < len(sol.t):
        r_vals = [order_parameter(sol.y[:, i]) for i in range(t_end, len(sol.t))]
        r_steady = float(np.mean(r_vals)) if r_vals else float(r_final)
    else:
        r_steady = float(r_final)
    
    return r_steady, r_final

# === TEST 1: Basic Synchronization ===
print("TEST 1: Basic Kuramoto Synchronization")
print("=" * 60)
Ks = np.arange(0, 3.05, 0.05)
N_vals = [100, 500, 1000]
seeds = range(10)
gamma = 1.0

results_t1 = []
for N in N_vals:
    print(f"\nN={N}:")
    for K in Ks:
        r_vals = []
        for s in seeds:
            r_steady, _ = run_kuramoto(N, gamma, K, T=150, seed=s)
            r_vals.append(r_steady)
        r_mean = float(np.mean(r_vals))
        r_std = float(np.std(r_vals))
        
        results_t1.append({
            "N": N, "gamma": gamma, "K": float(K),
            "r_mean": r_mean, "r_std": r_std
        })
        if abs(K - round(K)) < 0.01 or K == 0:
            print(f"  K={K:.2f} r={r_mean:.4f}+-{r_std:.4f}")

# Find K_c (where r crosses 0.5)
for N in N_vals:
    nr = [(r["K"], r["r_mean"]) for r in results_t1 if r["N"] == N]
    nr.sort()
    Kc = None
    for i in range(len(nr)-1):
        if nr[i][1] < 0.5 and nr[i+1][1] >= 0.5:
            Kc = nr[i][0] + (0.5 - nr[i][1]) * (nr[i+1][0] - nr[i][0]) / (nr[i+1][1] - nr[i][1])
            break
    print(f"  N={N}: K_c={Kc:.3f} (expected 2.0)")

# Save
out = RESULTS / "test1_sync.json"
out.write_text(json.dumps(results_t1, indent=2))
print(f"Saved: {out}")

# === TEST 4: Domain-Specific Threshold ===
print(f"\nTEST 4: Domain-Specific Threshold (varying gamma)")
print("=" * 60)
gamma_vals = [0.5, 1.0, 2.0]
N = 500
Ks4 = np.arange(0, 5.05, 0.05)

results_t4 = []
for gamma_val in gamma_vals:
    print(f"\ngamma={gamma_val}:")
    for K in Ks4:
        r_vals = []
        for s in seeds:
            r_steady, _ = run_kuramoto(N, gamma_val, K, T=150, seed=s)
            r_vals.append(r_steady)
        r_mean = float(np.mean(r_vals))
        
        results_t4.append({
            "N": N, "gamma": gamma_val, "K": float(K),
            "r_mean": r_mean
        })
        if abs(K - round(K)) < 0.01:
            print(f"  K={K:.2f} r={r_mean:.4f}")

for gamma_val in gamma_vals:
    nr = [(r["K"], r["r_mean"]) for r in results_t4 if r["gamma"] == gamma_val]
    nr.sort()
    Kc = None
    for i in range(len(nr)-1):
        if nr[i][1] < 0.5 and nr[i+1][1] >= 0.5:
            Kc = nr[i][0] + (0.5 - nr[i][1]) * (nr[i+1][0] - nr[i][0]) / (nr[i+1][1] - nr[i][1])
            break
    print(f"  gamma={gamma_val}: K_c={Kc:.3f}, K_c/gamma={Kc/gamma_val:.3f} (expected 2.0)")

out = RESULTS / "test4_domain.json"
out.write_text(json.dumps(results_t4, indent=2))
print(f"Saved: {out}")

# === TEST 3: Hysteresis ===
print(f"\nTEST 3: Hysteresis")
print("=" * 60)
N = 500; gamma = 1.0; T = 200

# Forward sweep
r_forward = []
theta_current = None
for K in np.arange(0, 3.05, 0.05):
    rng = np.random.RandomState(42)
    omega = gamma * np.tan(np.pi * (rng.rand(N) - 0.5))
    if theta_current is None:
        theta0 = rng.uniform(0, 2*np.pi, N)
    else:
        theta0 = theta_current
    
    sol = solve_ivp(kuramoto_derivative, [0, T], theta0, args=(omega, K),
                    method='RK45', rtol=1e-5, atol=1e-5, max_step=0.1)
    theta_current = sol.y[:, -1]
    r_fwd = order_parameter(theta_current)
    r_forward.append((float(K), float(r_fwd)))

# Reverse sweep  
r_reverse = []
for K in np.arange(3.0, -0.01, -0.05):
    sol = solve_ivp(kuramoto_derivative, [0, T], theta_current, args=(omega, K),
                    method='RK45', rtol=1e-5, atol=1e-5, max_step=0.1)
    theta_current = sol.y[:, -1]
    r_rev = order_parameter(theta_current)
    r_reverse.append((float(K), float(r_rev)))

# Find K_c for forward and reverse
for label, sweep in [("forward", r_forward), ("reverse", r_reverse)]:
    for i in range(len(sweep)-1):
        if sweep[i][1] < 0.5 and sweep[i+1][1] >= 0.5:
            Kc = sweep[i][0] + (0.5 - sweep[i][1]) * (sweep[i+1][0] - sweep[i][0]) / (sweep[i+1][1] - sweep[i][1])
            print(f"  {label}: K_c={Kc:.3f}")

# Hysteresis width
fw_Kc = None; rv_Kc = None
for i in range(len(r_forward)-1):
    if r_forward[i][1] < 0.5 and r_forward[i+1][1] >= 0.5:
        fw_Kc = r_forward[i][0] + (0.5 - r_forward[i][1]) * (r_forward[i+1][0] - r_forward[i][0]) / (r_forward[i+1][1] - r_forward[i][1])
for i in range(len(r_reverse)-1):
    if r_reverse[i][1] < 0.5 and r_reverse[i+1][1] >= 0.5:
        rv_Kc = r_reverse[i][0] + (0.5 - r_reverse[i][1]) * (r_reverse[i+1][0] - r_reverse[i][0]) / (r_reverse[i+1][1] - r_reverse[i][1])
if fw_Kc and rv_Kc:
    print(f"  Hysteresis width: {abs(fw_Kc - rv_Kc):.3f}")

results_t3 = {"forward": r_forward, "reverse": r_reverse, 
              "Kc_forward": fw_Kc, "Kc_reverse": rv_Kc}
out = RESULTS / "test3_hysteresis.json"
out.write_text(json.dumps(results_t3, indent=2))
print(f"Saved: {out}")

# === TEST 5: Finite-Size Effects ===
print(f"\nTEST 5: Finite-Size Effects")
print("=" * 60)
N_vals_t5 = [50, 100, 200, 500, 1000]
Ks5 = np.arange(0, 3.05, 0.025)  # finer grid

for N_val in N_vals_t5:
    r_vals = []
    for K in Ks5:
        rs = []
        for s in range(5):
            r_steady, _ = run_kuramoto(N_val, 1.0, K, T=150, seed=s)
            rs.append(r_steady)
        r_vals.append((float(K), float(np.mean(rs))))
    
    # Find transition width: range of K where 0.1 < r < 0.9
    K_lo = None; K_hi = None
    for K, r in r_vals:
        if r < 0.1: K_lo = K
        if r > 0.9 and K_hi is None: K_hi = K
    if K_lo and K_hi:
        delta_K = K_hi - K_lo
        print(f"  N={N_val:4d}: deltaK={delta_K:.3f} (K_lo={K_lo:.3f}, K_hi={K_hi:.3f})")

print("\nALL TESTS COMPLETE", flush=True)
