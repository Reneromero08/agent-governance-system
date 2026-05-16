"""Phase 5 precision: hysteresis at N=300, 30 seeds."""
import numpy as np, json, time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "LAW" / "CONTRACTS" / "_runs" / "hysteresis"
RESULTS.mkdir(parents=True, exist_ok=True)

N = 300; gamma = 1.0; dt = 0.03; T = 300; n_seeds = 30
Ks_forward = np.arange(0, 4.01, 0.05)
Ks_reverse = np.arange(4.0, -0.01, -0.05)

np.random.seed(42)
rng = np.random.RandomState(42)
omega = gamma * np.tan(np.pi * (rng.rand(N) - 0.5))
theta_init = rng.uniform(0, 2*np.pi, N)

t0 = time.perf_counter()

# FORWARD SWEEP
print("Forward sweep...", flush=True)
fwd_results = []
theta = theta_init.copy()
for K in Ks_forward:
    for s in range(n_seeds):
        rng2 = np.random.RandomState(s)
        th = theta.copy()
        steps = int(T / dt)
        for _ in range(steps):
            d = th[:, None] - th[None, :]
            th += dt * (omega + (K/N) * np.sum(np.sin(d), axis=1))
        r = float(abs(np.mean(np.exp(1j * th))))
        fwd_results.append({"K": float(K), "seed": s, "r": r})
        if s == 0:
            theta = th  # carry state forward for next K
            elapsed = time.perf_counter() - t0
            print(f"  K={K:.2f} r={r:.4f} [{elapsed:.0f}s]", flush=True)

theta_final = theta.copy()

# REVERSE SWEEP
print("Reverse sweep...", flush=True)
rev_results = []
theta = theta_final.copy()  # start from synchronized state
for K in Ks_reverse:
    for s in range(n_seeds):
        rng2 = np.random.RandomState(100 + s)
        th = theta.copy()
        steps = int(T / dt)
        for _ in range(steps):
            d = th[:, None] - th[None, :]
            th += dt * (omega + (K/N) * np.sum(np.sin(d), axis=1))
        r = float(abs(np.mean(np.exp(1j * th))))
        rev_results.append({"K": float(K), "seed": s, "r": r})
        if s == 0:
            theta = th
            elapsed = time.perf_counter() - t0
            print(f"  K={K:.2f} r={r:.4f} [{elapsed:.0f}s]", flush=True)

# Save
out = RESULTS / "hysteresis_N300_30seeds.json"
out.write_text(json.dumps({"fwd": fwd_results, "rev": rev_results}, indent=2))

# Summary: K_c for forward and reverse
fwd_by_K = {}
for r in fwd_results:
    fwd_by_K.setdefault(r["K"], []).append(r["r"])
rev_by_K = {}
for r in rev_results:
    rev_by_K.setdefault(r["K"], []).append(r["r"])

def find_Kc(by_K):
    avg = sorted((K, float(np.mean(rs))) for K, rs in by_K.items())
    for i in range(len(avg)-1):
        if avg[i][1] < 0.5 and avg[i+1][1] >= 0.5:
            return avg[i][0] + (0.5-avg[i][1])*(avg[i+1][0]-avg[i][0])/(avg[i+1][1]-avg[i][1])
    return None

Kc_fwd = find_Kc(fwd_by_K)
Kc_rev = find_Kc(rev_by_K)
print(f"\nK_c forward: {Kc_fwd:.3f}" if Kc_fwd else "\nK_c forward: not found")
print(f"K_c reverse: {Kc_rev:.3f}" if Kc_rev else "K_c reverse: not found")
if Kc_fwd and Kc_rev:
    print(f"Hysteresis width: {abs(Kc_fwd - Kc_rev):.3f}")

print(f"Total time: {time.perf_counter()-t0:.0f}s")
print(f"Saved: {out}")
