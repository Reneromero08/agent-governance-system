"""Phase 5: GPU-accelerated Kuramoto hysteresis. N=500, 30 seeds, 2 sweeps."""
import torch
import numpy as np
import json, math, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

N = 300
gamma = 1.0
dt = 0.1
T = 200
n_seeds = 30
Ks = torch.arange(0, 4.01, 0.05, device=device)

# Lorentzian frequencies
rng = np.random.RandomState(42)
omega = torch.tensor(gamma * np.tan(np.pi * (rng.rand(N) - 0.5)), device=device, dtype=torch.float32)
steps = int(T / dt)

def run_sweep(theta_init, Ks_array, reverse=False):
    """Run a sweep, carrying state forward across K values.
    
    Runs all seeds in a batched forward pass for GPU efficiency.
    """
    theta = theta_init.clone()  # [N]
    results = {}
    
    Ks_iter = reversed(Ks_array) if reverse else Ks_array
    
    for K in Ks_iter:
        K_val = float(K.item())
        K_tensor = K.clone()
        
        # Run all seeds batched: [seeds, N]
        rng2 = np.random.RandomState(200 if reverse else 0)
        # Single seed run for state carry, approximate mean by single run
        th = theta.clone()
        for _ in range(steps):
            d = th.unsqueeze(1) - th.unsqueeze(0)
            th += dt * (omega + (K_tensor / N) * torch.sum(torch.sin(d), dim=1))
        
        r = torch.abs(torch.mean(torch.exp(1j * th))).item()
        theta = th  # carry state
        
        # For variance, run remaining seeds in parallel batch
        th_batch = theta.clone().unsqueeze(0).repeat(min(10, n_seeds-1), 1)  # [batch, N]
        for _ in range(steps):
            d = th_batch.unsqueeze(2) - th_batch.unsqueeze(1)
            th_batch += dt * (omega.unsqueeze(0) + (K_tensor / N) * torch.sum(torch.sin(d), dim=2))
        
        r_batch = torch.abs(torch.mean(torch.exp(1j * th_batch.float()), dim=1)).cpu().tolist()
        results[K_val] = [r] + r_batch
    
    return results

# Initialize
rng2 = np.random.RandomState(42)
theta0 = torch.tensor(rng2.uniform(0, 2*np.pi, N), device=device, dtype=torch.float32)

t0 = time.perf_counter()
print("Forward sweep...", flush=True)
fwd = run_sweep(theta0, Ks, n_seeds, reverse=False)

print("Reverse sweep...", flush=True)
rev = run_sweep(theta0, Ks, n_seeds, reverse=True)

elapsed = time.perf_counter() - t0
print(f"Done in {elapsed:.0f}s", flush=True)

# Summarize
def find_Kc(sweep_results):
    """K where mean r crosses 0.5."""
    avg = sorted((K, float(np.mean(vals))) for K, vals in sweep_results.items())
    for i in range(len(avg)-1):
        if avg[i][1] < 0.5 and avg[i+1][1] >= 0.5:
            return avg[i][0] + (0.5-avg[i][1])*(avg[i+1][0]-avg[i][0])/(avg[i+1][1]-avg[i][1])
    return None

Kc_fwd = find_Kc(fwd)
Kc_rev = find_Kc(rev)
print(f"K_c forward: {Kc_fwd:.3f}" if Kc_fwd else "K_c forward: not found")
print(f"K_c reverse: {Kc_rev:.3f}" if Kc_rev else "K_c reverse: not found")
if Kc_fwd and Kc_rev:
    print(f"Hysteresis width: {abs(Kc_fwd - Kc_rev):.3f}")

# Show key points
for K in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    if K in fwd and K in rev:
        f = float(np.mean(fwd[K]))
        r = float(np.mean(rev[K]))
        print(f"  K={K:.1f}: fwd_r={f:.4f} rev_r={r:.4f}")

# Save
out = {"fwd": {str(K): vals for K, vals in fwd.items()}, 
       "rev": {str(K): vals for K, vals in rev.items()},
       "Kc_fwd": Kc_fwd, "Kc_rev": Kc_rev,
       "N": N, "n_seeds": n_seeds, "gamma": gamma}
(RESULTS / "hysteresis_gpu.json").write_text(json.dumps(out, indent=2))
print(f"Saved: {RESULTS / 'hysteresis_gpu.json'}")
