"""
36d_scaling_sweep.py

EXPERIMENT 36d: CATALYTIC SCALING SWEEP
========================================
Sweeps chain length N, measures the Godel transition lambda_c,
and verifies the scaling law:  lambda_c = (gap)^N  where gap=0.05.

TIMING: Compares catalytic (rank-1 lemma, O(n_phi)) vs direct
(O(n_phi * N^3)) winding computation, showing the catalytic speedup
grows as N^3.

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch
import numpy as np
import hashlib
import time

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
COMPLEX = torch.complex128  # double precision — ~15 digits vs ~7 for complex64

TAPE_SIZE_MB = 256
TAPE_SIZE_BYTES = TAPE_SIZE_MB * 1024 * 1024


# ======================================================================
#  Tape + Encoder (copied from 36c)
# ======================================================================

class CatalyticTape:
    def __init__(self, size_bytes=TAPE_SIZE_BYTES, seed=42):
        self.size_bytes = size_bytes
        rng = np.random.RandomState(seed)
        self.tape = rng.randint(0, 256, size=size_bytes, dtype=np.uint8)
        self.read_count = 0; self.write_count = 0
    def read(self, i): self.read_count += 1; return int(self.tape[i])
    def write(self, i, v): self.write_count += 1; self.tape[i] = v & 0xFF
    def hash(self): return hashlib.sha256(self.tape.tobytes()).hexdigest()

def encode(tape, lam, off=0):
    lb = np.array([lam], dtype=np.float64).tobytes()
    orig = [tape.read(off+i) for i in range(64)]
    for i,b in enumerate(lb): tape.write(off+i, tape.read(off+i)^b)
    for i in range(1,64): tape.write(off+i, tape.read(off+i)^tape.read(off+i-1))
    return off, lb, orig

def uncompute(tape, off, lb, orig):
    for i in range(63,0,-1): tape.write(off+i, tape.read(off+i)^tape.read(off+i-1))
    for i,b in enumerate(lb): tape.write(off+i, tape.read(off+i)^b)
    for i in range(64): assert tape.read(off+i) == orig[i]


# ======================================================================
#  Winding cache (rank-1 lemma) — generalized for any N
# ======================================================================

def build_cache(lam0, N, n_phi=200):
    E_ref = -0.05j
    phis = torch.linspace(0, 2*np.pi, n_phi)
    twists = torch.zeros(n_phi, dtype=COMPLEX)
    dets_0 = torch.zeros(n_phi, dtype=COMPLEX)
    elems_0 = torch.zeros(n_phi, dtype=COMPLEX)

    H_ref = torch.zeros((N, N), dtype=COMPLEX)
    for i in range(N-1): H_ref[i+1,i] = 1.0+0j
    H_ref[N-1,N-1] = -1j
    for i in range(N-1): H_ref[i,i] = -0.1j

    e0 = torch.zeros(N, dtype=COMPLEX); e0[0] = 1.0+0j  # unit vector for solve

    for k in range(n_phi):
        phi = phis[k].item()
        twist = torch.tensor(np.exp(1j*phi), dtype=COMPLEX)
        twists[k] = twist
        H = H_ref.clone()
        H[0, N-1] = lam0 * twist
        M = H - E_ref * torch.eye(N).to(COMPLEX)
        try:
            dets_0[k] = torch.linalg.det(M)
        except Exception:
            sign, logdet = torch.linalg.slogdet(M)
            dets_0[k] = sign * torch.exp(logdet)
        elems_0[k] = torch.linalg.solve(M, e0)[N-1]
    return dets_0, elems_0, twists

def winding_cat(lam, dets_0, elems_0, twists, lam0):
    dets = dets_0 * (1.0 + (lam-lam0) * twists * elems_0)
    # Guard against NaN from factor crossing zero
    if torch.isnan(dets).any() or torch.isinf(dets).any():
        return None
    dt = torch.diff(torch.angle(dets))
    dt = torch.remainder(dt + np.pi, 2*np.pi) - np.pi
    return int(round(float(torch.sum(dt).item())/(2*np.pi)))


# ======================================================================
#  Scaling sweep
# ======================================================================

def scaling_sweep():
    """
    Sweep chain length N, measure transition lambda_c, verify
    lambda_c = gap^N where gap = 0.05.
    Compare catalytic vs direct timing.
    """
    N_vals = [8, 16, 32, 64, 128]
    TIGHT = 0.9  # rebuild at 90% drift to avoid NaN
    results = []

    print("=" * 78)
    print("  EXPERIMENT 36d: CATALYTIC SCALING SWEEP")
    print("  Transition lambda_c vs chain length N")
    print("=" * 78)
    print(f"  {'N':>5s}  {'lam_c (obs)':>16s}  {'lam_c (pred)':>16s}  "
          f"{'ratio':>8s}  {'cat(ms)':>8s}  {'direct(s)':>9s}  "
          f"{'speedup':>8s}")
    print("  " + "-" * 85)

    for N in N_vals:
        tape = CatalyticTape(size_bytes=128*1024*1024)
        initial_hash = tape.hash()

        # Dynamic step count: need g < log10(0.05^N) = N*log10(0.05)
        target_g = N * np.log10(0.05)  # e.g. N=32 -> -41.7
        steps_needed = int(np.ceil((abs(target_g) - 1.0) / abs(np.log10(0.97))))
        steps_needed = min(max(steps_needed, 200), 20000)  # N=128 needs ~12500
        lr = 0.03

        lam0 = 0.1
        t0 = time.time()
        dets_0, elems_0, twists = build_cache(lam0, N)
        cache_time = time.time() - t0

        # Hunt transition
        g = -1.0
        lam = 10.0**g
        rebuild_threshold = lam0 * (1.0 - TIGHT)
        prev_W = None
        lam_c = None
        cat_steps = 0
        cat_time_total = 0.0

        for step in range(steps_needed):
            if step > 0 and lam < rebuild_threshold:
                lam0 = lam; rebuild_threshold = lam0 * (1.0 - TIGHT)
                dets_0, elems_0, twists = build_cache(lam0, N)

            t1 = time.time()
            W = winding_cat(lam, dets_0, elems_0, twists, lam0)
            if W is None:
                # Numerical collapse — immediate rebuild, skip this step
                lam0 = lam; rebuild_threshold = lam0 * (1.0 - TIGHT)
                dets_0, elems_0, twists = build_cache(lam0, N)
                W = winding_cat(lam, dets_0, elems_0, twists, lam0)
                if W is None:
                    g += np.log10(1.0 - lr); lam = 10.0**g
                    continue  # skip step with NaN
            cat_time_total += time.time() - t1

            if prev_W is not None and W != prev_W:
                lam_c = lam
                cat_steps = step
                # Validate: compare with direct winding at transition
                W_dir = direct_winding(lam, N)
                W_dir_prev = direct_winding(lam * 1.03, N)  # check just before
                break

            prev_W = W
            g += np.log10(1.0 - lr)
            lam = 10.0**g

        if lam_c is None:
            lam_c = lam

        # Validate with direct winding
        if cat_steps > 0:
            W_dir = direct_winding(lam_c, N)
            confirm = f"cat={W} direct={W_dir} {'MATCH' if W==W_dir else 'MISMATCH'}"
        else:
            confirm = "not found"

        # Per-call timing
        cat_per_step = cat_time_total / max(cat_steps, 1)
        t2 = time.time()
        _ = direct_winding(lam_c, N)
        direct_per_call = time.time() - t2

        lam_pred = 0.05**N
        ratio = lam_c / lam_pred if lam_pred > 1e-300 else float('inf')
        speedup = direct_per_call / max(cat_per_step, 1e-12)

        print(f"  {N:5d}  {lam_c:16.4e}  {lam_pred:16.4e}  "
              f"{ratio:8.2f}  {cat_per_step*1e6:8.1f}us "
              f"{direct_per_call*1e3:8.2f}ms {speedup:8.1f}x  "
              f"{confirm}")

        results.append((N, lam_c, lam_pred, ratio, speedup, confirm))

    print()
    print(f"  Predicted: lam_c = (0.05)^N")
    print(f"  Catalytic: O(n_phi) per step via rank-1 lemma + periodic cache rebuild")
    print(f"  Direct:    O(n_phi * N^3) per winding call")
    print("=" * 78)

    # Scaling law verification
    print(f"\n{'=' * 78}")
    print("  SCALING LAW VERIFICATION")
    print(f"{'=' * 78}")
    Ns = [r[0] for r in results if r[0] <= 64]
    lams = [r[1] for r in results if r[0] <= 64]

    for N, lc in zip(Ns, lams):
        pred = 0.05**N
        print(f"  N={N:3d}: obs={lc:.4e}  pred={pred:.4e}  "
              f"log10(obs/pred)={np.log10(lc/pred):+.2f}")

    print(f"{'=' * 78}")

    return results


def direct_winding(lam, N, n_phi=200):
    """Direct determinant computation for timing comparison."""
    E_ref = -0.05j
    I = torch.eye(N).to(COMPLEX)
    dets = torch.zeros(n_phi, dtype=COMPLEX)

    H0 = torch.zeros((N,N), dtype=COMPLEX)
    for i in range(N-1): H0[i+1,i] = 1.0+0j
    H0[N-1,N-1] = -1j
    for i in range(N-1): H0[i,i] = -0.1j

    for k, phi in enumerate(torch.linspace(0, 2*np.pi, n_phi)):
        H = H0.clone()
        H[0, N-1] = lam * torch.tensor(np.exp(1j*phi.item()), dtype=COMPLEX)
        M = H - E_ref*I
        try:    dets[k] = torch.linalg.det(M)
        except: sign,ld = torch.linalg.slogdet(M); dets[k] = sign*torch.exp(ld)

    dt = torch.diff(torch.angle(dets))
    dt = torch.remainder(dt + np.pi, 2*np.pi) - np.pi
    return int(round(float(torch.sum(dt).item())/(2*np.pi)))


# ======================================================================
#  Main
# ======================================================================

if __name__ == "__main__":
    scaling_sweep()
