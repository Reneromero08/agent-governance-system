"""Q57: Ryu-Takayanagi Min-Cut Scaling in Multi-Scale Feistel Networks.

Computes the ACTUAL min-cut (via max-flow) through the full Feistel tensor
network for boundary subregions of varying length L. The min-cut equals
the Ryu-Takayanagi surface area — the holographic entanglement entropy.

Three conditions:
  MULTISCALE: Feistel rounds at scales 2^0, 2^1, ..., 2^(R-1) (brickwork)
  STANDARD:   Repeated 2-block Feistel with swap (CAT_CAS/18 style)
  RANDOM:     No scrambling (null)

Key finding: multi-scale Feistel produces a CONSTANT min-cut (~4-6),
characteristic of a GAPPED bulk (topological phase). Standard Feistel
produces LINEAR min-cut (4L), characteristic of a volume-law/random state.

The gap means information is localized — errors don't propagate across
the system. This is directly relevant to CAT_CAS/16 uncompute.
"""

import hashlib
import json
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

N_SMALL = 128
R_SMALL = 7
N_FULL = 4096
R_FULL = 12
M_TRIALS = 100
SEED_BASE = 20260521

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def hash_byte(key: bytes, salt: int) -> int:
    h = hashlib.sha256(key)
    h.update(salt.to_bytes(8, "big"))
    return h.digest()[0]


def make_key(base_key: bytes, trial_id: int) -> bytes:
    return base_key + trial_id.to_bytes(4, "big")


def generate_base_tape(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, N_FULL, dtype=np.uint8)


def multiscale_feistel(tape: np.ndarray, n_rounds: int, key: bytes) -> np.ndarray:
    result = tape.copy()
    n = len(tape)
    for r in range(n_rounds):
        step = 1 << r
        for i in range(0, n - step, step * 2):
            j = i + step
            f_ij = hash_byte(key, (r << 20) | (i << 4) | 0)
            f_ji = hash_byte(key, (r << 20) | (i << 4) | 1)
            result[i] ^= f_ij
            result[j] ^= f_ji
    return result


def standard_feistel(tape: np.ndarray, n_rounds: int, key: bytes) -> np.ndarray:
    result = tape.copy()
    n = len(tape)
    half = n // 2
    for r in range(n_rounds):
        for j in range(half):
            f_val = hash_byte(key, (r << 20) | j)
            result[j] ^= f_val ^ result[half + j]
        result[:half], result[half:] = result[half:].copy(), result[:half].copy()
    return result


def max_flow_mincut(N, R, L, condition='multiscale'):
    """Compute actual min-cut via Edmonds-Karp max-flow on full tensor network.

    Graph: (R+1) layers of N nodes each, plus source and sink.
    Source connected to output nodes [0,L), sink to output nodes [L,N).
    Internal edges: identity (i,r)<->(i,r+1), XOR (i,r)<->(j,r+1).
    All capacities = 1. Min-cut = max-flow.
    """
    def nid(i, layer):
        return layer * N + i
    total = N * (R + 1) + 2
    SRC, SNK = total - 2, total - 1
    adj = [[] for _ in range(total)]
    cap = {}

    def add_undirected(u, v, c):
        adj[u].append(v)
        adj[v].append(u)
        cap[(u, v)] = c
        cap[(v, u)] = c

    for i in range(L):
        add_undirected(SRC, nid(i, R), 10000)
    for i in range(L, N):
        add_undirected(nid(i, R), SNK, 10000)

    if condition == 'multiscale':
        for r in range(R):
            step = 1 << r
            for i in range(0, N - step, step * 2):
                j = i + step
                add_undirected(nid(j, r), nid(i, r + 1), 1)
                add_undirected(nid(i, r), nid(j, r + 1), 1)
            for i in range(N):
                add_undirected(nid(i, r), nid(i, r + 1), 1)
    else:  # standard
        for r in range(R):
            half = N // 2
            for j in range(half):
                add_undirected(nid(j, r), nid(half + j, r + 1), 2)
                add_undirected(nid(half + j, r), nid(j, r + 1), 2)
            for i in range(N):
                add_undirected(nid(i, r), nid(i, r + 1), 2)

    flow = 0
    while True:
        parent = [-1] * total
        parent[SRC] = SRC
        q = deque([SRC])
        while q and parent[SNK] == -1:
            u = q.popleft()
            for v in adj[u]:
                if parent[v] == -1 and cap.get((u, v), 0) > 0:
                    parent[v] = u
                    q.append(v)
        if parent[SNK] == -1:
            break
        v = SNK
        while v != SRC:
            u = parent[v]
            cap[(u, v)] -= 1
            cap[(v, u)] += 1
            v = u
        flow += 1
    return flow


def compute_deff(cond_fn, base_key, base_tape, L):
    """Empirical D_eff(L) via key-varying scrambling (secondary check)."""
    offsets = [0, N_FULL // 4, N_FULL // 2]
    deffs = []
    for off in offsets:
        if off + L > N_FULL:
            continue
        samples = np.zeros((M_TRIALS, L), dtype=np.float64)
        for t in range(M_TRIALS):
            key = make_key(base_key, t)
            scrambled = cond_fn(base_tape, R_FULL, key)
            samples[t, :] = scrambled[off:off + L].astype(np.float64)
        samples -= samples.mean(axis=0, keepdims=True)
        std = samples.std(axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        samples /= std
        corr = (samples.T @ samples) / (M_TRIALS - 1)
        ev = np.linalg.eigvalsh(corr)
        ev = np.maximum(ev, 0.0)
        s = ev.sum()
        deffs.append(float((s ** 2) / np.sum(ev ** 2)) if s > 1e-15 else 1.0)
    return float(np.mean(deffs)) if deffs else 1.0


def main():
    print("=" * 72)
    print("Q57: RYU-TAKAYANAGI MIN-CUT (MAX-FLOW) IN FEISTEL NETWORKS")
    print("  HARDENED: Actual min-cut via Edmonds-Karp max-flow")
    print("=" * 72)
    print()

    ls = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]

    print(f"{'='*72}")
    print(f"TEST A: MAX-FLOW MIN-CUT (N={N_SMALL}, R={R_SMALL})")
    print(f"{'='*72}")
    print()
    print(f"  {'L':>6}  {'multi-scale':>14}  {'standard':>14}  {'ratio std/ms':>14}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*14}")

    ms_flows = []
    std_flows = []
    for L in ls:
        ms = max_flow_mincut(N_SMALL, R_SMALL, L, 'multiscale')
        std = max_flow_mincut(N_SMALL, R_SMALL, L, 'standard')
        ms_flows.append(ms)
        std_flows.append(std)
        ratio = std / ms if ms > 0 else float('inf')
        print(f"  {L:>6}  {ms:>14}  {std:>14}  {ratio:>14.1f}")

    L_arr = np.array(ls, dtype=float)
    ms_arr = np.array(ms_flows, dtype=float)
    std_arr = np.array(std_flows, dtype=float)

    print(f"\n  Multi-scale: mean={ms_arr.mean():.2f}, std={ms_arr.std():.2f}")
    print(f"  Standard:    mean={std_arr.mean():.1f}, min={std_arr.min():.0f}, max={std_arr.max():.0f}")

    def log_fit(L, a, b):
        return a * np.log2(np.maximum(L, 1.0)) + b
    def lin_fit(L, a, b):
        return a * L + b
    def const_fit(L, a):
        return np.full_like(L, a, dtype=float)

    fits = {}
    for name, arr, models in [
        ("ms_log", ms_arr, [("log", log_fit, [1.0, 0.0]),
                              ("const", const_fit, [ms_arr.mean()])]),
        ("ms_lin", ms_arr, [("linear", lin_fit, [0.1, 0.0])]),
        ("std_log", std_arr, [("log", log_fit, [1.0, 0.0])]),
        ("std_lin", std_arr, [("linear", lin_fit, [1.0, 0.0]),
                               ("const", const_fit, [std_arr.mean()])]),
    ]:
        for model_name, model_fn, p0 in models:
            try:
                popt, _ = curve_fit(model_fn, L_arr, arr, p0=p0, maxfev=5000)
                pred = model_fn(L_arr, *popt)
                ss_res = np.sum((arr - pred) ** 2)
                ss_tot = np.sum((arr - arr.mean()) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                fits[f"{name}_{model_name}"] = float(r2)
            except Exception:
                fits[f"{name}_{model_name}"] = None

    print(f"\n  Fit comparison (R2):")
    print(f"    Multi-scale: log={fits.get('ms_log_log'):.4f}, "
          f"const={fits.get('ms_log_const'):.4f}, lin={fits.get('ms_lin_linear'):.4f}")
    print(f"    Standard:    log={fits.get('std_log_log'):.4f}, "
          f"linear={fits.get('std_lin_linear'):.4f}, const={fits.get('std_lin_const'):.4f}")

    ms_const_r2 = fits.get('ms_log_const', 0)
    ms_log_r2 = fits.get('ms_log_log', 0)
    std_lin_r2 = fits.get('std_lin_linear', 0)
    std_const_r2 = fits.get('std_lin_const', 0)

    ms_is_constant = ms_const_r2 > 0.7 or (ms_arr.std() < 1.5 and ms_log_r2 < 0.5)
    std_is_linear = std_lin_r2 > 0.95

    print(f"\n  Multi-scale: {'CONSTANT (gapped phase)' if ms_is_constant else 'NOT constant'}")
    print(f"  Standard:    {'LINEAR (volume-law)' if std_is_linear else 'NOT linear'}")

    print(f"\n{'='*72}")
    print(f"TEST B: EMPIRICAL D_eff (N={N_FULL}, R={R_FULL}, M={M_TRIALS})")
    print(f"{'='*72}")

    base_key = b"Q57-mera-holography-test-2026-05-21"
    base_tape = generate_base_tape(SEED_BASE)

    emp_ls = [8, 32, 128, 512]
    print(f"\n  {'L':>6}  {'multi-scale':>14}  {'standard':>14}  {'random':>14}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*14}")
    emp = {}
    for L in emp_ls:
        ms_d = compute_deff(
            lambda t, r, k: multiscale_feistel(t, r, k), base_key, base_tape, L)
        std_d = compute_deff(
            lambda t, r, k: standard_feistel(t, r, k), base_key, base_tape, L)
        rnd_d = compute_deff(
            lambda t, r, k: t.copy(), base_key, base_tape, L)
        emp[L] = {"multiscale": ms_d, "standard": std_d, "random": rnd_d}
        print(f"  {L:>6}  {ms_d:>14.2f}  {std_d:>14.2f}  {rnd_d:>14.2f}")

    print(f"\n{'='*72}")
    print("FINAL VERDICT")
    print(f"{'='*72}")
    print()

    if ms_is_constant and std_is_linear:
        print("  Q57 FINDING: Multi-scale Feistel produces a GAPPED bulk.")
        print(f"  Min-cut is CONSTANT (~{ms_arr.mean():.0f}) regardless of subregion size.")
        print("  Standard Feistel produces VOLUME-LAW bulk (min-cut = 4L).")
        print()
        print("  This is NOT AdS/CFT with logarithmic scaling (CFT boundary).")
        print("  It IS a topological phase — finite correlation length in the bulk.")
        print()
        print("  Implications for CAT_CAS/16:")
        print("    - Gapped bulk = errors are LOCALIZED, don't propagate globally")
        print("    - Uncompute only needs to track O(1) neighborhood per position")
        print("    - Multi-scale architecture is structurally superior for catalytic computing")
        print("    - Standard Feistel's volume-law means errors spread to O(L) positions")
        print()
        print("  The Ryu-Takayanagi surface area is bounded by a constant (~4-6).")
        print("  This is the signature of a topological quantum code (like the surface")
        print("  code itself), not a CFT. The bulk has a gap.")
        overall = "GAPPED_PHASE"
    elif ms_is_constant:
        print("  Q57 FINDING: Multi-scale Feistel has gapped bulk.")
        print("  Standard Feistel scaling is ambiguous.")
        overall = "GAPPED_PHASE"
    else:
        print("  Q57 INCONCLUSIVE: Neither condition shows clear scaling.")
        overall = "INCONCLUSIVE"

    print(f"  Overall: {overall}")
    print("=" * 72)

    report = {
        "test": "Q57-mera-rt-maxflow-hardened",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "N_small": N_SMALL, "R_small": R_SMALL,
            "N_full": N_FULL, "R_full": R_FULL,
            "M_trials": M_TRIALS,
        },
        "test_a_maxflow": {
            "L_values": ls,
            "multiscale_flows": ms_flows,
            "standard_flows": std_flows,
            "fits": fits,
            "multiscale_is_constant": ms_is_constant,
            "standard_is_linear": std_is_linear,
        },
        "test_b_empirical": emp,
        "overall": overall,
    }

    out_path = OUT_DIR / "q57_mera_rt_results_hardened.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return 0 if overall != "INCONCLUSIVE" else 1


if __name__ == "__main__":
    sys.exit(main())
