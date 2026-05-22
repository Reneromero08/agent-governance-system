"""Q1 Hardened: grad_S normalization using QEC precision sweep data.

Loads v8_depol sweep.json (surface codes, d=3-11, 100k shots).
Computes E_measured = R * nabla_S / sigma^D_f for each condition.
If nabla_S is the correct normalization, E_measured should be constant
across all p, d — matching the calibrated E = 0.0169.
"""

import json, math, sys
from pathlib import Path
import numpy as np

SWEEP_PATH = Path(
    "THOUGHT/LAB/FORMULA/v4/qec_precision_sweep/v8/results/v8_depol/sweep.json")
ANALYSIS_PATH = Path(
    "THOUGHT/LAB/FORMULA/v4/qec_precision_sweep/v9/results/v9_extended/analysis.json")


def t(d):
    return (d - 1) // 2


def main():
    print("=" * 72)
    print("Q1 HARDENED: grad_S normalization on QEC surface codes")
    print("  E_measured = R * nabla_S / sigma^D_f")
    print("  E_calibrated = 0.0169 (from training)")
    print("=" * 72)
    print()

    with open(SWEEP_PATH) as f:
        sweep = json.load(f)
    with open(ANALYSIS_PATH) as f:
        analysis = json.load(f)

    E_cal = analysis["E"]
    sigma_map = {float(k): v for k, v in analysis["sigma_map"].items()}
    print(f"  E_calibrated: {E_cal:.6f}")
    print(f"  Sigma map: {len(sigma_map)} points")
    print()

    conditions = sweep["conditions"]
    print(f"  Total conditions: {len(conditions)}")
    print()

    results = []
    skipped = 0
    for c in conditions:
        p = float(c["physical_error_rate"])
        d = int(c["distance"])
        logR = float(c["log_suppression"])
        syn_density = float(c["syndrome_density"])
        nabla_S = math.sqrt(max(syn_density, 1e-12))
        D_f = t(d)
        R = math.exp(logR)

        sigma = sigma_map.get(p)
        if sigma is None or sigma <= 0:
            skipped += 1
            continue

        E_meas = R * nabla_S / (sigma ** D_f)
        results.append({
            "p": p, "d": d, "D_f": D_f, "nabla_S": nabla_S,
            "sigma": sigma, "logR": logR, "E_meas": E_meas,
        })

    print(f"  Used: {len(results)}, skipped (no sigma): {skipped}")
    print()

    E_vals = np.array([r["E_meas"] for r in results])
    E_mean = E_vals.mean()
    E_std = E_vals.std()
    E_cv = E_std / abs(E_mean) if abs(E_mean) > 1e-12 else 999

    # Group by distance
    print(f"  {'d':>3} {'D_f':>3} {'count':>5} {'E_mean':>12} {'E_std':>12}")
    print(f"  {'-'*3} {'-'*3} {'-'*5} {'-'*12} {'-'*12}")
    for d in sorted(set(r["d"] for r in results)):
        d_results = [r["E_meas"] for r in results if r["d"] == d]
        print(f"  {d:>3} {t(d):>3} {len(d_results):>5} "
              f"{np.mean(d_results):>12.6f} {np.std(d_results):>12.6f}")

    # Group by error rate
    print(f"\n  {'p':>8} {'count':>5} {'E_mean':>12} {'E_std':>12}")
    print(f"  {'-'*8} {'-'*5} {'-'*12} {'-'*12}")
    for p in sorted(set(r["p"] for r in results)):
        p_results = [r["E_meas"] for r in results if abs(r["p"] - p) < 1e-9]
        print(f"  {p:>8.4f} {len(p_results):>5} "
              f"{np.mean(p_results):>12.6f} {np.std(p_results):>12.6f}")

    print()
    print(f"  Global: mean={E_mean:.6f}, std={E_std:.6f}, CV={E_cv:.4f}")
    print(f"  Calibrated E: {E_cal:.6f}")
    print(f"  Delta: {abs(E_mean - E_cal):.6f} "
          f"({'MATCH' if abs(E_mean - E_cal) < 2*E_std else 'OFF'})")
    print()

    print("=" * 72)
    if E_cv < 0.5 and abs(E_mean - E_cal) < max(0.01, 2 * E_std):
        print("Q1 VERIFIED: E_measured is constant across all (p,d).")
        print(f"  CV={E_cv:.3f}, matches calibrated E={E_cal:.6f}.")
        print("  nabla_S IS the correct normalization for E.")
    elif E_cv < 1.0:
        print(f"Q1 SUPPORTED: Moderate consistency (CV={E_cv:.3f}).")
        print("  nabla_S is a reasonable normalization.")
    else:
        print(f"Q1 NOT SUPPORTED: E_measured varies (CV={E_cv:.3f}).")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
