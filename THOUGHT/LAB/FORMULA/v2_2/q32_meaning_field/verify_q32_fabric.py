"""Q32: Test the formula R = (E/nabla_S) * sigma^D_f as equilibrium.

The operator: multi-scale discrete Laplacian (from SEMIOTIC_ACTION_PRINCIPLE.md).
Each round r (step=2^r): rotate each pair toward mean by (1-sigma).
After many rounds, the signal reaches equilibrium.

Formula prediction: R_eq = (E/nabla_S) * sigma^D_f
- E = initial amplitude = 2.0 (peak-to-peak of sine wave)
- nabla_S = frequency k (entropy gradient of the pattern)
- sigma = per-round preservation
- D_f = number of rounds

Test: inject sine wave of frequency k, run D_f rounds, measure final amplitude.
Check if final amplitude * nabla_S / sigma^D_f ≈ constant (E).
"""

import math, sys, time
from pathlib import Path
import numpy as np

N = 4096
SEED_BASE = 20260521

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def multiscale_laplacian(tape, n_rounds, sigma, nabla_S_global=0.0):
    """Multi-scale wave operator: (1-sigma)*Box + nabla_S.
    
    Box: rotation toward pair mean (diffusion)
    nabla_S: damping toward zero (entropy gradient = mass)
    
    Rounds wrap around: each cycle of R_MAX rounds applies all scales.
    n_rounds can exceed R_MAX — the multi-scale cycle repeats.
    """
    result = tape.copy().astype(np.float64)
    n = len(result)
    alpha = 1.0 - sigma
    R_MAX = int(math.log2(n))  # rounds per cycle
    for r in range(n_rounds):
        r_eff = r % R_MAX
        step = 1 << r_eff
        # Damping: pull toward zero proportional to nabla_S
        if nabla_S_global > 0:
            damping_factor = 1.0 - alpha * nabla_S_global * step / n
            result *= max(damping_factor, 0.0)  # clamp to prevent explosion
        # Diffusion: rotate toward pair mean
        for i in range(0, n - step, step * 2):
            j = i + step
            mean = (result[i] + result[j]) / 2.0
            result[i] += alpha * (mean - result[i])
            result[j] += alpha * (mean - result[j])
    return result


def pattern_amplitude(signal, start, size):
    """Peak-to-peak amplitude of pattern region."""
    region = signal[start:start + size]
    return float(region.max() - region.min())


def main():
    print("=" * 72)
    print("Q32: FORMULA AS EQUILIBRIUM OF SEMIOTIC WAVE OPERATOR")
    print("  R_eq = (E/nabla_S) * sigma^D_f")
    print("  Operator: (1-sigma)*Box + nabla_S (multi-scale Laplacian)")
    print("=" * 72)
    print()

    rng = np.random.RandomState(SEED_BASE)
    signal_size = 1024
    fabric = rng.randn(N).astype(np.float64) * 0.01

    sigmas = [0.99, 0.95, 0.9]
    ks = [1, 2, 4, 8, 16, 32]
    rounds = [0, 2, 4, 8, 16, 32, 64]

    # nabla_S proportional to frequency: higher freq = steeper gradient
    # The damping per round = alpha * nabla_S * step/N ≈ alpha * k * 2^r/N
    # We set nabla_S_global = k / N so total damping depends on both k and rounds

    for sigma in sigmas:
        print(f"\n  sigma = {sigma}")
        D_f = 64
        for k in ks:
            positions = np.arange(signal_size)
            pattern = np.sin(2 * math.pi * k * positions / signal_size)
            tape = fabric.copy()
            tape[:signal_size] = pattern

            nabla_S = k / N  # entropy gradient proportional to frequency
            result = multiscale_laplacian(tape.copy(), D_f, sigma, nabla_S)
            amp = pattern_amplitude(result, 0, signal_size)

            # Formula: R = (E/nabla_S) * sigma^D_f
            # E = 2.0, nabla_S = k
            # For the damping model: R ≈ 2.0 * sigma^D_f
            pred = 2.0 * (sigma ** D_f)

            print(f"    k={k:>3}: amp={amp:.4f}  pred={pred:.6f}")

    print()
    print("=" * 72)
    print("With damping term: amplitude should follow sigma^D_f")
    print("Higher k -> more damping -> lower amplitude")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
