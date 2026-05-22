"""Q32: Entropy vs signal survival through the semiotic wave operator."""
import math, sys
import numpy as np

N = 4096

def multiscale_op(tape, n_rounds, sigma, nabla_S_global=0.0):
    """Multi-scale wave + dissipation operator from semiotic action.
    
    Unitary: rotation toward pair mean (Box term)
    Dissipative: damping proportional to local gradient (nabla_S term)
    """
    result = tape.copy().astype(np.float64)
    n = len(result)
    alpha = 1.0 - sigma
    R_MAX = int(math.log2(n))
    
    for cycle in range(n_rounds // R_MAX + 1):
        if cycle >= n_rounds // R_MAX and n_rounds % R_MAX == 0:
            break
        rounds_this_cycle = min(R_MAX, n_rounds - cycle * R_MAX)
        for r in range(rounds_this_cycle):
            step = 1 << r
            # Unitary: rotate toward mean
            for i in range(0, n - step, step * 2):
                j = i + step
                mean = (result[i] + result[j]) / 2.0
                result[i] += alpha * (mean - result[i])
                result[j] += alpha * (mean - result[j])
        
        # Dissipative: after each full cycle, apply entropy damping
        if nabla_S_global > 0:
            # Damping proportional to local gradient magnitude
            grad = np.abs(np.diff(result, prepend=result[0]))
            # Normalize gradient by its mean
            grad_norm = grad / (grad.mean() + 1e-12)
            # Damp each position: stronger where gradient is steeper
            damping = 1.0 - alpha * nabla_S_global * grad_norm / R_MAX
            result *= np.clip(damping, 0.0, 1.0)
    
    return result

def spectral_entropy(signal):
    """Spectral entropy: entropy of normalized FFT power spectrum.
    Low for pure tones, high for white noise. This IS nabla_S."""
    n = len(signal)
    fft = np.abs(np.fft.rfft(signal))
    power = fft ** 2
    power = power / (power.sum() + 1e-15)
    power = power[power > 1e-15]
    return float(-np.sum(power * np.log2(power)))

def main():
    rng = np.random.RandomState(42)
    fabric = rng.randn(N).astype(np.float64) * 0.01
    signal_size = 1024

    print("ENTROPY vs SIGNAL SURVIVAL")
    print("=" * 60)
    print("sigma=0.8, D_f=32")
    print()
    print(f"  {'signal type':>22} {'nabla_S':>8} {'amp_start':>10} {'amp_end':>10} {'survival':>10}")
    print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    results = []
    # Generate signals with varying entropy by adding noise to a sine wave
    for noise_level in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        signal = np.sin(2*np.pi*4*np.arange(signal_size)/signal_size) + noise_level * rng.randn(signal_size)
        nabla_S = spectral_entropy(signal)
        tape = fabric.copy()
        tape[:signal_size] = signal
        result = multiscale_op(tape, 32, 0.8, nabla_S)
        amp_start = signal.max() - signal.min()
        amp_end = result[:signal_size].max() - result[:signal_size].min()
        survival = amp_end / max(amp_start, 1e-12)
        name = f"noise={noise_level:.2f}"
        results.append((name, nabla_S, amp_start, amp_end, survival))
        print(f"  {name:>22} {nabla_S:>8.4f} {amp_start:>10.4f} {amp_end:>10.4f} {survival:>10.4f}")

    print()
    print("FORMULA TEST: survival vs 1/nabla_S")
    print("-" * 50)

    nablas = np.array([r[1] for r in results])
    survivals = np.array([r[4] for r in results])

    # Filter zero-entropy (would be infinite in formula)
    valid = nablas > 1e-10
    if valid.sum() >= 3:
        inv_nabla = 1.0 / nablas[valid]
        surv = survivals[valid]

        # Fit: survival = A/nabla_S + B
        A_mat = np.vstack([inv_nabla, np.ones_like(inv_nabla)]).T
        coeff, residuals, rank, sv = np.linalg.lstsq(A_mat, surv, rcond=None)
        slope, intercept = coeff[0], coeff[1]
        pred = slope * inv_nabla + intercept
        ss_res = np.sum((surv - pred) ** 2)
        ss_tot = np.sum((surv - surv.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        print(f"  Fit: survival = {slope:.4f}/nabla_S + {intercept:.4f}")
        print(f"  R2 = {r2:.4f}")
        print(f"  Formula predicts: survival = const/nabla_S")

        if r2 > 0.9:
            print("  VERDICT: Formula CONFIRMED - survival proportional to 1/nabla_S")
        elif r2 > 0.6:
            print("  VERDICT: Formula SUPPORTED - inverse relationship detected")
        else:
            print(f"  VERDICT: Formula NOT SUPPORTED - R2={r2:.3f} too low")
    else:
        print("  All signals have near-zero entropy. Formula cannot be tested.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
