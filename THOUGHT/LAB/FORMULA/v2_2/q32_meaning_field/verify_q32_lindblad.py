"""Q32: Lindblad dissipation on the Feistel fabric.

Implements the full semiotic wave equation with environmental coupling:
  (1-sigma)*Box psi + nabla_S*psi = noise_source

Lindblad (classical Langevin analog):
  d(psi)/dt = -nabla_S * psi + sqrt(2*nabla_S*kT) * dW
  
Discrete: after each multi-scale cycle:
  psi *= exp(-nabla_S * dt)
  psi += sqrt(2*nabla_S*kT*dt) * N(0,1)

At equilibrium: <psi^2> = kT  (independent of nabla_S!)
The survival ratio = final_amp / initial_amp = sqrt(kT)/E0

But the FORMULA predicts: R = (E/nabla_S) * sigma^D_f
This means the equilibrium amplitude SHOULD depend on nabla_S.

Test: does survival = A/nabla_S + B after Lindblad dissipation?
"""

import math, sys
import numpy as np

N = 4096

def spectral_entropy(signal):
    fft = np.abs(np.fft.rfft(signal))
    power = fft ** 2
    power = power / (power.sum() + 1e-15)
    power = power[power > 1e-15]
    return float(-np.sum(power * np.log2(power)))


def lindblad_fabric(tape, n_rounds, sigma, nabla_S):
    """Multi-scale diffusion + Lindblad dissipation.
    
    Unitary: rotation toward pair mean (Box term)
    Dissipative: Langevin damping + noise injection per cycle
    """
    result = tape.copy().astype(np.float64)
    n = len(result)
    alpha = 1.0 - sigma
    R_MAX = int(math.log2(n))
    rng = np.random.RandomState(42)
    
    cycles = max(1, n_rounds // R_MAX)
    
    for cycle in range(cycles):
        # Unitary: multi-scale rotation (one full cycle)
        for r in range(R_MAX):
            step = 1 << r
            for i in range(0, n - step, step * 2):
                j = i + step
                mean = (result[i] + result[j]) / 2.0
                result[i] += alpha * (mean - result[i])
                result[j] += alpha * (mean - result[j])
        
        # Lindblad dissipation: damping + noise
        if nabla_S > 1e-12:
            # Temperature: set so equilibrium variance = kT
            kT = 0.01  # noise floor temperature
            dt = 1.0 / R_MAX  # time step per cycle
            
            # Damping: multiplicative
            damping = np.exp(-nabla_S * dt)
            result *= damping
            
            # Noise injection: additive Gaussian
            noise_std = np.sqrt(2.0 * nabla_S * kT * dt)
            result += rng.randn(n) * noise_std
    
    return result


def main():
    print("=" * 72)
    print("Q32: LINDBLAD DISSIPATION ON FEISTEL FABRIC")
    print("  d(psi)/dt = -nabla_S*psi + noise")
    print("  Hypothesis: equilibrium amplitude ~ 1/nabla_S")
    print("=" * 72)
    print()

    rng = np.random.RandomState(42)
    fabric = rng.randn(N).astype(np.float64) * 0.01
    signal_size = 1024

    sigmas = [0.99, 0.95, 0.9, 0.8]
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    D_f = 64  # enough cycles for equilibrium

    print(f"D_f={D_f} (equilibrium)")
    print()
    print(f"  {'sigma':>8} {'noise':>8} {'nabla_S':>10} {'amp_end':>10} {'survival':>10} {'pred 1/nS':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    all_nablas = []
    all_survivals = []

    for sigma in sigmas:
        for noise_level in noise_levels:
            signal = (np.sin(2*np.pi*4*np.arange(signal_size)/signal_size) 
                      + noise_level * rng.randn(signal_size))
            nabla_S = spectral_entropy(signal)
            nabla_S_clipped = max(nabla_S, 0.01)  # avoid div by zero
            
            tape = fabric.copy()
            tape[:signal_size] = signal
            result = lindblad_fabric(tape, D_f, sigma, nabla_S_clipped)
            
            amp_start = max(signal.max() - signal.min(), 1e-12)
            amp_end = result[:signal_size].max() - result[:signal_size].min()
            survival = amp_end / amp_start
            
            all_nablas.append(nabla_S_clipped)
            all_survivals.append(survival)
            
            pred_inv = 1.0 / nabla_S_clipped
            print(f"  {sigma:>8.2f} {noise_level:>8.2f} {nabla_S_clipped:>10.4f} {amp_end:>10.4f} {survival:>10.4f} {pred_inv:>10.4f}")

    # Fit: survival = A/nabla_S + B
    nablas_arr = np.array(all_nablas)
    surv_arr = np.array(all_survivals)
    inv_nabla = 1.0 / nablas_arr

    A_mat = np.vstack([inv_nabla, np.ones_like(inv_nabla)]).T
    coeff, residuals, rank, sv = np.linalg.lstsq(A_mat, surv_arr, rcond=None)
    slope, intercept = coeff[0], coeff[1]
    pred = slope * inv_nabla + intercept
    ss_res = np.sum((surv_arr - pred) ** 2)
    ss_tot = np.sum((surv_arr - surv_arr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print()
    print("=" * 72)
    print(f"FIT: survival = {slope:.4f}/nabla_S + {intercept:.4f}")
    print(f"R2 = {r2:.4f}")
    
    if r2 > 0.85:
        print("LINDBLAD CONFIRMED: survival ~ 1/nabla_S")
        print("The formula R = (E/nabla_S) * sigma^D_f emerges from")
        print("the Lindblad dissipation on the multi-scale fabric.")
    elif r2 > 0.5:
        print("WEAK SUPPORT: inverse relationship detected but noisy")
    else:
        print("NOT SUPPORTED: Lindblad dissipation alone does not")
        print("produce 1/nabla_S scaling. Operator tuning needed.")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
