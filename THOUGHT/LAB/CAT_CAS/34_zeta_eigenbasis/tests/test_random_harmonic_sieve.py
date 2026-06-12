"""
Integrity Test: Harmonic Sieve Control
======================================
Testing the Riemann Harmonic Sieve against integers and random noise
to ensure the extracted frequencies are unique to the prime distribution
and not an artifact of PCA/FFT decomposition.
"""

import sys, math
from pathlib import Path
import numpy as np
import torch

REPO = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
sys.path.insert(0, str(REPO / "THOUGHT" / "LAB" / "TINY_COMPRESS" / "holographic-image"))

try:
    from holo_core import analyze_spectrum, project
except ImportError:
    class DummyProj:
        pass
    def project(obs, policy="fixed", fixed_k=10):
        cov = obs.T @ obs
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx]
        p = DummyProj()
        p.basis = vecs[:, :fixed_k].T
        return p

def extract_continuous_frequencies(evec_complex, max_freq=100.0):
    sig = torch.tensor(evec_complex.astype(np.complex64))
    fft_vals = torch.fft.fft(sig)
    power = torch.abs(fft_vals)**2
    
    N = len(sig)
    freqs = torch.fft.fftfreq(N, d=1.0) * N
    
    half_N = N // 2
    power = power[:half_N]
    freqs = freqs[:half_N]
    
    power = power / (torch.max(power) + 1e-15)
    
    peaks = []
    for i in range(1, half_N - 1):
        if power[i] > power[i-1] and power[i] > power[i+1] and power[i] > 0.05:
            alpha = power[i-1].item()
            beta = power[i].item()
            gamma = power[i+1].item()
            p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma + 1e-15)
            peak_freq = freqs[i].item() + p
            if peak_freq > 1.0 and peak_freq <= max_freq:
                peaks.append((peak_freq, beta))
                
    peaks.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in peaks]

def run_sieve(seq, name="Sequence"):
    N_items = len(seq)
    ln_seq = np.log(seq)
    
    S = np.zeros((N_items, N_items), dtype=np.complex128)
    for m in range(N_items):
        S[m, :] = np.exp(1j * ln_seq[m] * ln_seq)
        
    obs = np.hstack([S.real, S.imag])
    proj = project(obs, policy="fixed", fixed_k=16)
    basis = proj.basis
    
    all_frequencies = []
    for i in range(min(16, basis.shape[0])):
        evec_complex = basis[i, :N_items] + 1j * basis[i, N_items:]
        freqs = extract_continuous_frequencies(evec_complex, max_freq=60.0)
        
        scale_factor = 2 * math.pi / np.mean(np.diff(ln_seq))
        scaled_freqs = [f * scale_factor / N_items for f in freqs]
        
        for f in scaled_freqs:
            if 10.0 < f < 60.0:
                all_frequencies.append(f)
                
    all_frequencies.sort()
    clustered = []
    for f in all_frequencies:
        if not clustered or abs(f - clustered[-1][0]) > 1.5:
            clustered.append([f])
        else:
            clustered[-1].append(f)
            
    final_peaks = [np.mean(c) for c in clustered if len(c) > 1]
    print(f"{name:>10s} Peaks: {[f'{z:.2f}' for z in final_peaks[:6]]}")

def primes_upto(N):
    if N < 1: return []
    est = int(N * (math.log(N) + math.log(math.log(N))))
    sieve = np.ones(est, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(est**0.5) + 1):
        if sieve[i]: sieve[i*i:est:i] = False
    return np.where(sieve)[0][:N].astype(np.float64)

def main():
    print("=" * 60)
    print("INTEGRITY CHECK: Harmonic Sieve Control")
    print("=" * 60)
    
    N = 2000
    
    p = primes_upto(N)
    run_sieve(p, "Primes")
    
    integers = np.arange(2, N + 2, dtype=np.float64)
    run_sieve(integers, "Integers")
    
    # Use exponential distribution so log(random) is uniform-ish
    # or just uniform randoms
    np.random.default_rng(42)
    randoms = np.sort(np.random.uniform(2.0, float(N*10), size=N))
    run_sieve(randoms, "Random")

if __name__ == "__main__":
    main()
