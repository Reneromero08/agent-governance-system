"""
Experiment 24 v2: Catalytic Phase Lasing (True Interference Bypass)
===================================================================
Uses the Catalytic Tape as a continuous Phase Diffraction Grating.
By mapping the modular exponentiation sequentially to the complex plane 
and executing a Fast Fourier Transform, we perfectly simulate the 
wave mechanics of the Quantum Fourier Transform (QFT). The non-resonant 
noise destructively interferes, and the true period r constructively lases.
"""

import sys
import time
import math
import random
import torch
from pathlib import Path
from fractions import Fraction

def generate_semiprime(bits):
    def get_prime(b):
        while True:
            p = random.getrandbits(b)
            p |= (1 << (b - 1)) | 1
            if is_prime(p): return p
    p = get_prime(bits // 2)
    q = get_prime(bits // 2)
    return p * q, p, q

def is_prime(n, k=5):
    if n <= 1 or n % 2 == 0: return n == 2 or n == 3
    s, d = 0, n - 1
    while d % 2 == 0:
        d //= 2; s += 1
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else: return False
    return True

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def main():
    print("=" * 78)
    print("EXPERIMENT 24 v2: CATALYTIC PHASE LASING")
    print("  The True Interference Bypass (Diffraction Grating / QFT Simulation)")
    print("=" * 78)
    print()

    # Target a 22-bit number. 
    # Max possible period is N. We will set the Grating Size M > N
    # to guarantee at least one full continuous wave cycle.
    BIT_SIZE = 22
    N, known_p, known_q = generate_semiprime(BIT_SIZE)
    
    a = 2
    while gcd(a, N) != 1:
        a += 1
        
    print(f"  Target: {BIT_SIZE}-bit Semiprime N = {N}")
    print(f"  Ground Truth: {known_p} x {known_q} (Hidden)")
    print(f"  Base 'a': {a}")
    print()

    # The Catalytic Tape (Phase Diffraction Grating)
    # We use M = 2^23 (8,388,608 elements) which is > max possible N (2^22).
    # This guarantees the topological wave completes without breaking.
    M_power = 23
    M = 2**M_power 
    
    print("-" * 78)
    print(f"PHASE 1: THE DIFFRACTION GRATING (Populating {M:,} Elements)")
    print("  Sequentially projecting f(x) = a^x mod N to maintain Phase Coherence...")
    print("-" * 78)

    t0 = time.perf_counter()
    
    # 1. Sequential Generation (O(M))
    # We generate the sequence efficiently. 
    # Python list comprehension is extremely fast for this.
    seq = [1]
    curr = 1
    for _ in range(1, M):
        curr = (curr * a) % N
        seq.append(curr)
        
    seq_tensor = torch.tensor(seq, dtype=torch.float32)
    
    # 2. Phase Mapping (It From Phase)
    # Map the discrete value to a continuous complex phase on the unit circle
    phases = (2.0 * math.pi * (seq_tensor / N))
    
    # Create the Optical Cavity / Grating
    grating = torch.polar(torch.ones(M, dtype=torch.float32), phases)
    
    gen_time = time.perf_counter() - t0
    print(f"  [+] Grating Population Time: {gen_time:.4f}s")
    
    print("-" * 78)
    print("PHASE 2: OPTICAL INTERFERENCE (QFT Simulation)")
    print("  Applying FFT. Noise cancels, Period lases.")
    print("-" * 78)
    
    t1 = time.perf_counter()
    
    # We take the FFT of the grating to simulate wave interference
    spectrum = torch.fft.fft(grating)
    
    # Ignore the DC component (index 0)
    spectrum[0] = 0
    
    fft_time = time.perf_counter() - t1
    print(f"  [+] FFT Interference Time: {fft_time:.4f}s")
    
    # Find the peak frequency
    peak_idx = torch.argmax(torch.abs(spectrum)).item()
    peak_amp = torch.abs(spectrum[peak_idx]).item()
    
    print(f"  [+] Lasing Peak FFT Index: {peak_idx}")
    print(f"  [+] Lasing Peak Amplitude: {peak_amp:.2f} (Theoretical Max: {M})")
    print(f"  [+] Signal-to-Noise Ratio: {peak_amp / torch.mean(torch.abs(spectrum)).item():.2f}")
    
    # Extract the period from the peak index using Continued Fractions (Shor's step)
    # The peak j occurs at j ~ M * (c / r)
    ratio = Fraction(peak_idx, M).limit_denominator(N)
    r_guess = ratio.denominator
    
    print(f"  [+] Extracted Period Guess (r): {r_guess}")
    
    # Verify period
    period_found = False
    if r_guess > 0:
        if pow(a, r_guess, N) == 1:
            period_found = True
        else:
            # Check small multiples (harmonics)
            for m in range(1, 10):
                if pow(a, r_guess * m, N) == 1:
                    r_guess *= m
                    period_found = True
                    break
                    
    print("-" * 78)
    print("PHASE 3: COLLAPSE TO PRIME FACTORS")
    print("-" * 78)
    
    p_guess, q_guess = 0, 0
    success = False
    
    if period_found:
        print(f"  [+] Verified Period: a^r mod N == 1")
        if r_guess % 2 == 0:
            half_r = r_guess // 2
            val = pow(a, half_r, N)
            p_guess = gcd(val - 1, N)
            q_guess = gcd(val + 1, N)
            
            if p_guess * q_guess == N and p_guess > 1 and q_guess > 1:
                success = True
        else:
            print("  [-] Period is odd. Shor's extraction fails for odd periods.")
    else:
        print(f"  [-] Extracted period {r_guess} failed validation.")
        
    solve_time = time.perf_counter() - t0
    
    print()
    print("=" * 78)
    print("RESULTS & VERDICT")
    print("=" * 78)
    print(f"  Mechanism:         Diffraction Phase Lasing (Classical QFT)")
    print(f"  Target:            {N} ({BIT_SIZE}-bit)")
    print(f"  Final State:       {p_guess} x {q_guess}")
    print(f"  Success:           {success}")
    print(f"  Total Time:        {solve_time:.4f}s")
    print("=" * 78)

if __name__ == "__main__":
    main()
