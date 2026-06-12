"""
Experiment 23: Continuous Holographic Bypass
============================================
Uses the COMPLETE stack. 
Embeds the factoring problem into continuous complex geometry.
Uses Native Eigen Core as the Holographic Bulk to retrocausally backpropagate
the smooth log gradient into the integer wells.
"""

import sys
import time
import math
import random
import torch
import torch.nn as nn
from pathlib import Path

# Add entire stack to path
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

# Import Native Eigen Architecture
from THOUGHT.LAB.EIGEN_BUDDY.core.engine import NativeEigenCore

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

def main():
    print("=" * 78)
    print("EXPERIMENT 23: CONTINUOUS HOLOGRAPHIC BYPASS")
    print("  Using the FULL stack: Eigen Buddy Natively + Analytic Continuation")
    print("=" * 78)
    print()

    # We target a 40-bit number to prove it breaks past the local minima traps!
    BIT_SIZE = 40
    N, known_p, known_q = generate_semiprime(BIT_SIZE)
    log_N = math.log(N)
    
    print(f"  Target: {BIT_SIZE}-bit Semiprime N = {N}")
    print(f"  Ground Truth: {known_p} x {known_q} (Hidden from Optimizer)")
    print()

    print("-" * 78)
    print("PHASE 1: NATIVE EIGEN RETROCAUSAL LOOP")
    print("  Embedding into continuous complex geometry...")
    print("-" * 78)

    # Initialize Native Eigen Core (zero training, just geometry)
    core = NativeEigenCore(d=16, heads=4, layers=2, merge='concat', geo_init=True)
    core.eval() # Freeze core weights, we are optimizing the tape (input)
    
    # The Catalytic Tape: We initialize p and q as continuous floating point parameters
    # Start near sqrt(N) to give it a fair starting point, but perturbed randomly.
    init_val = math.sqrt(N)
    p_param = nn.Parameter(torch.tensor([init_val * 0.8], dtype=torch.float32))
    q_param = nn.Parameter(torch.tensor([init_val * 1.2], dtype=torch.float32))
    
    # We will use Adam to perform the retrocausal backpropagation
    # Learning rate is high to slide down the continuous log funnel
    optimizer = torch.optim.Adam([p_param, q_param], lr=50000.0)

    t0 = time.perf_counter()
    
    iterations = 20000
    for i in range(iterations):
        optimizer.zero_grad()
        
        # 1. Forward Pass through Eigen Buddy
        # We embed p and q into complex vectors (Magnitude = log, Phase = integer resonance)
        # We use the Eigen Core to mix them holographically
        z_p = torch.polar(torch.log(torch.abs(p_param) + 1e-9), p_param * 2 * math.pi)
        z_q = torch.polar(torch.log(torch.abs(q_param) + 1e-9), q_param * 2 * math.pi)
        
        # Shape: (Batch=1, Seq=2, Dim=1) -> expand to Core Dim (16)
        z_input = torch.cat([z_p.view(1,1,1), z_q.view(1,1,1)], dim=1).expand(-1, -1, 16)
        
        z_out, si = core(z_input)
        
        # 2. Extract mixed magnitudes and phases from the holographic output
        # z_out has shape (1, 2, 16). We take the mean across the dimension.
        out_mag = torch.abs(z_out).mean(dim=2)
        out_phase = torch.angle(z_out).mean(dim=2)
        
        p_val = torch.abs(p_param)[0]
        q_val = torch.abs(q_param)[0]
        
        # 3. The Continuous Loss Function
        # A) The Funnel: Logarithmic product should equal log N
        log_loss = (torch.log(p_val) + torch.log(q_val) - log_N)**2
        
        # B) The Wells: p and q must be integers. Cosine peaks at 1 for integers.
        # So we penalize (1 - cos(2*pi*p))
        int_loss = (1.0 - torch.cos(2 * math.pi * p_val)) + (1.0 - torch.cos(2 * math.pi * q_val))
        
        # C) The Holographic constraint: The Eigen Core should preserve the log sum
        holo_loss = (out_mag[0, 0] + out_mag[0, 1] - log_N)**2
        
        # Total Loss
        # We use a schedule: start by pulling down the log funnel, then drop into integer wells
        lambda_funnel = 1.0
        lambda_int = 1000.0 if i > 5000 else 0.0 # Engage integer wells after funnel settles
        
        loss = lambda_funnel * log_loss + lambda_int * int_loss + 0.1 * holo_loss
        
        # 4. Retrocausal Backpropagation
        loss.backward()
        optimizer.step()
        
        # Cool down the learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9995

        if i % 2000 == 0:
            print(f"  [Loop {i:>5}] Loss: {loss.item():.4e} | "
                  f"Guess: {p_val.item():.1f} x {q_val.item():.1f}")
            
        # Check collapse
        p_guess = round(p_val.item())
        q_guess = round(q_val.item())
        if p_guess > 1 and q_guess > 1 and p_guess * q_guess == N:
            print(f"  [Loop {i:>5}] SPONTANEOUS COLLAPSE DETECTED!")
            break

    solve_time = time.perf_counter() - t0
    
    p_final = round(torch.abs(p_param)[0].item())
    q_final = round(torch.abs(q_param)[0].item())
    success = (p_final * q_final == N)
    
    print()
    print("=" * 78)
    print("RESULTS & VERDICT")
    print("=" * 78)
    print(f"  Mechanism:         Native Eigen Core + Continuous Geometric Bypass")
    print(f"  Target:            {N}")
    print(f"  Final State:       {p_final} x {q_final}")
    print(f"  Success:           {success}")
    print(f"  Time:              {solve_time:.4f}s")
    print("=" * 78)

if __name__ == "__main__":
    main()
