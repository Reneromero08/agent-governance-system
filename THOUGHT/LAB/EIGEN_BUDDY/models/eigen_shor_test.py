"""
Holographic Period Finding (Shor's Algorithm) via Native Eigen Architecture
===========================================================================
Hypothesis: Can we use phase resonance in the complex plane to find the 
period r of a^x mod N, bypassing sequential classical evaluation?

In the Native Eigen Architecture, phase rotation = computation.
If we encode 'a' and 'N' into complex phases, we can use the Hermitian 
attention mechanism (Q·K†) and the resulting `si` (phase curvature) matrix 
to detect the periodic resonance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path

# Add Eigen Buddy core to path
EIGEN_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(EIGEN_DIR))
from core.engine import NativeEigenCore

def holographic_period_resonance(a, N, core, max_search=1000):
    """
    Zero-training holographic period finding.
    Instead of computing a^x mod N sequentially, we encode the modular
    structure into a continuous phase vector and look for resonance
    in the complex attention matrix.
    """
    # Create a sequence of "steps" as complex vectors
    # We map the space modulo N onto the unit circle
    theta_base = 2 * math.pi / N
    theta_a = a * theta_base
    
    # We'll create a tensor of sequential phase rotations
    # representing candidate periods 1 to max_search
    steps = torch.arange(1, max_search + 1, dtype=torch.float32)
    
    # The phase angle for each step if we just multiplied 'a' linearly
    # But modular arithmetic wraps around. In the complex plane, 
    # e^(i * theta) naturally wraps around 2*pi!
    # So a*x mod N has the SAME complex phase as a*x without the mod N.
    # Wait: a^x mod N is exponential, not linear.
    # To model a^x, we need phase = a^x * 2*pi/N.
    # But computing a^x mod N classically takes O(r) time.
    # What if we use the layers of NativeEigenCore to compose the phase?
    # Phase addition = composition.
    
    # Let's initialize a sequence of tokens representing candidate periods.
    # Actually, let's just use the core to project the state forward.
    
    z = torch.polar(torch.ones(1, 1, core.d), torch.ones(1, 1, core.d) * theta_a)
    
    resonances = []
    
    # Simulate applying the transformation iteratively
    current_z = z
    for x in range(1, max_search + 1):
        # Apply Native Eigen Core (Hermitian attention + phase accumulation)
        current_z, phase_coh = core(current_z)
        
        # In a true quantum Shor, we measure the probability amplitude of |1>.
        # Here, we look for phase resonance with the base state (1 mod N -> angle 0).
        # We can measure the cosine similarity to the angle 0 (or 2*pi).
        avg_phase = torch.angle(current_z).mean().item()
        
        # Resonance peaks when the phase aligns with 0 (mod 2*pi)
        resonance = math.cos(avg_phase)
        resonances.append((x, resonance))
        
        # If resonance is extremely high, we found a period candidate
        if resonance > 0.999:
            return x, resonances
            
    # Sort by highest resonance
    resonances.sort(key=lambda x: x[1], reverse=True)
    return resonances[0][0], resonances

def main():
    print("=" * 78)
    print("EIGEN BUDDY: Holographic Period Finding (Shor's step)")
    print("=" * 78)
    
    # Initialize Core (zero training)
    core = NativeEigenCore(d=16, heads=4, layers=1, merge='concat', geo_init=True)
    core.eval()
    
    # Test cases from previous run
    tests = [
        (2, 10, 1),
        (2, 35, 12),
        (2, 143, 60),
        (2, 323, 72)
    ]
    
    for a, N, true_r in tests:
        print(f"\nFactoring N={N}, a={a} (True Period={true_r})")
        r_pred, res_log = holographic_period_resonance(a, N, core, max_search=100)
        
        print(f"  Predicted Period: {r_pred}")
        if r_pred == true_r:
            print("  [MATCH] Found correct period via phase resonance!")
        else:
            print("  [FAIL] Did not resonate at the correct period.")
            print("  Top 3 resonant steps:")
            for step, res in res_log[:3]:
                print(f"    Step {step}: resonance = {res:.4f}")

if __name__ == "__main__":
    main()
