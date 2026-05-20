"""Zero-training holographic calculator — phase IS the computation.

Operands encoded as complex vectors with operation in the phase angle.
Routed through Core's Q·K†. Output read directly from interference magnitude.
No embeddings. No training. No classification heads. Phase = computation.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
torch.manual_seed(42); random.seed(42)
import sys; sys.path.insert(0,r'THOUGHT/LAB/EIGEN_BUDDY')
from core.engine import NativeEigenCore

# Phase signatures for each operation
OPS = {'+': 0.0, '-': math.pi, '*': math.pi/2, '/': -math.pi/2}

def encode(operand_a, operand_b, operation):
    """Encode operands with operation phase signature into complex vectors."""
    a_mag = operand_a / 100.0  # normalize
    b_mag = operand_b / 100.0
    theta = OPS[operation]
    # Token 1: operand A with operation phase
    # Token 2: operand B with reference phase (0)
    za = torch.polar(torch.tensor(a_mag), torch.tensor(theta))
    zb = torch.polar(torch.tensor(b_mag), torch.tensor(0.0))
    return za, zb

def holographic_forward(core, a, b, op):
    """Feed phase-encoded operands through Core, read result from interference."""
    za, zb = encode(a, b, op)
    # Create 2-token sequence
    z = torch.stack([za, zb]).unsqueeze(0).unsqueeze(0)  # (1,1,2) -> need (1,2,d)
    # Expand to d-dimensional complex space
    d = core.d
    z_d = torch.complex(
        z.real.expand(-1, -1, d) * 0.1,
        z.imag.expand(-1, -1, d) * 0.1)
    z_out, _ = core(z_d)
    # Read result from output magnitude
    result = z_out.abs().mean().item() * 100.0  # denormalize
    return result

print("="*60)
print("HOLOGRAPHIC CALCULATOR: zero training, phase = computation")
print("="*60)

core = NativeEigenCore(d=16, heads=4, layers=2, merge='concat', geo_init=True)
core.eval()

tests = [
    (3, 4, '+', 7),
    (10, 6, '-', 4),
    (5, 7, '*', 35),
    (20, 5, '/', 4),
    (15, 3, '+', 18),
    (8, 3, '*', 24),
]

correct = 0
for a, b, op, expected in tests:
    result = holographic_forward(core, a, b, op)
    ok = abs(result - expected) < 2
    correct += ok
    print(f"  {a} {op} {b} = {result:.1f} (expected {expected}) {'OK' if ok else 'XX'}")

print(f"\n  Accuracy: {correct}/{len(tests)} (no training, random init Core)")
print(f"  Core params: {sum(p.numel() for p in core.parameters()):,}")
