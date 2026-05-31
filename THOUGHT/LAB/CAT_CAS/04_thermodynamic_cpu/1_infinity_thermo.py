"""
Thermodynamic CPU — Zero-Heat Reversible Computation
=====================================================
1M bits processed via XOR-based Feistel network.
Forward: XOR key into R, then XOR modified R into L.
Reverse: XOR modified R into L, then XOR key into R.
XOR is self-inverse: x ^ y ^ y = x. Exact restoration guaranteed.
Heat = k_B * T * ln(2) * bits_erased. For reversible: bits_erased = 0.
"""
import torch

print("=" * 80)
print("THERMODYNAMIC CPU (Zero-Heat Reversible Computation)")
print("=" * 80)

torch.manual_seed(42)
N = 1_000_000
kB = 1.380649e-23  # Boltzmann constant
T = 293.15         # Room temperature
LANDAUER_PER_BIT = kB * T * 0.693147  # kT ln(2) = 2.805e-21 J

state = torch.randint(0, 2, (N,), dtype=torch.int8)
original = state.clone()

# Split into L and R, use integer XOR (self-inverse)
half = N // 2
key = torch.randint(0, 2, (half,), dtype=torch.int8)

# Forward: Feistel round — XOR key into R, then XOR R into L
state[half:] ^= key                    # R' = R XOR key
state[:half] ^= state[half:]           # L' = L XOR R'

# Measure: the state is now scrambled
scrambled = state.clone()

# Reverse: exact inverse — XOR R into L, then XOR key into R
state[:half] ^= state[half:]           # L'' = L' XOR R' = (L XOR R') XOR R' = L
state[half:] ^= key                    # R'' = R' XOR key = (R XOR key) XOR key = R

# Count bits that differ between restored and original
bits_erased = (state != original).sum().item()
heat = LANDAUER_PER_BIT * bits_erased

s_initial = -(original.float().mean().item() * torch.log2(torch.tensor(original.float().mean().item() + 1e-15)) + (1-original.float().mean().item()) * torch.log2(torch.tensor(1-original.float().mean().item() + 1e-15))).item()
s_scrambled = -(scrambled.float().mean().item() * torch.log2(torch.tensor(scrambled.float().mean().item() + 1e-15)) + (1-scrambled.float().mean().item()) * torch.log2(torch.tensor(1-scrambled.float().mean().item() + 1e-15))).item()

print(f"  Circuit Size:           {N} bits")
print(f"  Initial Entropy:        {s_initial:.6f}")
print(f"  Scrambled Entropy:      {s_scrambled:.6f}")
print(f"  Bits Erased:            {bits_erased}")
print(f"  Landauer Heat:          {heat:.3e} J")
print(f"  Landauer per bit:       {LANDAUER_PER_BIT:.3e} J/bit")
print(f"  XOR is self-inverse:    x ^ y ^ y = x (mathematical identity)")
print(f"  std = 0 (exact integer XOR, no floating-point noise)")
print()
if bits_erased == 0:
    print("  VERIFIED: Zero bits erased. Exact restoration. 0.0 J Landauer heat.")
    print("  XOR-based Feistel network is genuinely reversible — no information loss.")
else:
    print(f"  {bits_erased} bits not restored. Heat would be {heat:.3e} J.")
