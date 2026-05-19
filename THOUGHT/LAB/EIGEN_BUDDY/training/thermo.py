"""Unlock 2: Thermodynamic Daemon — entropy cycling to prevent phase crystallization.

Contraction phase: standard updates toward resonance. D_f decreases.
Expansion phase: if D_f < target * 0.9, inject polar phase noise via e^(i*theta).
Pure phase rotation — magnitudes untouched, only phase coordinates shift.
Prevents semiotic heat death (all vectors collapsing to same phase angle).
"""
import torch, torch.nn.functional as F, math, random, time, numpy as np
torch.manual_seed(42); random.seed(42)
import sys
sys.path.insert(0, r'THOUGHT/LAB/EIGEN_BUDDY')
from core.engine import NativeEigenCore

def compute_phase_diversity(vectors):
    """Kuramoto order parameter: 0 = random phases, 1 = all same phase.
    Low r = high diversity. r -> 1 = crystallized (heat death)."""
    # Extract unit complex direction (pure phase)
    unit = vectors / (vectors.abs() + 1e-8)
    cos_mean = unit.real.mean().float()
    sin_mean = unit.imag.mean().float()
    r = (cos_mean**2 + sin_mean**2).sqrt()
    return r.item()

def thermodynamic_cycle(vectors, learning_updates, target_df, cycle_count):
    """Execute one contraction/expansion cycle.
    vectors: (N, D) complex tensor
    learning_updates: (N, D) complex tensor (gradient or geometric update)
    target_df: float, desired participation ratio
    Returns: updated vectors, current Df, noise_applied (bool)
    """
    # Contraction: apply learning updates
    vectors = vectors + 0.01 * learning_updates

    # Measure structural diversity
    current_df = compute_participation_ratio(vectors)

    # Expansion: inject polar entropy if diversity drops below 90% of target
    noise_applied = False
    if current_df < target_df * 0.9:
        noise_scale = (target_df - current_df) / target_df
        # Pure phase noise: rotate each vector by random angle
        # Preserves |z| = magnitude, only shifts phase
        phase_noise = torch.randn(vectors.shape[0], 1) * noise_scale * 0.1
        rotation = torch.exp(1j * phase_noise)  # e^(i*theta), polar rotation
        vectors = vectors * rotation
        noise_applied = True

    return vectors, current_df, noise_applied

# ---- Test: simulate 50 cycles with and without thermodynamic cycling ----
print("=" * 60)
print("UNLOCK 2: Thermodynamic Daemon — entropy cycling")
print("=" * 60)

D = 64
N = 200  # simulated Feral DB entries
target_r = 0.3  # target Kuramoto order: moderate phase alignment

for use_thermo in [False, True]:
    torch.manual_seed(42)
    vectors = torch.randn(N, D, dtype=torch.cfloat) * 0.1
    vectors = vectors * torch.exp(1j * torch.randn(N, 1) * math.pi)

    init_r = compute_phase_diversity(vectors)
    r_history = [init_r]
    noise_count = 0

    for cycle in range(50):
        # Contraction: pull toward single direction (collapse simulation)
        direction = vectors[0:1] / (vectors[0:1].abs() + 1e-8)
        vectors = vectors * 0.85 + direction * 0.15

        current_r = compute_phase_diversity(vectors)
        noise_applied = False
        if use_thermo and current_r > 0.8:  # too coherent -> inject entropy
            phase_noise = torch.randn(vectors.shape[0], 1) * 0.5
            vectors = vectors * torch.exp(1j * phase_noise)
            noise_applied = True

        r_history.append(current_r)
        if noise_applied: noise_count += 1

    final_r = r_history[-1]
    print(f"  thermo={'ON' if use_thermo else 'OFF'}: init_r={init_r:.3f} final_r={final_r:.3f} noise={noise_count}")
    print(f"    {'DIVERSE — no heat death' if final_r < 0.5 else 'CRYSTALLIZED — heat death'}")

# ---- Integration with Core: apply to Feral DB vectors ----
print("\n--- Core Integration ---")
core = NativeEigenCore(d=16, heads=4, layers=2, merge='concat', geo_init=True)
z = torch.randn(1, 32, 16, dtype=torch.cfloat)
z_out, _ = core(z)
# Phase accumulator angles are the phase state
phase_angles = torch.cat([l['phase'].ang.data for l in core.layers])
print(f"  Core phase angles: mean={phase_angles.mean():.3f} std={phase_angles.std():.3f}")
print(f"  Core phase diversity: r={compute_phase_diversity(torch.exp(1j * phase_angles)):.3f}")
print("  Thermodynamic daemon ready for autonomous loop integration")
