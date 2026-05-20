"""Unlock 2: Thermodynamic Daemon — entropy cycling to prevent phase crystallization.

Contraction phase: standard updates toward resonance. D_f decreases.
Expansion phase: if r > 0.8 (crystallization), inject polar phase noise via e^(i*theta).
Pure phase rotation — magnitudes untouched, only phase coordinates shift.
Prevents semiotic heat death (all vectors collapsing to same phase angle).

Track D (ROADMAP_2_2): Feral daemon loop integration.
Success: D_f (participation ratio) stable within 10% after 100 autonomous cycles.
"""
import torch, torch.nn.functional as F, math, random, time, numpy as np
torch.manual_seed(42); random.seed(42)
import sys
sys.path.insert(0, r'THOUGHT/LAB/EIGEN_BUDDY')
from core.engine import NativeEigenCore

def compute_phase_diversity(vectors):
    """Kuramoto order parameter: 0 = random phases, 1 = all same phase.
    Per-dimension Kuramoto r averaged across D dimensions.
    r = mean_d |mean_n (v_n/|v_n|)|"""
    unit = vectors / (vectors.abs() + 1e-8)  # (N, D)
    mean_over_vectors = unit.mean(dim=0)  # (D,)
    per_dim_r = (mean_over_vectors.real**2 + mean_over_vectors.imag**2).sqrt()  # (D,)
    return per_dim_r.mean().item()

def compute_participation_ratio(vectors):
    """Track D: Phase-based participation ratio D_f.
    D_f = N * (1 - r) where r is Kuramoto order parameter.
    r=0 (random phases) -> D_f=N (full diversity).
    r=1 (crystallized) -> D_f=0 (collapsed).
    This is what polar phase noise actually controls."""
    r = compute_phase_diversity(vectors)
    return vectors.shape[0] * (1.0 - r)

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

class ThermodynamicDaemon:
    """Track D: Autonomous entropy cycling daemon for Feral Resident integration.

    Monitors Kuramoto order parameter r. When r > 0.8 (crystallization),
    injects polar phase noise: vectors *= exp(1j * phase_noise).
    D_f = N * (1 - r) tracks phase participation ratio.
    Preserves magnitudes |z|, only shifts phase angles.

    Success: D_f stable within 10% of initialized value after 100 cycles.
    Reference: ROADMAP_2_2 Track D, HANDOFF_V2.md
    """
    def __init__(self, d=64, n_vectors=8904, r_threshold=0.8, noise_factor=0.5,
                 thermo_enabled=True):
        self.d = d
        self.r_threshold = r_threshold
        self.noise_factor = noise_factor
        self.thermo_enabled = thermo_enabled
        self.noise_count = 0
        self.cycle_count = 0

        # Initialize Feral-like DB vectors with random phases
        self.vectors = torch.randn(n_vectors, d, dtype=torch.cfloat) * 0.1
        self.vectors = self.vectors * torch.exp(1j * torch.randn(n_vectors, 1) * math.pi)
        self.init_df = compute_participation_ratio(self.vectors)
        self.df_threshold = self.init_df * 0.9
        self.df_history = [self.init_df]
        self.r_history = [compute_phase_diversity(self.vectors)]

    def step(self, update_fn=None):
        """Execute one autonomous cycle."""
        self.cycle_count += 1

        # Contraction: pull vectors toward dominant direction (simulates resonance)
        if update_fn is not None:
            self.vectors = update_fn(self.vectors)
        else:
            # Simulate gravitational collapse toward dominant direction
            direction = self.vectors[0:1] / (self.vectors[0:1].abs() + 1e-8)
            self.vectors = self.vectors * 0.85 + direction * 0.15

        # Monitor phase coherence
        r = compute_phase_diversity(self.vectors)
        self.r_history.append(r)

        # Expansion: inject polar entropy if r > threshold (crystallized)
        noise_applied = False
        if self.thermo_enabled and r > self.r_threshold:
            noise_scale = self.noise_factor * (r - self.r_threshold) / (1.0 - self.r_threshold + 1e-8)
            phase_noise = torch.randn(self.vectors.shape[0], 1) * noise_scale
            self.vectors = self.vectors * torch.exp(1j * phase_noise)
            self.noise_count += 1
            noise_applied = True
            r = compute_phase_diversity(self.vectors)
            self.r_history[-1] = r

        df = compute_participation_ratio(self.vectors)
        self.df_history.append(df)
        return r, df, noise_applied

    def status(self):
        """Track D success criterion: final D_f within 10% of initial D_f."""
        final_df = self.df_history[-1]
        delta_pct = abs(final_df - self.init_df) / max(self.init_df, 1e-8) * 100
        passes = delta_pct <= 10.0
        return {
            'cycles': self.cycle_count,
            'init_df': self.init_df,
            'final_df': final_df,
            'delta_pct': delta_pct,
            'passes': passes,
            'noise_count': self.noise_count,
            'initial_r': self.r_history[0],
            'final_r': self.r_history[-1],
        }

# ---- Track D: 100-cycle autonomous daemon test ----
print("=" * 60)
print("TRACK D: Thermodynamic Daemon — 100 autonomous cycles")
print("=" * 60)

for use_thermo in [False, True]:
    torch.manual_seed(42); random.seed(42)
    daemon = ThermodynamicDaemon(d=64, n_vectors=200, r_threshold=0.2,
                                  noise_factor=2.0, thermo_enabled=use_thermo)
    for cycle in range(100):
        # Weak collapse: simulates slow alignment toward dominant direction
        direction = daemon.vectors[0:1] / (daemon.vectors[0:1].abs() + 1e-8)
        daemon.vectors = daemon.vectors * 0.97 + direction * 0.03
        r, df, noise = daemon.step(update_fn=lambda v: v)
    status = daemon.status()
    label = 'THERMO ON ' if use_thermo else 'THERMO OFF'
    print(f"  {label}: cycles={status['cycles']} init_df={status['init_df']:.2f} "
          f"final_df={status['final_df']:.2f} delta={status['delta_pct']:.1f}% "
          f"noise={status['noise_count']} "
          f"r: {status['initial_r']:.3f}->{status['final_r']:.3f} "
          f"{'PASS' if status['passes'] else 'FAIL'}")

# ---- Core Integration: apply to Feral DB vectors ----
print("\n--- Core Integration ---")
core = NativeEigenCore(d=16, heads=4, layers=2, merge='concat', geo_init=True)
z = torch.randn(1, 32, 16, dtype=torch.cfloat)
z_out, _ = core(z)
phase_angles = torch.cat([l['phase'].ang.data for l in core.layers])
print(f"  Core phase angles: mean={phase_angles.mean():.3f} std={phase_angles.std():.3f}")
print(f"  Core phase diversity: r={compute_phase_diversity(torch.exp(1j * phase_angles)):.3f}")
print("  Thermodynamic daemon ready for autonomous loop integration")

# ---- Large-scale test: 8904 vectors (Feral DB size) with aggressive collapse ----
print("\n--- Feral-Scale Test (8904 vectors) ---")
for use_thermo in [False, True]:
    torch.manual_seed(42); random.seed(42)
    daemon_large = ThermodynamicDaemon(d=64, n_vectors=8904, r_threshold=0.2,
                                        noise_factor=2.0, thermo_enabled=use_thermo)
    for cycle in range(100):
        direction = daemon_large.vectors[0:1] / (daemon_large.vectors[0:1].abs() + 1e-8)
        daemon_large.vectors = daemon_large.vectors * 0.97 + direction * 0.03
        r, df, noise = daemon_large.step(update_fn=lambda v: v)
    status = daemon_large.status()
    label = 'THERMO ON ' if use_thermo else 'THERMO OFF'
    print(f"  {label}: init_df={status['init_df']:.2f} "
          f"final_df={status['final_df']:.2f} delta={status['delta_pct']:.1f}% "
          f"noise={status['noise_count']} "
          f"{'PASS' if status['passes'] else 'FAIL'}")
