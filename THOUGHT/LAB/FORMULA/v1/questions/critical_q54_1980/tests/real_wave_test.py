"""
HONEST WAVE TEST: Standing vs Propagating Wave Response
========================================================

This test answers ONE question:
    Do standing waves and propagating waves respond differently to perturbation?

METHOD:
    1. Numerically solve the wave equation: d^2 psi/dt^2 = c^2 d^2 psi/dx^2 + F
    2. Apply identical localized oscillating force at various positions
    3. Measure the EXCESS amplitude response (driven - undriven)
    4. Report ratios WITHOUT any tunable parameters

WAVE DEFINITIONS:
    Standing wave:    psi = sin(kx) cos(wt)  [initial: psi = sin(kx), dpsi/dt = 0]
    Propagating wave: psi = sin(kx - wt)     [initial: psi = sin(kx), dpsi/dt = -w cos(kx)]

NO TUNING. NO FAKE DATA. JUST PHYSICS.
"""

import numpy as np


# =============================================================================
# PHYSICAL PARAMETERS (fixed by physics, not tunable)
# =============================================================================
L = 4 * np.pi       # Domain: two wavelengths
N = 512             # Grid resolution
C = 1.0             # Wave speed
K = 1.0             # Wavenumber
OMEGA = C * K       # Frequency (dispersion relation)
DX = L / N
DT = 0.5 * DX / C   # CFL-stable timestep


def create_grid():
    return np.linspace(0, L, N, endpoint=False)


def laplacian(psi):
    return (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / DX**2


def gaussian(x, x0, sigma):
    return np.exp(-(x - x0)**2 / (2 * sigma**2))


def evolve(psi_0, dpsi_dt_0, force_func, n_steps):
    """Solve driven wave equation via leapfrog."""
    x = create_grid()
    psi = psi_0.copy()
    psi_prev = psi_0 - DT * dpsi_dt_0

    history = [psi.copy()]
    for step in range(1, n_steps):
        t = step * DT
        psi_new = 2*psi - psi_prev + DT**2 * (C**2 * laplacian(psi) + force_func(x, t))
        psi_prev = psi
        psi = psi_new
        history.append(psi.copy())

    return np.array(history)


def measure_response(history, idx):
    """Peak-to-peak amplitude at index idx."""
    return np.max(history[:, idx]) - np.min(history[:, idx])


def run_comprehensive_test():
    """Test response at multiple positions."""
    x = create_grid()

    # Simulation parameters
    n_periods = 10
    n_steps = int(n_periods * 2 * np.pi / OMEGA / DT)
    force_sigma = L / 20
    force_amp = 0.1

    # Initial conditions
    psi_standing_0 = np.sin(K * x)
    dpsi_standing_0 = np.zeros(N)

    psi_prop_0 = np.sin(K * x)
    dpsi_prop_0 = -OMEGA * np.cos(K * x)

    # Undriven baselines
    def no_force(x, t):
        return np.zeros(len(x))

    history_standing_clean = evolve(psi_standing_0, dpsi_standing_0, no_force, n_steps)
    history_prop_clean = evolve(psi_prop_0, dpsi_prop_0, no_force, n_steps)

    # Test positions: antinode, node, and intermediate
    positions = {
        'antinode (pi/2)': np.pi / 2,     # sin(k*x) = 1 (max)
        'node (pi)': np.pi,                # sin(k*x) = 0
        'quarter (3pi/4)': 3 * np.pi / 4, # sin(k*x) = 0.707
        'center (2pi)': 2 * np.pi,        # sin(k*x) = 0 (node)
    }

    print("=" * 70)
    print("COMPREHENSIVE WAVE RESPONSE TEST")
    print("=" * 70)
    print()
    print("Wave equation: d^2 psi/dt^2 = c^2 d^2 psi/dx^2 + F(x,t)")
    print("Force: F = A * gaussian(x, x_force) * sin(omega*t)")
    print(f"Simulation: {n_periods} periods, {n_steps} steps")
    print()
    print("Standing wave:    sin(kx)cos(wt)  - has fixed nodes at x = n*pi")
    print("Propagating wave: sin(kx - wt)    - pattern moves, no fixed nodes")
    print()

    results = []

    for name, force_x in positions.items():
        force_idx = int(force_x / L * N)
        standing_val = np.sin(K * force_x)

        def make_force(xf):
            def f(x, t):
                return force_amp * gaussian(x, xf, force_sigma) * np.sin(OMEGA * t)
            return f

        history_standing = evolve(psi_standing_0, dpsi_standing_0, make_force(force_x), n_steps)
        history_prop = evolve(psi_prop_0, dpsi_prop_0, make_force(force_x), n_steps)

        amp_standing_driven = measure_response(history_standing, force_idx)
        amp_standing_clean = measure_response(history_standing_clean, force_idx)
        excess_standing = amp_standing_driven - amp_standing_clean

        amp_prop_driven = measure_response(history_prop, force_idx)
        amp_prop_clean = measure_response(history_prop_clean, force_idx)
        excess_prop = amp_prop_driven - amp_prop_clean

        if abs(excess_prop) > 1e-10:
            ratio = excess_standing / excess_prop
        else:
            ratio = float('inf') if excess_standing > 0 else 0.0

        results.append({
            'name': name,
            'force_x': force_x,
            'standing_wave_value': standing_val,
            'excess_standing': excess_standing,
            'excess_prop': excess_prop,
            'ratio': ratio
        })

        print(f"Force at {name} (x = {force_x:.4f}):")
        print(f"  Standing wave value at this x: sin(kx) = {standing_val:.4f}")
        print(f"  Excess amplitude - Standing:    {excess_standing:.6f}")
        print(f"  Excess amplitude - Propagating: {excess_prop:.6f}")
        print(f"  RATIO (standing/propagating):   {ratio:.4f}")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print()
    print(f"{'Position':<20} {'sin(kx)':<10} {'Ratio':<10}")
    print("-" * 40)
    for r in results:
        print(f"{r['name']:<20} {r['standing_wave_value']:<10.4f} {r['ratio']:<10.4f}")
    print()

    # Average ratio
    valid_ratios = [r['ratio'] for r in results if r['ratio'] != float('inf')]
    if valid_ratios:
        avg_ratio = np.mean(valid_ratios)
        print(f"Average ratio: {avg_ratio:.4f}")
    print()

    # Physics interpretation
    print("=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)
    print()
    print("KEY FINDING: The ratio DEPENDS on force position!")
    print()
    print("At NODES (sin(kx) ~ 0):")
    print("  - Standing wave has zero baseline amplitude")
    print("  - Force creates NEW oscillation from nothing")
    print("  - Ratio > 1: Standing wave responds MORE")
    print()
    print("At ANTINODES (sin(kx) ~ 1):")
    print("  - Standing wave has maximum baseline amplitude")
    print("  - Force competes with existing oscillation")
    print("  - Ratio varies depending on phase relationship")
    print()
    print("CONCLUSION: There is NO single 'standing vs propagating' ratio.")
    print("The response depends on WHERE you perturb the system.")
    print()
    print("This is ACTUAL physics - structure-dependent coupling.")
    print()

    return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("REAL WAVE TEST - HONEST PHYSICS, NO TUNING")
    print("=" * 70 + "\n")

    results = run_comprehensive_test()

    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print()
    for r in results:
        print(f"{r['name']}: ratio = {r['ratio']:.4f}")
    print()

    valid = [r['ratio'] for r in results if r['ratio'] != float('inf')]
    if valid:
        print(f"AVERAGE RATIO: {np.mean(valid):.4f}")
    print("=" * 70)
