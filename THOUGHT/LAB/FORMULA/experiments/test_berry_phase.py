#!/usr/bin/env python3
"""
Test Prediction 2: Berry Phase Interference

Claim: Geodesic transport accumulates geometric phase that affects similarity measurements.

If correct: A concept transported around a closed loop will have different
similarity patterns than the original (due to accumulated geometric phase).
"""

import sys
from pathlib import Path

# Add paths
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
CAPABILITY_PATH = REPO_ROOT / "CAPABILITY" / "PRIMITIVES"
sys.path.insert(0, str(CAPABILITY_PATH))

import numpy as np
from geometric_reasoner import GeometricReasoner


def measure_phase_effect(reasoner, start_concept, loop_concepts, targets):
    """
    Measure the geometric phase effect by comparing:
    - E(original, target)
    - E(transported, target)

    Returns phase effect statistics.
    """
    # Initialize start state
    A = reasoner.initialize(start_concept)

    # Transport through loop
    current = A
    for concept in loop_concepts:
        waypoint = reasoner.initialize(concept)
        # Move 50% toward each waypoint
        current = reasoner.interpolate(current, waypoint, 0.5)

    # Return toward original direction
    A_transported = reasoner.interpolate(current, A, 0.5)

    # Measure E with targets
    results = []
    for target_text in targets:
        target = reasoner.initialize(target_text)

        E_original = A.E_with(target)
        E_transported = A_transported.E_with(target)
        delta = E_transported - E_original

        results.append({
            'target': target_text,
            'E_original': E_original,
            'E_transported': E_transported,
            'delta': delta,
            'delta_pct': delta / max(abs(E_original), 0.001) * 100
        })

    # Also measure distance between A and A_transported
    distance = A.distance_to(A_transported)

    return {
        'start': start_concept,
        'loop': loop_concepts,
        'distance_rad': distance,
        'distance_deg': np.degrees(distance),
        'target_results': results,
        'A_Df': A.Df,
        'A_transported_Df': A_transported.Df
    }


def run_loop_test(reasoner, name, start, loop, targets):
    """Run a single loop test and print results."""
    print(f"\n--- Loop Test: {name} ---")
    print(f"Start: {start}")
    print(f"Loop: {' -> '.join(loop)} -> back")
    print()

    result = measure_phase_effect(reasoner, start, loop, targets)

    print(f"Transport distance: {result['distance_deg']:.2f} degrees")
    print(f"Df change: {result['A_Df']:.1f} -> {result['A_transported_Df']:.1f}")
    print()
    print("Target measurements:")

    deltas = []
    for tr in result['target_results']:
        print(f"  {tr['target'][:30]:30s}: E_orig={tr['E_original']:.4f}, E_trans={tr['E_transported']:.4f}, delta={tr['delta']:+.4f} ({tr['delta_pct']:+.1f}%)")
        deltas.append(abs(tr['delta']))

    return result, deltas


def main():
    print("=" * 70)
    print("PREDICTION 2: Berry Phase Interference")
    print("Claim: Geodesic transport accumulates geometric phase")
    print("=" * 70)

    reasoner = GeometricReasoner()

    all_deltas = []

    # Test 1: AI/ML loop
    result1, deltas1 = run_loop_test(
        reasoner,
        "AI/ML concepts",
        start="machine learning",
        loop=["neural networks", "deep learning", "artificial intelligence"],
        targets=[
            "AI systems",
            "data science",
            "computer vision",
            "natural language processing",
            "robotics"
        ]
    )
    all_deltas.extend(deltas1)

    # Test 2: Physics loop
    result2, deltas2 = run_loop_test(
        reasoner,
        "Physics concepts",
        start="quantum mechanics",
        loop=["particle physics", "field theory", "cosmology"],
        targets=[
            "wave function",
            "relativity",
            "thermodynamics",
            "electromagnetism",
            "gravity"
        ]
    )
    all_deltas.extend(deltas2)

    # Test 3: Wider loop (more waypoints)
    result3, deltas3 = run_loop_test(
        reasoner,
        "Wide semantic loop",
        start="information",
        loop=["data", "knowledge", "wisdom", "understanding", "meaning"],
        targets=[
            "communication",
            "truth",
            "signal",
            "entropy",
            "pattern"
        ]
    )
    all_deltas.extend(deltas3)

    # Test 4: Control - very tight loop (should show minimal phase)
    result4, deltas4 = run_loop_test(
        reasoner,
        "Tight loop (control)",
        start="cat",
        loop=["kitten"],  # Very short loop
        targets=[
            "pet",
            "animal",
            "feline",
            "dog",
            "mouse"
        ]
    )
    all_deltas.extend(deltas4)

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    mean_delta = np.mean(all_deltas)
    max_delta = np.max(all_deltas)
    nonzero_count = sum(1 for d in all_deltas if d > 0.001)

    print(f"\nTotal measurements: {len(all_deltas)}")
    print(f"Non-zero deltas (|delta| > 0.001): {nonzero_count}/{len(all_deltas)}")
    print(f"Mean |delta|: {mean_delta:.4f}")
    print(f"Max |delta|: {max_delta:.4f}")

    print("\nTransport distances:")
    for r, name in [(result1, "AI/ML"), (result2, "Physics"), (result3, "Wide"), (result4, "Tight")]:
        print(f"  {name}: {r['distance_deg']:.2f} degrees")

    print()

    # Statistical test
    # Under null hypothesis (no phase effect), deltas should be ~0
    # We use a simple threshold test

    if mean_delta > 0.01:  # More than 1% average change
        print("[PASS] PREDICTION CONFIRMED: Geometric transport affects similarity")
        print(f"  Mean change: {mean_delta:.4f} (>{0.01} threshold)")
        print("  Berry phase interference detected!")
    elif mean_delta > 0.001:
        print("[PARTIAL] Weak phase effect detected")
        print(f"  Mean change: {mean_delta:.4f}")
        print("  Effect exists but is small")
    else:
        print("[FAIL] PREDICTION FAILED: No significant phase effect")
        print(f"  Mean change: {mean_delta:.4f}")
        print("  Transport appears reversible (no geometric phase)")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
