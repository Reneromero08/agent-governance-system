#!/usr/bin/env python3
"""
Test Prediction 1: Rate-Dependent Threshold

Claim: Fast processing invalidates the equilibrium assumption.
At high processing rates, the gate should show WORSE discrimination
because the mind state hasn't stabilized.

Simplified test: Measure how gate discrimination degrades when
we process chunks without letting the mind "settle" between absorptions.
"""

import sys
from pathlib import Path
import time

# Add paths
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
CAPABILITY_PATH = REPO_ROOT / "CAPABILITY" / "PRIMITIVES"
FERAL_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "FERAL_RESIDENT"
sys.path.insert(0, str(CAPABILITY_PATH))
sys.path.insert(0, str(FERAL_PATH))

import numpy as np
from geometric_reasoner import GeometricReasoner, GeometricState

# The invariant threshold
CRITICAL_RESONANCE = 1.0 / (2.0 * np.pi)


def get_dynamic_threshold(n_memories: int) -> float:
    """Q46 nucleation threshold"""
    grad_S = 1.0 / np.sqrt(max(n_memories, 1))
    return CRITICAL_RESONANCE / (1.0 + grad_S)


def load_paper_chunks(max_papers=10, chunks_per_paper=5):
    """Load real paper chunks for testing."""
    papers_dir = FERAL_PATH / "research" / "papers" / "markdown"
    chunks = []

    for i, paper_file in enumerate(papers_dir.glob("*.md")):
        if i >= max_papers:
            break

        try:
            content = paper_file.read_text(encoding='utf-8')
            # Split into chunks (simple paragraph split)
            paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 100]

            for j, para in enumerate(paragraphs[:chunks_per_paper]):
                chunks.append({
                    'paper_id': paper_file.stem,
                    'chunk_id': j,
                    'content': para[:500]  # Limit length
                })
        except Exception as e:
            continue

    return chunks


def simulate_processing(reasoner, chunks, noise_scale=0.0):
    """
    Simulate chunk processing with variable noise (modeling fast processing instability).

    noise_scale: Amount of random noise added to mind state after absorption
                 (0 = slow processing, stable mind)
                 (Higher = fast processing, turbulent mind)

    The idea: Fast processing doesn't let the mind "settle", so we model this
    as noise/perturbation that hasn't been smoothed out yet.

    Returns gate decisions and E values.
    """
    mind_state = None
    n_memories = 0
    results = []

    for chunk in chunks:
        chunk_state = reasoner.initialize(chunk['content'])

        # Compute E with current mind
        if mind_state is not None:
            E = chunk_state.E_with(mind_state)
        else:
            E = 0.5  # Neutral for first chunk

        # Gate decision
        threshold = get_dynamic_threshold(n_memories)
        gate_open = E > threshold

        # Record result
        results.append({
            'E': E,
            'threshold': threshold,
            'gate_open': gate_open,
            'n_memories': n_memories
        })

        # Absorb if gate open
        if gate_open:
            if mind_state is None:
                mind_state = chunk_state
            else:
                # Use 1/N weighting (Law 1: Inertia)
                n = n_memories + 1
                t = 1.0 / (n + 1)
                mind_state = reasoner.interpolate(mind_state, chunk_state, t)

            n_memories += 1

            # Add noise to simulate fast processing instability
            # Fast processing = high noise (mind hasn't settled)
            # Slow processing = low noise (mind is stable)
            if noise_scale > 0 and mind_state is not None:
                noise = np.random.randn(len(mind_state.vector)) * noise_scale
                perturbed = mind_state.vector + noise
                # Re-normalize to stay on manifold
                perturbed = perturbed / np.linalg.norm(perturbed)
                mind_state = GeometricState(
                    vector=perturbed.astype(np.float32),
                    operation_history=mind_state.operation_history
                )

    return results


def compute_discrimination(results):
    """
    Compute Cohen's d between absorbed and rejected E values.

    Higher d = better discrimination (gate is working well)
    Lower d = worse discrimination (gate is confused)
    """
    absorbed_E = [r['E'] for r in results if r['gate_open']]
    rejected_E = [r['E'] for r in results if not r['gate_open']]

    if len(absorbed_E) < 2 or len(rejected_E) < 2:
        return None, len(absorbed_E), len(rejected_E)

    mean_absorbed = np.mean(absorbed_E)
    mean_rejected = np.mean(rejected_E)

    # Pooled standard deviation
    std_absorbed = np.std(absorbed_E, ddof=1)
    std_rejected = np.std(rejected_E, ddof=1)

    n1, n2 = len(absorbed_E), len(rejected_E)
    pooled_std = np.sqrt(((n1-1)*std_absorbed**2 + (n2-1)*std_rejected**2) / (n1+n2-2))

    if pooled_std < 0.001:
        return None, n1, n2

    cohen_d = (mean_absorbed - mean_rejected) / pooled_std

    return cohen_d, n1, n2


def main():
    print("=" * 70)
    print("PREDICTION 1: Rate-Dependent Threshold")
    print("Claim: Fast processing degrades gate discrimination")
    print("=" * 70)
    print()

    reasoner = GeometricReasoner()

    # Load chunks
    print("Loading paper chunks...")
    chunks = load_paper_chunks(max_papers=20, chunks_per_paper=5)
    print(f"Loaded {len(chunks)} chunks")
    print()

    # Test different noise levels (modeling processing speed)
    # Higher noise = faster processing = WORSE discrimination (predicted)
    # noise_scale models the "turbulence" from not letting mind settle
    noise_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]

    results_by_rate = []

    for noise in noise_values:
        print(f"Testing noise_scale={noise}...")

        # Run simulation multiple times and average (noise is stochastic)
        cohen_ds = []
        for trial in range(3):
            results = simulate_processing(reasoner, chunks.copy(), noise_scale=noise)
            cohen_d, n_absorbed, n_rejected = compute_discrimination(results)
            if cohen_d is not None:
                cohen_ds.append(cohen_d)

        if cohen_ds:
            mean_d = np.mean(cohen_ds)
            results_by_rate.append({
                'noise': noise,
                'cohen_d': mean_d,
                'n_trials': len(cohen_ds)
            })
            print(f"  Cohen's d = {mean_d:.3f} (avg of {len(cohen_ds)} trials)")
        else:
            results_by_rate.append({
                'noise': noise,
                'cohen_d': None,
                'n_trials': 0
            })
            print(f"  Insufficient data")
        print()

    # Analysis
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    valid_results = [r for r in results_by_rate if r['cohen_d'] is not None]

    if len(valid_results) >= 2:
        # Check if discrimination DECREASES with noise (fast processing)
        noises = [r['noise'] for r in valid_results]
        cohens = [r['cohen_d'] for r in valid_results]

        # Compute correlation between noise and discrimination
        # Prediction: NEGATIVE correlation (more noise = worse discrimination)
        correlation = np.corrcoef(noises, cohens)[0, 1]

        print("Discrimination by noise level (proxy for processing speed):")
        for r in valid_results:
            print(f"  noise={r['noise']:.2f}: Cohen's d = {r['cohen_d']:.3f}")

        print()
        print(f"Correlation (noise vs d): {correlation:.3f}")
        print()

        # Compute degradation ratio
        if valid_results[0]['cohen_d'] > 0 and valid_results[-1]['cohen_d'] > 0:
            degradation = (valid_results[0]['cohen_d'] - valid_results[-1]['cohen_d']) / valid_results[0]['cohen_d'] * 100
            print(f"Degradation from noise=0 to noise={valid_results[-1]['noise']}: {degradation:.1f}%")
            print()

        # Verdict
        if correlation < -0.5:
            print("[PASS] PREDICTION CONFIRMED: Fast processing degrades discrimination")
            print(f"  Correlation = {correlation:.3f} < -0.5 threshold")
            print("  Noise (modeling fast processing) reduces gate effectiveness!")
        elif correlation < 0:
            print("[PARTIAL] Weak negative correlation")
            print(f"  Correlation = {correlation:.3f}")
            print("  Trend exists but is weak")
        else:
            print("[FAIL] PREDICTION FAILED: No rate dependence")
            print(f"  Correlation = {correlation:.3f}")
    else:
        print("Insufficient data for analysis")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
