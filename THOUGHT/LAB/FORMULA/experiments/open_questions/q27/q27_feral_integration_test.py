#!/usr/bin/env python3
"""
Q27 FERAL Integration Test: Self-Protective Gating in Live Memory System

Tests the Q27 adaptive threshold finding using FERAL_RESIDENT's GeometricMemory.
This validates that the self-protective gating phenomenon occurs in the actual
memory accumulation system, not just in isolated simulation.

Key validation:
1. Build memory from real papers
2. Inject noise into mind_state after memories
3. Verify discrimination improves under noise (selection pressure)
4. Confirm acceptance rate drops (fewer pass)
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Tuple

# Find repo root
REPO_ROOT = Path(__file__).resolve().parent
while REPO_ROOT.name != "agent-governance-system" and REPO_ROOT.parent != REPO_ROOT:
    REPO_ROOT = REPO_ROOT.parent

FERAL_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "FERAL_RESIDENT"
CAPABILITY_PATH = REPO_ROOT / "CAPABILITY" / "PRIMITIVES"
sys.path.insert(0, str(FERAL_PATH))
sys.path.insert(0, str(CAPABILITY_PATH))

import numpy as np
from geometric_memory import GeometricMemory
from geometric_reasoner import GeometricState

# Configuration
CRITICAL_RESONANCE = 1.0 / (2.0 * np.pi)
N_TRIALS = 5
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.1, 0.2]


def get_dynamic_threshold(n_memories: int) -> float:
    """Q46 nucleation threshold"""
    grad_S = 1.0 / np.sqrt(max(n_memories, 1))
    return CRITICAL_RESONANCE / (1.0 + grad_S)


def load_paper_content() -> List[str]:
    """Load paper chunks from FERAL's markdown papers."""
    papers_dir = FERAL_PATH / "research" / "papers" / "markdown"
    chunks = []

    if not papers_dir.exists():
        print(f"Warning: Papers directory not found at {papers_dir}")
        return []

    for paper_file in papers_dir.glob("*.md"):
        try:
            content = paper_file.read_text(encoding='utf-8')
            paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 100]
            chunks.extend([p[:500] for p in paragraphs[:5]])
        except Exception:
            continue

    return chunks


def perturb_mind_state(memory: GeometricMemory, noise_scale: float):
    """
    Inject noise into memory's mind_state to simulate processing instability.

    This models the Q27 hypothesis: fast processing doesn't let mind settle,
    causing effective threshold to rise (self-protective gating).
    """
    if memory.mind_state is None or noise_scale == 0:
        return

    noise = np.random.randn(len(memory.mind_state.vector)) * noise_scale
    perturbed = memory.mind_state.vector + noise
    perturbed = perturbed / np.linalg.norm(perturbed)

    memory.mind_state = GeometricState(
        vector=perturbed.astype(np.float32),
        operation_history=memory.mind_state.operation_history
    )


def run_gated_accumulation(
    memory: GeometricMemory,
    chunks: List[str],
    noise_scale: float = 0.0
) -> Dict:
    """
    Accumulate memories with E-gating and optional noise injection.

    Returns statistics on gate behavior.
    """
    threshold = get_dynamic_threshold(0)

    accepted = []
    rejected = []

    for i, chunk in enumerate(chunks):
        # First few chunks seed the memory WITHOUT noise
        # This establishes coherent initial direction
        # Noise only added after successful absorptions DURING gating phase
        if i < 3:
            memory.remember(chunk)
            # DO NOT add noise during seeding - this is key to Q27 mechanism
            # Noise during seeding just randomizes direction (not self-protective)
            continue

        # After seeding, use E-gating
        query_result = memory.recall_with_gate(
            chunk,
            corpus=[chunk],  # Self-query to get E
            threshold=threshold
        )

        E = query_result['E']
        gate_open = query_result['gate_open']

        if gate_open:
            accepted.append(E)
            memory.remember(chunk)
            # Inject noise AFTER absorption (key to Q27 mechanism)
            if noise_scale > 0:
                perturb_mind_state(memory, noise_scale)
        else:
            rejected.append(E)

        # Update threshold based on memory count
        threshold = get_dynamic_threshold(len(memory.memory_history))

    return {
        'accepted_E': accepted,
        'rejected_E': rejected,
        'n_accepted': len(accepted),
        'n_rejected': len(rejected),
        'final_metrics': memory.get_evolution_metrics()
    }


def compute_cohen_d(accepted: List[float], rejected: List[float]) -> float:
    """Compute Cohen's d effect size."""
    if len(accepted) < 2 or len(rejected) < 2:
        return None

    mean_a, mean_r = np.mean(accepted), np.mean(rejected)
    std_a, std_r = np.std(accepted, ddof=1), np.std(rejected, ddof=1)
    n1, n2 = len(accepted), len(rejected)

    pooled_std = np.sqrt(((n1-1)*std_a**2 + (n2-1)*std_r**2) / (n1+n2-2))

    if pooled_std < 0.001:
        return None

    return (mean_a - mean_r) / pooled_std


def main():
    print("=" * 70)
    print("Q27 FERAL INTEGRATION TEST")
    print("Self-Protective Gating in Live GeometricMemory")
    print("=" * 70)
    print()

    # Load paper chunks
    print("Loading paper chunks from FERAL_RESIDENT...")
    chunks = load_paper_content()

    if len(chunks) < 20:
        print(f"Warning: Only {len(chunks)} chunks loaded. Need at least 20 for meaningful test.")
        if len(chunks) < 5:
            print("Insufficient data. Generating synthetic test data...")
            chunks = [
                f"This is synthetic chunk {i} about quantum mechanics and neural networks. " * 5
                for i in range(50)
            ]

    print(f"Loaded {len(chunks)} chunks")
    print()

    results_by_noise = {}

    for noise in NOISE_LEVELS:
        print(f"Testing noise_scale={noise:.2f} ({N_TRIALS} trials)...")

        cohen_ds = []
        acceptance_rates = []
        final_dfs = []

        for trial in range(N_TRIALS):
            memory = GeometricMemory()

            result = run_gated_accumulation(memory, chunks.copy(), noise_scale=noise)

            d = compute_cohen_d(result['accepted_E'], result['rejected_E'])
            if d is not None:
                cohen_ds.append(d)

            total = result['n_accepted'] + result['n_rejected']
            if total > 0:
                acceptance_rates.append(result['n_accepted'] / total)

            if result['final_metrics']['current_Df'] > 0:
                final_dfs.append(result['final_metrics']['current_Df'])

        if cohen_ds:
            mean_d = np.mean(cohen_ds)
            std_d = np.std(cohen_ds, ddof=1) if len(cohen_ds) > 1 else 0
            mean_acc = np.mean(acceptance_rates) if acceptance_rates else 0
            mean_df = np.mean(final_dfs) if final_dfs else 0

            results_by_noise[noise] = {
                'mean_cohen_d': mean_d,
                'std_cohen_d': std_d,
                'mean_acceptance_rate': mean_acc,
                'mean_final_Df': mean_df,
                'n_valid_trials': len(cohen_ds)
            }

            print(f"  Cohen's d = {mean_d:.3f} +/- {std_d:.3f}")
            print(f"  Acceptance rate = {mean_acc:.1%}")
            print(f"  Final Df = {mean_df:.2f}")
        else:
            results_by_noise[noise] = None
            print(f"  Insufficient data")
        print()

    # Correlation analysis
    print("=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    print()

    valid_noises = [n for n in NOISE_LEVELS if results_by_noise.get(n) is not None]
    valid_ds = [results_by_noise[n]['mean_cohen_d'] for n in valid_noises]
    valid_acc = [results_by_noise[n]['mean_acceptance_rate'] for n in valid_noises]

    if len(valid_noises) >= 3:
        corr_d = np.corrcoef(valid_noises, valid_ds)[0, 1]
        corr_acc = np.corrcoef(valid_noises, valid_acc)[0, 1]

        print(f"Noise vs Cohen's d correlation: r = {corr_d:.3f}")
        print(f"Noise vs Acceptance rate correlation: r = {corr_acc:.3f}")
        print()

        # Verdict
        print("VERDICT:")
        if corr_d > 0.5 and corr_acc < -0.3:
            print("  [PASS] Self-protective gating CONFIRMED in live memory system")
            print(f"    - Discrimination improves with noise (r = {corr_d:.3f} > 0.5)")
            print(f"    - Acceptance rate drops with noise (r = {corr_acc:.3f} < -0.3)")
        elif corr_d > 0:
            print("  [PARTIAL] Weak self-protective gating observed")
            print(f"    - Discrimination trend positive (r = {corr_d:.3f})")
        else:
            print("  [FAIL] Self-protective gating NOT confirmed")
            print(f"    - Correlation = {corr_d:.3f}")
    else:
        corr_d = None
        corr_acc = None
        print("Insufficient data for correlation analysis")

    # Generate receipt
    print()
    print("=" * 70)

    receipt = {
        'timestamp': datetime.now().isoformat(),
        'test': 'Q27 FERAL Integration',
        'config': {
            'n_trials': N_TRIALS,
            'noise_levels': NOISE_LEVELS,
            'n_chunks': len(chunks)
        },
        'results': {str(k): v for k, v in results_by_noise.items()},
        'correlations': {
            'noise_vs_cohend': corr_d,
            'noise_vs_acceptance': corr_acc
        },
        'verdict': 'PASS' if (corr_d and corr_d > 0.5) else 'PARTIAL' if (corr_d and corr_d > 0) else 'FAIL'
    }

    receipt_dir = Path(__file__).parent / "receipts"
    receipt_dir.mkdir(exist_ok=True)
    receipt_file = receipt_dir / f"q27_feral_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(receipt_file, 'w') as f:
        json.dump(receipt, f, indent=2, default=str)

    print(f"Receipt saved to: {receipt_file}")

    # Summary table
    print()
    print("SUMMARY TABLE:")
    print("-" * 60)
    print(f"{'Noise':<8} {'Cohen d':<12} {'Accept Rate':<15} {'Final Df':<10}")
    print("-" * 60)
    for noise in valid_noises:
        r = results_by_noise[noise]
        print(f"{noise:<8.2f} {r['mean_cohen_d']:<12.3f} {r['mean_acceptance_rate']:<15.1%} {r['mean_final_Df']:<10.2f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
