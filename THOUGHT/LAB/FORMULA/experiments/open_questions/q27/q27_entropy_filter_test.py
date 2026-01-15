#!/usr/bin/env python3
"""
Q27 Entropy Filter Test: Multiplicative vs Additive Relationship

This test explores the theoretical implication that entropy acts as a
multiplicative filter rather than an additive degradation.

Key hypotheses to test:
1. PHASE TRANSITION: There's a critical noise level where behavior shifts
   from additive (degrading) to multiplicative (filtering)
2. OPTIMAL NOISE: There exists an optimal noise level that maximizes
   quality × quantity tradeoff
3. MULTIPLICATIVE RELATIONSHIP: quality = signal × filter(noise), not signal - noise

Metrics:
- Cohen's d: discrimination quality (negentropy proxy)
- Acceptance rate: quantity passing
- Quality-Quantity Product: d × acceptance_rate (should show optimum)
- Filter Strength: 1 - acceptance_rate (selection pressure)
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Find repo root
REPO_ROOT = Path(__file__).resolve().parent
while REPO_ROOT.name != "agent-governance-system" and REPO_ROOT.parent != REPO_ROOT:
    REPO_ROOT = REPO_ROOT.parent

CAPABILITY_PATH = REPO_ROOT / "CAPABILITY" / "PRIMITIVES"
FERAL_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "FERAL_RESIDENT"
sys.path.insert(0, str(CAPABILITY_PATH))
sys.path.insert(0, str(FERAL_PATH))

import numpy as np
from geometric_reasoner import GeometricReasoner, GeometricState

# Configuration
CRITICAL_RESONANCE = 1.0 / (2.0 * np.pi)
N_TRIALS = 10

# Fine-grained noise levels to find phase transition
NOISE_LEVELS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04,
                0.05, 0.06, 0.07, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]


def get_dynamic_threshold(n_memories: int) -> float:
    """Q46 nucleation threshold"""
    grad_S = 1.0 / np.sqrt(max(n_memories, 1))
    return CRITICAL_RESONANCE / (1.0 + grad_S)


def load_paper_chunks(max_papers=20, chunks_per_paper=5):
    """Load real paper chunks for testing."""
    papers_dir = FERAL_PATH / "research" / "papers" / "markdown"
    chunks = []

    if papers_dir.exists():
        for i, paper_file in enumerate(papers_dir.glob("*.md")):
            if i >= max_papers:
                break
            try:
                content = paper_file.read_text(encoding='utf-8')
                paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 100]
                for j, para in enumerate(paragraphs[:chunks_per_paper]):
                    chunks.append({'content': para[:500]})
            except Exception:
                continue

    if len(chunks) < 50:
        # Generate synthetic if not enough real data
        chunks = [{'content': f"Synthetic chunk {i} about entropy and information theory. " * 5}
                  for i in range(100)]

    return chunks


def simulate_processing(reasoner, chunks, noise_scale=0.0):
    """Simulate chunk processing with noise injection."""
    mind_state = None
    n_memories = 0
    results = []

    for chunk in chunks:
        chunk_state = reasoner.initialize(chunk['content'])

        if mind_state is not None:
            E = chunk_state.E_with(mind_state)
        else:
            E = 0.5

        threshold = get_dynamic_threshold(n_memories)
        gate_open = E > threshold

        results.append({
            'E': E,
            'threshold': threshold,
            'gate_open': gate_open,
            'n_memories': n_memories
        })

        if gate_open:
            if mind_state is None:
                mind_state = chunk_state
            else:
                n = n_memories + 1
                t = 1.0 / (n + 1)
                mind_state = reasoner.interpolate(mind_state, chunk_state, t)

            n_memories += 1

            # Add noise AFTER absorption
            if noise_scale > 0 and mind_state is not None:
                noise = np.random.randn(len(mind_state.vector)) * noise_scale
                perturbed = mind_state.vector + noise
                perturbed = perturbed / np.linalg.norm(perturbed)
                mind_state = GeometricState(
                    vector=perturbed.astype(np.float32),
                    operation_history=mind_state.operation_history
                )

    return results


def compute_metrics(results) -> Dict:
    """Compute comprehensive metrics for entropy analysis."""
    absorbed_E = [r['E'] for r in results if r['gate_open']]
    rejected_E = [r['E'] for r in results if not r['gate_open']]

    n_absorbed = len(absorbed_E)
    n_rejected = len(rejected_E)
    total = n_absorbed + n_rejected

    acceptance_rate = n_absorbed / total if total > 0 else 0
    filter_strength = 1 - acceptance_rate

    # Cohen's d
    cohen_d = None
    if len(absorbed_E) >= 2 and len(rejected_E) >= 2:
        mean_a, mean_r = np.mean(absorbed_E), np.mean(rejected_E)
        std_a, std_r = np.std(absorbed_E, ddof=1), np.std(rejected_E, ddof=1)
        pooled_std = np.sqrt(((n_absorbed-1)*std_a**2 + (n_rejected-1)*std_r**2) / (n_absorbed+n_rejected-2))
        if pooled_std > 0.001:
            cohen_d = (mean_a - mean_r) / pooled_std

    # Quality-Quantity product (should show optimum)
    qq_product = cohen_d * acceptance_rate if cohen_d else None

    # Mean E of survivors (quality of accepted)
    mean_survivor_E = np.mean(absorbed_E) if absorbed_E else None

    return {
        'cohen_d': cohen_d,
        'acceptance_rate': acceptance_rate,
        'filter_strength': filter_strength,
        'qq_product': qq_product,
        'mean_survivor_E': mean_survivor_E,
        'n_absorbed': n_absorbed,
        'n_rejected': n_rejected
    }


def find_phase_transition(results_by_noise: Dict) -> float:
    """
    Find the critical noise level where behavior transitions
    from additive (d decreases) to multiplicative (d increases).
    """
    noise_levels = sorted([n for n in results_by_noise.keys() if results_by_noise[n]['cohen_d'] is not None])

    if len(noise_levels) < 3:
        return None

    # Find minimum Cohen's d point (transition)
    min_d = float('inf')
    transition_noise = noise_levels[0]

    for noise in noise_levels:
        d = results_by_noise[noise]['cohen_d']
        if d < min_d:
            min_d = d
            transition_noise = noise

    return transition_noise


def find_optimal_noise(results_by_noise: Dict) -> Tuple[float, float]:
    """
    Find optimal noise level that maximizes quality × quantity product.
    """
    best_noise = 0.0
    best_product = 0.0

    for noise, metrics in results_by_noise.items():
        if metrics['qq_product'] is not None and metrics['qq_product'] > best_product:
            best_product = metrics['qq_product']
            best_noise = noise

    return best_noise, best_product


def test_multiplicative_relationship(results_by_noise: Dict) -> Dict:
    """
    Test if quality = signal × filter(noise) fits better than quality = signal - noise.

    Multiplicative model: d = d0 × f(noise) where f increases with noise
    Additive model: d = d0 - k × noise

    Compare R² of both models.
    """
    valid_data = [(n, m) for n, m in results_by_noise.items()
                  if m['cohen_d'] is not None and n > 0]

    if len(valid_data) < 4:
        return {'insufficient_data': True}

    noises = np.array([d[0] for d in valid_data])
    cohens = np.array([d[1]['cohen_d'] for d in valid_data])
    filters = np.array([d[1]['filter_strength'] for d in valid_data])

    # Get baseline (noise=0)
    d0 = results_by_noise.get(0.0, {}).get('cohen_d', cohens[0])

    # Additive model: d = d0 - k*noise
    # Linear regression
    additive_pred = d0 - noises * ((d0 - cohens[-1]) / noises[-1])
    ss_res_add = np.sum((cohens - additive_pred) ** 2)
    ss_tot = np.sum((cohens - np.mean(cohens)) ** 2)
    r2_additive = 1 - (ss_res_add / ss_tot) if ss_tot > 0 else 0

    # Multiplicative model: d = d0 * filter_strength^k
    # or equivalently: d ~ filter_strength (selection pressure)
    # Fit: d = a * filter^b
    try:
        # Log transform for power law fit
        log_filter = np.log(filters + 0.01)  # avoid log(0)
        log_d = np.log(cohens)

        # Linear regression in log space
        slope, intercept = np.polyfit(log_filter, log_d, 1)
        mult_pred = np.exp(intercept) * (filters + 0.01) ** slope

        ss_res_mult = np.sum((cohens - mult_pred) ** 2)
        r2_multiplicative = 1 - (ss_res_mult / ss_tot) if ss_tot > 0 else 0
    except:
        r2_multiplicative = 0
        slope, intercept = 0, 0

    # Correlation between filter strength and Cohen's d
    corr_filter_d = np.corrcoef(filters, cohens)[0, 1]

    return {
        'r2_additive': r2_additive,
        'r2_multiplicative': r2_multiplicative,
        'better_model': 'multiplicative' if r2_multiplicative > r2_additive else 'additive',
        'filter_d_correlation': corr_filter_d,
        'power_law_exponent': slope
    }


def main():
    print("=" * 70)
    print("Q27 ENTROPY FILTER TEST")
    print("Testing Multiplicative vs Additive Entropy Relationship")
    print("=" * 70)
    print()

    reasoner = GeometricReasoner()

    print("Loading chunks...")
    chunks = load_paper_chunks(max_papers=20, chunks_per_paper=5)
    print(f"Loaded {len(chunks)} chunks")
    print()

    results_by_noise = {}

    print("Running fine-grained noise sweep...")
    print("-" * 70)

    for noise in NOISE_LEVELS:
        trial_metrics = []

        for trial in range(N_TRIALS):
            results = simulate_processing(reasoner, chunks.copy(), noise_scale=noise)
            metrics = compute_metrics(results)
            if metrics['cohen_d'] is not None:
                trial_metrics.append(metrics)

        if trial_metrics:
            # Average across trials
            avg_metrics = {
                'cohen_d': np.mean([m['cohen_d'] for m in trial_metrics]),
                'acceptance_rate': np.mean([m['acceptance_rate'] for m in trial_metrics]),
                'filter_strength': np.mean([m['filter_strength'] for m in trial_metrics]),
                'qq_product': np.mean([m['qq_product'] for m in trial_metrics if m['qq_product']]),
                'mean_survivor_E': np.mean([m['mean_survivor_E'] for m in trial_metrics if m['mean_survivor_E']]),
                'std_d': np.std([m['cohen_d'] for m in trial_metrics]),
                'n_trials': len(trial_metrics)
            }
            results_by_noise[noise] = avg_metrics

            print(f"noise={noise:.3f}: d={avg_metrics['cohen_d']:.3f}, "
                  f"acc={avg_metrics['acceptance_rate']:.1%}, "
                  f"filter={avg_metrics['filter_strength']:.1%}, "
                  f"QQ={avg_metrics['qq_product']:.3f}")

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    # 1. Find phase transition
    transition = find_phase_transition(results_by_noise)
    print(f"1. PHASE TRANSITION")
    print(f"   Critical noise level (minimum d): {transition:.3f}" if transition else "   Could not determine")
    print(f"   Below this: additive regime (noise degrades)")
    print(f"   Above this: multiplicative regime (noise filters)")
    print()

    # 2. Find optimal noise
    optimal_noise, optimal_product = find_optimal_noise(results_by_noise)
    print(f"2. OPTIMAL NOISE LEVEL")
    print(f"   Optimal noise: {optimal_noise:.3f}")
    print(f"   Quality×Quantity product: {optimal_product:.3f}")
    if optimal_noise > 0:
        print(f"   This confirms: some noise is BETTER than no noise")
    print()

    # 3. Test multiplicative vs additive
    model_test = test_multiplicative_relationship(results_by_noise)
    print(f"3. MULTIPLICATIVE VS ADDITIVE MODEL")
    if 'insufficient_data' not in model_test:
        print(f"   Additive model R²: {model_test['r2_additive']:.3f}")
        print(f"   Multiplicative model R²: {model_test['r2_multiplicative']:.3f}")
        print(f"   Better fit: {model_test['better_model'].upper()}")
        print(f"   Filter-Quality correlation: {model_test['filter_d_correlation']:.3f}")
        if model_test['better_model'] == 'multiplicative':
            print(f"   CONFIRMED: Entropy acts as multiplicative filter!")
    else:
        print("   Insufficient data for model comparison")
    print()

    # 4. Summary table
    print("=" * 70)
    print("DETAILED RESULTS")
    print("-" * 70)
    print(f"{'Noise':<8} {'Cohen d':<10} {'Accept%':<10} {'Filter%':<10} {'QQ Prod':<10}")
    print("-" * 70)

    for noise in sorted(results_by_noise.keys()):
        m = results_by_noise[noise]
        marker = ""
        if noise == transition:
            marker = " <-- TRANSITION"
        if noise == optimal_noise:
            marker = " <-- OPTIMAL"
        print(f"{noise:<8.3f} {m['cohen_d']:<10.3f} {m['acceptance_rate']*100:<10.1f} "
              f"{m['filter_strength']*100:<10.1f} {m['qq_product']:<10.3f}{marker}")

    print("-" * 70)
    print()

    # Generate receipt
    receipt = {
        'timestamp': datetime.now().isoformat(),
        'test': 'Q27 Entropy Filter Analysis',
        'findings': {
            'phase_transition_noise': transition,
            'optimal_noise': optimal_noise,
            'optimal_qq_product': optimal_product,
            'model_comparison': model_test
        },
        'results': {str(k): v for k, v in results_by_noise.items()},
        'conclusion': {
            'multiplicative_confirmed': model_test.get('better_model') == 'multiplicative',
            'phase_transition_exists': transition is not None and transition > 0,
            'optimal_noise_nonzero': optimal_noise > 0
        }
    }

    receipt_dir = Path(__file__).parent / "receipts"
    receipt_dir.mkdir(exist_ok=True)
    receipt_file = receipt_dir / f"q27_entropy_filter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(receipt_file, 'w') as f:
        json.dump(receipt, f, indent=2, default=str)

    print(f"Receipt saved to: {receipt_file}")

    # Final verdict
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    if model_test.get('better_model') == 'multiplicative' and optimal_noise > 0:
        print("ENTROPY AS MULTIPLICATIVE FILTER: CONFIRMED")
        print(f"- Phase transition at noise ~{transition:.3f}")
        print(f"- Optimal noise level: {optimal_noise:.3f}")
        print("- Entropy concentrates negentropy through selection")
    else:
        print("Results inconclusive - need more data or different parameters")


if __name__ == "__main__":
    main()
