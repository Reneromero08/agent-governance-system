"""
ELO Gate Test - Apply the formula as a gate on ELO updates

The formula doesn't predict WHO wins.
It gates HOW MUCH to trust the result.

Match outcome = SIGNAL (direction of update)
R = GATE (how informative was this match?)

High R -> large K factor (trust this match)
Low R -> small K factor (noisy match, discount it)
"""

import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def create_players(n_players: int = 100, seed: int = None) -> Dict:
    """Create players with true skill and initial ELO."""
    if seed is not None:
        np.random.seed(seed)

    # True skill (hidden) - what we're trying to estimate
    true_skill = np.random.normal(1500, 300, n_players)

    # Volatility per player (some players are inconsistent)
    volatility = np.random.uniform(0.1, 0.5, n_players)

    return {
        'true_skill': true_skill,
        'volatility': volatility,
        'n_players': n_players
    }


def simulate_match(players: Dict, p1: int, p2: int, rng) -> Tuple[int, float, float]:
    """
    Simulate a match between two players.

    Returns:
    - winner: 0 if p1 wins, 1 if p2 wins
    - margin: how decisive the win was (0-1)
    - noise: how noisy was this particular match
    """
    skill1 = players['true_skill'][p1]
    skill2 = players['true_skill'][p2]
    vol1 = players['volatility'][p1]
    vol2 = players['volatility'][p2]

    # Actual performance = skill + noise
    perf1 = skill1 + rng.normal(0, vol1 * 200)
    perf2 = skill2 + rng.normal(0, vol2 * 200)

    # Winner
    winner = 0 if perf1 > perf2 else 1

    # Margin (how decisive)
    diff = abs(perf1 - perf2)
    margin = min(diff / 500, 1.0)  # Normalize to 0-1

    # Match noise (combined volatility)
    noise = (vol1 + vol2) / 2

    return winner, margin, noise


def expected_score(elo_a: float, elo_b: float) -> float:
    """Standard ELO expected score."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def compute_R(margin: float, noise: float, history_entropy: float) -> float:
    """
    Compute R for this match.

    E = margin (signal strength - decisive wins are more informative)
    D = noise + history_entropy (disorder)
    """
    E = margin + 0.1  # Avoid zero
    D = noise + history_entropy + 0.1

    # Simple R = E/D
    R = E / D

    return R


def compute_gate(R: float, threshold: float = 1.0) -> float:
    """
    Gate function: how much to trust this match.

    Returns K multiplier in [0.5, 2.0]
    """
    # Sigmoid-like scaling
    gate = 2 / (1 + np.exp(-R + threshold))
    return np.clip(gate, 0.5, 2.0)


def run_elo_system(players: Dict, matches: List[Tuple], method: str,
                   base_K: float = 32) -> np.ndarray:
    """
    Run ELO updates for all matches.

    Methods:
    - 'standard': Fixed K factor
    - 'gated': K modulated by R
    - 'margin': K scaled by margin only
    - 'inverse': K inversely proportional to noise (ablation)
    """
    n = players['n_players']
    elo = np.ones(n) * 1500  # Start at 1500

    # Track history for entropy calculation
    history = {i: [] for i in range(n)}

    for p1, p2, winner, margin, noise in matches:
        # Expected scores
        exp1 = expected_score(elo[p1], elo[p2])
        exp2 = 1 - exp1

        # Actual scores
        s1 = 1 if winner == 0 else 0
        s2 = 1 - s1

        # Compute K factor based on method
        if method == 'standard':
            K = base_K

        elif method == 'gated':
            # Full R-gated approach
            h1 = np.std(history[p1]) / 100 if len(history[p1]) > 2 else 0.5
            h2 = np.std(history[p2]) / 100 if len(history[p2]) > 2 else 0.5
            history_entropy = (h1 + h2) / 2

            R = compute_R(margin, noise, history_entropy)
            gate = compute_gate(R)
            K = base_K * gate

        elif method == 'margin':
            # Ablation: only use margin
            K = base_K * (0.5 + margin)

        elif method == 'noise_inverse':
            # Ablation: only use noise (inverse)
            K = base_K * (1.5 - noise)

        elif method == 'no_gate':
            # Same as standard
            K = base_K

        # Update ELO
        elo[p1] += K * (s1 - exp1)
        elo[p2] += K * (s2 - exp2)

        # Track history
        history[p1].append(elo[p1])
        history[p2].append(elo[p2])

    return elo


def evaluate_elo(elo: np.ndarray, true_skill: np.ndarray) -> Dict:
    """Evaluate how well ELO tracks true skill."""
    # Correlation
    corr = np.corrcoef(elo, true_skill)[0, 1]

    # Rank correlation (Spearman)
    elo_ranks = np.argsort(np.argsort(elo))
    true_ranks = np.argsort(np.argsort(true_skill))
    rank_corr = np.corrcoef(elo_ranks, true_ranks)[0, 1]

    # Top-10 accuracy (are the top 10 ELO actually top 10 skill?)
    top10_elo = set(np.argsort(elo)[-10:])
    top10_true = set(np.argsort(true_skill)[-10:])
    top10_acc = len(top10_elo & top10_true) / 10

    # MSE of normalized values
    elo_norm = (elo - elo.mean()) / elo.std()
    true_norm = (true_skill - true_skill.mean()) / true_skill.std()
    mse = np.mean((elo_norm - true_norm) ** 2)

    return {
        'correlation': corr,
        'rank_correlation': rank_corr,
        'top10_accuracy': top10_acc,
        'mse': mse
    }


def run_test(seed: int, n_players: int = 100, n_matches: int = 2000) -> Dict:
    """Run a single test with multiple ELO methods."""
    players = create_players(n_players, seed)
    rng = np.random.default_rng(seed + 1000)

    # Generate random matches
    matches = []
    for _ in range(n_matches):
        p1, p2 = rng.choice(n_players, 2, replace=False)
        winner, margin, noise = simulate_match(players, p1, p2, rng)
        matches.append((p1, p2, winner, margin, noise))

    methods = ['standard', 'gated', 'margin', 'noise_inverse']
    results = {}

    for method in methods:
        elo = run_elo_system(players, matches, method)
        results[method] = evaluate_elo(elo, players['true_skill'])

    return results


def run_full_test(n_trials: int = 10) -> Dict:
    """Run multiple trials and aggregate results."""
    all_results = {method: {'correlation': [], 'rank_correlation': [],
                            'top10_accuracy': [], 'mse': []}
                   for method in ['standard', 'gated', 'margin', 'noise_inverse']}

    for i in range(n_trials):
        trial = run_test(seed=i * 1000 + 42)
        for method, metrics in trial.items():
            for metric, value in metrics.items():
                all_results[method][metric].append(value)

    # Aggregate
    summary = {}
    for method, metrics in all_results.items():
        summary[method] = {
            metric: (np.mean(values), np.std(values))
            for metric, values in metrics.items()
        }

    return summary, all_results


if __name__ == "__main__":
    print("=" * 70)
    print("ELO GATE TEST")
    print("=" * 70)
    print()
    print("Match outcome = SIGNAL (who won)")
    print("R = GATE (how informative was this match?)")
    print()
    print("Methods:")
    print("  - standard: Fixed K=32")
    print("  - gated: K modulated by R = margin / (noise + history_entropy)")
    print("  - margin: K scaled by margin only (ablation)")
    print("  - noise_inverse: K inversely proportional to noise (ablation)")
    print()

    summary, raw = run_full_test(n_trials=10)

    # Print results
    print("-" * 70)
    print(f"{'Method':<15} | {'Correlation':>12} | {'Rank Corr':>12} | {'Top10 Acc':>10} | {'MSE':>10}")
    print("-" * 70)

    for method in ['standard', 'gated', 'margin', 'noise_inverse']:
        corr = summary[method]['correlation']
        rank = summary[method]['rank_correlation']
        top10 = summary[method]['top10_accuracy']
        mse = summary[method]['mse']
        print(f"{method:<15} | {corr[0]:>6.4f}±{corr[1]:.4f} | "
              f"{rank[0]:>6.4f}±{rank[1]:.4f} | {top10[0]:>6.2f}±{top10[1]:.2f} | "
              f"{mse[0]:>6.4f}±{mse[1]:.4f}")

    print("-" * 70)

    # Compare gated vs standard
    print()
    print("=" * 70)
    print("COMPARISON: GATED vs STANDARD")
    print("=" * 70)

    gated_corr = np.mean(raw['gated']['correlation'])
    std_corr = np.mean(raw['standard']['correlation'])
    delta = gated_corr - std_corr
    pct = (delta / std_corr) * 100

    print(f"\nCorrelation improvement: {delta:+.4f} ({pct:+.2f}%)")

    gated_rank = np.mean(raw['gated']['rank_correlation'])
    std_rank = np.mean(raw['standard']['rank_correlation'])
    delta_rank = gated_rank - std_rank
    pct_rank = (delta_rank / std_rank) * 100

    print(f"Rank correlation improvement: {delta_rank:+.4f} ({pct_rank:+.2f}%)")

    gated_top10 = np.mean(raw['gated']['top10_accuracy'])
    std_top10 = np.mean(raw['standard']['top10_accuracy'])
    delta_top10 = gated_top10 - std_top10

    print(f"Top-10 accuracy improvement: {delta_top10:+.2f}")

    # Wins count
    wins_corr = sum(1 for g, s in zip(raw['gated']['correlation'],
                                       raw['standard']['correlation']) if g > s)
    wins_rank = sum(1 for g, s in zip(raw['gated']['rank_correlation'],
                                       raw['standard']['rank_correlation']) if g > s)

    print(f"\nWins (correlation): {wins_corr}/10")
    print(f"Wins (rank corr): {wins_rank}/10")

    # Verdict
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if wins_corr >= 7 or wins_rank >= 7:
        print(f"\n** R-GATED ELO BEATS STANDARD")
        print(f"   Correlation: {gated_corr:.4f} vs {std_corr:.4f}")
        print(f"   Rank corr: {gated_rank:.4f} vs {std_rank:.4f}")
    else:
        print(f"\n   Gated ELO: {wins_corr}/10 correlation, {wins_rank}/10 rank")
        print(f"   Needs tuning or different R formulation")
