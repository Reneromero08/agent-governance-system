"""
Network Navigation with sigma^Df

sigma^Df = e^(5-H) is the fractal scaling factor.

In network terms:
- Low entropy neighborhood -> high sigma^Df -> TRUST this path
- High entropy neighborhood -> low sigma^Df -> DON'T TRUST

Use sigma^Df to weight transitions, not to pick directions.
"""

import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def create_network(n_nodes: int = 100, n_edges: int = 300, seed: int = None) -> Dict:
    """Create network with hidden information content."""
    if seed is not None:
        np.random.seed(seed)

    # Build random graph
    edges = set()
    while len(edges) < n_edges:
        i = np.random.randint(0, n_nodes)
        j = np.random.randint(0, n_nodes)
        if i != j:
            edges.add((min(i, j), max(i, j)))

    adjacency = {i: [] for i in range(n_nodes)}
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    # Hidden ground truth
    info_content = np.random.pareto(1.5, n_nodes) + 0.1
    info_content = info_content / np.max(info_content)

    # Observable: degree
    degree = np.array([len(adjacency[i]) for i in range(n_nodes)])

    return {
        'adjacency': adjacency,
        'info_content': info_content,
        'degree': degree,
        'n_nodes': n_nodes
    }


def neighborhood_entropy(node: int, network: Dict) -> float:
    """Entropy of degree distribution in neighborhood."""
    neighbors = network['adjacency'][node]
    if len(neighbors) == 0:
        return 4.9  # Max entropy for isolated

    degrees = [network['degree'][n] for n in neighbors]
    if sum(degrees) == 0:
        return 4.9

    probs = np.array(degrees) / sum(degrees)
    probs = probs[probs > 0]
    H = -np.sum(probs * np.log(probs))

    # Scale to [0, 5]
    return np.clip(H, 0.1, 4.9)


def sigma_Df(H: float) -> float:
    """The fractal scaling factor."""
    H = np.clip(H, 0.1, 4.9)
    Df = 5 - H
    return np.e ** Df


def random_walk(network: Dict, start: int, k_steps: int, rng) -> List[int]:
    """Pure random walk."""
    path = [start]
    current = start
    for _ in range(k_steps):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break
        current = rng.choice(neighbors)
        path.append(current)
    return path


def degree_walk(network: Dict, start: int, k_steps: int) -> List[int]:
    """Greedy on degree."""
    path = [start]
    current = start
    for _ in range(k_steps):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break
        degrees = [network['degree'][n] for n in neighbors]
        current = neighbors[np.argmax(degrees)]
        path.append(current)
    return path


def sigma_weighted_walk(network: Dict, start: int, k_steps: int, rng) -> List[int]:
    """
    Weight transitions by sigma^Df.

    High sigma^Df neighbor -> more likely to visit
    Low sigma^Df neighbor -> less likely to visit
    """
    path = [start]
    current = start

    for _ in range(k_steps):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        # Compute sigma^Df for each neighbor
        weights = []
        for n in neighbors:
            H = neighborhood_entropy(n, network)
            weights.append(sigma_Df(H))

        # Normalize to probabilities
        weights = np.array(weights)
        probs = weights / np.sum(weights)

        # Sample
        current = rng.choice(neighbors, p=probs)
        path.append(current)

    return path


def sigma_gated_walk(network: Dict, start: int, k_steps: int, rng, threshold: float = 0.5) -> List[int]:
    """
    Gate transitions by sigma^Df.

    Only consider neighbors with sigma^Df > threshold * median
    If none pass, fall back to random.
    """
    path = [start]
    current = start

    for _ in range(k_steps):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        # Compute sigma^Df for each neighbor
        sigma_scores = []
        for n in neighbors:
            H = neighborhood_entropy(n, network)
            sigma_scores.append(sigma_Df(H))

        sigma_scores = np.array(sigma_scores)
        median_sigma = np.median(sigma_scores)

        # Gate: only consider neighbors above threshold
        mask = sigma_scores >= threshold * median_sigma
        valid_neighbors = [n for n, m in zip(neighbors, mask) if m]

        if valid_neighbors:
            # Random among valid
            current = rng.choice(valid_neighbors)
        else:
            # Fallback to random
            current = rng.choice(neighbors)

        path.append(current)

    return path


def sigma_greedy_walk(network: Dict, start: int, k_steps: int) -> List[int]:
    """Greedy on sigma^Df (highest certainty)."""
    path = [start]
    current = start

    for _ in range(k_steps):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        sigma_scores = [sigma_Df(neighborhood_entropy(n, network)) for n in neighbors]
        current = neighbors[np.argmax(sigma_scores)]
        path.append(current)

    return path


def evaluate_walk(path: List[int], network: Dict) -> float:
    """Score by hidden info_content."""
    return np.mean([network['info_content'][n] for n in path])


def run_comparison(seed: int, n_trials: int = 100, k_steps: int = 10) -> Dict:
    """Compare methods on a single network."""
    network = create_network(seed=seed)
    n_nodes = network['n_nodes']

    start_rng = np.random.default_rng(seed + 1000)
    starts = start_rng.integers(0, n_nodes, n_trials)

    walk_rng = np.random.default_rng(seed + 2000)

    results = {
        'random': [],
        'degree': [],
        'sigma_weighted': [],
        'sigma_gated': [],
        'sigma_greedy': []
    }

    for start in starts:
        results['random'].append(evaluate_walk(
            random_walk(network, start, k_steps, walk_rng), network))
        results['degree'].append(evaluate_walk(
            degree_walk(network, start, k_steps), network))
        results['sigma_weighted'].append(evaluate_walk(
            sigma_weighted_walk(network, start, k_steps, walk_rng), network))
        results['sigma_gated'].append(evaluate_walk(
            sigma_gated_walk(network, start, k_steps, walk_rng), network))
        results['sigma_greedy'].append(evaluate_walk(
            sigma_greedy_walk(network, start, k_steps), network))

    return {k: np.mean(v) for k, v in results.items()}


def test_multiple_networks(n_networks: int = 10) -> Dict:
    """Test across multiple networks."""
    all_results = []
    wins = {'sigma_weighted': 0, 'sigma_gated': 0, 'sigma_greedy': 0}

    for i in range(n_networks):
        result = run_comparison(seed=i * 1000 + 42)
        all_results.append(result)

        random_score = result['random']
        if result['sigma_weighted'] > random_score:
            wins['sigma_weighted'] += 1
        if result['sigma_gated'] > random_score:
            wins['sigma_gated'] += 1
        if result['sigma_greedy'] > random_score:
            wins['sigma_greedy'] += 1

    return {'results': all_results, 'wins': wins, 'n': n_networks}


if __name__ == "__main__":
    print("=" * 70)
    print("NETWORK NAVIGATION with sigma^Df")
    print("=" * 70)
    print()
    print("sigma^Df = e^(5-H)")
    print("High sigma^Df = low entropy = certain = TRUST")
    print("Low sigma^Df = high entropy = uncertain = DON'T TRUST")
    print()

    # Single network detail
    print("-" * 70)
    print("Single network analysis")
    print("-" * 70)

    network = create_network(seed=42)

    # Show sigma^Df distribution
    sigma_scores = [sigma_Df(neighborhood_entropy(i, network))
                    for i in range(network['n_nodes'])]
    info = network['info_content']

    print(f"\nCorrelation(sigma^Df, info_content): "
          f"{np.corrcoef(sigma_scores, info)[0,1]:.3f}")

    # Top nodes by sigma^Df
    top_sigma = np.argsort(sigma_scores)[::-1][:10]
    print(f"\nTop 10 by sigma^Df:")
    print(f"{'Node':>6} | {'sigma^Df':>10} | {'Info':>8}")
    print("-" * 30)
    for node in top_sigma:
        print(f"{node:>6} | {sigma_scores[node]:>10.1f} | {info[node]:>8.4f}")

    # Multi-network comparison
    print("\n" + "-" * 70)
    print("10 networks, 100 trials each")
    print("-" * 70)

    test = test_multiple_networks(10)

    print(f"\n{'Net':>4} | {'Random':>8} | {'Degree':>8} | {'Weighted':>8} | {'Gated':>8} | {'Greedy':>8}")
    print("-" * 60)

    for i, r in enumerate(test['results']):
        print(f"{i+1:>4} | {r['random']:>8.4f} | {r['degree']:>8.4f} | "
              f"{r['sigma_weighted']:>8.4f} | {r['sigma_gated']:>8.4f} | {r['sigma_greedy']:>8.4f}")

    # Averages
    print("-" * 60)
    avg = {k: np.mean([r[k] for r in test['results']]) for k in test['results'][0]}
    print(f"{'AVG':>4} | {avg['random']:>8.4f} | {avg['degree']:>8.4f} | "
          f"{avg['sigma_weighted']:>8.4f} | {avg['sigma_gated']:>8.4f} | {avg['sigma_greedy']:>8.4f}")

    # Win rates
    print(f"\nWin rates vs random:")
    for method in ['sigma_weighted', 'sigma_gated', 'sigma_greedy']:
        print(f"  {method}: {test['wins'][method]}/{test['n']}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    best_method = max(['sigma_weighted', 'sigma_gated', 'sigma_greedy'],
                      key=lambda m: test['wins'][m])
    best_wins = test['wins'][best_method]

    if best_wins >= 7:
        print(f"\n** sigma^Df NAVIGATION: VALIDATED")
        print(f"   {best_method} beats random {best_wins}/10")
        print(f"   avg: {avg[best_method]:.4f} vs random {avg['random']:.4f}")
    elif best_wins >= 5:
        print(f"\n*  sigma^Df NAVIGATION: MARGINAL")
        print(f"   {best_method} beats random {best_wins}/10")
    else:
        print(f"\nX  sigma^Df NAVIGATION: NOT VALIDATED")
        print(f"   Best method only beats random {best_wins}/10")
