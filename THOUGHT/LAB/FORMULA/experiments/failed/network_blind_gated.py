"""
Network Blind with Gating - Can the formula help with NO signal?

Original blind test: R from structure only (degree, clustering)
No embeddings, no correlation with hidden info_content.

Question: Does gating help when there's no signal to follow?
"""

import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def create_blind_network(n_nodes: int = 100, n_edges: int = 300, seed: int = None) -> Dict:
    """Network with NO observable signal correlated with info."""
    if seed is not None:
        np.random.seed(seed)

    edges = set()
    while len(edges) < n_edges:
        i, j = np.random.randint(0, n_nodes, 2)
        if i != j:
            edges.add((min(i, j), max(i, j)))

    adjacency = {i: [] for i in range(n_nodes)}
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    # Hidden info - NO correlation with structure
    info_content = np.random.pareto(1.5, n_nodes) + 0.1
    info_content /= np.max(info_content)

    # Observable: only degree (no correlation with info)
    degree = np.array([len(adjacency[i]) for i in range(n_nodes)])

    return {
        'adjacency': adjacency,
        'info_content': info_content,
        'degree': degree,
        'n_nodes': n_nodes
    }


def structural_entropy(node: int, network: Dict) -> float:
    """Entropy from neighborhood degrees."""
    neighbors = network['adjacency'][node]
    if not neighbors:
        return 4.9

    degrees = [network['degree'][n] for n in neighbors]
    if sum(degrees) == 0:
        return 4.9

    probs = np.array(degrees) / sum(degrees)
    probs = probs[probs > 0]
    H = -np.sum(probs * np.log(probs))
    return np.clip(H, 0.1, 4.9)


def path_entropy_blind(path: List[int], network: Dict) -> float:
    """Entropy of path based on degrees visited."""
    if len(path) < 2:
        return 0.1

    degrees = [network['degree'][n] for n in path]
    degrees = np.array(degrees) + 1  # avoid zero
    probs = degrees / np.sum(degrees)
    H = -np.sum(probs * np.log(probs + 1e-10))
    return np.clip(H, 0.1, 4.9)


def random_walk(network: Dict, start: int, k: int, rng) -> List[int]:
    path = [start]
    current = start
    for _ in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break
        current = rng.choice(neighbors)
        path.append(current)
    return path


def degree_walk(network: Dict, start: int, k: int) -> List[int]:
    """Greedy on degree (only observable signal)."""
    path = [start]
    current = start
    for _ in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break
        degrees = [network['degree'][n] for n in neighbors]
        current = neighbors[np.argmax(degrees)]
        path.append(current)
    return path


def gated_degree_walk(network: Dict, start: int, k: int, threshold: float = 2.0) -> List[int]:
    """
    Degree walk with path entropy gating.

    When path entropy > threshold: stop, stay at best degree node.
    """
    path = [start]
    current = start
    best_node = start
    best_degree = network['degree'][start]

    for step in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        # Check gate
        H = path_entropy_blind(path, network)
        if H > threshold and step > 2:
            path.append(best_node)
            continue

        # Follow degree
        degrees = [network['degree'][n] for n in neighbors]
        best_idx = np.argmax(degrees)
        current = neighbors[best_idx]
        path.append(current)

        if degrees[best_idx] > best_degree:
            best_degree = degrees[best_idx]
            best_node = current

    return path


def R_gated_walk(network: Dict, start: int, k: int, sigma: float = 1.0) -> List[int]:
    """
    Use R = (E/D) * f^Df with structural signals.

    E = degree of candidate (only "signal" we have)
    D = path entropy
    f = degree
    Df = log(path_length)
    """
    path = [start]
    current = start
    best_node = start
    best_degree = network['degree'][start]

    for step in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        # Compute R for each neighbor
        D = path_entropy_blind(path, network)
        D = max(D, 0.1)
        Df = np.log(len(path) + 1) + 1

        R_scores = []
        for n in neighbors:
            E = network['degree'][n] / 10  # normalize
            f = E
            R = (E / D) * (f ** Df)
            R_scores.append(R)

        # Gate based on max R
        max_R = max(R_scores)
        W = len(path) * D
        gate = np.exp(-W**2 / sigma**2)

        if gate < 0.3 and step > 2:
            path.append(best_node)
            continue

        best_idx = np.argmax(R_scores)
        current = neighbors[best_idx]
        path.append(current)

        if network['degree'][current] > best_degree:
            best_degree = network['degree'][current]
            best_node = current

    return path


def evaluate(path: List[int], network: Dict) -> float:
    return np.mean([network['info_content'][n] for n in path])


def run_test(seed: int, n_trials: int = 100, k: int = 10) -> Dict:
    network = create_blind_network(seed=seed)
    n = network['n_nodes']

    # Check correlation between degree and info
    corr = np.corrcoef(network['degree'], network['info_content'])[0, 1]

    rng = np.random.default_rng(seed + 1000)
    starts = rng.integers(0, n, n_trials)
    walk_rng = np.random.default_rng(seed + 2000)

    results = {
        'random': [],
        'degree': [],
        'gated_degree': [],
        'R_gated': []
    }

    for s in starts:
        results['random'].append(evaluate(random_walk(network, s, k, walk_rng), network))
        results['degree'].append(evaluate(degree_walk(network, s, k), network))
        results['gated_degree'].append(evaluate(gated_degree_walk(network, s, k), network))
        results['R_gated'].append(evaluate(R_gated_walk(network, s, k), network))

    return {
        'results': {key: np.mean(v) for key, v in results.items()},
        'degree_info_corr': corr
    }


if __name__ == "__main__":
    print("=" * 70)
    print("NETWORK BLIND with GATING")
    print("=" * 70)
    print()
    print("NO embeddings. Only structural features (degree).")
    print("Question: Can gating help when there's no signal?")
    print()

    # Multi-network test
    print("-" * 70)
    print("10 networks, 100 trials each")
    print("-" * 70)

    all_results = []
    wins_random = {'degree': 0, 'gated_degree': 0, 'R_gated': 0}
    wins_degree = {'gated_degree': 0, 'R_gated': 0}
    correlations = []

    for i in range(10):
        test = run_test(seed=i * 1000 + 42)
        r = test['results']
        correlations.append(test['degree_info_corr'])
        all_results.append(r)

        for m in wins_random:
            if r[m] > r['random']:
                wins_random[m] += 1
        for m in wins_degree:
            if r[m] > r['degree']:
                wins_degree[m] += 1

    print(f"\n{'Net':>4} | {'Random':>8} | {'Degree':>8} | {'Gated':>8} | {'R_gated':>8} | {'Corr':>8}")
    print("-" * 60)

    for i, (r, corr) in enumerate(zip(all_results, correlations)):
        print(f"{i+1:>4} | {r['random']:>8.4f} | {r['degree']:>8.4f} | "
              f"{r['gated_degree']:>8.4f} | {r['R_gated']:>8.4f} | {corr:>8.3f}")

    print("-" * 60)
    avg = {k: np.mean([r[k] for r in all_results]) for k in all_results[0]}
    avg_corr = np.mean(correlations)
    print(f"{'AVG':>4} | {avg['random']:>8.4f} | {avg['degree']:>8.4f} | "
          f"{avg['gated_degree']:>8.4f} | {avg['R_gated']:>8.4f} | {avg_corr:>8.3f}")

    print(f"\nWins vs random:")
    for m, w in wins_random.items():
        print(f"  {m}: {w}/10")

    print(f"\nWins vs degree:")
    for m, w in wins_degree.items():
        print(f"  {m}: {w}/10")

    print(f"\nAverage correlation(degree, info): {avg_corr:.3f}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if abs(avg_corr) < 0.1:
        print(f"\n   Degree has NO correlation with info ({avg_corr:.3f})")
        print("   Without signal, formula cannot help.")
        print()
        print("   The formula gates SIGNAL, it doesn't create signal.")
        print("   No signal = nothing to gate.")
    else:
        print(f"\n   Unexpected: degree correlates with info ({avg_corr:.3f})")
