"""
Network Navigation - Full R formula properly interpreted

R = (E / nabla_H) * sigma^Df

E = embedding similarity (the signal)
nabla_H = gradient (how different from neighbors)
sigma^Df = certainty scaling

A node STANDS OUT if:
- High similarity to query (high E)
- Different from neighbors (high nabla_H in denominator... wait)

Actually: nabla_H is in denominator, so LOW gradient = HIGH R
This means: similar to neighbors = higher R

Hmm, that seems wrong for navigation. Let me think...

In gradient descent:
- nabla_H = how much we're changing (gradient magnitude)
- Low nabla_H means small changes
- R = E / nabla_H: high R when getting good signal (E) per unit change

For navigation:
- nabla_H = how different this node is from context
- Low nabla_H = this node is consistent with neighbors
- High R = good signal AND consistent with context

So R rewards nodes that:
1. Have high similarity (E)
2. Are CONSISTENT with their neighborhood (low nabla_H)
3. In certain regions (high sigma^Df)

This is about RELIABILITY, not outliers!
"""

import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def create_network(n_nodes: int = 100, n_edges: int = 300,
                   embedding_dim: int = 32, seed: int = None) -> Dict:
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

    info_content = np.random.pareto(1.5, n_nodes) + 0.1
    info_content /= np.max(info_content)

    signal_dir = np.random.randn(embedding_dim)
    signal_dir /= np.linalg.norm(signal_dir)

    embeddings = np.zeros((n_nodes, embedding_dim))
    for i in range(n_nodes):
        base = np.random.randn(embedding_dim)
        base /= np.linalg.norm(base)
        embeddings[i] = info_content[i] * signal_dir + 0.3 * base
        embeddings[i] /= np.linalg.norm(embeddings[i])

    query = signal_dir + 0.1 * np.random.randn(embedding_dim)
    query /= np.linalg.norm(query)

    degree = np.array([len(adjacency[i]) for i in range(n_nodes)])

    return {
        'adjacency': adjacency,
        'info_content': info_content,
        'embeddings': embeddings,
        'query': query,
        'degree': degree,
        'n_nodes': n_nodes
    }


def compute_node_R(node: int, network: Dict) -> float:
    """
    Full R formula for a node.

    E = similarity to query
    nabla_H = |similarity - mean_neighbor_similarity|
    H = entropy of neighbor similarities
    sigma^Df = e^(5-H)
    """
    emb = network['embeddings']
    query = network['query']
    neighbors = network['adjacency'][node]

    # E = this node's similarity
    E = np.dot(emb[node], query)
    E = max(E, 0.01)

    if not neighbors:
        return E * 100  # Isolated but similar = trust

    # Neighbor similarities
    neighbor_sims = np.array([np.dot(emb[n], query) for n in neighbors])
    neighbor_sims = np.clip(neighbor_sims, 0.01, 1.0)

    # nabla_H = gradient (difference from neighbors)
    mean_neighbor_sim = np.mean(neighbor_sims)
    nabla_H = abs(E - mean_neighbor_sim) + 0.01

    # H = entropy of neighbor distribution
    probs = neighbor_sims / np.sum(neighbor_sims)
    H = -np.sum(probs * np.log(probs + 1e-10))
    H = np.clip(H, 0.1, 4.9)

    # sigma^Df
    Df = 5 - H
    sigma_df = np.e ** Df

    # R formula
    R = (E / nabla_H) * sigma_df

    return R


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


def similarity_walk(network: Dict, start: int, k: int) -> List[int]:
    path = [start]
    current = start
    for _ in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break
        sims = [np.dot(network['embeddings'][n], network['query']) for n in neighbors]
        current = neighbors[np.argmax(sims)]
        path.append(current)
    return path


def R_walk(network: Dict, start: int, k: int) -> List[int]:
    """Greedy on full R score."""
    path = [start]
    current = start
    for _ in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break
        R_scores = [compute_node_R(n, network) for n in neighbors]
        current = neighbors[np.argmax(R_scores)]
        path.append(current)
    return path


def R_weighted_walk(network: Dict, start: int, k: int, rng) -> List[int]:
    """Probabilistic weighted by R."""
    path = [start]
    current = start
    for _ in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break
        R_scores = np.array([compute_node_R(n, network) for n in neighbors])
        R_scores = np.clip(R_scores, 0.01, None)
        probs = R_scores / np.sum(R_scores)
        current = rng.choice(neighbors, p=probs)
        path.append(current)
    return path


def R_gated_similarity(network: Dict, start: int, k: int, threshold: float = 0.5) -> List[int]:
    """
    Gate by R, then pick by similarity.

    Only consider neighbors where R > threshold * median_R
    Among those, pick highest similarity.
    """
    path = [start]
    current = start

    for _ in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        R_scores = [compute_node_R(n, network) for n in neighbors]
        sims = [np.dot(network['embeddings'][n], network['query']) for n in neighbors]

        median_R = np.median(R_scores)
        valid = [(n, s) for n, r, s in zip(neighbors, R_scores, sims)
                 if r >= threshold * median_R]

        if valid:
            current = max(valid, key=lambda x: x[1])[0]
        else:
            current = neighbors[np.argmax(sims)]

        path.append(current)

    return path


def evaluate(path: List[int], network: Dict) -> float:
    return np.mean([network['info_content'][n] for n in path])


def run_test(seed: int, n_trials: int = 100, k: int = 10) -> Dict:
    network = create_network(seed=seed)
    n = network['n_nodes']

    start_rng = np.random.default_rng(seed + 1000)
    starts = start_rng.integers(0, n, n_trials)
    walk_rng = np.random.default_rng(seed + 2000)

    results = {
        'random': [],
        'similarity': [],
        'R_greedy': [],
        'R_weighted': [],
        'R_gated_sim': []
    }

    for s in starts:
        results['random'].append(evaluate(random_walk(network, s, k, walk_rng), network))
        results['similarity'].append(evaluate(similarity_walk(network, s, k), network))
        results['R_greedy'].append(evaluate(R_walk(network, s, k), network))
        results['R_weighted'].append(evaluate(R_weighted_walk(network, s, k, walk_rng), network))
        results['R_gated_sim'].append(evaluate(R_gated_similarity(network, s, k), network))

    return {k: np.mean(v) for k, v in results.items()}


def test_multiple(n_networks: int = 10):
    all_results = []
    wins_random = {'R_greedy': 0, 'R_weighted': 0, 'R_gated_sim': 0}
    wins_sim = {'R_greedy': 0, 'R_weighted': 0, 'R_gated_sim': 0}

    for i in range(n_networks):
        r = run_test(seed=i * 1000 + 42)
        all_results.append(r)

        for m in wins_random:
            if r[m] > r['random']:
                wins_random[m] += 1
            if r[m] > r['similarity']:
                wins_sim[m] += 1

    return {
        'results': all_results,
        'wins_random': wins_random,
        'wins_sim': wins_sim,
        'n': n_networks
    }


if __name__ == "__main__":
    print("=" * 70)
    print("NETWORK NAVIGATION - Full R formula")
    print("=" * 70)
    print()
    print("R = (E / nabla_H) * sigma^Df")
    print("E = similarity, nabla_H = gradient from neighbors, sigma^Df = certainty")
    print()
    print("R rewards: HIGH similarity + CONSISTENT with neighbors + CERTAIN")
    print()

    # Analyze what R correlates with
    print("-" * 70)
    print("R correlation analysis (single network)")
    print("-" * 70)

    network = create_network(seed=42)
    R_scores = [compute_node_R(i, network) for i in range(network['n_nodes'])]
    sims = [np.dot(network['embeddings'][i], network['query']) for i in range(network['n_nodes'])]
    info = network['info_content']

    print(f"\nCorr(R, info_content):  {np.corrcoef(R_scores, info)[0,1]:.3f}")
    print(f"Corr(sim, info_content): {np.corrcoef(sims, info)[0,1]:.3f}")
    print(f"Corr(R, similarity):     {np.corrcoef(R_scores, sims)[0,1]:.3f}")

    # Multi-network test
    print("\n" + "-" * 70)
    print("10 networks, 100 trials each")
    print("-" * 70)

    test = test_multiple(10)

    print(f"\n{'Net':>4} | {'Random':>8} | {'Sim':>8} | {'R_greedy':>8} | {'R_weight':>8} | {'R_gated':>8}")
    print("-" * 60)

    for i, r in enumerate(test['results']):
        print(f"{i+1:>4} | {r['random']:>8.4f} | {r['similarity']:>8.4f} | "
              f"{r['R_greedy']:>8.4f} | {r['R_weighted']:>8.4f} | {r['R_gated_sim']:>8.4f}")

    print("-" * 60)
    avg = {k: np.mean([r[k] for r in test['results']]) for k in test['results'][0]}
    print(f"{'AVG':>4} | {avg['random']:>8.4f} | {avg['similarity']:>8.4f} | "
          f"{avg['R_greedy']:>8.4f} | {avg['R_weighted']:>8.4f} | {avg['R_gated_sim']:>8.4f}")

    print(f"\nWins vs RANDOM:")
    for m, w in test['wins_random'].items():
        print(f"  {m}: {w}/{test['n']}")

    print(f"\nWins vs SIMILARITY:")
    for m, w in test['wins_sim'].items():
        print(f"  {m}: {w}/{test['n']}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    best_vs_sim = max(test['wins_sim'].items(), key=lambda x: x[1])

    if best_vs_sim[1] >= 7:
        print(f"\n** FULL R: BEATS SIMILARITY")
        print(f"   {best_vs_sim[0]} beats similarity {best_vs_sim[1]}/10")
    elif best_vs_sim[1] >= 5:
        print(f"\n*  FULL R: MARGINAL vs similarity")
        print(f"   {best_vs_sim[0]} beats similarity {best_vs_sim[1]}/10")
    else:
        print(f"\nX  FULL R: DOES NOT BEAT SIMILARITY")
        print(f"   Best method only beats similarity {best_vs_sim[1]}/10")
