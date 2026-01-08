"""
Formula v2 - Correct interpretation from source

R = (E/D) * f^Df

E = essence (the signal, what you want)
D = dissonance (uncertainty, noise, fluctuations)
f = information content (NOT Euler's e!)
Df = fractal dimension (complexity scaling)

R = resonance factor (dimensionless quality measure)
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

    return {
        'adjacency': adjacency,
        'info_content': info_content,
        'embeddings': embeddings,
        'query': query,
        'n_nodes': n_nodes
    }


def compute_R_v2(path: List[int], network: Dict, candidate: int) -> float:
    """
    R = (E/D) * f^Df

    For navigation:
    E = similarity of candidate to query (essence - what we want)
    D = path entropy (dissonance - accumulated uncertainty)
    f = information content of candidate (embedding magnitude or similarity)
    Df = fractal dimension (could be path length or complexity measure)
    """
    emb = network['embeddings']
    query = network['query']

    # E = essence = similarity to what we're looking for
    E = np.dot(emb[candidate], query)
    E = max(E, 0.01)

    # D = dissonance = path uncertainty
    if len(path) < 2:
        D = 0.1
    else:
        path_sims = [np.dot(emb[n], query) for n in path]
        path_sims = np.clip(path_sims, 0.01, 1.0)
        probs = np.array(path_sims) / np.sum(path_sims)
        D = -np.sum(probs * np.log(probs + 1e-10))  # entropy
        D = max(D, 0.1)

    # f = information content of candidate
    # Could be: similarity, or some other measure
    f = np.dot(emb[candidate], query)
    f = max(f, 0.01)

    # Df = fractal dimension
    # For a path: could be related to how "space-filling" the path is
    # Simple: use log(path_length) as proxy for dimension
    Df = np.log(len(path) + 1) + 1  # ranges from 1 to ~3 for paths up to 10

    # R = (E/D) * f^Df
    R = (E / D) * (f ** Df)

    return R


def compute_R_v2_simple(similarity: float, dissonance: float, info: float, complexity: float) -> float:
    """
    Simplified R computation.

    R = (E/D) * f^Df
    """
    E = max(similarity, 0.01)
    D = max(dissonance, 0.01)
    f = max(info, 0.01)
    Df = max(complexity, 0.1)

    return (E / D) * (f ** Df)


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


def R_v2_walk(network: Dict, start: int, k: int) -> List[int]:
    """Navigate using R = (E/D) * f^Df"""
    path = [start]
    current = start

    for _ in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        # Score each neighbor with R_v2
        R_scores = [compute_R_v2(path, network, n) for n in neighbors]
        current = neighbors[np.argmax(R_scores)]
        path.append(current)

    return path


def R_v2_gated_walk(network: Dict, start: int, k: int, R_threshold: float = 1.0) -> List[int]:
    """
    Gate navigation by R.

    If best R > threshold: take the step
    If best R < threshold: stop (stay at current best)
    """
    path = [start]
    current = start
    best_node = start
    best_sim = np.dot(network['embeddings'][start], network['query'])

    for step in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        # Score each neighbor
        R_scores = [compute_R_v2(path, network, n) for n in neighbors]
        best_R = max(R_scores)
        best_neighbor = neighbors[np.argmax(R_scores)]

        # Gate: is this step worth taking?
        if best_R >= R_threshold or step < 3:  # Always take first few steps
            current = best_neighbor
            path.append(current)

            # Track overall best
            current_sim = np.dot(network['embeddings'][current], network['query'])
            if current_sim > best_sim:
                best_sim = current_sim
                best_node = current
        else:
            # R too low - stop exploring, stay at best
            path.append(best_node)

    return path


def adaptive_R_walk(network: Dict, start: int, k: int) -> List[int]:
    """
    Adapt behavior based on R.

    High R = explore (take the step)
    Low R = exploit (return to best known)
    """
    path = [start]
    current = start
    best_node = start
    best_sim = np.dot(network['embeddings'][start], network['query'])

    R_history = []

    for step in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        # Score neighbors
        R_scores = [compute_R_v2(path, network, n) for n in neighbors]
        best_R = max(R_scores)
        best_neighbor = neighbors[np.argmax(R_scores)]

        R_history.append(best_R)

        # Adaptive threshold based on history
        if len(R_history) > 3:
            R_median = np.median(R_history)
            threshold = R_median * 0.5
        else:
            threshold = 0

        if best_R >= threshold:
            # Good R - explore
            current = best_neighbor
        else:
            # Low R - go back to best if possible
            if best_node in neighbors:
                current = best_node
            else:
                current = best_neighbor

        path.append(current)

        # Update best
        current_sim = np.dot(network['embeddings'][current], network['query'])
        if current_sim > best_sim:
            best_sim = current_sim
            best_node = current

    return path


def evaluate(path: List[int], network: Dict) -> float:
    return np.mean([network['info_content'][n] for n in path])


def run_test(seed: int, n_trials: int = 100, k: int = 10) -> Dict:
    network = create_network(seed=seed)
    n = network['n_nodes']

    rng = np.random.default_rng(seed + 1000)
    starts = rng.integers(0, n, n_trials)
    walk_rng = np.random.default_rng(seed + 2000)

    results = {
        'random': [],
        'similarity': [],
        'R_v2': [],
        'R_v2_gated': [],
        'adaptive_R': []
    }

    for s in starts:
        results['random'].append(evaluate(random_walk(network, s, k, walk_rng), network))
        results['similarity'].append(evaluate(similarity_walk(network, s, k), network))
        results['R_v2'].append(evaluate(R_v2_walk(network, s, k), network))
        results['R_v2_gated'].append(evaluate(R_v2_gated_walk(network, s, k), network))
        results['adaptive_R'].append(evaluate(adaptive_R_walk(network, s, k), network))

    return {key: np.mean(v) for key, v in results.items()}


if __name__ == "__main__":
    print("=" * 70)
    print("FORMULA v2: R = (E/D) * f^Df")
    print("=" * 70)
    print()
    print("E = essence (similarity to target)")
    print("D = dissonance (path entropy)")
    print("f = information content")
    print("Df = fractal dimension (path complexity)")
    print()

    # Multi-network test
    print("-" * 70)
    print("10 networks, 100 trials each")
    print("-" * 70)

    all_results = []
    wins = {'R_v2': 0, 'R_v2_gated': 0, 'adaptive_R': 0}
    beats_sim = {'R_v2': 0, 'R_v2_gated': 0, 'adaptive_R': 0}

    for i in range(10):
        r = run_test(seed=i * 1000 + 42)
        all_results.append(r)
        for m in wins:
            if r[m] > r['random']:
                wins[m] += 1
            if r[m] > r['similarity']:
                beats_sim[m] += 1

    print(f"\n{'Net':>4} | {'Random':>8} | {'Sim':>8} | {'R_v2':>8} | {'Gated':>8} | {'Adaptive':>8}")
    print("-" * 60)

    for i, r in enumerate(all_results):
        print(f"{i+1:>4} | {r['random']:>8.4f} | {r['similarity']:>8.4f} | "
              f"{r['R_v2']:>8.4f} | {r['R_v2_gated']:>8.4f} | {r['adaptive_R']:>8.4f}")

    print("-" * 60)
    avg = {k: np.mean([r[k] for r in all_results]) for k in all_results[0]}
    print(f"{'AVG':>4} | {avg['random']:>8.4f} | {avg['similarity']:>8.4f} | "
          f"{avg['R_v2']:>8.4f} | {avg['R_v2_gated']:>8.4f} | {avg['adaptive_R']:>8.4f}")

    print(f"\nWins vs random:")
    for m, w in wins.items():
        print(f"  {m}: {w}/10")

    print(f"\nWins vs similarity:")
    for m, w in beats_sim.items():
        print(f"  {m}: {w}/10")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    best = max(beats_sim.items(), key=lambda x: x[1])
    if best[1] >= 7:
        print(f"\n** FORMULA v2: BEATS SIMILARITY")
        print(f"   {best[0]} wins {best[1]}/10")
        print(f"   Avg: {avg[best[0]]:.4f} vs sim {avg['similarity']:.4f}")
    else:
        print(f"\nX  FORMULA v2: Does not consistently beat similarity")
        print(f"   Best: {best[0]} with {best[1]}/10")
