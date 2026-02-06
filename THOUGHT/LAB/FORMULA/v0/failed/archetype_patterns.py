"""
Archetype Patterns - sigma^Df as pattern recognition

"Symbols are archetypes" - not numbers, patterns.

sigma^Df = e^(5-H) where H is neighborhood entropy

This identifies PATTERNS in neighborhoods:
- Low H (low entropy) = clear pattern (one dominant similarity)
- High H (high entropy) = no clear pattern (uniform similarities)

Archetypal patterns for navigation:
1. "BEACON" - node with high sim, neighbors with low sim (stands out)
2. "PATH" - gradient of similarities pointing toward target
3. "HUB" - uniformly high similarities (promising region)
4. "NOISE" - random similarities (avoid)

Can sigma^Df distinguish these?
"""

import numpy as np
from typing import Dict, List, Tuple
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


def classify_neighborhood_pattern(node: int, network: Dict) -> Tuple[str, float]:
    """
    Classify the archetypal pattern of a node's neighborhood.

    Returns (pattern_name, confidence)
    """
    neighbors = network['adjacency'][node]
    if not neighbors:
        return ('ISOLATED', 1.0)

    emb = network['embeddings']
    query = network['query']

    node_sim = np.dot(emb[node], query)
    neighbor_sims = np.array([np.dot(emb[n], query) for n in neighbors])

    # Statistics
    mean_n = np.mean(neighbor_sims)
    std_n = np.std(neighbor_sims)
    max_n = np.max(neighbor_sims)
    min_n = np.min(neighbor_sims)

    # Pattern detection
    if node_sim > mean_n + std_n and std_n < 0.1:
        # Node stands out above uniform neighbors
        return ('BEACON', node_sim - mean_n)

    if max_n - min_n > 0.3 and max_n > node_sim:
        # Clear gradient pointing somewhere
        return ('PATH', max_n - min_n)

    if mean_n > 0.5 and std_n < 0.15:
        # Uniformly high = good region
        return ('HUB', mean_n)

    if std_n > 0.2:
        # High variance = noise
        return ('NOISE', std_n)

    return ('UNKNOWN', 0.5)


def compute_sigma_df(node: int, network: Dict) -> float:
    """sigma^Df = e^(5-H)"""
    neighbors = network['adjacency'][node]
    if not neighbors:
        return np.e ** 5

    neighbor_sims = np.array([np.dot(network['embeddings'][n], network['query'])
                              for n in neighbors])
    neighbor_sims = np.clip(neighbor_sims, 0.01, 1.0)

    probs = neighbor_sims / np.sum(neighbor_sims)
    H = -np.sum(probs * np.log(probs + 1e-10))
    H = np.clip(H, 0.1, 4.9)

    return np.e ** (5 - H)


def analyze_patterns(network: Dict):
    """Analyze sigma^Df by pattern type."""
    patterns = {}
    sigma_by_pattern = {}
    info_by_pattern = {}

    for node in range(network['n_nodes']):
        pattern, conf = classify_neighborhood_pattern(node, network)
        sigma = compute_sigma_df(node, network)
        info = network['info_content'][node]

        if pattern not in patterns:
            patterns[pattern] = 0
            sigma_by_pattern[pattern] = []
            info_by_pattern[pattern] = []

        patterns[pattern] += 1
        sigma_by_pattern[pattern].append(sigma)
        info_by_pattern[pattern].append(info)

    return patterns, sigma_by_pattern, info_by_pattern


def pattern_guided_walk(network: Dict, start: int, k: int) -> List[int]:
    """
    Navigate using pattern recognition.

    Priority:
    1. PATH neighbors (follow the gradient)
    2. BEACON neighbors (isolated high signal)
    3. HUB neighbors (good region)
    4. Avoid NOISE
    """
    path = [start]
    current = start

    pattern_priority = {'PATH': 4, 'BEACON': 3, 'HUB': 2, 'UNKNOWN': 1, 'NOISE': 0, 'ISOLATED': 0}

    for _ in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        # Score each neighbor by pattern
        scored = []
        for n in neighbors:
            pattern, conf = classify_neighborhood_pattern(n, network)
            sim = np.dot(network['embeddings'][n], network['query'])
            priority = pattern_priority.get(pattern, 1)
            score = priority * 0.3 + sim * 0.7  # Blend pattern and similarity
            scored.append((n, score, pattern))

        # Pick best
        best = max(scored, key=lambda x: x[1])
        current = best[0]
        path.append(current)

    return path


def sigma_pattern_walk(network: Dict, start: int, k: int) -> List[int]:
    """
    Use sigma^Df to identify archetypal (reliable) neighbors.

    High sigma^Df = clear pattern = trustworthy
    Then pick highest similarity among trustworthy neighbors.
    """
    path = [start]
    current = start

    for _ in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        # Compute sigma^Df for each neighbor
        sigmas = [compute_sigma_df(n, network) for n in neighbors]
        sims = [np.dot(network['embeddings'][n], network['query']) for n in neighbors]

        # Find "archetypal" neighbors (high sigma^Df = clear pattern)
        sigma_threshold = np.percentile(sigmas, 70)  # Top 30%
        archetypal = [(n, s) for n, sig, s in zip(neighbors, sigmas, sims)
                      if sig >= sigma_threshold]

        if archetypal:
            # Among archetypal, pick highest similarity
            current = max(archetypal, key=lambda x: x[1])[0]
        else:
            # Fallback to highest similarity
            current = neighbors[np.argmax(sims)]

        path.append(current)

    return path


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
        'pattern': [],
        'sigma_pattern': []
    }

    for s in starts:
        results['random'].append(evaluate(random_walk(network, s, k, walk_rng), network))
        results['similarity'].append(evaluate(similarity_walk(network, s, k), network))
        results['pattern'].append(evaluate(pattern_guided_walk(network, s, k), network))
        results['sigma_pattern'].append(evaluate(sigma_pattern_walk(network, s, k), network))

    return {k: np.mean(v) for k, v in results.items()}


if __name__ == "__main__":
    print("=" * 70)
    print("ARCHETYPE PATTERNS - sigma^Df as pattern recognition")
    print("=" * 70)
    print()
    print("Patterns: BEACON (stands out), PATH (gradient), HUB (good region), NOISE")
    print()

    # Analyze patterns in single network
    print("-" * 70)
    print("Pattern analysis (single network)")
    print("-" * 70)

    network = create_network(seed=42)
    patterns, sigma_by_pattern, info_by_pattern = analyze_patterns(network)

    print(f"\n{'Pattern':>10} | {'Count':>6} | {'Avg sigma^Df':>12} | {'Avg Info':>10}")
    print("-" * 50)

    for p in sorted(patterns.keys()):
        avg_sigma = np.mean(sigma_by_pattern[p])
        avg_info = np.mean(info_by_pattern[p])
        print(f"{p:>10} | {patterns[p]:>6} | {avg_sigma:>12.1f} | {avg_info:>10.4f}")

    # Multi-network test
    print("\n" + "-" * 70)
    print("10 networks, 100 trials each")
    print("-" * 70)

    all_results = []
    wins = {'pattern': 0, 'sigma_pattern': 0}
    beats_sim = {'pattern': 0, 'sigma_pattern': 0}

    for i in range(10):
        r = run_test(seed=i * 1000 + 42)
        all_results.append(r)
        for m in wins:
            if r[m] > r['random']:
                wins[m] += 1
            if r[m] > r['similarity']:
                beats_sim[m] += 1

    print(f"\n{'Net':>4} | {'Random':>8} | {'Sim':>8} | {'Pattern':>8} | {'Sigma_Pat':>10}")
    print("-" * 50)

    for i, r in enumerate(all_results):
        print(f"{i+1:>4} | {r['random']:>8.4f} | {r['similarity']:>8.4f} | "
              f"{r['pattern']:>8.4f} | {r['sigma_pattern']:>10.4f}")

    print("-" * 50)
    avg = {k: np.mean([r[k] for r in all_results]) for k in all_results[0]}
    print(f"{'AVG':>4} | {avg['random']:>8.4f} | {avg['similarity']:>8.4f} | "
          f"{avg['pattern']:>8.4f} | {avg['sigma_pattern']:>10.4f}")

    print(f"\nWins vs random: pattern={wins['pattern']}/10, sigma_pattern={wins['sigma_pattern']}/10")
    print(f"Wins vs similarity: pattern={beats_sim['pattern']}/10, sigma_pattern={beats_sim['sigma_pattern']}/10")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    best = max(beats_sim.items(), key=lambda x: x[1])
    if best[1] >= 7:
        print(f"\n** PATTERN RECOGNITION: BEATS SIMILARITY")
        print(f"   {best[0]} wins {best[1]}/10")
    elif best[1] >= 5:
        print(f"\n*  PATTERN RECOGNITION: MARGINAL")
    else:
        print(f"\nX  PATTERN RECOGNITION: DOES NOT BEAT SIMILARITY")
