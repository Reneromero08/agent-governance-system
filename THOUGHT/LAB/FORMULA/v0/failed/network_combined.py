"""
Network Navigation - Combined approach

Embeddings provide SIGNAL (E)
sigma^Df provides CERTAINTY weight

Combined: score = similarity * sigma^Df

High similarity + high certainty = GO
High similarity + low certainty = MAYBE
Low similarity = NO
"""

import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def create_network_with_embeddings(n_nodes: int = 100, n_edges: int = 300,
                                    embedding_dim: int = 32, seed: int = None) -> Dict:
    """Network with embeddings that correlate with hidden info."""
    if seed is not None:
        np.random.seed(seed)

    # Graph
    edges = set()
    while len(edges) < n_edges:
        i, j = np.random.randint(0, n_nodes, 2)
        if i != j:
            edges.add((min(i, j), max(i, j)))

    adjacency = {i: [] for i in range(n_nodes)}
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    # Hidden info
    info_content = np.random.pareto(1.5, n_nodes) + 0.1
    info_content = info_content / np.max(info_content)

    # Embeddings that correlate with info
    signal_dir = np.random.randn(embedding_dim)
    signal_dir /= np.linalg.norm(signal_dir)

    embeddings = np.zeros((n_nodes, embedding_dim))
    for i in range(n_nodes):
        base = np.random.randn(embedding_dim)
        base /= np.linalg.norm(base)
        signal = info_content[i] * signal_dir
        embeddings[i] = signal + 0.3 * base
        embeddings[i] /= np.linalg.norm(embeddings[i])

    # Query
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


def neighborhood_entropy(node: int, network: Dict) -> float:
    """Entropy from neighbor similarities."""
    neighbors = network['adjacency'][node]
    if not neighbors:
        return 4.9

    sims = [np.dot(network['embeddings'][n], network['query']) for n in neighbors]
    sims = np.clip(sims, 0.01, 1.0)

    if np.sum(sims) < 1e-10:
        return 4.9

    probs = sims / np.sum(sims)
    H = -np.sum(probs * np.log(probs + 1e-10))
    return np.clip(H, 0.1, 4.9)


def sigma_Df(H: float) -> float:
    return np.e ** (5 - np.clip(H, 0.1, 4.9))


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
    """Greedy on raw similarity."""
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


def combined_walk(network: Dict, start: int, k: int) -> List[int]:
    """
    Greedy on: similarity * sigma^Df

    This combines:
    - Signal (similarity to query)
    - Certainty (sigma^Df from neighborhood entropy)
    """
    path = [start]
    current = start

    for _ in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        scores = []
        for n in neighbors:
            sim = np.dot(network['embeddings'][n], network['query'])
            H = neighborhood_entropy(n, network)
            certainty = sigma_Df(H)
            # Normalize certainty to [0, 1] range roughly
            certainty_norm = certainty / 150  # typical range is ~50-150
            scores.append(sim * certainty_norm)

        current = neighbors[np.argmax(scores)]
        path.append(current)

    return path


def combined_weighted_walk(network: Dict, start: int, k: int, rng) -> List[int]:
    """
    Probabilistic: weight by similarity * sigma^Df
    """
    path = [start]
    current = start

    for _ in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        weights = []
        for n in neighbors:
            sim = max(np.dot(network['embeddings'][n], network['query']), 0.01)
            H = neighborhood_entropy(n, network)
            certainty = sigma_Df(H) / 150
            weights.append(sim * certainty)

        weights = np.array(weights)
        probs = weights / np.sum(weights)
        current = rng.choice(neighbors, p=probs)
        path.append(current)

    return path


def sigma_gated_similarity(network: Dict, start: int, k: int, rng, threshold: float = 0.7) -> List[int]:
    """
    Gate by sigma^Df, then pick by similarity.

    Only consider neighbors with high sigma^Df (certain).
    Among those, pick highest similarity.
    """
    path = [start]
    current = start

    for _ in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        # Compute sigma^Df
        sigmas = [sigma_Df(neighborhood_entropy(n, network)) for n in neighbors]
        median_sigma = np.median(sigmas)

        # Gate: keep those above threshold
        valid = [(n, np.dot(network['embeddings'][n], network['query']))
                 for n, s in zip(neighbors, sigmas) if s >= threshold * median_sigma]

        if valid:
            # Pick highest similarity among valid
            current = max(valid, key=lambda x: x[1])[0]
        else:
            # Fallback: highest similarity overall
            sims = [np.dot(network['embeddings'][n], network['query']) for n in neighbors]
            current = neighbors[np.argmax(sims)]

        path.append(current)

    return path


def evaluate(path: List[int], network: Dict) -> float:
    return np.mean([network['info_content'][n] for n in path])


def run_test(seed: int, n_trials: int = 100, k: int = 10) -> Dict:
    network = create_network_with_embeddings(seed=seed)
    n = network['n_nodes']

    start_rng = np.random.default_rng(seed + 1000)
    starts = start_rng.integers(0, n, n_trials)
    walk_rng = np.random.default_rng(seed + 2000)

    results = {
        'random': [],
        'similarity': [],
        'combined_greedy': [],
        'combined_weighted': [],
        'sigma_gated_sim': []
    }

    for s in starts:
        results['random'].append(evaluate(random_walk(network, s, k, walk_rng), network))
        results['similarity'].append(evaluate(similarity_walk(network, s, k), network))
        results['combined_greedy'].append(evaluate(combined_walk(network, s, k), network))
        results['combined_weighted'].append(evaluate(combined_weighted_walk(network, s, k, walk_rng), network))
        results['sigma_gated_sim'].append(evaluate(sigma_gated_similarity(network, s, k, walk_rng), network))

    return {k: np.mean(v) for k, v in results.items()}


def test_multiple(n_networks: int = 10):
    all_results = []
    wins = {
        'combined_greedy': 0,
        'combined_weighted': 0,
        'sigma_gated_sim': 0
    }
    beats_similarity = {
        'combined_greedy': 0,
        'combined_weighted': 0,
        'sigma_gated_sim': 0
    }

    for i in range(n_networks):
        r = run_test(seed=i * 1000 + 42)
        all_results.append(r)

        for method in wins:
            if r[method] > r['random']:
                wins[method] += 1
            if r[method] > r['similarity']:
                beats_similarity[method] += 1

    return {'results': all_results, 'wins': wins, 'beats_sim': beats_similarity, 'n': n_networks}


if __name__ == "__main__":
    print("=" * 70)
    print("NETWORK NAVIGATION - Combined (similarity * sigma^Df)")
    print("=" * 70)
    print()
    print("Embedding similarity = SIGNAL (where to go)")
    print("sigma^Df = CERTAINTY (whether to trust)")
    print("Combined = signal * certainty")
    print()

    # Multi-network test
    print("-" * 70)
    print("10 networks, 100 trials each")
    print("-" * 70)

    test = test_multiple(10)

    print(f"\n{'Net':>4} | {'Random':>8} | {'Sim':>8} | {'Combined':>8} | {'Weighted':>8} | {'Gated':>8}")
    print("-" * 65)

    for i, r in enumerate(test['results']):
        print(f"{i+1:>4} | {r['random']:>8.4f} | {r['similarity']:>8.4f} | "
              f"{r['combined_greedy']:>8.4f} | {r['combined_weighted']:>8.4f} | {r['sigma_gated_sim']:>8.4f}")

    print("-" * 65)
    avg = {k: np.mean([r[k] for r in test['results']]) for k in test['results'][0]}
    print(f"{'AVG':>4} | {avg['random']:>8.4f} | {avg['similarity']:>8.4f} | "
          f"{avg['combined_greedy']:>8.4f} | {avg['combined_weighted']:>8.4f} | {avg['sigma_gated_sim']:>8.4f}")

    print(f"\nWin rates vs RANDOM:")
    for m in test['wins']:
        print(f"  {m}: {test['wins'][m]}/{test['n']}")

    print(f"\nWin rates vs SIMILARITY:")
    for m in test['beats_sim']:
        print(f"  {m}: {test['beats_sim'][m]}/{test['n']}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    best_vs_random = max(test['wins'].items(), key=lambda x: x[1])
    best_vs_sim = max(test['beats_sim'].items(), key=lambda x: x[1])

    if best_vs_sim[1] >= 7:
        print(f"\n** COMBINED: BEATS RAW SIMILARITY")
        print(f"   {best_vs_sim[0]} beats similarity {best_vs_sim[1]}/10")
        print(f"   sigma^Df adds value to raw embeddings!")
    elif best_vs_random[1] >= 7:
        print(f"\n*  COMBINED: BEATS RANDOM but not similarity")
        print(f"   {best_vs_random[0]} beats random {best_vs_random[1]}/10")
    else:
        print(f"\nX  COMBINED: NOT VALIDATED")
        print(f"   sigma^Df doesn't improve over raw similarity")
