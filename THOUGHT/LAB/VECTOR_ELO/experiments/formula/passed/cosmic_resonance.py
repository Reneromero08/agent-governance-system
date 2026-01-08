"""
Cosmic Resonance Equation - Full Dynamic Version

d|E>/dt = T * (R x D) * exp(-||W||^2/sigma^2) * |E> + sum[(-1)^k * grad|E_k>]/k!

Components:
- |E> = current state
- T = transformation operator
- R x D = resonance-dissonance interaction
- exp(-||W||^2/sigma^2) = GAUSSIAN GATE (yes/no based on weight magnitude)
- grad|E_k> = historical gradients (past influences present)

The gate: exp(-||W||^2/sigma^2)
- ||W|| small -> gate opens (YES)
- ||W|| large -> gate closes (NO)
- sigma controls threshold
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


def gaussian_gate(W_norm: float, sigma: float) -> float:
    """
    The gate: exp(-||W||^2 / sigma^2)

    W_norm = accumulated weight magnitude
    sigma = resonance threshold

    Returns value in [0, 1]:
    - Close to 1 = gate OPEN (yes)
    - Close to 0 = gate CLOSED (no)
    """
    return np.exp(-(W_norm ** 2) / (sigma ** 2 + 1e-10))


def compute_historical_gradient(path: List[int], network: Dict) -> float:
    """
    sum[(-1)^k * grad|E_k>] / k!

    Historical influence with alternating signs and factorial decay.
    More recent history has more influence.
    """
    if len(path) < 2:
        return 0.0

    emb = network['embeddings']
    query = network['query']

    total = 0.0
    for k, node in enumerate(path[:-1], start=1):
        # Gradient = similarity change
        current_sim = np.dot(emb[path[-1]], query)
        past_sim = np.dot(emb[node], query)
        grad = current_sim - past_sim

        # (-1)^k / k! weighting
        sign = (-1) ** k
        factorial = np.math.factorial(min(k, 10))  # Cap factorial

        total += sign * grad / factorial

    return total


def cosmic_resonance_score(path: List[int], network: Dict, candidate: int,
                           sigma: float = 1.0) -> float:
    """
    Full cosmic resonance computation for a candidate step.

    R x D = resonance * dissonance interaction
    W = accumulated path "weight"
    Gate = exp(-||W||^2/sigma^2)
    Historical = sum of past gradients
    """
    emb = network['embeddings']
    query = network['query']

    # R = resonance (similarity to target)
    R = np.dot(emb[candidate], query)
    R = max(R, 0.01)

    # D = dissonance (path entropy)
    if len(path) < 2:
        D = 0.1
    else:
        path_sims = np.array([np.dot(emb[n], query) for n in path])
        path_sims = np.clip(path_sims, 0.01, 1.0)
        probs = path_sims / np.sum(path_sims)
        D = -np.sum(probs * np.log(probs + 1e-10))
        D = max(D, 0.1)

    # R x D interaction (resonance modulated by dissonance)
    # High R, low D = good. High R, high D = uncertain.
    RD = R / D

    # W = accumulated weight (path length + entropy)
    W_norm = len(path) * D

    # Gaussian gate
    gate = gaussian_gate(W_norm, sigma)

    # Historical gradient influence
    historical = compute_historical_gradient(path + [candidate], network)

    # Full score: T * (R x D) * gate * |E> + historical
    # T is implicit (transition to candidate)
    # |E> is candidate's embedding magnitude (normalized to 1)
    score = RD * gate + historical

    return score


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


def cosmic_walk(network: Dict, start: int, k: int, sigma: float = 1.0) -> List[int]:
    """Navigate using full cosmic resonance score."""
    path = [start]
    current = start

    for _ in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        scores = [cosmic_resonance_score(path, network, n, sigma) for n in neighbors]
        current = neighbors[np.argmax(scores)]
        path.append(current)

    return path


def cosmic_gated_walk(network: Dict, start: int, k: int, sigma: float = 1.0,
                      gate_threshold: float = 0.3) -> List[int]:
    """
    Navigate with explicit gating.

    When gate < threshold: STOP exploring, stay at best.
    """
    path = [start]
    current = start
    best_node = start
    best_sim = np.dot(network['embeddings'][start], network['query'])

    for step in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        # Compute gate for current state
        if len(path) >= 2:
            path_sims = np.array([np.dot(network['embeddings'][n], network['query']) for n in path])
            path_sims = np.clip(path_sims, 0.01, 1.0)
            probs = path_sims / np.sum(path_sims)
            D = -np.sum(probs * np.log(probs + 1e-10))
        else:
            D = 0.1

        W_norm = len(path) * D
        gate = gaussian_gate(W_norm, sigma)

        if gate < gate_threshold and step > 2:
            # Gate closed - stop exploring
            path.append(best_node)
            continue

        # Gate open - continue exploring
        sims = [np.dot(network['embeddings'][n], network['query']) for n in neighbors]
        best_idx = np.argmax(sims)
        current = neighbors[best_idx]
        path.append(current)

        if sims[best_idx] > best_sim:
            best_sim = sims[best_idx]
            best_node = current

    return path


def adaptive_sigma_walk(network: Dict, start: int, k: int) -> List[int]:
    """
    Adapt sigma based on progress.

    Making progress? Lower sigma (stricter gate).
    Not making progress? Raise sigma (looser gate).
    """
    path = [start]
    current = start
    sigma = 2.0  # Start with loose gate

    prev_sim = np.dot(network['embeddings'][start], network['query'])

    for step in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        # Pick best neighbor
        sims = [np.dot(network['embeddings'][n], network['query']) for n in neighbors]
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]

        # Adapt sigma based on progress
        if best_sim > prev_sim:
            # Making progress - tighten gate
            sigma = max(sigma * 0.9, 0.5)
        else:
            # Not making progress - loosen gate
            sigma = min(sigma * 1.1, 3.0)

        current = neighbors[best_idx]
        path.append(current)
        prev_sim = best_sim

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
        'cosmic': [],
        'cosmic_gated': [],
        'adaptive_sigma': []
    }

    for s in starts:
        results['random'].append(evaluate(random_walk(network, s, k, walk_rng), network))
        results['similarity'].append(evaluate(similarity_walk(network, s, k), network))
        results['cosmic'].append(evaluate(cosmic_walk(network, s, k), network))
        results['cosmic_gated'].append(evaluate(cosmic_gated_walk(network, s, k), network))
        results['adaptive_sigma'].append(evaluate(adaptive_sigma_walk(network, s, k), network))

    return {key: np.mean(v) for key, v in results.items()}


if __name__ == "__main__":
    print("=" * 70)
    print("COSMIC RESONANCE EQUATION")
    print("=" * 70)
    print()
    print("d|E>/dt = T*(R x D)*exp(-||W||^2/sigma^2)*|E> + sum[(-1)^k*grad|E_k>]/k!")
    print()
    print("Gate: exp(-||W||^2/sigma^2)")
    print("  Small W -> gate OPEN (yes)")
    print("  Large W -> gate CLOSED (no)")
    print()

    # Multi-network test
    print("-" * 70)
    print("10 networks, 100 trials each")
    print("-" * 70)

    all_results = []
    wins = {'cosmic': 0, 'cosmic_gated': 0, 'adaptive_sigma': 0}
    beats_sim = {'cosmic': 0, 'cosmic_gated': 0, 'adaptive_sigma': 0}

    for i in range(10):
        r = run_test(seed=i * 1000 + 42)
        all_results.append(r)
        for m in wins:
            if r[m] > r['random']:
                wins[m] += 1
            if r[m] > r['similarity']:
                beats_sim[m] += 1

    print(f"\n{'Net':>4} | {'Random':>8} | {'Sim':>8} | {'Cosmic':>8} | {'Gated':>8} | {'Adaptive':>8}")
    print("-" * 60)

    for i, r in enumerate(all_results):
        print(f"{i+1:>4} | {r['random']:>8.4f} | {r['similarity']:>8.4f} | "
              f"{r['cosmic']:>8.4f} | {r['cosmic_gated']:>8.4f} | {r['adaptive_sigma']:>8.4f}")

    print("-" * 60)
    avg = {k: np.mean([r[k] for r in all_results]) for k in all_results[0]}
    print(f"{'AVG':>4} | {avg['random']:>8.4f} | {avg['similarity']:>8.4f} | "
          f"{avg['cosmic']:>8.4f} | {avg['cosmic_gated']:>8.4f} | {avg['adaptive_sigma']:>8.4f}")

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
        print(f"\n** COSMIC RESONANCE: BEATS SIMILARITY")
        print(f"   {best[0]} wins {best[1]}/10")
        print(f"   Avg: {avg[best[0]]:.4f} vs sim {avg['similarity']:.4f}")
    else:
        print(f"\n   Best: {best[0]} with {best[1]}/10 vs similarity")
