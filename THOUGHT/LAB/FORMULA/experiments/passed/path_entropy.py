"""
Path Entropy - H as journey entropy, not local

sigma^Df where H = entropy of the PATH so far

The archetype: accumulated information along the journey.

As you walk:
- Path entropy increases (more places visited)
- sigma^Df decreases (less certainty)
- At some point: STOP (diminishing returns)

The gate: YES if path still has low entropy (focused)
         NO if path has high entropy (wandering)
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


def path_entropy(path: List[int], network: Dict) -> float:
    """
    Entropy of the path based on similarities visited.

    Low entropy = focused path (similar nodes)
    High entropy = wandering path (diverse nodes)
    """
    if len(path) < 2:
        return 0.1

    sims = [np.dot(network['embeddings'][n], network['query']) for n in path]
    sims = np.clip(sims, 0.01, 1.0)

    # Normalize to probabilities
    probs = np.array(sims) / np.sum(sims)
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


def path_gated_walk(network: Dict, start: int, k: int, entropy_threshold: float = 2.0) -> List[int]:
    """
    Walk that stops when path entropy exceeds threshold.

    sigma^Df = e^(5-H) where H = path entropy
    When H > threshold, sigma^Df gets small = STOP

    After stopping, stay at best node found.
    """
    path = [start]
    current = start
    best_node = start
    best_sim = np.dot(network['embeddings'][start], network['query'])

    for step in range(k):
        # Check path entropy
        H = path_entropy(path, network)
        scale = sigma_Df(H)

        # If entropy too high (scale too low), stop exploring
        if H > entropy_threshold and step > 2:
            # Stay at best node
            path.append(best_node)
            continue

        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        # Pick highest similarity
        sims = [np.dot(network['embeddings'][n], network['query']) for n in neighbors]
        best_idx = np.argmax(sims)
        current = neighbors[best_idx]
        path.append(current)

        # Track best
        if sims[best_idx] > best_sim:
            best_sim = sims[best_idx]
            best_node = current

    return path


def adaptive_walk(network: Dict, start: int, k: int) -> List[int]:
    """
    Adapt step size by sigma^Df.

    High sigma^Df (low path entropy) = explore further
    Low sigma^Df (high path entropy) = exploit current best

    Implementation: probability of exploration vs exploitation
    """
    path = [start]
    current = start
    best_node = start
    best_sim = np.dot(network['embeddings'][start], network['query'])

    for step in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        # Path entropy and sigma^Df
        H = path_entropy(path, network)
        scale = sigma_Df(H)

        # Normalize scale to [0, 1] probability
        # High scale = explore, low scale = exploit
        explore_prob = min(scale / 150, 0.95)  # cap at 0.95

        if np.random.random() < explore_prob:
            # EXPLORE: follow similarity gradient
            sims = [np.dot(network['embeddings'][n], network['query']) for n in neighbors]
            current = neighbors[np.argmax(sims)]
        else:
            # EXPLOIT: return toward best node if possible
            # Find path back to best_node
            if best_node in neighbors:
                current = best_node
            else:
                # Can't get back, just pick highest similarity
                sims = [np.dot(network['embeddings'][n], network['query']) for n in neighbors]
                current = neighbors[np.argmax(sims)]

        path.append(current)

        # Update best
        current_sim = np.dot(network['embeddings'][current], network['query'])
        if current_sim > best_sim:
            best_sim = current_sim
            best_node = current

    return path


def R_guided_exploration(network: Dict, start: int, k: int) -> List[int]:
    """
    Use R = (E / nabla_H) * sigma^Df for step decisions.

    E = similarity gain from this step
    nabla_H = change in path entropy
    sigma^Df = scale based on current path entropy

    High R = good step (high gain, low entropy change, certain path)
    Low R = bad step (skip/backtrack)
    """
    path = [start]
    current = start
    prev_H = path_entropy([start], network)

    for step in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        # Evaluate each neighbor
        best_neighbor = None
        best_R = -float('inf')

        current_sim = np.dot(network['embeddings'][current], network['query'])

        for n in neighbors:
            # Hypothetical path
            hypo_path = path + [n]
            hypo_H = path_entropy(hypo_path, network)

            # E = similarity of neighbor
            E = np.dot(network['embeddings'][n], network['query'])
            E = max(E, 0.01)

            # nabla_H = entropy change (absolute)
            nabla_H = abs(hypo_H - prev_H) + 0.01

            # sigma^Df from current path
            sigma_df = sigma_Df(prev_H)

            # R formula
            R = (E / nabla_H) * sigma_df

            if R > best_R:
                best_R = R
                best_neighbor = n

        current = best_neighbor
        path.append(current)
        prev_H = path_entropy(path, network)

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
        'path_gated': [],
        'adaptive': [],
        'R_guided': []
    }

    for s in starts:
        results['random'].append(evaluate(random_walk(network, s, k, walk_rng), network))
        results['similarity'].append(evaluate(similarity_walk(network, s, k), network))
        results['path_gated'].append(evaluate(path_gated_walk(network, s, k), network))
        results['adaptive'].append(evaluate(adaptive_walk(network, s, k), network))
        results['R_guided'].append(evaluate(R_guided_exploration(network, s, k), network))

    return {key: np.mean(v) for key, v in results.items()}


if __name__ == "__main__":
    print("=" * 70)
    print("PATH ENTROPY - H as journey entropy")
    print("=" * 70)
    print()
    print("H = entropy of path (accumulated journey)")
    print("sigma^Df = e^(5-H) = exploration budget")
    print()

    # Multi-network test
    print("-" * 70)
    print("10 networks, 100 trials each")
    print("-" * 70)

    all_results = []
    wins = {'path_gated': 0, 'adaptive': 0, 'R_guided': 0}
    beats_sim = {'path_gated': 0, 'adaptive': 0, 'R_guided': 0}

    for i in range(10):
        r = run_test(seed=i * 1000 + 42)
        all_results.append(r)
        for m in wins:
            if r[m] > r['random']:
                wins[m] += 1
            if r[m] > r['similarity']:
                beats_sim[m] += 1

    print(f"\n{'Net':>4} | {'Random':>8} | {'Sim':>8} | {'PathGate':>8} | {'Adaptive':>8} | {'R_guided':>8}")
    print("-" * 65)

    for i, r in enumerate(all_results):
        print(f"{i+1:>4} | {r['random']:>8.4f} | {r['similarity']:>8.4f} | "
              f"{r['path_gated']:>8.4f} | {r['adaptive']:>8.4f} | {r['R_guided']:>8.4f}")

    print("-" * 65)
    avg = {k: np.mean([r[k] for r in all_results]) for k in all_results[0]}
    print(f"{'AVG':>4} | {avg['random']:>8.4f} | {avg['similarity']:>8.4f} | "
          f"{avg['path_gated']:>8.4f} | {avg['adaptive']:>8.4f} | {avg['R_guided']:>8.4f}")

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
        print(f"\n** PATH ENTROPY: BEATS SIMILARITY")
        print(f"   {best[0]} wins {best[1]}/10")
    else:
        print(f"\nX  PATH ENTROPY: DOES NOT BEAT SIMILARITY")
        print(f"   Best: {best[0]} with {best[1]}/10")
