"""
Network Test - BLIND VERSION (GPT critique FIXED)

Fixes:
1. Remove internal np.random.seed(42) from create_information_network()
2. Generate start nodes ahead of time, use same starts for all methods
3. No per-trial reseeding during walks
"""

import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def create_information_network(n_nodes: int = 100, n_edges: int = 300, seed: int = None) -> Dict:
    """
    Create network with HIDDEN information content.
    Seed is passed in from caller - NO internal seeding.
    """
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

    # HIDDEN ground truth: information content
    info_content = np.random.pareto(1.5, n_nodes) + 0.1
    info_content = info_content / np.max(info_content)

    # OBSERVABLE: degree of each node
    degree = np.array([len(adjacency[i]) for i in range(n_nodes)])

    # OBSERVABLE: local clustering coefficient
    clustering = np.zeros(n_nodes)
    for i in range(n_nodes):
        neighbors = adjacency[i]
        if len(neighbors) < 2:
            clustering[i] = 0
        else:
            neighbor_edges = 0
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 < n2 and n2 in adjacency[n1]:
                        neighbor_edges += 1
            max_edges = len(neighbors) * (len(neighbors) - 1) / 2
            clustering[i] = neighbor_edges / max_edges if max_edges > 0 else 0

    return {
        'adjacency': adjacency,
        'info_content': info_content,
        'degree': degree,
        'clustering': clustering,
        'n_nodes': n_nodes
    }


def compute_R_blind(node: int, network: Dict) -> float:
    """Compute R using ONLY observable network properties."""
    neighbors = network['adjacency'][node]
    degree = network['degree']

    if len(neighbors) == 0:
        return 0.01

    neighbor_degrees = [degree[n] for n in neighbors]
    if sum(neighbor_degrees) > 0:
        probs = np.array(neighbor_degrees) / sum(neighbor_degrees)
        probs = probs[probs > 0]
        H = -np.sum(probs * np.log(probs + 1e-10))
    else:
        H = 0.1

    H = max(H, 0.1)
    H = min(H, 4.9)

    node_degree = degree[node]
    mean_neighbor_degree = np.mean(neighbor_degrees)
    nabla_H = abs(node_degree - mean_neighbor_degree) / (mean_neighbor_degree + 1) + 0.01

    alpha = 3 ** (1 / 2 - 1)
    E = H ** alpha
    Df = max(5 - H, 0.1)

    sigma = np.e
    R = (E / nabla_H) * (sigma ** Df)

    return R


def random_walk(network: Dict, start: int, k_steps: int, rng: np.random.Generator) -> List[int]:
    """Random walk using provided RNG (no internal seeding)."""
    path = [start]
    current = start

    for _ in range(k_steps):
        neighbors = network['adjacency'][current]
        if len(neighbors) == 0:
            break
        current = rng.choice(neighbors)
        path.append(current)

    return path


def R_guided_walk(network: Dict, start: int, k_steps: int, R_scores: np.ndarray) -> List[int]:
    """Deterministic walk guided by R scores."""
    path = [start]
    current = start

    for _ in range(k_steps):
        neighbors = network['adjacency'][current]
        if len(neighbors) == 0:
            break
        neighbor_R = [R_scores[n] for n in neighbors]
        best_idx = np.argmax(neighbor_R)
        current = neighbors[best_idx]
        path.append(current)

    return path


def degree_guided_walk(network: Dict, start: int, k_steps: int) -> List[int]:
    """Deterministic walk guided by degree."""
    path = [start]
    current = start
    degree = network['degree']

    for _ in range(k_steps):
        neighbors = network['adjacency'][current]
        if len(neighbors) == 0:
            break
        neighbor_degrees = [degree[n] for n in neighbors]
        best_idx = np.argmax(neighbor_degrees)
        current = neighbors[best_idx]
        path.append(current)

    return path


def evaluate_walk(path: List[int], network: Dict) -> float:
    """Score a walk by the HIDDEN info_content found."""
    info = network['info_content']
    return np.mean([info[node] for node in path])


def run_fair_comparison(network_seed: int, n_trials: int = 100, k_steps: int = 10) -> Dict:
    """
    Fair comparison:
    - Single network with given seed
    - Generate all start nodes FIRST
    - Use SAME starts for all methods
    - Random walk uses its own RNG, seeded once
    """
    # Create network with explicit seed
    network = create_information_network(n_nodes=100, n_edges=300, seed=network_seed)
    n_nodes = network['n_nodes']

    # Compute R scores (BLIND)
    R_scores = np.array([compute_R_blind(i, network) for i in range(n_nodes)])

    # Generate all start nodes AHEAD of time
    start_rng = np.random.default_rng(seed=network_seed + 1000)
    start_nodes = start_rng.integers(0, n_nodes, size=n_trials)

    # Separate RNG for random walks (seeded once, not per trial)
    walk_rng = np.random.default_rng(seed=network_seed + 2000)

    random_scores = []
    R_scores_list = []
    degree_scores = []

    for start in start_nodes:
        # All methods use SAME start
        random_path = random_walk(network, start, k_steps, walk_rng)
        R_path = R_guided_walk(network, start, k_steps, R_scores)
        degree_path = degree_guided_walk(network, start, k_steps)

        random_scores.append(evaluate_walk(random_path, network))
        R_scores_list.append(evaluate_walk(R_path, network))
        degree_scores.append(evaluate_walk(degree_path, network))

    return {
        'random_mean': np.mean(random_scores),
        'R_mean': np.mean(R_scores_list),
        'degree_mean': np.mean(degree_scores),
        'R_vs_random': np.mean(R_scores_list) / np.mean(random_scores),
        'R_vs_degree': np.mean(R_scores_list) / np.mean(degree_scores),
        'degree_vs_random': np.mean(degree_scores) / np.mean(random_scores)
    }


def test_multiple_networks(n_networks: int = 10, n_trials: int = 100) -> Dict:
    """
    Test across ACTUALLY DIFFERENT networks.
    Each network gets a unique seed.
    """
    R_beats_random = 0
    R_beats_degree = 0
    degree_beats_random = 0

    all_results = []

    for net_idx in range(n_networks):
        # Each network is TRULY DIFFERENT (unique seed)
        network_seed = net_idx * 1000 + 42

        result = run_fair_comparison(network_seed, n_trials=n_trials)
        all_results.append(result)

        if result['R_mean'] > result['random_mean']:
            R_beats_random += 1
        if result['R_mean'] > result['degree_mean']:
            R_beats_degree += 1
        if result['degree_mean'] > result['random_mean']:
            degree_beats_random += 1

    return {
        'R_beats_random': R_beats_random,
        'R_beats_degree': R_beats_degree,
        'degree_beats_random': degree_beats_random,
        'n_networks': n_networks,
        'all_results': all_results
    }


if __name__ == "__main__":
    print("=" * 60)
    print("NETWORK BLIND TEST - FIXED (GPT critique addressed)")
    print("=" * 60)
    print()
    print("Fixes applied:")
    print("1. No internal seeding in create_information_network()")
    print("2. Same start nodes for all methods")
    print("3. Single RNG for random walks (no per-trial reseeding)")
    print()

    # Test on multiple TRULY DIFFERENT networks
    print("-" * 60)
    print("Test: 10 DIFFERENT networks, 100 trials each")
    print("-" * 60)

    results = test_multiple_networks(n_networks=10, n_trials=100)

    print(f"\n{'Network':>8} | {'Random':>10} | {'Degree':>10} | {'R-guided':>10} | {'R/Rand':>8} | {'R/Deg':>8}")
    print("-" * 70)

    for i, r in enumerate(results['all_results']):
        print(f"{i+1:>8} | {r['random_mean']:>10.4f} | {r['degree_mean']:>10.4f} | {r['R_mean']:>10.4f} | {r['R_vs_random']:>8.2f}x | {r['R_vs_degree']:>8.2f}x")

    # Aggregate
    print("-" * 70)
    avg_random = np.mean([r['random_mean'] for r in results['all_results']])
    avg_degree = np.mean([r['degree_mean'] for r in results['all_results']])
    avg_R = np.mean([r['R_mean'] for r in results['all_results']])
    print(f"{'AVG':>8} | {avg_random:>10.4f} | {avg_degree:>10.4f} | {avg_R:>10.4f} | {avg_R/avg_random:>8.2f}x | {avg_R/avg_degree:>8.2f}x")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n  R beats Random:  {results['R_beats_random']}/{results['n_networks']} networks")
    print(f"  R beats Degree:  {results['R_beats_degree']}/{results['n_networks']} networks")
    print(f"  Degree beats Random: {results['degree_beats_random']}/{results['n_networks']} networks")

    # Final verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if results['R_beats_random'] >= 7:
        if results['R_beats_degree'] >= 7:
            print("\n** BLIND NETWORK TEST: FULLY VALIDATED")
            print(f"   R beats random {results['R_beats_random']}/10, beats degree {results['R_beats_degree']}/10")
        else:
            print("\n*  BLIND NETWORK TEST: PARTIALLY VALIDATED")
            print(f"   R beats random {results['R_beats_random']}/10, but NOT consistently better than degree")
    else:
        print("\nX  BLIND NETWORK TEST: NOT VALIDATED")
        print(f"   R only beats random {results['R_beats_random']}/10 times")
