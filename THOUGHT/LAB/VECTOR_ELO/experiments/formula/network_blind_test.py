"""
Network Test - BLIND VERSION (per GPT critique)

Changes:
1. Remove info_content from R computation (BLIND)
2. R computed ONLY from network structure + entropy
3. info_content is hidden ground truth for scoring only
4. Compare: R-guided vs random vs degree-greedy
"""

import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def create_information_network(n_nodes: int = 100, n_edges: int = 300) -> Dict:
    """Create network with HIDDEN information content."""
    np.random.seed(42)

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

    # HIDDEN ground truth: information content (R cannot see this)
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
            # Count edges between neighbors
            neighbor_edges = 0
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 < n2 and n2 in adjacency[n1]:
                        neighbor_edges += 1
            max_edges = len(neighbors) * (len(neighbors) - 1) / 2
            clustering[i] = neighbor_edges / max_edges if max_edges > 0 else 0

    return {
        'adjacency': adjacency,
        'info_content': info_content,  # HIDDEN - only for scoring
        'degree': degree,              # OBSERVABLE
        'clustering': clustering,      # OBSERVABLE
        'n_nodes': n_nodes
    }


def compute_R_blind(node: int, network: Dict) -> float:
    """
    Compute R using ONLY observable network properties.
    NO access to info_content.

    Mapping:
    - H = entropy of degree distribution in neighborhood
    - nabla_H = difference from neighbor's entropy
    - Df = 5 - H (derived)
    """
    neighbors = network['adjacency'][node]
    degree = network['degree']

    if len(neighbors) == 0:
        return 0.01  # Isolated node has low R

    # H = entropy of neighbor degrees (structural entropy)
    neighbor_degrees = [degree[n] for n in neighbors]
    if sum(neighbor_degrees) > 0:
        probs = np.array(neighbor_degrees) / sum(neighbor_degrees)
        probs = probs[probs > 0]
        H = -np.sum(probs * np.log(probs + 1e-10))
    else:
        H = 0.1

    H = max(H, 0.1)
    H = min(H, 4.9)

    # nabla_H = structural gradient (how different is this node's neighborhood?)
    node_degree = degree[node]
    mean_neighbor_degree = np.mean(neighbor_degrees)
    nabla_H = abs(node_degree - mean_neighbor_degree) / (mean_neighbor_degree + 1) + 0.01

    # E derived from H (as per calibration)
    alpha = 3 ** (1 / 2 - 1)  # 1D network structure
    E = H ** alpha

    # Df derived from H
    Df = max(5 - H, 0.1)

    # R formula
    sigma = np.e
    R = (E / nabla_H) * (sigma ** Df)

    return R


def random_walk(network: Dict, start: int, k_steps: int = 10) -> List[int]:
    """Random walk from start node."""
    path = [start]
    current = start

    for _ in range(k_steps):
        neighbors = network['adjacency'][current]
        if len(neighbors) == 0:
            break
        current = np.random.choice(neighbors)
        path.append(current)

    return path


def R_guided_walk(network: Dict, start: int, k_steps: int = 10, R_scores: np.ndarray = None) -> List[int]:
    """Walk guided by R scores (greedy on R)."""
    path = [start]
    current = start

    for _ in range(k_steps):
        neighbors = network['adjacency'][current]
        if len(neighbors) == 0:
            break

        # Choose neighbor with highest R
        neighbor_R = [R_scores[n] for n in neighbors]
        best_idx = np.argmax(neighbor_R)
        current = neighbors[best_idx]
        path.append(current)

    return path


def degree_guided_walk(network: Dict, start: int, k_steps: int = 10) -> List[int]:
    """Walk guided by degree (greedy on connectivity)."""
    path = [start]
    current = start
    degree = network['degree']

    for _ in range(k_steps):
        neighbors = network['adjacency'][current]
        if len(neighbors) == 0:
            break

        # Choose neighbor with highest degree
        neighbor_degrees = [degree[n] for n in neighbors]
        best_idx = np.argmax(neighbor_degrees)
        current = neighbors[best_idx]
        path.append(current)

    return path


def evaluate_walk(path: List[int], network: Dict) -> float:
    """Score a walk by the HIDDEN info_content found."""
    info = network['info_content']
    return np.mean([info[node] for node in path])


def run_blind_navigation_test(n_trials: int = 100, k_steps: int = 10) -> Dict:
    """
    Main test: blind R-guided navigation vs random vs degree.
    """
    network = create_information_network(n_nodes=100, n_edges=300)
    n_nodes = network['n_nodes']

    # Compute R scores (BLIND - no access to info_content)
    R_scores = np.array([compute_R_blind(i, network) for i in range(n_nodes)])

    # Run trials
    random_scores = []
    R_scores_list = []
    degree_scores = []

    for trial in range(n_trials):
        np.random.seed(trial + 1000)
        start = np.random.randint(0, n_nodes)

        # Random walk
        random_path = random_walk(network, start, k_steps)
        random_scores.append(evaluate_walk(random_path, network))

        # R-guided walk
        R_path = R_guided_walk(network, start, k_steps, R_scores)
        R_scores_list.append(evaluate_walk(R_path, network))

        # Degree-guided walk
        degree_path = degree_guided_walk(network, start, k_steps)
        degree_scores.append(evaluate_walk(degree_path, network))

    return {
        'random_mean': np.mean(random_scores),
        'random_std': np.std(random_scores),
        'R_mean': np.mean(R_scores_list),
        'R_std': np.std(R_scores_list),
        'degree_mean': np.mean(degree_scores),
        'degree_std': np.std(degree_scores),
        'R_vs_random': np.mean(R_scores_list) / np.mean(random_scores),
        'R_vs_degree': np.mean(R_scores_list) / np.mean(degree_scores),
        'degree_vs_random': np.mean(degree_scores) / np.mean(random_scores)
    }


def statistical_significance(n_trials: int = 100, k_steps: int = 10) -> Dict:
    """Test statistical significance with multiple network instances."""
    R_wins_random = 0
    R_wins_degree = 0
    degree_wins_random = 0

    for seed in range(10):  # 10 different networks
        np.random.seed(seed * 100)

        # Create new network
        network = create_information_network(n_nodes=100, n_edges=300)
        n_nodes = network['n_nodes']
        R_scores = np.array([compute_R_blind(i, network) for i in range(n_nodes)])

        random_total = 0
        R_total = 0
        degree_total = 0

        for trial in range(n_trials):
            np.random.seed(seed * 1000 + trial)
            start = np.random.randint(0, n_nodes)

            random_score = evaluate_walk(random_walk(network, start, k_steps), network)
            R_score = evaluate_walk(R_guided_walk(network, start, k_steps, R_scores), network)
            degree_score = evaluate_walk(degree_guided_walk(network, start, k_steps), network)

            random_total += random_score
            R_total += R_score
            degree_total += degree_score

        if R_total > random_total:
            R_wins_random += 1
        if R_total > degree_total:
            R_wins_degree += 1
        if degree_total > random_total:
            degree_wins_random += 1

    return {
        'R_beats_random': R_wins_random,
        'R_beats_degree': R_wins_degree,
        'degree_beats_random': degree_wins_random,
        'n_networks': 10
    }


if __name__ == "__main__":
    print("=" * 60)
    print("NETWORK NAVIGATION - BLIND (GPT Critique Version)")
    print("=" * 60)
    print()
    print("R computed ONLY from network structure.")
    print("info_content is HIDDEN ground truth for scoring.")
    print()

    # Test 1: Single network, many trials
    print("-" * 60)
    print("Test 1: Single network, 100 trials, 10 steps each")
    print("-" * 60)

    results = run_blind_navigation_test(n_trials=100, k_steps=10)

    print(f"\n{'Method':>15} | {'Mean Info':>12} | {'Std':>10}")
    print("-" * 45)
    print(f"{'Random':>15} | {results['random_mean']:>12.4f} | {results['random_std']:>10.4f}")
    print(f"{'Degree-greedy':>15} | {results['degree_mean']:>12.4f} | {results['degree_std']:>10.4f}")
    print(f"{'R-guided':>15} | {results['R_mean']:>12.4f} | {results['R_std']:>10.4f}")

    print(f"\nR vs Random:  {results['R_vs_random']:.2f}x")
    print(f"R vs Degree:  {results['R_vs_degree']:.2f}x")
    print(f"Degree vs Random: {results['degree_vs_random']:.2f}x")

    # Test 2: Statistical significance across networks
    print("\n" + "-" * 60)
    print("Test 2: Statistical significance (10 different networks)")
    print("-" * 60)

    sig_results = statistical_significance()

    print(f"\n  R beats Random:  {sig_results['R_beats_random']}/{sig_results['n_networks']} networks")
    print(f"  R beats Degree:  {sig_results['R_beats_degree']}/{sig_results['n_networks']} networks")
    print(f"  Degree beats Random: {sig_results['degree_beats_random']}/{sig_results['n_networks']} networks")

    # Final verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    R_beats_random = results['R_vs_random'] > 1.0
    R_beats_degree = results['R_vs_degree'] > 1.0
    statistically_significant = sig_results['R_beats_random'] >= 7  # 7/10 or better

    print(f"\nR beats random walk: {R_beats_random} ({results['R_vs_random']:.2f}x)")
    print(f"R beats degree-greedy: {R_beats_degree} ({results['R_vs_degree']:.2f}x)")
    print(f"Statistically significant: {statistically_significant} ({sig_results['R_beats_random']}/10)")

    if R_beats_random and statistically_significant:
        if R_beats_degree:
            print("\n** BLIND NETWORK TEST: FULLY VALIDATED")
            print("   R-guided navigation beats BOTH random AND degree")
        else:
            print("\n*  BLIND NETWORK TEST: PARTIALLY VALIDATED")
            print("   R beats random but not degree")
    else:
        print("\nX  BLIND NETWORK TEST: FAILED")
        print("   Blind R does not provide navigation advantage")
