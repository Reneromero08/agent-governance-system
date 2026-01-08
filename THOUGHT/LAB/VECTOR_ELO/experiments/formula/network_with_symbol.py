"""
Network Test - WITH SYMBOL (actual E, not structural proxy)

The blind test failed because R had no signal - only structure.
Now: Give R access to a SYMBOL (embedding) that correlates with info.

This models AGS use case:
- We have embeddings that capture semantic content
- Embeddings are noisy signals of "true meaning"
- R uses embedding similarity as E (actual signal)

The embedding is NOT the same as info_content, but CORRELATES with it.
This is the realistic scenario.
"""

import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def create_network_with_symbols(
    n_nodes: int = 100,
    n_edges: int = 300,
    embedding_dim: int = 32,
    embedding_noise: float = 0.3,
    seed: int = None
) -> Dict:
    """
    Create network where each node has:
    - info_content (hidden ground truth)
    - embedding (observable symbol, correlates with info)
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

    # Hidden ground truth: information content
    info_content = np.random.pareto(1.5, n_nodes) + 0.1
    info_content = info_content / np.max(info_content)

    # SYMBOL: Embedding that correlates with info_content
    # High-info nodes have embeddings that cluster together
    # Low-info nodes have more random embeddings

    # Create a "true signal" direction in embedding space
    signal_direction = np.random.randn(embedding_dim)
    signal_direction = signal_direction / np.linalg.norm(signal_direction)

    embeddings = np.zeros((n_nodes, embedding_dim))
    for i in range(n_nodes):
        # Base: random embedding
        base = np.random.randn(embedding_dim)
        base = base / np.linalg.norm(base)

        # Signal: proportional to info_content, aligned with signal_direction
        signal = info_content[i] * signal_direction

        # Combine with noise
        embeddings[i] = signal + embedding_noise * base
        embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])

    # Query embedding (what we're looking for)
    # High-quality query aligned with signal direction
    query = signal_direction + 0.1 * np.random.randn(embedding_dim)
    query = query / np.linalg.norm(query)

    degree = np.array([len(adjacency[i]) for i in range(n_nodes)])

    return {
        'adjacency': adjacency,
        'info_content': info_content,  # Hidden
        'embeddings': embeddings,       # Observable SYMBOL
        'query': query,                 # Observable
        'degree': degree,
        'n_nodes': n_nodes
    }


def compute_R_with_symbol(node: int, network: Dict) -> float:
    """
    Compute R using SYMBOL (embedding similarity) as E.

    E = cosine similarity between node embedding and query
    nabla_H = entropy gradient (how different from neighbors)
    Df = 5 - H (derived)
    """
    embeddings = network['embeddings']
    query = network['query']
    neighbors = network['adjacency'][node]

    # E = embedding similarity to query (THIS IS THE SYMBOL)
    E = np.dot(embeddings[node], query)
    E = max(E, 0.01)  # Keep positive

    if len(neighbors) == 0:
        return E * 10  # Isolated high-similarity node

    # H = entropy of neighbor similarities
    neighbor_sims = [np.dot(embeddings[n], query) for n in neighbors]
    neighbor_sims = np.array(neighbor_sims)
    neighbor_sims = np.clip(neighbor_sims, 0.01, 1.0)

    if np.sum(neighbor_sims) > 0:
        probs = neighbor_sims / np.sum(neighbor_sims)
        H = -np.sum(probs * np.log(probs + 1e-10))
    else:
        H = 0.1

    H = max(H, 0.1)
    H = min(H, 4.9)

    # nabla_H = similarity gradient (am I more similar than neighbors?)
    my_sim = np.dot(embeddings[node], query)
    mean_neighbor_sim = np.mean(neighbor_sims)
    nabla_H = abs(my_sim - mean_neighbor_sim) + 0.01

    # Df derived from H
    Df = max(5 - H, 0.1)

    # R formula
    sigma = np.e
    R = (E / nabla_H) * (sigma ** Df)

    return R


def random_walk(network: Dict, start: int, k_steps: int, rng) -> List[int]:
    """Random walk."""
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
    """Walk guided by R scores."""
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


def similarity_guided_walk(network: Dict, start: int, k_steps: int) -> List[int]:
    """Walk guided by raw embedding similarity (baseline without R formula)."""
    path = [start]
    current = start
    embeddings = network['embeddings']
    query = network['query']

    for _ in range(k_steps):
        neighbors = network['adjacency'][current]
        if len(neighbors) == 0:
            break
        neighbor_sims = [np.dot(embeddings[n], query) for n in neighbors]
        best_idx = np.argmax(neighbor_sims)
        current = neighbors[best_idx]
        path.append(current)

    return path


def degree_guided_walk(network: Dict, start: int, k_steps: int) -> List[int]:
    """Walk guided by degree."""
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
    """Score by hidden info_content."""
    info = network['info_content']
    return np.mean([info[node] for node in path])


def run_comparison(network_seed: int, n_trials: int = 100, k_steps: int = 10) -> Dict:
    """Compare R-guided vs similarity-guided vs random vs degree."""
    network = create_network_with_symbols(seed=network_seed)
    n_nodes = network['n_nodes']

    # Compute R scores with symbol
    R_scores = np.array([compute_R_with_symbol(i, network) for i in range(n_nodes)])

    # Generate starts
    start_rng = np.random.default_rng(seed=network_seed + 1000)
    start_nodes = start_rng.integers(0, n_nodes, size=n_trials)
    walk_rng = np.random.default_rng(seed=network_seed + 2000)

    random_scores = []
    R_scores_list = []
    sim_scores_list = []
    degree_scores = []

    for start in start_nodes:
        random_scores.append(evaluate_walk(random_walk(network, start, k_steps, walk_rng), network))
        R_scores_list.append(evaluate_walk(R_guided_walk(network, start, k_steps, R_scores), network))
        sim_scores_list.append(evaluate_walk(similarity_guided_walk(network, start, k_steps), network))
        degree_scores.append(evaluate_walk(degree_guided_walk(network, start, k_steps), network))

    return {
        'random': np.mean(random_scores),
        'R_guided': np.mean(R_scores_list),
        'similarity': np.mean(sim_scores_list),
        'degree': np.mean(degree_scores),
        'R_vs_random': np.mean(R_scores_list) / np.mean(random_scores),
        'R_vs_sim': np.mean(R_scores_list) / np.mean(sim_scores_list),
        'sim_vs_random': np.mean(sim_scores_list) / np.mean(random_scores)
    }


def test_multiple_networks(n_networks: int = 10) -> Dict:
    """Test across different networks."""
    R_beats_random = 0
    R_beats_sim = 0
    sim_beats_random = 0

    all_results = []

    for net_idx in range(n_networks):
        result = run_comparison(network_seed=net_idx * 1000 + 42)
        all_results.append(result)

        if result['R_guided'] > result['random']:
            R_beats_random += 1
        if result['R_guided'] > result['similarity']:
            R_beats_sim += 1
        if result['similarity'] > result['random']:
            sim_beats_random += 1

    return {
        'R_beats_random': R_beats_random,
        'R_beats_similarity': R_beats_sim,
        'similarity_beats_random': sim_beats_random,
        'n_networks': n_networks,
        'results': all_results
    }


if __name__ == "__main__":
    print("=" * 70)
    print("NETWORK TEST - WITH SYMBOL (embedding as E)")
    print("=" * 70)
    print()
    print("R now has actual signal: embedding similarity to query")
    print("This models AGS use case (vectors with semantic content)")
    print()

    # Test
    print("-" * 70)
    print("Test: 10 different networks, 100 trials each")
    print("-" * 70)

    results = test_multiple_networks(n_networks=10)

    print(f"\n{'Net':>4} | {'Random':>10} | {'Degree':>10} | {'Sim':>10} | {'R-guided':>10} | {'R/Rand':>8} | {'R/Sim':>8}")
    print("-" * 80)

    for i, r in enumerate(results['results']):
        print(f"{i+1:>4} | {r['random']:>10.4f} | {r['degree']:>10.4f} | {r['similarity']:>10.4f} | {r['R_guided']:>10.4f} | {r['R_vs_random']:>8.2f}x | {r['R_vs_sim']:>8.2f}x")

    # Averages
    print("-" * 80)
    avg_rand = np.mean([r['random'] for r in results['results']])
    avg_deg = np.mean([r['degree'] for r in results['results']])
    avg_sim = np.mean([r['similarity'] for r in results['results']])
    avg_R = np.mean([r['R_guided'] for r in results['results']])
    print(f"{'AVG':>4} | {avg_rand:>10.4f} | {avg_deg:>10.4f} | {avg_sim:>10.4f} | {avg_R:>10.4f} | {avg_R/avg_rand:>8.2f}x | {avg_R/avg_sim:>8.2f}x")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  R beats Random:     {results['R_beats_random']}/{results['n_networks']} networks")
    print(f"  R beats Similarity: {results['R_beats_similarity']}/{results['n_networks']} networks")
    print(f"  Similarity beats Random: {results['similarity_beats_random']}/{results['n_networks']} networks")

    # Key question: Does R add value OVER raw similarity?
    print("\n" + "=" * 70)
    print("KEY QUESTION: Does R add value over raw similarity?")
    print("=" * 70)

    if results['R_beats_similarity'] >= 7:
        print("\n** YES: R formula adds value over raw embedding similarity")
        print("   The formula's structure (E/nabla_H * sigma^Df) helps")
    elif results['R_beats_similarity'] >= 5:
        print("\n*  MARGINAL: R sometimes beats raw similarity")
    else:
        print("\nX  NO: Raw similarity works as well or better than R")
        print("   The formula structure doesn't add value in this domain")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if results['R_beats_random'] >= 7:
        print("\n** WITH SYMBOL: R-GUIDED NAVIGATION VALIDATED")
        print(f"   R beats random: {results['R_beats_random']}/10")
        print(f"   Average improvement: {avg_R/avg_rand:.2f}x")
    else:
        print("\nX  WITH SYMBOL: STILL NOT VALIDATED")
        print(f"   R only beats random {results['R_beats_random']}/10")
