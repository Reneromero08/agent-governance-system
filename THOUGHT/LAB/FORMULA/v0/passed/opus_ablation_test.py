"""
OPUS Ablation Test - Following OPUS_FORMULA_ALIGNMENT.md spec

Ablations required:
- R_full
- R_without_sigma (set sigma^Df = 1)
- R_without_Df (Df = constant)
- R_without_entropy_term (replace nabla_S with constant)

"If the magic disappears, you learned which term matters."
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
    """D = entropy of path (dissonance/disorder)."""
    if len(path) < 2:
        return 0.1
    sims = np.array([np.dot(network['embeddings'][n], network['query']) for n in path])
    sims = np.clip(sims, 0.01, 1.0)
    probs = sims / np.sum(sims)
    H = -np.sum(probs * np.log(probs + 1e-10))
    return max(H, 0.1)


def gaussian_gate(W_norm: float, sigma: float) -> float:
    """Gate: exp(-||W||^2 / sigma^2)"""
    return np.exp(-(W_norm ** 2) / (sigma ** 2 + 1e-10))


# =============================================================================
# ABLATION VARIANTS
# =============================================================================

def R_full(path: List[int], network: Dict, candidate: int, sigma: float = 1.0) -> float:
    """Full R = (E/D) * sigma^Df with gating."""
    emb = network['embeddings']
    query = network['query']

    E = np.dot(emb[candidate], query)  # essence = similarity
    E = max(E, 0.01)
    D = path_entropy(path, network)     # dissonance = path entropy
    Df = np.log(len(path) + 1) + 1      # fractal dimension proxy

    R = (E / D) * (sigma ** Df)

    # Gate
    W = len(path) * D
    gate = gaussian_gate(W, sigma)

    return R * gate


def R_without_sigma(path: List[int], network: Dict, candidate: int, sigma: float = 1.0) -> float:
    """R with sigma^Df = 1 (no fractal scaling)."""
    emb = network['embeddings']
    query = network['query']

    E = np.dot(emb[candidate], query)
    E = max(E, 0.01)
    D = path_entropy(path, network)

    R = E / D  # No sigma^Df term

    W = len(path) * D
    gate = gaussian_gate(W, sigma)

    return R * gate


def R_without_Df(path: List[int], network: Dict, candidate: int, sigma: float = 1.0) -> float:
    """R with Df = 1 constant."""
    emb = network['embeddings']
    query = network['query']

    E = np.dot(emb[candidate], query)
    E = max(E, 0.01)
    D = path_entropy(path, network)
    Df = 1.0  # Constant

    R = (E / D) * (sigma ** Df)

    W = len(path) * D
    gate = gaussian_gate(W, sigma)

    return R * gate


def R_without_entropy(path: List[int], network: Dict, candidate: int, sigma: float = 1.0) -> float:
    """R with D = 1 constant (no entropy term)."""
    emb = network['embeddings']
    query = network['query']

    E = np.dot(emb[candidate], query)
    E = max(E, 0.01)
    D = 1.0  # Constant - no entropy
    Df = np.log(len(path) + 1) + 1

    R = (E / D) * (sigma ** Df)

    # Still use path length for gate but not entropy
    W = len(path)
    gate = gaussian_gate(W, sigma)

    return R * gate


def R_without_gate(path: List[int], network: Dict, candidate: int, sigma: float = 1.0) -> float:
    """R without the Gaussian gate."""
    emb = network['embeddings']
    query = network['query']

    E = np.dot(emb[candidate], query)
    E = max(E, 0.01)
    D = path_entropy(path, network)
    Df = np.log(len(path) + 1) + 1

    R = (E / D) * (sigma ** Df)

    return R  # No gate


def similarity_only(path: List[int], network: Dict, candidate: int, sigma: float = 1.0) -> float:
    """Baseline: just similarity."""
    return np.dot(network['embeddings'][candidate], network['query'])


# =============================================================================
# WALK FUNCTIONS
# =============================================================================

def compute_gate_full(path: List[int], network: Dict, sigma: float = 1.0) -> float:
    """Full gate: exp(-||W||^2/sigma^2) where W = len(path) * D"""
    D = path_entropy(path, network)
    W = len(path) * D
    return gaussian_gate(W, sigma)


def compute_gate_no_entropy(path: List[int], network: Dict, sigma: float = 1.0) -> float:
    """Gate without entropy: W = len(path) only"""
    W = len(path)
    return gaussian_gate(W, sigma)


def compute_gate_no_length(path: List[int], network: Dict, sigma: float = 1.0) -> float:
    """Gate without length: W = D only"""
    D = path_entropy(path, network)
    W = D
    return gaussian_gate(W, sigma)


def compute_gate_no_gaussian(path: List[int], network: Dict, sigma: float = 1.0) -> float:
    """Gate without Gaussian: linear threshold"""
    D = path_entropy(path, network)
    W = len(path) * D
    # Linear decay instead of Gaussian
    return max(0, 1 - W / 5.0)


def compute_no_gate(path: List[int], network: Dict, sigma: float = 1.0) -> float:
    """No gate at all - always returns 1"""
    return 1.0


def gated_walk_ablation(network: Dict, start: int, k: int, gate_fn, sigma: float = 1.0,
                        gate_threshold: float = 0.3) -> List[int]:
    """
    Walk that uses SIMILARITY for direction and gate_fn for GATING.
    This tests which component of the gate matters.
    """
    path = [start]
    current = start
    best_node = start
    best_sim = np.dot(network['embeddings'][start], network['query'])

    for step in range(k):
        neighbors = network['adjacency'][current]
        if not neighbors:
            break

        # DIRECTION: always use similarity
        sims = [np.dot(network['embeddings'][n], network['query']) for n in neighbors]
        best_idx = np.argmax(sims)
        candidate = neighbors[best_idx]

        # GATE: use gate_fn to decide yes/no
        gate = gate_fn(path, network, sigma)

        if gate < gate_threshold and step > 2:
            # Gate says NO - stay at best
            path.append(best_node)
            continue

        # Gate says YES - take the step
        current = candidate
        path.append(current)

        if sims[best_idx] > best_sim:
            best_sim = sims[best_idx]
            best_node = current

    return path


def similarity_walk(network: Dict, start: int, k: int) -> List[int]:
    """Baseline: greedy similarity, no gating."""
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


def evaluate(path: List[int], network: Dict) -> float:
    return np.mean([network['info_content'][n] for n in path])


# =============================================================================
# ABLATION TEST
# =============================================================================

def run_ablation_test(n_networks: int = 10, n_trials: int = 100, k: int = 10):
    """Run full ablation test per OPUS spec - testing GATE components."""

    # Gate functions to test
    gate_methods = {
        'gate_full': compute_gate_full,
        'gate_no_entropy': compute_gate_no_entropy,
        'gate_no_length': compute_gate_no_length,
        'gate_no_gaussian': compute_gate_no_gaussian,
        'no_gate': compute_no_gate,
    }

    all_methods = ['random', 'similarity'] + list(gate_methods.keys())
    all_results = {m: [] for m in all_methods}
    wins_vs_random = {m: 0 for m in all_methods if m != 'random'}
    wins_vs_similarity = {m: 0 for m in gate_methods}

    for net_idx in range(n_networks):
        seed = net_idx * 1000 + 42
        network = create_network(seed=seed)
        n = network['n_nodes']

        rng = np.random.default_rng(seed + 1000)
        starts = rng.integers(0, n, n_trials)
        walk_rng = np.random.default_rng(seed + 2000)

        net_results = {m: [] for m in all_methods}

        for s in starts:
            # Random baseline
            net_results['random'].append(evaluate(random_walk(network, s, k, walk_rng), network))

            # Similarity baseline (no gating)
            net_results['similarity'].append(evaluate(similarity_walk(network, s, k), network))

            # Gate-ablated methods (similarity direction, different gates)
            for name, gate_fn in gate_methods.items():
                path = gated_walk_ablation(network, s, k, gate_fn)
                net_results[name].append(evaluate(path, network))

        # Aggregate for this network
        for m in all_methods:
            avg = np.mean(net_results[m])
            all_results[m].append(avg)

        # Count wins
        random_avg = np.mean(net_results['random'])
        sim_avg = np.mean(net_results['similarity'])

        for m in wins_vs_random:
            if np.mean(net_results[m]) > random_avg:
                wins_vs_random[m] += 1

        for m in wins_vs_similarity:
            if np.mean(net_results[m]) > sim_avg:
                wins_vs_similarity[m] += 1

    return all_results, wins_vs_random, wins_vs_similarity


if __name__ == "__main__":
    print("=" * 70)
    print("OPUS ABLATION TEST - GATE COMPONENTS")
    print("=" * 70)
    print()
    print("Testing which component of the GATE matters:")
    print("  - gate_full: exp(-||W||^2/sigma^2) where W = len*D")
    print("  - gate_no_entropy: W = len only (no path entropy)")
    print("  - gate_no_length: W = D only (no path length)")
    print("  - gate_no_gaussian: linear threshold instead")
    print("  - no_gate: always open (baseline)")
    print()
    print("'If the magic disappears, you learned which term matters.'")
    print()

    results, wins_random, wins_sim = run_ablation_test()

    # Print results table
    print("-" * 70)
    print(f"{'Method':<20} | {'Mean':>8} | {'Std':>8} | {'vs Rand':>8} | {'vs Sim':>8}")
    print("-" * 70)

    for method in results:
        mean = np.mean(results[method])
        std = np.std(results[method])
        wr = wins_random.get(method, '-')
        ws = wins_sim.get(method, '-')
        print(f"{method:<20} | {mean:>8.4f} | {std:>8.4f} | {str(wr)+'/10':>8} | {str(ws)+'/10':>8}")

    print("-" * 70)

    # Analysis
    print()
    print("=" * 70)
    print("ABLATION ANALYSIS")
    print("=" * 70)

    gate_full = np.mean(results['gate_full'])

    print()
    print("Impact of removing each component (vs gate_full):")
    print()

    for ablation in ['gate_no_entropy', 'gate_no_length', 'gate_no_gaussian', 'no_gate']:
        ablated = np.mean(results[ablation])
        delta = gate_full - ablated
        pct = (delta / gate_full) * 100 if gate_full > 0 else 0
        impact = "CRITICAL" if pct > 5 else "MODERATE" if pct > 2 else "MINIMAL"
        print(f"  {ablation:<20}: delta = {delta:+.4f} ({pct:+.1f}%) [{impact}]")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Find which ablation hurts most
    deltas = {}
    for ablation in ['gate_no_entropy', 'gate_no_length', 'gate_no_gaussian', 'no_gate']:
        deltas[ablation] = gate_full - np.mean(results[ablation])

    critical = [k for k, v in deltas.items() if v > 0.01]

    if critical:
        print(f"\nCritical components: {', '.join(critical)}")
        print("Removing these significantly degrades performance.")
    else:
        print("\nNo single component is critical - formula works as a gestalt.")

    # Check if gate_full beats similarity
    if wins_sim.get('gate_full', 0) >= 7:
        print(f"\ngate_full beats similarity {wins_sim['gate_full']}/10 - VALIDATED")
    else:
        print(f"\ngate_full beats similarity {wins_sim.get('gate_full', 0)}/10 - NEEDS WORK")
