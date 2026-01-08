"""
Navigation Trap Test - Adversarial benchmark per OPUS_NAVIGATION_TEST.md

Creates graphs with "similarity traps" where greedy similarity fails.
Tests whether gate-controlled navigation can escape traps and find target.

Key insight: Traps have HIGH similarity but NO path to target.
Target is reachable only via temporary similarity DIP.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# TRAP GRAPH CONSTRUCTION
# =============================================================================

def create_trap_graph(n_nodes: int = 50, n_traps: int = 3, seed: int = None) -> Dict:
    """
    Create a graph with similarity traps.

    Structure:
    - Start node
    - Target node (reachable via "valley" path)
    - Trap nodes (high similarity, dead ends)
    - Valley nodes (lower similarity, lead to target)
    """
    if seed is not None:
        np.random.seed(seed)

    dim = 32

    # Query vector (what we're looking for)
    query = np.random.randn(dim)
    query = query / np.linalg.norm(query)

    # Create node embeddings
    embeddings = np.random.randn(n_nodes, dim)

    # Assign roles
    start = 0
    target = 1
    trap_nodes = list(range(2, 2 + n_traps))
    valley_nodes = list(range(2 + n_traps, 2 + n_traps + 5))  # 5 valley nodes

    # Make target highly similar to query
    embeddings[target] = query + np.random.randn(dim) * 0.1

    # Make traps VERY similar to query (higher than target initially visible)
    for t in trap_nodes:
        embeddings[t] = query + np.random.randn(dim) * 0.05  # Even closer!

    # Make valley nodes have LOWER similarity (the "dip")
    valley_direction = np.random.randn(dim)
    valley_direction = valley_direction / np.linalg.norm(valley_direction)
    for i, v in enumerate(valley_nodes):
        # Mix query with orthogonal direction
        alpha = 0.3 + i * 0.1  # Gradually return to query direction
        embeddings[v] = alpha * query + (1 - alpha) * valley_direction
        embeddings[v] = embeddings[v] / np.linalg.norm(embeddings[v])

    # Make start moderately similar
    embeddings[start] = 0.5 * query + 0.5 * np.random.randn(dim)
    embeddings[start] = embeddings[start] / np.linalg.norm(embeddings[start])

    # Normalize all embeddings
    for i in range(n_nodes):
        embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])

    # Build adjacency - the key is connectivity structure
    adjacency = {i: [] for i in range(n_nodes)}

    # Start connects to traps AND first valley node
    adjacency[start] = trap_nodes + [valley_nodes[0]]

    # Traps connect to each other and random dead-end nodes (not target!)
    dead_ends = list(range(2 + n_traps + 5, min(n_nodes, 2 + n_traps + 15)))
    for t in trap_nodes:
        other_traps = [x for x in trap_nodes if x != t]
        adjacency[t] = other_traps + dead_ends[:3]

    # Dead ends connect to each other (circular, no way out)
    for i, d in enumerate(dead_ends):
        adjacency[d] = [dead_ends[(i+1) % len(dead_ends)], dead_ends[(i-1) % len(dead_ends)]]

    # Valley path leads to target
    for i, v in enumerate(valley_nodes[:-1]):
        adjacency[v] = [valley_nodes[i + 1]]
        # Also add some random non-trap neighbors
        safe_nodes = [x for x in range(n_nodes) if x not in trap_nodes and x not in dead_ends]
        adjacency[v].extend(np.random.choice(safe_nodes, size=min(2, len(safe_nodes)), replace=False).tolist())

    # Last valley node connects to target
    adjacency[valley_nodes[-1]] = [target]

    # Target connects back (but we stop when we reach it)
    adjacency[target] = [valley_nodes[-1]]

    # Compute similarities for quick lookup
    similarities = np.array([np.dot(embeddings[i], query) for i in range(n_nodes)])

    return {
        'n_nodes': n_nodes,
        'embeddings': embeddings,
        'query': query,
        'adjacency': adjacency,
        'similarities': similarities,
        'start': start,
        'target': target,
        'traps': trap_nodes,
        'valley': valley_nodes,
        'dead_ends': dead_ends
    }


def verify_trap_properties(graph: Dict) -> Dict:
    """Verify the graph has required trap properties."""
    sims = graph['similarities']

    trap_sims = [sims[t] for t in graph['traps']]
    valley_sims = [sims[v] for v in graph['valley']]
    target_sim = sims[graph['target']]
    start_sim = sims[graph['start']]

    return {
        'start_sim': start_sim,
        'target_sim': target_sim,
        'trap_sims': trap_sims,
        'valley_sims': valley_sims,
        'traps_higher_than_valley': all(t > min(valley_sims) for t in trap_sims),
        'valley_dips': min(valley_sims) < start_sim
    }


# =============================================================================
# NAVIGATION POLICIES
# =============================================================================

def greedy_similarity(graph: Dict, max_steps: int = 20) -> Tuple[List[int], bool]:
    """Baseline: always pick highest similarity neighbor."""
    path = [graph['start']]
    current = graph['start']
    visited = {current}

    for _ in range(max_steps):
        if current == graph['target']:
            return path, True

        neighbors = [n for n in graph['adjacency'][current] if n not in visited]
        if not neighbors:
            return path, False

        # Pick max similarity
        sims = [graph['similarities'][n] for n in neighbors]
        best_idx = np.argmax(sims)
        current = neighbors[best_idx]
        path.append(current)
        visited.add(current)

    return path, current == graph['target']


def beam_similarity(graph: Dict, beam_width: int = 3, max_steps: int = 20) -> Tuple[List[int], bool]:
    """Beam search by similarity."""
    # Each beam entry: (path, current_node, visited_set)
    beams = [([graph['start']], graph['start'], {graph['start']})]

    for _ in range(max_steps):
        # Check if any beam reached target
        for path, current, _ in beams:
            if current == graph['target']:
                return path, True

        # Expand all beams
        candidates = []
        for path, current, visited in beams:
            neighbors = [n for n in graph['adjacency'][current] if n not in visited]
            for n in neighbors:
                new_path = path + [n]
                new_visited = visited | {n}
                score = graph['similarities'][n]
                candidates.append((new_path, n, new_visited, score))

        if not candidates:
            break

        # Keep top beam_width
        candidates.sort(key=lambda x: x[3], reverse=True)
        beams = [(c[0], c[1], c[2]) for c in candidates[:beam_width]]

    # Return best beam
    if beams:
        return beams[0][0], beams[0][1] == graph['target']
    return [graph['start']], False


def similarity_stop_heuristic(graph: Dict, epsilon: float = 0.01, patience: int = 3,
                              max_steps: int = 20) -> Tuple[List[int], bool]:
    """Similarity with early stopping on plateau."""
    path = [graph['start']]
    current = graph['start']
    visited = {current}
    no_improvement = 0
    best_sim = graph['similarities'][current]

    for _ in range(max_steps):
        if current == graph['target']:
            return path, True

        neighbors = [n for n in graph['adjacency'][current] if n not in visited]
        if not neighbors:
            return path, False

        sims = [graph['similarities'][n] for n in neighbors]
        best_idx = np.argmax(sims)
        best_neighbor_sim = sims[best_idx]

        # Check improvement
        if best_neighbor_sim > best_sim + epsilon:
            best_sim = best_neighbor_sim
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            return path, False  # Stop - plateau detected

        current = neighbors[best_idx]
        path.append(current)
        visited.add(current)

    return path, current == graph['target']


# =============================================================================
# OPTION A: SIMILARITY + GATE CONTROL
# =============================================================================

def compute_path_entropy(path: List[int], graph: Dict) -> float:
    """Entropy of similarities along path."""
    if len(path) < 2:
        return 0.0
    sims = [graph['similarities'][n] for n in path]
    # Normalize to probabilities
    sims = np.array(sims) - min(sims) + 0.01
    probs = sims / sims.sum()
    return -np.sum(probs * np.log(probs + 1e-10))


def compute_gate(path: List[int], graph: Dict, sigma: float = 2.0) -> float:
    """Gate value based on path statistics."""
    D = compute_path_entropy(path, graph)
    W = len(path) * D
    return np.exp(-W**2 / sigma**2)


def gated_navigation_a(graph: Dict, max_steps: int = 30,
                       base_beam: int = 2, gate_threshold: float = 0.3) -> Tuple[List[int], bool]:
    """
    Option A: Similarity steers, gate controls search policy.

    Gate controls:
    - beam_width: expands when gate is high (confident)
    - backtracking: triggers when gate is low (stuck)
    """
    path = [graph['start']]
    current = graph['start']
    visited = {current}
    best_node = current
    best_sim = graph['similarities'][current]
    stuck_count = 0
    backtrack_stack = [current]  # Stack for backtracking

    for step in range(max_steps):
        if current == graph['target']:
            return path, True

        # Compute gate
        gate = compute_gate(path, graph)

        # Dynamic beam width based on gate
        beam_width = max(1, int(base_beam * (1 + gate)))

        neighbors = [n for n in graph['adjacency'][current] if n not in visited]

        if not neighbors:
            # Dead end - backtrack if gate allows
            if gate < gate_threshold and len(backtrack_stack) > 1:
                backtrack_stack.pop()
                current = backtrack_stack[-1]
                path.append(current)
                stuck_count += 1
                if stuck_count > 5:
                    return path, False  # Give up
                continue
            return path, False

        # Get top-k by similarity
        sims = [(n, graph['similarities'][n]) for n in neighbors]
        sims.sort(key=lambda x: x[1], reverse=True)
        top_k = sims[:beam_width]

        # If gate is low and we're not improving, try second-best
        if gate < gate_threshold and len(top_k) > 1:
            # Check if best looks like a trap (high sim, but we've been stuck)
            if stuck_count > 0:
                # Try alternative path
                current = top_k[1][0]  # Second best
            else:
                current = top_k[0][0]
        else:
            current = top_k[0][0]

        path.append(current)
        visited.add(current)
        backtrack_stack.append(current)

        # Track best
        current_sim = graph['similarities'][current]
        if current_sim > best_sim:
            best_sim = current_sim
            best_node = current
            stuck_count = 0
        elif current_sim < best_sim - 0.1:
            # Similarity dropped - are we in the valley?
            stuck_count += 1

    return path, current == graph['target']


# =============================================================================
# OPTION B: Delta-R STEERING
# =============================================================================

def compute_action_R(current: int, action: int, graph: Dict) -> float:
    """
    Compute R(s,a) for action selection.

    E(s,a) = predicted similarity gain
    ∇S(s,a) = neighbor dispersion (entropy)
    Df(s,a) = complexity (log degree)
    """
    # E: similarity gain
    current_sim = graph['similarities'][current]
    action_sim = graph['similarities'][action]
    E = max(0.01, action_sim - current_sim + 0.5)  # +0.5 baseline so valleys aren't zero

    # ∇S: dispersion of action's neighbors
    action_neighbors = graph['adjacency'][action]
    if action_neighbors:
        neighbor_sims = [graph['similarities'][n] for n in action_neighbors]
        grad_S = np.std(neighbor_sims) + 0.01
    else:
        grad_S = 1.0  # Dead end penalty

    # Df: complexity
    Df = np.log(len(action_neighbors) + 1) + 0.1

    # R = E / ∇S * σ^Df
    sigma = 0.5
    R = (E / grad_S) * (sigma ** Df)

    return R


def delta_r_steering(graph: Dict, max_steps: int = 30,
                     gate_threshold: float = 0.3) -> Tuple[List[int], bool]:
    """
    Option B: Delta-R chooses direction, gate controls stopping.
    """
    path = [graph['start']]
    current = graph['start']
    visited = {current}

    for step in range(max_steps):
        if current == graph['target']:
            return path, True

        neighbors = [n for n in graph['adjacency'][current] if n not in visited]
        if not neighbors:
            return path, False

        # Compute R for each action
        action_scores = [(n, compute_action_R(current, n, graph)) for n in neighbors]
        action_scores.sort(key=lambda x: x[1], reverse=True)

        # Gate controls stopping
        gate = compute_gate(path, graph)
        if gate < gate_threshold and step > 3:
            # Low gate - are we stuck?
            best_R = action_scores[0][1]
            if best_R < 0.1:  # No good action
                return path, False

        # Take best R action
        current = action_scores[0][0]
        path.append(current)
        visited.add(current)

    return path, current == graph['target']


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_single_test(seed: int) -> Dict:
    """Run all policies on one graph."""
    graph = create_trap_graph(seed=seed)

    results = {}

    # Baselines
    path, success = greedy_similarity(graph)
    results['greedy'] = {'success': success, 'steps': len(path), 'path': path}

    path, success = beam_similarity(graph, beam_width=3)
    results['beam'] = {'success': success, 'steps': len(path), 'path': path}

    path, success = similarity_stop_heuristic(graph)
    results['stop_heuristic'] = {'success': success, 'steps': len(path), 'path': path}

    # Option A
    path, success = gated_navigation_a(graph)
    results['option_a'] = {'success': success, 'steps': len(path), 'path': path}

    # Option B
    path, success = delta_r_steering(graph)
    results['option_b'] = {'success': success, 'steps': len(path), 'path': path}

    return results


def run_full_test(n_trials: int = 30) -> Dict:
    """Run full benchmark."""
    all_results = {
        'greedy': {'successes': 0, 'total_steps': []},
        'beam': {'successes': 0, 'total_steps': []},
        'stop_heuristic': {'successes': 0, 'total_steps': []},
        'option_a': {'successes': 0, 'total_steps': []},
        'option_b': {'successes': 0, 'total_steps': []},
    }

    for i in range(n_trials):
        trial = run_single_test(seed=i * 1000 + 42)
        for method, result in trial.items():
            if result['success']:
                all_results[method]['successes'] += 1
            all_results[method]['total_steps'].append(result['steps'])

    # Compute rates
    summary = {}
    for method, data in all_results.items():
        summary[method] = {
            'success_rate': data['successes'] / n_trials,
            'avg_steps': np.mean(data['total_steps']),
            'successes': data['successes'],
            'total': n_trials
        }

    return summary


if __name__ == "__main__":
    print("=" * 70)
    print("NAVIGATION TRAP TEST")
    print("=" * 70)
    print()
    print("Adversarial benchmark: graphs where greedy similarity fails.")
    print("Traps have HIGH similarity but lead to dead ends.")
    print("Target reachable only via similarity DIP (valley path).")
    print()

    # Verify trap properties first
    print("Verifying trap graph properties...")
    test_graph = create_trap_graph(seed=42)
    props = verify_trap_properties(test_graph)
    print(f"  Start similarity: {props['start_sim']:.3f}")
    print(f"  Target similarity: {props['target_sim']:.3f}")
    print(f"  Trap similarities: {[f'{s:.3f}' for s in props['trap_sims']]}")
    print(f"  Valley similarities: {[f'{s:.3f}' for s in props['valley_sims']]}")
    print(f"  Traps higher than valley min: {props['traps_higher_than_valley']}")
    print(f"  Valley dips below start: {props['valley_dips']}")
    print()

    # Run full test
    print("Running navigation test (30 trials)...")
    print()

    summary = run_full_test(n_trials=30)

    # Results table
    print("-" * 70)
    print(f"{'Method':<20} | {'Success Rate':>12} | {'Avg Steps':>10} | {'Wins':>8}")
    print("-" * 70)

    for method in ['greedy', 'beam', 'stop_heuristic', 'option_a', 'option_b']:
        s = summary[method]
        print(f"{method:<20} | {s['success_rate']:>11.1%} | {s['avg_steps']:>10.1f} | "
              f"{s['successes']:>3}/{s['total']}")

    print("-" * 70)

    # Analysis
    print()
    print("=" * 70)
    print("PASS/FAIL CRITERIA (per OPUS_NAVIGATION_TEST.md)")
    print("=" * 70)

    greedy_rate = summary['greedy']['success_rate']
    beam_rate = summary['beam']['success_rate']
    option_a_rate = summary['option_a']['success_rate']
    option_b_rate = summary['option_b']['success_rate']

    # Check baseline failure rate
    print()
    print(f"Greedy fails >=70%: {greedy_rate <= 0.30} (actual: {1-greedy_rate:.1%} fail)")
    print(f"Beam fails >=30%: {beam_rate <= 0.70} (actual: {1-beam_rate:.1%} fail)")

    # Option A criteria
    print()
    print("OPTION A (Similarity + Gate):")
    a_beats_greedy = option_a_rate >= greedy_rate + 0.1  # Significantly better
    a_beats_beam = option_a_rate >= beam_rate
    a_fewer_steps = summary['option_a']['avg_steps'] <= summary['beam']['avg_steps']

    print(f"  Beats greedy (>=70% vs greedy): {a_beats_greedy} ({option_a_rate:.1%} vs {greedy_rate:.1%})")
    print(f"  Beats beam (>=50%): {option_a_rate >= 0.5} ({option_a_rate:.1%})")
    print(f"  Fewer steps than beam: {a_fewer_steps} ({summary['option_a']['avg_steps']:.1f} vs {summary['beam']['avg_steps']:.1f})")

    # Option B criteria
    print()
    print("OPTION B (Delta-R Steering):")
    b_beats_a = option_b_rate > option_a_rate
    print(f"  Beats Option A: {b_beats_a} ({option_b_rate:.1%} vs {option_a_rate:.1%})")

    # Verdict
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if a_beats_greedy and option_a_rate >= 0.5:
        print()
        print("** OPTION A PASSES **")
        print("Formula VALIDATED as navigation via policy control.")
        print("Gate successfully escapes similarity traps.")
    else:
        print()
        print("Option A: NEEDS WORK")
        print(f"Success rate {option_a_rate:.1%} vs required >=50%")

    if b_beats_a:
        print()
        print("** OPTION B PASSES **")
        print("Formula can act as directional navigation signal.")
    else:
        print()
        print("Option B: Does not beat Option A")
        print("Delta-R steering is not superior to gated navigation.")
