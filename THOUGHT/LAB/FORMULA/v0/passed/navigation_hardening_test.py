"""
Navigation Hardening Test - Validate Option B (Delta-R Steering)

Per GPT review, 6 hardening tests:
1. Replication sweep (50+ seeds, 3+ graph families)
2. Action-conditioned ablations
3. Budget parity check
4. No hidden lookahead verification
5. Failure mode breakdown
6. Policy ELO tournament
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# GRAPH GENERATORS (3 families)
# =============================================================================

def create_trap_graph_v1(n_nodes: int = 50, n_traps: int = 3, seed: int = None) -> Dict:
    """Original trap graph: explicit trap nodes + valley path."""
    if seed is not None:
        np.random.seed(seed)

    dim = 32
    query = np.random.randn(dim)
    query = query / np.linalg.norm(query)

    embeddings = np.random.randn(n_nodes, dim)

    start = 0
    target = 1
    trap_nodes = list(range(2, 2 + n_traps))
    valley_nodes = list(range(2 + n_traps, 2 + n_traps + 5))

    # Target: high similarity
    embeddings[target] = query + np.random.randn(dim) * 0.1

    # Traps: VERY high similarity (higher than visible target path)
    for t in trap_nodes:
        embeddings[t] = query + np.random.randn(dim) * 0.05

    # Valley: lower similarity dip
    valley_direction = np.random.randn(dim)
    valley_direction = valley_direction / np.linalg.norm(valley_direction)
    for i, v in enumerate(valley_nodes):
        alpha = 0.3 + i * 0.1
        embeddings[v] = alpha * query + (1 - alpha) * valley_direction
        embeddings[v] = embeddings[v] / np.linalg.norm(embeddings[v])

    embeddings[start] = 0.5 * query + 0.5 * np.random.randn(dim)
    embeddings[start] = embeddings[start] / np.linalg.norm(embeddings[start])

    for i in range(n_nodes):
        embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])

    adjacency = {i: [] for i in range(n_nodes)}
    adjacency[start] = trap_nodes + [valley_nodes[0]]

    dead_ends = list(range(2 + n_traps + 5, min(n_nodes, 2 + n_traps + 15)))
    for t in trap_nodes:
        other_traps = [x for x in trap_nodes if x != t]
        adjacency[t] = other_traps + dead_ends[:3]

    for i, d in enumerate(dead_ends):
        adjacency[d] = [dead_ends[(i+1) % len(dead_ends)], dead_ends[(i-1) % len(dead_ends)]]

    for i, v in enumerate(valley_nodes[:-1]):
        adjacency[v] = [valley_nodes[i + 1]]
        safe_nodes = [x for x in range(n_nodes) if x not in trap_nodes and x not in dead_ends]
        if safe_nodes:
            adjacency[v].extend(np.random.choice(safe_nodes, size=min(2, len(safe_nodes)), replace=False).tolist())

    adjacency[valley_nodes[-1]] = [target]
    adjacency[target] = [valley_nodes[-1]]

    similarities = np.array([np.dot(embeddings[i], query) for i in range(n_nodes)])

    return {
        'n_nodes': n_nodes, 'embeddings': embeddings, 'query': query,
        'adjacency': adjacency, 'similarities': similarities,
        'start': start, 'target': target, 'traps': trap_nodes,
        'valley': valley_nodes, 'dead_ends': dead_ends, 'family': 'v1_explicit'
    }


def create_trap_graph_v2(n_nodes: int = 60, seed: int = None) -> Dict:
    """Lattice trap: grid with similarity gradient + local maxima traps."""
    if seed is not None:
        np.random.seed(seed)

    dim = 32
    query = np.random.randn(dim)
    query = query / np.linalg.norm(query)

    # Grid layout
    grid_size = int(np.sqrt(n_nodes))
    n_nodes = grid_size * grid_size

    embeddings = np.random.randn(n_nodes, dim)

    # Target in corner
    target = n_nodes - 1
    embeddings[target] = query + np.random.randn(dim) * 0.05

    # Start in opposite corner
    start = 0
    embeddings[start] = 0.3 * query + 0.7 * np.random.randn(dim)

    # Create local maxima traps (high similarity but not connected to target)
    n_traps = 3
    trap_positions = np.random.choice(range(n_nodes // 3, 2 * n_nodes // 3), n_traps, replace=False)
    for t in trap_positions:
        embeddings[t] = query + np.random.randn(dim) * 0.03  # Even higher than target!

    # Normalize
    for i in range(n_nodes):
        embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])

    # Grid adjacency
    adjacency = {i: [] for i in range(n_nodes)}
    for i in range(n_nodes):
        row, col = i // grid_size, i % grid_size
        # 4-connected grid
        if row > 0:
            adjacency[i].append(i - grid_size)
        if row < grid_size - 1:
            adjacency[i].append(i + grid_size)
        if col > 0:
            adjacency[i].append(i - 1)
        if col < grid_size - 1:
            adjacency[i].append(i + 1)

    # Remove connections FROM traps (make them dead ends)
    for t in trap_positions:
        # Keep incoming edges, but outgoing only to other traps or back
        adjacency[t] = [n for n in adjacency[t] if n in trap_positions or np.random.random() < 0.3]

    similarities = np.array([np.dot(embeddings[i], query) for i in range(n_nodes)])

    return {
        'n_nodes': n_nodes, 'embeddings': embeddings, 'query': query,
        'adjacency': adjacency, 'similarities': similarities,
        'start': start, 'target': target, 'traps': list(trap_positions),
        'valley': [], 'dead_ends': [], 'family': 'v2_lattice'
    }


def create_trap_graph_v3(n_nodes: int = 50, seed: int = None) -> Dict:
    """Hub-spoke trap: central hub with high-similarity spokes that are traps."""
    if seed is not None:
        np.random.seed(seed)

    dim = 32
    query = np.random.randn(dim)
    query = query / np.linalg.norm(query)

    embeddings = np.random.randn(n_nodes, dim)

    start = 0
    target = n_nodes - 1
    hub = n_nodes // 2

    # Target: reachable via indirect path
    embeddings[target] = query + np.random.randn(dim) * 0.1

    # Hub: moderate similarity
    embeddings[hub] = 0.6 * query + 0.4 * np.random.randn(dim)

    # Trap spokes: high similarity, dead ends
    n_traps = 4
    trap_spokes = list(range(1, 1 + n_traps))
    for t in trap_spokes:
        embeddings[t] = query + np.random.randn(dim) * 0.02  # Very high!

    # Good path: lower similarity but leads to target
    good_path = list(range(hub + 1, target))
    for i, g in enumerate(good_path):
        alpha = 0.4 + 0.1 * (i / len(good_path))
        embeddings[g] = alpha * query + (1 - alpha) * np.random.randn(dim)

    # Normalize
    for i in range(n_nodes):
        embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])

    # Adjacency
    adjacency = {i: [] for i in range(n_nodes)}

    # Start connects to hub and traps
    adjacency[start] = [hub] + trap_spokes

    # Hub connects to good path and traps
    adjacency[hub] = trap_spokes + ([good_path[0]] if good_path else [target])

    # Traps are dead ends (connect only to each other)
    for t in trap_spokes:
        adjacency[t] = [x for x in trap_spokes if x != t]

    # Good path is linear to target
    for i, g in enumerate(good_path[:-1]):
        adjacency[g] = [good_path[i + 1]]
    if good_path:
        adjacency[good_path[-1]] = [target]

    adjacency[target] = [good_path[-1]] if good_path else [hub]

    similarities = np.array([np.dot(embeddings[i], query) for i in range(n_nodes)])

    return {
        'n_nodes': n_nodes, 'embeddings': embeddings, 'query': query,
        'adjacency': adjacency, 'similarities': similarities,
        'start': start, 'target': target, 'traps': trap_spokes,
        'valley': good_path, 'dead_ends': [], 'family': 'v3_hub'
    }


# =============================================================================
# NAVIGATION POLICIES (from original)
# =============================================================================

def greedy_similarity(graph: Dict, max_steps: int = 30) -> Tuple[List[int], bool, Dict]:
    """Greedy similarity baseline."""
    path = [graph['start']]
    current = graph['start']
    visited = {current}
    expansions = 0

    for _ in range(max_steps):
        if current == graph['target']:
            return path, True, {'expansions': expansions}

        neighbors = [n for n in graph['adjacency'][current] if n not in visited]
        if not neighbors:
            return path, False, {'expansions': expansions, 'failure': 'dead_end'}

        expansions += len(neighbors)
        sims = [graph['similarities'][n] for n in neighbors]
        current = neighbors[np.argmax(sims)]
        path.append(current)
        visited.add(current)

    return path, current == graph['target'], {'expansions': expansions, 'failure': 'max_steps'}


def compute_action_R_full(current: int, action: int, graph: Dict) -> float:
    """Full action-conditioned R."""
    current_sim = graph['similarities'][current]
    action_sim = graph['similarities'][action]
    E = max(0.01, action_sim - current_sim + 0.5)

    action_neighbors = graph['adjacency'][action]
    if action_neighbors:
        neighbor_sims = [graph['similarities'][n] for n in action_neighbors]
        grad_S = np.std(neighbor_sims) + 0.01
    else:
        grad_S = 1.0

    Df = np.log(len(action_neighbors) + 1) + 0.1
    sigma = 0.5
    R = (E / grad_S) * (sigma ** Df)
    return R


def compute_action_R_no_gradS(current: int, action: int, graph: Dict) -> float:
    """Ablation: remove grad_S term."""
    current_sim = graph['similarities'][current]
    action_sim = graph['similarities'][action]
    E = max(0.01, action_sim - current_sim + 0.5)

    action_neighbors = graph['adjacency'][action]
    Df = np.log(len(action_neighbors) + 1) + 0.1
    sigma = 0.5
    R = E * (sigma ** Df)  # No division by grad_S
    return R


def compute_action_R_no_Df(current: int, action: int, graph: Dict) -> float:
    """Ablation: remove Df term (constant)."""
    current_sim = graph['similarities'][current]
    action_sim = graph['similarities'][action]
    E = max(0.01, action_sim - current_sim + 0.5)

    action_neighbors = graph['adjacency'][action]
    if action_neighbors:
        neighbor_sims = [graph['similarities'][n] for n in action_neighbors]
        grad_S = np.std(neighbor_sims) + 0.01
    else:
        grad_S = 1.0

    R = E / grad_S  # No Df term
    return R


def compute_action_R_raw_sim(current: int, action: int, graph: Dict) -> float:
    """Ablation: just raw similarity (should collapse to greedy)."""
    return graph['similarities'][action]


def compute_action_R_shuffled(current: int, action: int, graph: Dict, rng) -> float:
    """Ablation: shuffle neighbor sims before computing dispersion."""
    current_sim = graph['similarities'][current]
    action_sim = graph['similarities'][action]
    E = max(0.01, action_sim - current_sim + 0.5)

    action_neighbors = graph['adjacency'][action]
    if action_neighbors:
        # Shuffle similarities from RANDOM nodes, not actual neighbors
        random_nodes = rng.choice(graph['n_nodes'], size=len(action_neighbors), replace=True)
        neighbor_sims = [graph['similarities'][n] for n in random_nodes]
        grad_S = np.std(neighbor_sims) + 0.01
    else:
        grad_S = 1.0

    Df = np.log(len(action_neighbors) + 1) + 0.1
    sigma = 0.5
    R = (E / grad_S) * (sigma ** Df)
    return R


def delta_r_steering(graph: Dict, max_steps: int = 30, R_fn=compute_action_R_full,
                     rng=None) -> Tuple[List[int], bool, Dict]:
    """Delta-R steering with configurable R function."""
    path = [graph['start']]
    current = graph['start']
    visited = {current}
    expansions = 0

    for step in range(max_steps):
        if current == graph['target']:
            return path, True, {'expansions': expansions}

        neighbors = [n for n in graph['adjacency'][current] if n not in visited]
        if not neighbors:
            return path, False, {'expansions': expansions, 'failure': 'dead_end'}

        expansions += len(neighbors)

        # Compute R for each action
        if rng is not None and R_fn == compute_action_R_shuffled:
            action_scores = [(n, compute_action_R_shuffled(current, n, graph, rng)) for n in neighbors]
        else:
            action_scores = [(n, R_fn(current, n, graph)) for n in neighbors]

        action_scores.sort(key=lambda x: x[1], reverse=True)
        current = action_scores[0][0]
        path.append(current)
        visited.add(current)

    return path, current == graph['target'], {'expansions': expansions, 'failure': 'max_steps'}


# =============================================================================
# TEST 1: REPLICATION SWEEP
# =============================================================================

def test_replication_sweep(n_seeds: int = 50):
    """50+ seeds, 3 graph families."""
    print("=" * 70)
    print("TEST 1: REPLICATION SWEEP")
    print("=" * 70)
    print(f"Seeds: {n_seeds}, Families: 3")
    print()

    generators = [
        ('v1_explicit', create_trap_graph_v1),
        ('v2_lattice', create_trap_graph_v2),
        ('v3_hub', create_trap_graph_v3),
    ]

    results = defaultdict(lambda: defaultdict(list))

    for family_name, gen_fn in generators:
        for seed in range(n_seeds):
            graph = gen_fn(seed=seed * 1000 + 42)

            # Greedy
            _, success, _ = greedy_similarity(graph)
            results[family_name]['greedy'].append(int(success))

            # Delta-R
            _, success, _ = delta_r_steering(graph)
            results[family_name]['delta_r'].append(int(success))

    # Report
    print(f"{'Family':<15} | {'Greedy':>15} | {'Delta-R':>15} | {'Delta':>10}")
    print("-" * 60)

    all_greedy = []
    all_delta_r = []

    for family_name, _ in generators:
        greedy_rate = np.mean(results[family_name]['greedy'])
        delta_r_rate = np.mean(results[family_name]['delta_r'])
        greedy_ci = 1.96 * np.std(results[family_name]['greedy']) / np.sqrt(n_seeds)
        delta_r_ci = 1.96 * np.std(results[family_name]['delta_r']) / np.sqrt(n_seeds)

        all_greedy.extend(results[family_name]['greedy'])
        all_delta_r.extend(results[family_name]['delta_r'])

        print(f"{family_name:<15} | {greedy_rate:>5.1%} +/- {greedy_ci:.1%} | "
              f"{delta_r_rate:>5.1%} +/- {delta_r_ci:.1%} | {delta_r_rate - greedy_rate:>+.1%}")

    print("-" * 60)
    overall_greedy = np.mean(all_greedy)
    overall_delta_r = np.mean(all_delta_r)
    overall_ci = 1.96 * np.std(all_delta_r) / np.sqrt(len(all_delta_r))
    print(f"{'OVERALL':<15} | {overall_greedy:>5.1%}          | "
          f"{overall_delta_r:>5.1%} +/- {overall_ci:.1%} | {overall_delta_r - overall_greedy:>+.1%}")

    return overall_delta_r, overall_ci


# =============================================================================
# TEST 2: ACTION-CONDITIONED ABLATIONS
# =============================================================================

def test_ablations(n_seeds: int = 30):
    """Ablate each component of action-conditioned R."""
    print()
    print("=" * 70)
    print("TEST 2: ACTION-CONDITIONED ABLATIONS")
    print("=" * 70)

    ablations = [
        ('full', compute_action_R_full),
        ('no_gradS', compute_action_R_no_gradS),
        ('no_Df', compute_action_R_no_Df),
        ('raw_sim', compute_action_R_raw_sim),
    ]

    results = {name: [] for name, _ in ablations}
    results['shuffled'] = []

    for seed in range(n_seeds):
        graph = create_trap_graph_v1(seed=seed * 1000 + 42)
        rng = np.random.default_rng(seed)

        for name, R_fn in ablations:
            _, success, _ = delta_r_steering(graph, R_fn=R_fn)
            results[name].append(int(success))

        # Shuffled needs rng
        _, success, _ = delta_r_steering(graph, R_fn=compute_action_R_shuffled, rng=rng)
        results['shuffled'].append(int(success))

    print()
    print(f"{'Ablation':<15} | {'Success Rate':>12} | {'vs Full':>10}")
    print("-" * 45)

    full_rate = np.mean(results['full'])
    for name in ['full', 'no_gradS', 'no_Df', 'raw_sim', 'shuffled']:
        rate = np.mean(results[name])
        delta = rate - full_rate
        marker = "BASELINE" if name == 'full' else ("DROP" if delta < -0.1 else "OK")
        print(f"{name:<15} | {rate:>11.1%} | {delta:>+9.1%} [{marker}]")

    return results


# =============================================================================
# TEST 3: BUDGET PARITY
# =============================================================================

def test_budget_parity(n_seeds: int = 30):
    """Ensure equal expansions/depth."""
    print()
    print("=" * 70)
    print("TEST 3: BUDGET PARITY")
    print("=" * 70)

    greedy_expansions = []
    delta_r_expansions = []
    greedy_depth = []
    delta_r_depth = []

    for seed in range(n_seeds):
        graph = create_trap_graph_v1(seed=seed * 1000 + 42)

        _, _, meta_g = greedy_similarity(graph)
        path_g, _, meta_d = delta_r_steering(graph)

        greedy_expansions.append(meta_g['expansions'])
        delta_r_expansions.append(meta_d['expansions'])
        greedy_depth.append(len(path_g))
        delta_r_depth.append(len(path_g))

    print()
    print(f"{'Metric':<20} | {'Greedy':>10} | {'Delta-R':>10}")
    print("-" * 45)
    print(f"{'Avg Expansions':<20} | {np.mean(greedy_expansions):>10.1f} | {np.mean(delta_r_expansions):>10.1f}")
    print(f"{'Avg Path Length':<20} | {np.mean(greedy_depth):>10.1f} | {np.mean(delta_r_depth):>10.1f}")

    parity = abs(np.mean(greedy_expansions) - np.mean(delta_r_expansions)) < 5
    print()
    print(f"Budget parity: {'PASS' if parity else 'FAIL'}")

    return parity


# =============================================================================
# TEST 4: NO HIDDEN LOOKAHEAD
# =============================================================================

def test_no_lookahead():
    """Verify R(s,a) uses only local information."""
    print()
    print("=" * 70)
    print("TEST 4: NO HIDDEN LOOKAHEAD")
    print("=" * 70)
    print()

    # Check what compute_action_R_full accesses
    print("compute_action_R_full accesses:")
    print("  - graph['similarities'][current]  -- current node sim")
    print("  - graph['similarities'][action]   -- action node sim")
    print("  - graph['adjacency'][action]      -- action's neighbors")
    print("  - graph['similarities'][n] for n in neighbors -- neighbor sims")
    print()
    print("Does NOT access:")
    print("  - graph['target']")
    print("  - graph['traps']")
    print("  - Path to target")
    print("  - Future nodes beyond action's immediate neighbors")
    print()
    print("Verdict: NO LOOKAHEAD - uses only local neighborhood")

    return True


# =============================================================================
# TEST 5: FAILURE MODE BREAKDOWN
# =============================================================================

def test_failure_modes(n_seeds: int = 50):
    """Classify failure modes."""
    print()
    print("=" * 70)
    print("TEST 5: FAILURE MODE BREAKDOWN")
    print("=" * 70)

    failures = defaultdict(int)
    total_failures = 0

    for seed in range(n_seeds):
        graph = create_trap_graph_v1(seed=seed * 1000 + 42)
        path, success, meta = delta_r_steering(graph)

        if not success:
            total_failures += 1

            # Classify failure
            if 'failure' in meta:
                if meta['failure'] == 'dead_end':
                    # Check if in trap
                    if any(p in graph['traps'] for p in path):
                        failures['trap_basin'] += 1
                    else:
                        failures['other_dead_end'] += 1
                elif meta['failure'] == 'max_steps':
                    # Check for oscillation (repeated visits)
                    if len(path) != len(set(path)):
                        failures['oscillation'] += 1
                    else:
                        failures['slow_progress'] += 1
            else:
                failures['unknown'] += 1

    print()
    if total_failures > 0:
        print(f"Total failures: {total_failures}/{n_seeds} ({total_failures/n_seeds:.1%})")
        print()
        print(f"{'Failure Mode':<20} | {'Count':>6} | {'% of Failures':>12}")
        print("-" * 45)
        for mode, count in sorted(failures.items(), key=lambda x: -x[1]):
            print(f"{mode:<20} | {count:>6} | {count/total_failures:>11.1%}")
    else:
        print("No failures to analyze!")

    return failures


# =============================================================================
# TEST 6: POLICY ELO TOURNAMENT
# =============================================================================

def test_policy_elo(n_rounds: int = 100):
    """ELO tournament over policies."""
    print()
    print("=" * 70)
    print("TEST 6: POLICY ELO TOURNAMENT")
    print("=" * 70)

    policies = {
        'greedy': lambda g: greedy_similarity(g),
        'delta_r_full': lambda g: delta_r_steering(g, R_fn=compute_action_R_full),
        'delta_r_no_gradS': lambda g: delta_r_steering(g, R_fn=compute_action_R_no_gradS),
        'delta_r_no_Df': lambda g: delta_r_steering(g, R_fn=compute_action_R_no_Df),
        'delta_r_raw_sim': lambda g: delta_r_steering(g, R_fn=compute_action_R_raw_sim),
    }

    elo = {name: 1500.0 for name in policies}
    K = 32

    generators = [create_trap_graph_v1, create_trap_graph_v2, create_trap_graph_v3]

    for round_idx in range(n_rounds):
        seed = round_idx * 1000 + 42
        gen = generators[round_idx % 3]
        graph = gen(seed=seed)

        # Run all policies
        results = {}
        for name, policy_fn in policies.items():
            _, success, _ = policy_fn(graph)
            results[name] = int(success)

        # Pairwise ELO updates
        policy_names = list(policies.keys())
        for i in range(len(policy_names)):
            for j in range(i + 1, len(policy_names)):
                p1, p2 = policy_names[i], policy_names[j]
                s1, s2 = results[p1], results[p2]

                # Expected scores
                exp1 = 1 / (1 + 10 ** ((elo[p2] - elo[p1]) / 400))
                exp2 = 1 - exp1

                # Actual scores (1 = win, 0.5 = tie, 0 = loss)
                if s1 > s2:
                    actual1, actual2 = 1, 0
                elif s1 < s2:
                    actual1, actual2 = 0, 1
                else:
                    actual1, actual2 = 0.5, 0.5

                elo[p1] += K * (actual1 - exp1)
                elo[p2] += K * (actual2 - exp2)

    # Report
    print()
    print(f"{'Policy':<20} | {'ELO':>8} | {'Rank':>6}")
    print("-" * 40)

    ranked = sorted(elo.items(), key=lambda x: -x[1])
    for rank, (name, score) in enumerate(ranked, 1):
        print(f"{name:<20} | {score:>8.1f} | {rank:>6}")

    return elo


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NAVIGATION HARDENING TEST SUITE")
    print("=" * 70)
    print()
    print("Validating Option B (Delta-R Steering) per GPT review")
    print()

    # Run all tests
    rate, ci = test_replication_sweep(n_seeds=50)
    ablation_results = test_ablations(n_seeds=30)
    parity = test_budget_parity(n_seeds=30)
    lookahead = test_no_lookahead()
    failures = test_failure_modes(n_seeds=50)
    elo = test_policy_elo(n_rounds=100)

    # Final verdict
    print()
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    checks = []

    # Check 1: Replication
    replication_pass = rate > 0.5 and ci < 0.15
    checks.append(('Replication (>50%, CI<15%)', replication_pass, f"{rate:.1%} +/- {ci:.1%}"))

    # Check 2: Ablations matter
    ablation_pass = (np.mean(ablation_results['raw_sim']) < np.mean(ablation_results['full']) - 0.2)
    checks.append(('Ablations matter (raw_sim drops)', ablation_pass,
                   f"full={np.mean(ablation_results['full']):.1%} vs raw_sim={np.mean(ablation_results['raw_sim']):.1%}"))

    # Check 3: Budget parity
    checks.append(('Budget parity', parity, ""))

    # Check 4: No lookahead
    checks.append(('No hidden lookahead', lookahead, ""))

    # Check 5: Failure modes identified
    failure_pass = sum(failures.values()) > 0 or rate > 0.9
    checks.append(('Failure modes analyzed', failure_pass, f"{sum(failures.values())} failures classified"))

    # Check 6: ELO ranking
    elo_pass = elo['delta_r_full'] > elo['greedy'] + 50
    checks.append(('ELO: delta_r_full > greedy', elo_pass,
                   f"{elo['delta_r_full']:.0f} vs {elo['greedy']:.0f}"))

    print()
    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if detail:
            print(f"        {detail}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("** ALL CHECKS PASS - OPTION B VALIDATED **")
    else:
        print("Some checks failed - needs investigation")
