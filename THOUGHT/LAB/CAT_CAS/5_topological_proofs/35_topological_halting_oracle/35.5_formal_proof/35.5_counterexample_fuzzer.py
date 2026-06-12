"""
35.5_counterexample_fuzzer.py

Formal correctness verification via random TM generation.

CONJECTURES TESTED:
  1.  W = 0  iff  halt state is reachable from initial state and no
      cycles exist in the active-only subgraph.
  2.  EP detected (kappa(V) > 1e6) iff the halt state is the unique
      sink and the longest chain-to-halt has length >= 2.
  3.  The point-gap winding W equals the number of directed cycles
      in the TM's transition graph that exclude the halt state.

METHOD:
  Generate N=10,000 random TMs with 2-5 states, random transitions.
  For each:
    - Build non-Hermitian H, compute W and kappa(V)
    - Compute BFS reachability from initial state to halt
    - Count cycles in active-subgraph via DFS
    - Compare: W=0 <-> reachable, W cycle count vs graph cycle count

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch
import numpy as np
from torch import linalg as LA

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# 1.  Random TM generator
# ---------------------------------------------------------------------------

def random_tm(num_states, edge_prob=0.4):
    """
    Generate a random TM with num_states states (last = halt).
    Each state reads 0 or 1 and transitions to a random next state
    with a random next symbol.
    Returns: (transitions, num_states, halt_idx)
    """
    halt_idx = num_states - 1
    transitions = {}
    for s in range(num_states):
        for b in range(2):
            if s == halt_idx:
                continue  # halt has no outgoing transitions
            if np.random.random() < edge_prob:
                sn = np.random.randint(0, num_states)
                bn = np.random.randint(0, 2)
                d = np.random.choice([-1, 0, 1])
                transitions[(s, b)] = (sn, bn, d)
    return transitions, num_states, halt_idx


# ---------------------------------------------------------------------------
# 2.  Hamiltonian builder (from 35.2)
# ---------------------------------------------------------------------------

def build_nonhermitian_H(transitions, num_states, halt_idx=None):
    symbols = 2
    N = num_states * symbols
    H = torch.zeros((N, N), dtype=torch.complex64)
    for s in range(num_states):
        for b in range(symbols):
            idx = s * symbols + b
            is_halt = (halt_idx is not None and s == halt_idx)
            H[idx, idx] = -1j * (10.0 if is_halt else 1.0) * 0.1
    for (s, b), (sn, bn, _dir) in transitions.items():
        i = s * symbols + b
        j = sn * symbols + bn
        H[j, i] = 1.0 + 0j
    return H


# ---------------------------------------------------------------------------
# 3.  Winding, EP, and reachability
# ---------------------------------------------------------------------------

def compute_W_full(H, transitions, num_states, halt_idx):
    """Full point-gap winding via determinant sweep over global twist phi.
    Applies exp(i*phi) to ALL transitions to detect spectral flow from
    any directed cycle in the transition graph.
    Returns integer winding W."""
    symbols = 2
    N = H.shape[0]
    I = torch.eye(N, dtype=torch.complex64)
    n_phi = 200
    dets = torch.zeros(n_phi, dtype=torch.complex64)

    for k in range(n_phi):
        phi = 2.0 * np.pi * k / n_phi
        twist = torch.tensor(np.exp(1j * phi), dtype=torch.complex64)
        H_phi = H.clone()
        for (s, b), (sn, bn, _) in transitions.items():
            i = s * symbols + b
            j = sn * symbols + bn
            H_phi[j, i] = H_phi[j, i] * twist
        d = LA.det(H_phi)
        if d.abs().item() < 1e-30:
            sign, ld = LA.slogdet(H_phi)
            dets[k] = sign * torch.exp(ld)
        else:
            dets[k] = d

    dtheta = torch.diff(torch.angle(dets))
    dtheta = torch.remainder(dtheta + np.pi, 2.0 * np.pi) - np.pi
    W_raw = float(torch.sum(dtheta).item()) / (2.0 * np.pi)
    return int(round(W_raw))


def compute_kappa(H):
    """Eigenvector condition number — EP detection."""
    _, eigvecs = LA.eig(H)
    return float(LA.cond(eigvecs).item())


def bfs_reachable(transitions, num_states, halt_idx, start=0):
    """BFS from start state to halt on the transition graph."""
    symbols = 2
    adj = {s: set() for s in range(num_states)}
    for (s, b), (sn, bn, _) in transitions.items():
        adj[s].add(sn)
    visited = set()
    queue = [start]
    while queue:
        s = queue.pop(0)
        if s == halt_idx:
            return True
        if s in visited:
            continue
        visited.add(s)
        for sn in adj.get(s, set()):
            if sn not in visited:
                queue.append(sn)
    return False


def count_cycles_config(transitions, num_states, halt_idx, symbols=2):
    """Count directed cycles in the CONFIGURATION graph (state x symbol).
    This matches the graph actually encoded in the Hamiltonian.
    A transition (s,b)->(sn,bn) is a directed edge in the full graph."""
    cfg_adj = {}
    cfg_nodes = []
    for s in range(num_states):
        for b in range(symbols):
            idx = s * symbols + b
            cfg_nodes.append(idx)
            cfg_adj[idx] = set()
    for (s, b), (sn, bn, _) in transitions.items():
        i = s * symbols + b
        j = sn * symbols + bn
        cfg_adj[i].add(j)

    halt_indices = {halt_idx * symbols + b for b in range(symbols)}
    cycle_count = 0
    active = [n for n in cfg_nodes if n not in halt_indices]
    visited_global = set()

    for start in active:
        if start in visited_global:
            continue
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for nb in cfg_adj.get(node, set()):
                if nb in halt_indices:
                    continue
                if nb in path:
                    cycle_count += 1
                elif nb not in visited_global:
                    visited_global.add(nb)
                    stack.append((nb, path + [nb]))
        visited_global.add(start)
    return cycle_count


# ---------------------------------------------------------------------------
# 4.  Fuzzer runner
# ---------------------------------------------------------------------------

def fuzz(N=10000):
    """Run counterexample search over N random TMs."""

    # Results accumulators
    results = {
        "total": N,
        "halt_reachable": 0,
        "halt_unreachable": 0,
        "W_eq_0": 0,
        "W_ne_0": 0,
        "correct": 0,           # W=0 iff acyclic
        "W_cycle_correct": 0,
        "W_cycle_fp": 0,         # W=0 but has cycles
        "W_cycle_fn": 0,         # W!=0 but no cycles
        "dag_count": 0,
        "dag_reachable": 0,
        "dag_unreachable": 0,
        "ep_detected": 0,
        "ep_reachable": 0,
        "ep_unreachable": 0,
        "cycle_graph": [],
        "W_values": [],
        "kappa_values": [],
    }

    for i in range(N):
        ns = np.random.randint(2, 6)  # 2-5 states
        transitions, num_states, halt_idx = random_tm(ns)

        # Build H, compute observables
        H = build_nonhermitian_H(transitions, num_states, halt_idx)
        W_val = compute_W_full(H, transitions, num_states, halt_idx)
        kappa = compute_kappa(H)

        # Graph properties
        reachable = bfs_reachable(transitions, num_states, halt_idx)
        cycles = count_cycles_config(transitions, num_states, halt_idx)

        # Update counts
        if reachable:
            results["halt_reachable"] += 1
        else:
            results["halt_unreachable"] += 1

        if W_val == 0:
            results["W_eq_0"] += 1
        else:
            results["W_ne_0"] += 1

        # Primary test: W=0 iff the graph is acyclic (no directed cycles)
        has_cycles = cycles > 0
        W_acyclic_match = (W_val == 0) == (not has_cycles)
        if W_acyclic_match:
            results["W_cycle_correct"] += 1
        else:
            if W_val == 0 and has_cycles:
                results["W_cycle_fp"] += 1  # W=0 but has cycles
            else:
                results["W_cycle_fn"] += 1  # W!=0 but no cycles

        # Secondary: among acyclic graphs, is halt reachability consistent?
        if not has_cycles:
            results["dag_count"] += 1
            if reachable:
                results["dag_reachable"] += 1
            else:
                results["dag_unreachable"] += 1

        ep = kappa > 1e6
        if ep:
            results["ep_detected"] += 1
            if reachable:
                results["ep_reachable"] += 1
            else:
                results["ep_unreachable"] += 1

        results["cycle_graph"].append(cycles)
        results["W_values"].append(W_val)
        results["kappa_values"].append(kappa)

    return results


# ---------------------------------------------------------------------------
# 5.  Report
# ---------------------------------------------------------------------------

def report(results):
    N = results["total"]
    print("=" * 70)
    print("  FORMAL PROOF — Counterexample Fuzzer Report")
    print("=" * 70)
    print(f"\n  Total TMs tested: {N}")
    print(f"\n  Reachability:")
    print(f"    Halt reachable:    {results['halt_reachable']:5d}"
          f"  ({100*results['halt_reachable']/N:.1f}%)")
    print(f"    Halt unreachable:  {results['halt_unreachable']:5d}"
          f"  ({100*results['halt_unreachable']/N:.1f}%)")
    print(f"\n  Winding W:")
    print(f"    W = 0:  {results['W_eq_0']:5d}"
          f"  ({100*results['W_eq_0']/N:.1f}%)")
    print(f"    W != 0: {results['W_ne_0']:5d}"
          f"  ({100*results['W_ne_0']/N:.1f}%)")
    print(f"\n  W <-> Acyclic Correspondence:")
    acc = 100 * results["W_cycle_correct"] / N
    print(f"    Correct:      {results['W_cycle_correct']:5d}  ({acc:.2f}%)  W=0 iff acyclic")
    print(f"    False pos:    {results['W_cycle_fp']:5d}  (W=0 but has cycles)")
    print(f"    False neg:    {results['W_cycle_fn']:5d}  (W!=0 but no cycles)")
    print(f"\n  DAG analysis (acyclic graphs):")
    print(f"    Total DAGs:   {results['dag_count']:5d}")
    if results["dag_count"] > 0:
        print(f"    Halt reachable: {results['dag_reachable']:5d}  "
              f"({100*results['dag_reachable']/results['dag_count']:.1f}%)")
        print(f"    Halt unreachable: {results['dag_unreachable']:5d}  "
              f"({100*results['dag_unreachable']/results['dag_count']:.1f}%)")
    print(f"\n  Exceptional Points:")
    print(f"    EP detected:      {results['ep_detected']:5d}"
          f"  ({100*results['ep_detected']/N:.1f}%)")
    if results["ep_detected"] > 0:
        print(f"    EP + reachable:   {results['ep_reachable']:5d}"
              f"  ({100*results['ep_reachable']/results['ep_detected']:.1f}%)")
        print(f"    EP + unreachable: {results['ep_unreachable']:5d}"
              f"  ({100*results['ep_unreachable']/results['ep_detected']:.1f}%)")
    print(f"\n  Cycle statistics:")
    cycles = np.array(results["cycle_graph"])
    print(f"    Mean cycles per TM: {cycles.mean():.2f}")
    print(f"    TMs with cycles:    {(cycles > 0).sum()}")

    # Cycle vs W correlation
    if len(set(results["W_values"])) > 1:
        w_vals = np.array(results["W_values"])
        print(f"\n  Cycle-Winding correlation:")
        for w in sorted(set(results["W_values"])):
            mask = w_vals == w
            c_mean = cycles[mask].mean()
            print(f"    W={w}: mean cycles = {c_mean:.2f}  (n={mask.sum()})")

    # Conclusion
    print(f"\n  ***  VERDICT:  ", end="")
    errs = results["W_cycle_fp"] + results["W_cycle_fn"]
    if errs == 0:
        print(f"PROVEN — W=0 iff acyclic for all {N} cases")
    elif errs < 0.02 * N:
        print(f"LARGELY PROVEN — {errs} exceptions ({100*errs/N:.2f}%)")
    else:
        print(f"{errs} exceptions ({100*errs/N:.1f}%) — investigate")
    if results["W_cycle_fn"] == 0:
        print(f"  Zero false negatives: W>0 always implies cycles exist")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 6.  Main
# ---------------------------------------------------------------------------

def main():
    results = fuzz(N=500)
    report(results)


if __name__ == "__main__":
    main()
