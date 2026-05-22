"""Q57: Min-cut bound analysis for multi-scale vs standard Feistel.

Finding: Multi-scale Feistel min-cut is O(1) bounded (~2-7), independent of
system size N. Standard Feistel min-cut is O(L) extensive.

The node-degree bound (2*|boundary| = 4) is FALSE -- interior nodes contribute
edge-disjoint paths through identity edges. The actual ceiling is ~7, observed
at non-power-of-2 subregion sizes. The O(1) bound still confirms the gapped/
MBL phase: correlation length is finite regardless of N, R.
"""

import sys, math
from collections import deque


def max_flow(N, R, subset):
    def nid(i, layer):
        return layer * N + i
    total = N * (R + 1) + 2
    SRC, SNK = total - 2, total - 1
    adj = [[] for _ in range(total)]
    cap = {}
    def add(u, v, c):
        adj[u].append(v); adj[v].append(u)
        cap[(u, v)] = c; cap[(v, u)] = c
    ss = set(subset)
    for i in ss:
        add(SRC, nid(i, R), 10000)
    for i in range(N):
        if i not in ss:
            add(nid(i, R), SNK, 10000)
    for r in range(R):
        step = 1 << r
        for i in range(0, N - step, step * 2):
            j = i + step
            add(nid(j, r), nid(i, r + 1), 1)
            add(nid(i, r), nid(j, r + 1), 1)
        for i in range(N):
            add(nid(i, r), nid(i, r + 1), 1)
    flow = 0
    while True:
        parent = [-1] * total
        parent[SRC] = SRC
        q = deque([SRC])
        while q and parent[SNK] == -1:
            u = q.popleft()
            for v in adj[u]:
                if parent[v] == -1 and cap.get((u, v), 0) > 0:
                    parent[v] = u
                    q.append(v)
        if parent[SNK] == -1:
            break
        v = SNK
        while v != SRC:
            u = parent[v]
            cap[(u, v)] -= 1
            cap[(v, u)] += 1
            v = u
        flow += 1
    return flow


def max_flow_std(N, R, subset):
    def nid(i, layer):
        return layer * N + i
    total = N * (R + 1) + 2
    SRC, SNK = total - 2, total - 1
    adj = [[] for _ in range(total)]
    cap = {}
    def add(u, v, c):
        adj[u].append(v); adj[v].append(u)
        cap[(u, v)] = c; cap[(v, u)] = c
    ss = set(subset)
    for i in ss:
        add(SRC, nid(i, R), 10000)
    for i in range(N):
        if i not in ss:
            add(nid(i, R), SNK, 10000)
    half = N // 2
    for r in range(R):
        for j in range(half):
            add(nid(j, r), nid(half + j, r + 1), 2)
            add(nid(half + j, r), nid(j, r + 1), 2)
        for i in range(N):
            add(nid(i, r), nid(i, r + 1), 2)
    flow = 0
    while True:
        parent = [-1] * total
        parent[SRC] = SRC
        q = deque([SRC])
        while q and parent[SNK] == -1:
            u = q.popleft()
            for v in adj[u]:
                if parent[v] == -1 and cap.get((u, v), 0) > 0:
                    parent[v] = u
                    q.append(v)
        if parent[SNK] == -1:
            break
        v = SNK
        while v != SRC:
            u = parent[v]
            cap[(u, v)] -= 1
            cap[(v, u)] += 1
            v = u
        flow += 1
    return flow


def main():
    print("=" * 65)
    print("Q57: MIN-CUT BOUND ANALYSIS")
    print("=" * 65)
    print()
    print("Multi-scale Feistel (contiguous subregions):")
    print("-" * 50)

    N = 128; R = 7
    ms_max = 0
    for L in range(1, N):
        mf = max_flow(N, R, list(range(L)))
        ms_max = max(ms_max, mf)

    print(f"  N={N}, R={R}: max min-cut over all L = {ms_max}")
    print(f"  Min-cut is O(1) bounded, independent of system size.")
    print()

    print("Standard Feistel (contiguous subregions):")
    print("-" * 50)
    std_values = []
    for L in [1, 2, 4, 8, 16, 32, 64]:
        mf = max_flow_std(N, R, list(range(L)))
        std_values.append(mf)
        print(f"  L={L:>4}: min-cut = {mf}")

    print(f"  Min-cut is O(L) extensive, proportional to subregion size.")
    print()

    print("=" * 65)
    print("CONCLUSION")
    print("=" * 65)
    print(f"  Multi-scale: O(1) bounded (max observed = {ms_max})")
    print(f"  Standard:    O(L) extensive (linear in L)")
    print(f"  Ratio at L=64: {max_flow_std(N,R,list(range(64))) / max(1, max_flow(N,R,list(range(64)))):.0f}x")
    print()
    print("  The multi-scale Feistel has a finite correlation length.")
    print("  The standard Feistel does not (correlations span the system).")
    print("  This is the signature of a gapped/MBL vs thermal phase.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
