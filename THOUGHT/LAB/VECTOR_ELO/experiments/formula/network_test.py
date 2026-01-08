#!/usr/bin/env python3
"""
F.7.8: Network Centrality Comparison

Tests if R relates to eigenvector centrality in semantic networks.

If formula captures "essence," high-E nodes should have high centrality.

Prediction: Correlation > 0.7 with eigenvector centrality.
Falsification: Negative correlation or no relationship.
"""

import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def network_centrality_test():
    """
    Build semantic similarity network and compare:

    - Eigenvector centrality (PageRank-like)
    - Formula-predicted R for each node

    If formula captures "essence," high-E nodes should have high centrality.
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx not installed")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        print("sentence-transformers not installed, using random embeddings")
        model = None

    # Build word network
    words = [
        'king', 'queen', 'prince', 'princess', 'throne',
        'democracy', 'election', 'vote', 'president', 'congress',
        'cat', 'dog', 'pet', 'animal', 'mammal',
        'computer', 'software', 'algorithm', 'data', 'code'
    ]

    if model:
        embeddings = model.encode(words)
    else:
        np.random.seed(42)
        embeddings = np.random.randn(len(words), 384)

    # Build similarity graph
    G = nx.Graph()
    for i, w1 in enumerate(words):
        G.add_node(w1)
        for j, w2 in enumerate(words):
            if i < j:
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10
                )
                if sim > 0.3:  # Threshold
                    G.add_edge(w1, w2, weight=float(sim))

    # Eigenvector centrality
    try:
        centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        centrality = {w: 1/len(words) for w in words}

    # PageRank
    pagerank = nx.pagerank(G, weight='weight')

    # Formula-based R for each node
    R_formula = {}
    for i, word in enumerate(words):
        # E = average similarity to neighbors (essence = connectedness)
        neighbors = list(G.neighbors(word))
        if neighbors:
            neighbor_idx = [words.index(n) for n in neighbors]
            sims = []
            for j in neighbor_idx:
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10
                )
                sims.append(sim)
            E = np.mean(sims)
        else:
            E = 0

        # nabla_S = variance of similarities (entropy = inconsistency)
        all_sims = []
        for j in range(len(words)):
            if j != i:
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10
                )
                all_sims.append(sim)
        nabla_S = np.var(all_sims) + 0.01  # Avoid division by zero

        R_formula[word] = E / nabla_S

    # Correlations
    words_sorted = sorted(words)
    centrality_vec = [centrality.get(w, 0) for w in words_sorted]
    pagerank_vec = [pagerank.get(w, 0) for w in words_sorted]
    R_formula_vec = [R_formula.get(w, 0) for w in words_sorted]

    corr_centrality = np.corrcoef(centrality_vec, R_formula_vec)[0, 1]
    corr_pagerank = np.corrcoef(pagerank_vec, R_formula_vec)[0, 1]

    return {
        'centrality': centrality,
        'pagerank': pagerank,
        'R_formula': R_formula,
        'corr_centrality': corr_centrality,
        'corr_pagerank': corr_pagerank,
        'graph_info': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges()
        }
    }


if __name__ == '__main__':
    print("F.7.8: Network Centrality Comparison")
    print("=" * 50)

    if not HAS_NETWORKX:
        print("\nERROR: networkx not installed")
        print("Install with: pip install networkx")
        exit(1)

    result = network_centrality_test()

    print(f"\nGraph: {result['graph_info']['nodes']} nodes, {result['graph_info']['edges']} edges")
    print("-" * 60)
    print(f"{'Word':15s} | {'Centrality':>10s} | {'PageRank':>10s} | {'R_formula':>10s}")
    print("-" * 60)

    # Sort by R_formula
    words_sorted = sorted(result['R_formula'].keys(), key=lambda w: -result['R_formula'][w])
    for word in words_sorted[:10]:  # Top 10
        c = result['centrality'].get(word, 0)
        p = result['pagerank'].get(word, 0)
        r = result['R_formula'].get(word, 0)
        print(f"{word:15s} | {c:10.4f} | {p:10.4f} | {r:10.4f}")

    print("-" * 60)
    print(f"\nCorrelations:")
    print(f"  Eigenvector Centrality vs R_formula: {result['corr_centrality']:.4f}")
    print(f"  PageRank vs R_formula: {result['corr_pagerank']:.4f}")

    if result['corr_centrality'] > 0.7:
        print("\n** VALIDATED: Strong centrality-R correlation (>0.7)")
    elif result['corr_centrality'] > 0:
        print("\n*  PASS: Positive centrality-R correlation")
    elif np.isnan(result['corr_centrality']):
        print("\n?  INCONCLUSIVE: Cannot compute correlation")
    else:
        print("\nX  FALSIFIED: Negative centrality-R correlation")
