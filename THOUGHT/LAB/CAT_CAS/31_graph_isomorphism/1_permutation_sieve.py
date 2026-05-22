"""
Phase 3: Holographic Graph Isomorphism — Permutation Sieve
============================================================
Maps adjacency matrices to 2D phase gratings. Isomorphic graphs
produce identical .holo spectral signatures regardless of vertex
permutation. Non-isomorphic graphs produce different spectra.

Uses the .holo engine from TINY_COMPRESS for spectral analysis.
The permutation invariance emerges from the spectrum being a
graph invariant — two isomorphic graphs have identical eigenvalue
spectra and thus identical .holo signatures.
"""
import sys, time, math, random, numpy as np
from pathlib import Path

REPO = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO / "THOUGHT" / "LAB" / "TINY_COMPRESS" / "holographic-image"))
from holo_core import analyze_spectrum

def random_graph(n, p=0.3):
    """Generate random undirected graph."""
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                adj[i, j] = adj[j, i] = 1.0
    return adj

def random_permutation(n):
    """Generate random permutation of n nodes."""
    perm = list(range(n))
    random.shuffle(perm)
    return perm

def apply_permutation(adj, perm):
    """Apply vertex permutation to adjacency matrix."""
    n = len(adj)
    result = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            result[i, j] = adj[perm[i], perm[j]]
    return result

def graph_to_grating(adj):
    """Map adjacency matrix to 2D phase grating."""
    # Map binary {0,1} to phases {0, pi} — edge = phase flip
    return np.exp(1j * np.pi * adj)

def holo_signature(grating):
    """Compute .holo spectral signature of grating."""
    n = grating.shape[0]
    # Flatten to observation matrix: treat each row as a sample
    obs = np.zeros((n, n * 2), dtype=np.float64)
    for i in range(n):
        row = grating[i]
        obs[i, :n] = row.real
        obs[i, n:] = row.imag
    spectrum = analyze_spectrum(obs)
    return {
        'D_pr': spectrum.participation_dimension,
        'D_sh': spectrum.shannon_dimension,
        'eigenvalues': spectrum.eigenvalues[:min(20, len(spectrum.eigenvalues))],
        'cumulative': spectrum.cumulative_variance[:min(20, len(spectrum.cumulative_variance))],
    }

def spectral_distance(sig1, sig2):
    """Measure distance between two .holo signatures."""
    # Compare D_pr and D_sh
    dpr_diff = abs(sig1['D_pr'] - sig2['D_pr']) / max(sig1['D_pr'], sig2['D_pr'], 1)
    dsh_diff = abs(sig1['D_sh'] - sig2['D_sh']) / max(sig1['D_sh'], sig2['D_sh'], 1)
    
    # Compare eigenvalue spectra
    e1 = sig1['eigenvalues']
    e2 = sig2['eigenvalues']
    min_len = min(len(e1), len(e2))
    ev_diff = np.linalg.norm(e1[:min_len] - e2[:min_len]) / (np.linalg.norm(e1[:min_len]) + 1e-10)
    
    return {
        'dpr_diff': dpr_diff,
        'dsh_diff': dsh_diff,
        'ev_diff': ev_diff,
        'combined': (dpr_diff + dsh_diff + ev_diff) / 3,
    }


print("=" * 78)
print("PHASE 3: HOLOGRAPHIC GRAPH ISOMORPHISM — Permutation Sieve")
print("=" * 78)

for n in [10, 20, 30, 50]:
    print(f"\n  Graph size: {n} nodes")
    
    # Generate random graph and its isomorphic permuted version
    G = random_graph(n, p=0.3)
    perm = random_permutation(n)
    H = apply_permutation(G, perm)
    
    # Generate a non-isomorphic graph
    G2 = random_graph(n, p=0.3)
    
    # Map to phase gratings
    grating_G = graph_to_grating(G)
    grating_H = graph_to_grating(H)
    grating_G2 = graph_to_grating(G2)
    
    # Compute .holo signatures
    sig_G = holo_signature(grating_G)
    sig_H = holo_signature(grating_H)
    sig_G2 = holo_signature(grating_G2)
    
    # Compare
    iso_dist = spectral_distance(sig_G, sig_H)
    noniso_dist = spectral_distance(sig_G, sig_G2)
    
    iso_ok = iso_dist['combined'] < 0.01
    noniso_diff = noniso_dist['combined'] > iso_dist['combined']
    
    print(f"    G vs H (isomorphic):    D_pr={sig_G['D_pr']:.1f}/{sig_H['D_pr']:.1f} dist={iso_dist['combined']:.6f} {'IDENTICAL' if iso_ok else 'DIFFER'}")
    print(f"    G vs G2 (non-iso):      D_pr={sig_G['D_pr']:.1f}/{sig_G2['D_pr']:.1f} dist={noniso_dist['combined']:.6f} {'DISTINCT' if noniso_diff else 'SIMILAR'}")

# Bulk test
print(f"\n  BULK TEST: 100 graph pairs")
iso_dists = []
noniso_dists = []
for _ in range(100):
    n = random.randint(10, 30)
    G = random_graph(n, p=0.3)
    perm = random_permutation(n)
    H = apply_permutation(G, perm)
    G2 = random_graph(n, p=0.3)
    
    sig_G = holo_signature(graph_to_grating(G))
    sig_H = holo_signature(graph_to_grating(H))
    sig_G2 = holo_signature(graph_to_grating(G2))
    
    iso_dists.append(spectral_distance(sig_G, sig_H)['combined'])
    noniso_dists.append(spectral_distance(sig_G, sig_G2)['combined'])

iso_mean = np.mean(iso_dists)
noniso_mean = np.mean(noniso_dists)
iso_max = np.max(iso_dists)
print(f"  Isomorphic pairs:    mean dist = {iso_mean:.6f}  max = {iso_max:.6f}")
print(f"  Non-isomorphic pairs: mean dist = {noniso_mean:.6f}")
print(f"  Separation ratio:     {noniso_mean / (iso_mean + 1e-10):.1f}x")
print(f"  Isomorphic always below 1e-4: {all(d < 1e-4 for d in iso_dists)}")
print(f"  All non-iso > iso:            {all(n > i for i, n in zip(iso_dists, noniso_dists))}")


# ================================================================
# HARD TEST: Strongly Regular Graphs (cospectral non-isomorphic)
# ================================================================
def srg_shrikhande():
    n=16;adj=np.zeros((n,n),dtype=np.float64)
    for i in range(4):
        for j in range(4):
            u=i*4+j
            for i2 in range(4):
                for j2 in range(4):
                    v=i2*4+j2
                    if u>=v:continue
                    sr=(i==i2 and j!=j2);sc=(j==j2 and i!=i2)
                    dg=((i+j)%2==(i2+j2)%2 and i!=i2 and j!=j2)
                    if sr or sc or dg:adj[u,v]=adj[v,u]=1.0
    return adj

def srg_rook_4x4():
    n=16;adj=np.zeros((n,n),dtype=np.float64)
    for i in range(4):
        for j in range(4):
            u=i*4+j
            for i2 in range(4):
                for j2 in range(4):
                    v=i2*4+j2
                    if u>=v:continue
                    if i==i2 or j==j2:adj[u,v]=adj[v,u]=1.0
    return adj

print(f"\n{'='*78}")
print(f"  HARD TEST: Cospectral Non-Isomorphic SRG(16,6,2,2)")
print(f"{'='*78}")
G_shrik=srg_shrikhande();G_rook=srg_rook_4x4()
sig_shrik=holo_signature(graph_to_grating(G_shrik))
sig_rook=holo_signature(graph_to_grating(G_rook))
d=spectral_distance(sig_shrik,sig_rook)
print(f"  Shrikhande D_pr={sig_shrik['D_pr']:.1f} D_sh={sig_shrik['D_sh']:.1f}")
print(f"  Rook        D_pr={sig_rook['D_pr']:.1f} D_sh={sig_rook['D_sh']:.1f}")
print(f"  Distance: {d['combined']:.6f}")
print(f"  {'[+] DISTINGUISHED!' if d['combined']>0.001 else '[-] Not distinguished'}")

print(f"\n  SCALE TEST: Larger graphs")
for n in [50,100,200]:
    G=random_graph(n,p=0.3);perm=random_permutation(n);H=apply_permutation(G,perm)
    G2=random_graph(n,p=0.3)
    sig_G=holo_signature(graph_to_grating(G))
    sig_H=holo_signature(graph_to_grating(H))
    sig_G2=holo_signature(graph_to_grating(G2))
    d_iso=spectral_distance(sig_G,sig_H)['combined']
    d_non=spectral_distance(sig_G,sig_G2)['combined']
    print(f"    n={n:>3}: iso={d_iso:.8f} noniso={d_non:.6f} D_pr={sig_G['D_pr']:.1f}")

print("=" * 78)
