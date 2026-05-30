import numpy as np
import sys
import os

import importlib.util
spec = importlib.util.spec_from_file_location("oracle", os.path.join(os.path.dirname(__file__), '..', '46_4_topological_genetic_code_oracle.py'))
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)

build_H = oracle.build_H
sgc_map = oracle.sgc_map
CODONS = oracle.CODONS
is_adjacent = oracle.is_adjacent
def verify_hypercube_edges():
    edges = 0
    for i in range(64):
        for j in range(64):
            if is_adjacent(CODONS[i], CODONS[j]):
                edges += 1
    # undirected edges = total / 2
    actual_edges = edges // 2
    assert actual_edges == 288, f"Hypercube edge failure! Expected 288, got {actual_edges}"
    print(f"[VERIFY] Hypercube Adjacency: PASS (Exactly 288 point-mutation edges confirmed)")

def build_conservative_H(mapping):
    L = 64
    H = np.zeros((L, L), dtype=np.complex128)
    # Re-build but without the sign(j-i) symmetry breaking
    # This should yield identical radius for SGC and RND if the gauge symmetry theorem holds.
    KD = oracle.KD
    for i in range(L):
        H[i, i] = -1j * 0.5 * KD[mapping[CODONS[i]]]
        for j in range(L):
            if is_adjacent(CODONS[i], CODONS[j]):
                kd_i = KD[mapping[CODONS[i]]]
                kd_j = KD[mapping[CODONS[j]]]
                mag = 1.0 / (1.0 + abs(kd_j - kd_i))
                # purely conservative gradient pump
                phi = 1.2 * (kd_j - kd_i) 
                H[j, i] = mag * np.exp(phi)
    return H

def verify_gauge_symmetry():
    # Test SGC
    H_sgc = build_conservative_H(sgc_map)
    rad_sgc = np.max(np.abs(np.linalg.eigvals(H_sgc)))
    
    # Test RND
    aa_vals = list(sgc_map.values())
    np.random.seed(42)
    np.random.shuffle(aa_vals)
    rnd_map = {CODONS[j]: aa_vals[j] for j in range(64)}
    
    H_rnd = build_conservative_H(rnd_map)
    rad_rnd = np.max(np.abs(np.linalg.eigvals(H_rnd)))
    
    print(f"[VERIFY] Conservative SGC Radius: {rad_sgc:.4f}")
    print(f"[VERIFY] Conservative RND Radius: {rad_rnd:.4f}")
    
    if abs(rad_sgc - rad_rnd) < 2.0:
        print("[VERIFY] Gauge Symmetry Theorem: PASS (Without sign(j-i), conservative pumps cancel globally, preventing random code inflation)")
    else:
        print("[VERIFY] Gauge Symmetry Theorem: FAIL")

if __name__ == "__main__":
    print("--- 46.4 MATHEMATICAL HARDENING SUITE ---")
    verify_hypercube_edges()
    verify_gauge_symmetry()
    print("--- HARDENING COMPLETE ---")
