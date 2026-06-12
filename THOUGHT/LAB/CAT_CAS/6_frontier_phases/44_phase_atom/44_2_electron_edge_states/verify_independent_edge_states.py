"""
Independent verification of Exp 47.2 thesis: Non-Hermitian skin effect produces
edge-localized eigenstates that a Hermitian control lattice cannot produce.
"""
import numpy as np
import os, sys

def build_hamiltonian(L, gamma, mu_boundary=0.0):
    N = L * L
    H = np.zeros((N, N), dtype=complex)
    t = 1.0
    core_min = L // 2 - 1
    core_max = L // 2 + 1
    boundary_indices = []
    for x in range(L):
        for y in range(L):
            i = x * L + y
            if x == 0 or x == L - 1 or y == 0 or y == L - 1:
                boundary_indices.append(i)
                H[i, i] += mu_boundary
            if core_min <= x <= core_max and core_min <= y <= core_max:
                H[i, i] += -100.0j
            if x < L - 1:
                j = (x + 1) * L + y
                H[i, j] += t + gamma
                H[j, i] += t - gamma
            if y < L - 1:
                j = x * L + (y + 1)
                H[i, j] += t + 1j * gamma
                H[j, i] += t - 1j * gamma
    return H, boundary_indices

def count_edge_states(H, boundary_indices, edge_threshold=0.5):
    evals, evecs = np.linalg.eig(H)
    N = H.shape[0]
    count = 0
    for i in range(N):
        v = evecs[:, i]
        prob = np.abs(v)**2 / np.sum(np.abs(v)**2)
        if np.sum(prob[boundary_indices]) > edge_threshold:
            count += 1
    return count

def run():
    lines = []
    lines.append("=" * 90)
    lines.append("INDEPENDENT VERIFICATION: EXP 47.2 EDGE STATE PHYSICS")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Thesis: Non-Hermitian skin effect (gamma>0) produces edge-localized")
    lines.append("eigenstates that do NOT exist in a Hermitian control (gamma=0).")
    lines.append("")

    for L in [8, 12, 15]:
        lines.append(f"--- Lattice {L}x{L} ---")
        for gamma in [0.0, 0.6]:
            H, boundary = build_hamiltonian(L, gamma)
            n_edge = count_edge_states(H, boundary)
            label = "Hermitian (gamma=0.0)" if gamma == 0.0 else "Non-Hermitian (gamma=0.6)"
            lines.append(f"  {label}: {n_edge} edge states")
        H0, _ = build_hamiltonian(L, 0.0)
        H6, _ = build_hamiltonian(L, 0.6)
        n0 = count_edge_states(H0, boundary)
        n6 = count_edge_states(H6, boundary)
        lines.append(f"  Non-Hermitian / Hermitian ratio: {n6/max(n0,1):.1f}x")
        lines.append("")

    lines.append("--- CONCLUSION ---")
    lines.append("If the non-Hermitian skin effect is real, gamma=0.6 should produce")
    lines.append("significantly MORE edge states than gamma=0.0 at all lattice sizes.")
    lines.append("If edge states are purely geometric, both should be similar.")

    report = "\n".join(lines)
    print(report)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "TELEMETRY_44_2_INDEPENDENT_VERIFY.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport: {path}")

if __name__ == "__main__":
    run()
