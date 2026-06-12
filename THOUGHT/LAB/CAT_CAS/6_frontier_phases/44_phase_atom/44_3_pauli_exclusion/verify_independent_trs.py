"""
Independent verification of Exp 47.3 thesis: TRS breaking via chiral Peierls pump
produces level repulsion (Pauli exclusion). Random perturbation should NOT replicate this.
"""
import numpy as np
import os

def build_base(L, mu_val=10.0):
    N = L * L
    t = 1.0
    core_min, core_max = L//2 - 1, L//2 + 1
    boundary_indices = [x*L+y for x in range(L) for y in range(L)
                        if x==0 or x==L-1 or y==0 or y==L-1]
    mu_nodes = {i: mu_val for i in boundary_indices}
    return N, t, core_min, core_max, boundary_indices, mu_nodes

def build_bosonic(L, mu_nodes, core_min, core_max, N, t):
    H = np.zeros((N,N), dtype=complex)
    for x in range(L):
        for y in range(L):
            i = x*L+y
            if i in mu_nodes: H[i,i] += mu_nodes[i]
            if core_min<=x<=core_max and core_min<=y<=core_max: H[i,i] += -100j
            if x<L-1: j=(x+1)*L+y; H[i,j]+=t; H[j,i]+=t
            if y<L-1: j=x*L+(y+1); H[i,j]+=t; H[j,i]+=t
    return H

def build_fermionic(L, mu_nodes, core_min, core_max, N, t):
    H = np.zeros((N,N), dtype=complex)
    alpha = 1.0/3.0
    for x in range(L):
        for y in range(L):
            i = x*L+y
            if i in mu_nodes: H[i,i] += mu_nodes[i]
            if core_min<=x<=core_max and core_min<=y<=core_max: H[i,i] += -100j
            if x<L-1: j=(x+1)*L+y; H[i,j]+=t; H[j,i]+=t
            if y<L-1: j=x*L+(y+1); phase=2*np.pi*alpha*x; H[i,j]+=t*np.exp(1j*phase); H[j,i]+=t*np.exp(-1j*phase)
    return H

def get_min_edge_gap(H, boundary_indices):
    evals, evecs = np.linalg.eig(H)
    edge_evals = []
    for i in range(len(evals)):
        if np.imag(evals[i]) < -50:
            continue
        v = evecs[:, i]
        prob = np.abs(v)**2 / np.sum(np.abs(v)**2)
        if np.sum(prob[boundary_indices]) > 0.5:
            edge_evals.append(evals[i])
    edge_evals = sorted(edge_evals, key=lambda x: np.real(x))
    if len(edge_evals) < 2:
        return float('nan')
    return min(abs(edge_evals[i] - edge_evals[i-1]) for i in range(1, len(edge_evals)))

def run():
    for L in [10, 12, 15]:
        N, t, core_min, core_max, boundary_indices, mu_nodes = build_base(L)

        H_b = build_bosonic(L, mu_nodes, core_min, core_max, N, t)
        H_f = build_fermionic(L, mu_nodes, core_min, core_max, N, t)

        # Random complex perturbation on the boundary (Hermitian base + noise on boundary edges)
        H_r = H_b.copy()
        np.random.default_rng(137)
        for i in boundary_indices:
            for j in boundary_indices:
                if i != j and abs(H_b[i,j]) < 0.001:
                    H_r[i,j] += np.random.normal(0, 0.5) + 1j*np.random.normal(0, 0.5)

        mg_b = get_min_edge_gap(H_b, boundary_indices)
        mg_f = get_min_edge_gap(H_f, boundary_indices)
        mg_r = get_min_edge_gap(H_r, boundary_indices)

        print(f"L={L}: bosonic={mg_b:.2e}  fermionic={mg_f:.2e}  random={mg_r:.2e}")
        if mg_b and mg_f and mg_r:
            if mg_b < 1e-8 and mg_f > 0.001 and mg_r < 1e-8:
                print(f"  -> VERIFIED: Only chiral pump breaks degeneracy. Random perturbation does NOT.")
            elif mg_r > 0.001:
                print(f"  -> WARNING: Random perturbation ALSO lifts degeneracy. Effect may not be topological.")
            else:
                print(f"  -> Mixed result: bosonic={mg_b} fermionic={mg_f} random={mg_r}")

    print("\nCONCLUSION: If random perturbation preserves degeneracy while Peierls pump")
    print("breaks it, the level repulsion is genuinely topological (TRS-breaking specific).")

if __name__ == "__main__":
    run()
