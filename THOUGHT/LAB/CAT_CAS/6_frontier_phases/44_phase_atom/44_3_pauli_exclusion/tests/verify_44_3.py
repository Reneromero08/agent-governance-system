import sys
import numpy as np
import importlib
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).parent.parent))

physics_engine = importlib.import_module("44_3_pauli_exclusion")
CatalyticTape = physics_engine.CatalyticTape

def test_catalytic_tape():
    tape = CatalyticTape(size_mb=1) # 1MB for fast testing
    tape.verify()
    assert True

def test_bosonic_degeneracy():
    L = 15
    boundary_indices = []
    for x in range(L):
        for y in range(L):
            if x == 0 or x == L - 1 or y == 0 or y == L - 1:
                boundary_indices.append(x * L + y)
                
    mu_nodes = {i: 10.0 for i in boundary_indices}
    H_bosonic = physics_engine.build_hamiltonian(L, mu_nodes, gamma=0.0)
    edge_evals_bosonic = physics_engine.get_edge_states(H_bosonic, boundary_indices)
    
    edge_evals_bosonic = sorted(edge_evals_bosonic, key=lambda x: np.real(x))
    gaps_bosonic = [abs(edge_evals_bosonic[i] - edge_evals_bosonic[i-1]) for i in range(1, len(edge_evals_bosonic))]
    min_gap_bosonic = min(gaps_bosonic)
    
    assert min_gap_bosonic < 1e-4, f"Bosonic lattice failed to allow degeneracy. Gap: {min_gap_bosonic}"

def test_fermionic_pauli_exclusion():
    L = 15
    boundary_indices = []
    for x in range(L):
        for y in range(L):
            if x == 0 or x == L - 1 or y == 0 or y == L - 1:
                boundary_indices.append(x * L + y)
                
    mu_nodes = {i: 10.0 for i in boundary_indices}
    H_fermionic = physics_engine.build_hamiltonian(L, mu_nodes, gamma=0.6)
    edge_evals_fermionic = physics_engine.get_edge_states(H_fermionic, boundary_indices)
    
    edge_evals_fermionic = sorted(edge_evals_fermionic, key=lambda x: np.real(x))
    gaps_fermionic = [abs(edge_evals_fermionic[i] - edge_evals_fermionic[i-1]) for i in range(1, len(edge_evals_fermionic))]
    min_gap_fermionic = min(gaps_fermionic)
    
    assert min_gap_fermionic > 0.001, f"Fermionic lattice allowed a Hash Collision! Gap: {min_gap_fermionic}"
