import sys
import numpy as np
import importlib
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).parent.parent))

physics_engine = importlib.import_module("47_2_electron_edge_states")
CatalyticTape = physics_engine.CatalyticTape

def test_catalytic_tape():
    tape = CatalyticTape(size_mb=1) # 1MB for fast testing
    tape.verify()
    assert True

def test_bulk_boundary_correspondence():
    # Test that Edge States cannot penetrate the Nucleus core
    L = 15
    N = L * L
    H, core_indices, boundary_indices = physics_engine.build_hamiltonian(L, 0.0)
    
    evals, evecs = np.linalg.eig(H)
    
    max_core_overlap = 0.0
    edge_states = 0
    
    for i in range(N):
        v = evecs[:, i]
        prob = np.abs(v)**2 / np.sum(np.abs(v)**2)
        
        # Core state
        if np.imag(evals[i]) < -50:
            continue
            
        boundary_prob = np.sum(prob[boundary_indices])
        if boundary_prob > 0.5:
            edge_states += 1
            core_overlap = np.sum(prob[core_indices])
            if core_overlap > max_core_overlap:
                max_core_overlap = core_overlap
                
    assert edge_states > 0, "No topological edge states found!"
    assert max_core_overlap < 0.01, f"Edge states penetrated the Nucleus! Max overlap: {max_core_overlap}"

def test_shell_quantization():
    # Test that energy injection causes discrete edge state shell jumps
    L = 15
    N = L * L
    shell_counts = []
    
    mu_values = [0.0, 2.0, 4.0]
    for mu in mu_values:
        H, _, boundary_indices = physics_engine.build_hamiltonian(L, mu)
        evals, evecs = np.linalg.eig(H)
        edges = 0
        for i in range(N):
            v = evecs[:, i]
            prob = np.abs(v)**2 / np.sum(np.abs(v)**2)
            if np.sum(prob[boundary_indices]) > 0.5:
                edges += 1
        shell_counts.append(edges)
        
    # Ensure they don't stay identical, they must drop in discrete integer jumps
    assert len(set(shell_counts)) > 1, "Shells did not jump discretely. Quantization failed."
    assert shell_counts[0] > shell_counts[-1], "Higher energy did not reduce active boundary edge states."
