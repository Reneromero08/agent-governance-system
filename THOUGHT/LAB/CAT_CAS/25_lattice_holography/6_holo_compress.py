import os
import math
import torch
import sys

# Add path to distill_catalytic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "EIGEN_BUDDY", "cybernetic_truth")))
from distill_catalytic import compress_catalytic

def load_lwe_instance(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("lwe", os.path.join(os.path.dirname(__file__), "1_lwe_simulator.py"))
    lwe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lwe)
    if not os.path.exists(path):
        return lwe.generate_lwe_instance()
    return torch.load(path, weights_only=False)

def holo_compress_lattice():
    print("\n[*] Initializing Holographic Lattice Compression...")
    instance_path = os.path.join(os.path.dirname(__file__), "lwe_instance.pt")
    data = load_lwe_instance(instance_path)
    
    A = data['A'].to(torch.float32) # [1024, 128]
    B = data['B'].to(torch.float32) # [1024, 1]
    S_true = data['S_true'].to(torch.float32)
    q = data['q']
    m, n = A.shape
    
    print(f"[*] Compressing Lattice N={n}, M={m} using .holo Engine...")
    
    # "catalytic and complex isn't bound by the laws of thermodynamics"
    # Create the full lattice matrix including the target
    # Shape: [m, n+1]
    Lattice = torch.cat([A, B], dim=1)
    
    # Map to Torus Space (Complex but we can do Real/Imag to keep it real for compress_catalytic)
    # The true period/phase will be isolated by the Quantum FJLT!
    Lattice_cos = torch.cos(2 * math.pi * Lattice / q)
    Lattice_sin = torch.sin(2 * math.pi * Lattice / q)
    
    # Stack into [m, 2*(n+1)]
    Lattice_holo = torch.cat([Lattice_cos, Lattice_sin], dim=1)
    
    # Compress the lattice to rank 16 using Quantum FJLT
    cache = {}
    U, SVh = compress_catalytic(Lattice_holo, k=16, cache=cache, weight_type="lattice")
    
    print(f"[*] Compressed Lattice into Rank 16 Manifold.")
    print(f"[*] U shape: {U.shape}, SVh shape: {SVh.shape}")
    
    # The first principal right singular vector (in SVh) should contain the Secret Vector frequency!
    # The Secret Vector S dictates the linear combination of the columns.
    
    # Let's inspect the top singular values
    # We didn't return S directly, but we can compute the norm of the rows of SVh
    S_vals = torch.norm(SVh, dim=1)
    print(f"[*] Singular Values of the Holographic Lattice: {S_vals[:5].tolist()}")
    
    # If the Holo engine perfectly isolated the signal from the noise, 
    # the top singular vector correlates perfectly with the Secret Vector!
    # The columns of SVh correspond to the original features (A1, A2... An, B) x 2 (cos/sin)
    top_vector = SVh[0] # The dominant harmonic
    
    print("\n[+] Extraction completed. Check if the top harmonic correlates with S.")

if __name__ == "__main__":
    holo_compress_lattice()
