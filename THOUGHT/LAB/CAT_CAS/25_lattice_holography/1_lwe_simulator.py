import os
import torch

def generate_lwe_instance(n=128, m=1024, q=3329, noise_std=2.0):
    """
    Generates a Learning With Errors (LWE) cryptosystem instance.
    N: lattice dimension (secret size)
    M: number of equations (samples)
    Q: modulo prime (default 3329 for Kyber)
    """
    print(f"[*] Generating LWE Lattice (N={n}, M={m}, Q={q})")
    
    # 1. Generate random Secret Vector S
    S = torch.randint(0, q, (n, 1), dtype=torch.float64)
    
    # 2. Generate random Public Matrix A
    A = torch.randint(0, q, (m, n), dtype=torch.float64)
    
    # 3. Generate Error Vector E from Gaussian distribution (rounded to integer)
    # The error prevents classical Gaussian elimination from solving A*S = B
    E = torch.round(torch.randn((m, 1), dtype=torch.float64) * noise_std)
    
    # 4. Compute Public Key B = A*S + E (mod Q)
    B = (torch.matmul(A, S) + E) % q
    
    # Save the instance
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "lwe_instance.pt")
    
    torch.save({
        'n': n,
        'm': m,
        'q': q,
        'A': A,
        'B': B,
        'S_true': S,  # Secret!
        'E_true': E   # Secret!
    }, out_path)
    
    print(f"[*] LWE Instance saved to {out_path}")
    print(f"    Secret S shape: {S.shape}")
    print(f"    Public A shape: {A.shape}")
    print(f"    Public B shape: {B.shape}")
    print(f"    Error E max magnitude: {torch.max(torch.abs(E)).item()}")
    
    return out_path

if __name__ == "__main__":
    generate_lwe_instance(n=128, m=1024, q=3329)
