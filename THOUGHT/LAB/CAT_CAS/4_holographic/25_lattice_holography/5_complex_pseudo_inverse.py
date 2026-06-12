import os
import math
import torch

def load_lwe_instance(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("lwe", os.path.join(os.path.dirname(__file__), "1_lwe_simulator.py"))
    lwe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lwe)
    if not os.path.exists(path):
        return lwe.generate_lwe_instance()
    return torch.load(path, weights_only=False)

def complex_pseudo_inverse_attack():
    print("\n[*] Initializing Complex Torus Pseudo-Inverse Attack...")
    instance_path = os.path.join(os.path.dirname(__file__), "lwe_instance.pt")
    data = load_lwe_instance(instance_path)
    
    A = data['A'].to(torch.float64) # [1024, 128]
    B = data['B'].to(torch.float64) # [1024, 1]
    S_true = data['S_true'].to(torch.float64)
    q = data['q']
    m, n = A.shape
    
    # Map to Torus Space
    Z_A = torch.exp(1j * 2 * math.pi * A / q)
    Z_B = torch.exp(1j * 2 * math.pi * B / q)
    
    print(f"[*] Solving N={n}, M={m} via Torus Pseudo-Inverse...")
    # S_complex = pseudo_inverse(Z_A) @ Z_B
    Z_A_pinv = torch.linalg.pinv(Z_A)
    S_complex = torch.matmul(Z_A_pinv, Z_B)
    
    # Extract phase and map back to modulo q
    S_phase = torch.angle(S_complex)
    
    # Adjust phase from [-pi, pi] to [0, 2pi]
    S_phase = (S_phase + 2 * math.pi) % (2 * math.pi)
    
    S_pred = torch.round(S_phase / (2 * math.pi) * q)
    
    # Wait, the pseudo-inverse solves Z_B = Z_A * S_complex
    # But the actual LWE equation is B = A * S mod q
    # Z_B = exp(i 2pi (A S)/q)
    # Z_A = exp(i 2pi A/q)
    # Z_A * S_complex performs complex matrix multiplication, which is fundamentally different from phase multiplication!
    # A * S in integers means summing up the columns of A weighted by S.
    # In the Torus space, Z_A * S_complex means summing up the complex vectors.
    # But phase summation requires MULTIPLYING the complex elements!
    # Z_B[j] = Product_{k} (Z_A[j, k])^S[k]
    # This is highly non-linear in complex space. Pseudo-inverse solves the linear complex system, not the multiplicative one.
    
    # But let's see if the continuous approximation holds anyway!
    S_diff = (S_pred - S_true) % q
    S_diff[S_diff > q/2] -= q
    error_norm = torch.norm(S_diff).item()
    
    print(f"[*] Post-Quantum S Error Norm: {error_norm:.2f}")

if __name__ == "__main__":
    complex_pseudo_inverse_attack()
