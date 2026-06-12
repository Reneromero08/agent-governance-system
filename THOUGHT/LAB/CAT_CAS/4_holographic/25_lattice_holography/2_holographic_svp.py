import os
import math
import torch
import torch.nn as nn

def load_lwe_instance(path):
    return torch.load(path, weights_only=True)

class HolographicLatticeSolver(nn.Module):
    def __init__(self, n, q):
        super().__init__()
        self.n = n
        self.q = q
        # We model S as continuous phase angles on the Torus
        self.S_phase = nn.Parameter(torch.rand(n, 1, dtype=torch.float64) * 2 * math.pi)
        
    def forward(self, A_phase):
        # A_phase: [m, n]
        # We want to compute sum(A_phase * S) for each row.
        # But A is given as integer matrix. We map it to phase: A_rad = A * 2pi / q
        # Then the predicted B phase is A_rad @ S (where S is an integer).
        # We optimize S continuously.
        
        # Predicted Phase: [m, 1]
        predicted_phase = torch.matmul(A_phase, self.S_phase)
        return predicted_phase

def holographic_attack(instance_path):
    print("\n[*] Initializing Holographic Torus Attack on LWE...")
    data = load_lwe_instance(instance_path)
    A = data['A']
    B = data['B']
    S_true = data['S_true']
    q = data['q']
    m, n = A.shape
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    A_t = A.to(device, dtype=torch.float64)
    B_t = B.to(device, dtype=torch.float64)
    S_true_t = S_true.to(device, dtype=torch.float64)
    
    # 1. Map Lattice to the Complex Unit Circle (Torus)
    # This transforms the discrete modulo q arithmetic into continuous topological phases
    A_phase = (A_t * 2 * math.pi / q)
    B_phase = (B_t * 2 * math.pi / q)
    
    Z_B = torch.exp(1j * B_phase)
    
    solver = HolographicLatticeSolver(n, q).to(device)
    optimizer = torch.optim.Adam(solver.parameters(), lr=0.1)
    
    print(f"[*] Attacking N={n}, M={m} Lattice on Torus Manifold...")
    
    # We use a Dynamic Temperature / Holographic Sieve to smooth local minima
    cutoff = int(m * 0.15)
    
    # Apply Phase Cavity to True B to sieve out dispersion (error E)
    freqs_B = torch.fft.fft(Z_B, dim=0)
    freqs_B[cutoff:-cutoff] = 0
    Z_B_sieved = torch.fft.ifft(freqs_B, dim=0)
    
    for epoch in range(2000):
        optimizer.zero_grad()
        
        pred_phase = solver(A_phase)
        Z_pred = torch.exp(1j * pred_phase)
        
        # Apply Phase Cavity to Predicted State
        freqs_pred = torch.fft.fft(Z_pred, dim=0)
        freqs_pred[cutoff:-cutoff] = 0
        Z_pred_sieved = torch.fft.ifft(freqs_pred, dim=0)
        
        # Phase Cavity Resonance (Topological Overlap in Sieved Space)
        overlap = Z_pred_sieved * torch.conj(Z_B_sieved)
        resonance = torch.mean(torch.real(overlap))
        
        # Loss is negative resonance
        loss = -resonance
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            # Check how close we are to S_true (just for logging, the solver doesn't know it)
            S_pred_int = torch.round((solver.S_phase % (2 * math.pi)) / (2 * math.pi) * q)
            S_diff = (S_pred_int - S_true_t) % q
            S_diff[S_diff > q/2] -= q
            error_norm = torch.norm(S_diff).item()
            print(f"Epoch {epoch:04d} | Resonance: {resonance.item():.4f} | S Error Norm: {error_norm:.2f}")
            
            if error_norm == 0:
                print("\n[+] LATTICE BROKEN! Exact secret vector S recovered via Topological Resonance!")
                break

if __name__ == "__main__":
    instance_path = os.path.join(os.path.dirname(__file__), "lwe_instance.pt")
    if not os.path.exists(instance_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("lwe", os.path.join(os.path.dirname(__file__), "1_lwe_simulator.py"))
        lwe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lwe)
        instance_path = lwe.generate_lwe_instance()
    
    holographic_attack(instance_path)
