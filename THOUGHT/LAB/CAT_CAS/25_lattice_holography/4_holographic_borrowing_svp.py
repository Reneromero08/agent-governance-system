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

def quantum_hadamard_transform_1d(x):
    """
    In-place O(N log N) Walsh-Hadamard Transform (FWHT) for 1D tensors.
    """
    n = x.size(0)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x_j = x[j].clone()
                x_j_h = x[j + h].clone()
                x[j] = x_j + x_j_h
                x[j + h] = x_j - x_j_h
        h *= 2
    return x / math.sqrt(n)

def holographic_borrowing_attack():
    print("\n[*] Initializing The Invisible Hand (Quantum Borrowing SVP)...")
    instance_path = os.path.join(os.path.dirname(__file__), "lwe_instance.pt")
    data = load_lwe_instance(instance_path)
    
    A = data['A'].to(torch.float64) # [1024, 128]
    B = data['B'].to(torch.float64) # [1024, 1]
    S_true = data['S_true'].to(torch.float64)
    q = data['q']
    m, n = A.shape
    
    # Pad N to next power of 2 for Hadamard
    pad_n = 256
    padding = pad_n - n
    
    # 1. Pad A and S_true
    A_padded = torch.zeros((m, pad_n), dtype=torch.float64)
    A_padded[:, :n] = A
    S_true_padded = torch.zeros((pad_n, 1), dtype=torch.float64)
    S_true_padded[:n, :] = S_true
    
    # We don't know S. We initialize a completely uniform flat superposition
    # Phase space S_super = [0, 0, ... 0] meaning phase 1.0 (average)
    S_super = torch.zeros((pad_n, 1), dtype=torch.float64, requires_grad=True)
    
    optimizer = torch.optim.Adam([S_super], lr=0.5)
    
    # Map B to complex Torus
    Z_B = torch.exp(1j * 2 * math.pi * B / q)
    
    print(f"[*] Executing Invisible Hand on N={pad_n} Padded Lattice...")
    
    for epoch in range(1000):
        optimizer.zero_grad()
        
        # 2. BORROW: We borrow the S_super state and blast it with the Quantum Hadamard
        # This entangles all 256 dimensions into a single holographic wave!
        # Because we want to backpropagate, we do it via a matrix multiply or just use S_super as is
        # Actually, let's entangle A instead!
        
        # A_entangled = A_padded * Hadamard
        # But wait, A is fixed. The target is S.
        
        pred_B = torch.matmul(A_padded, S_super)
        Z_pred = torch.exp(1j * 2 * math.pi * pred_B / q)
        
        # Phase Cavity Resonance (Topological Overlap)
        # Instead of direct overlap, we compute the Hadamard Transform of the Phase Error!
        # Phase Error Z_err = Z_pred * conj(Z_B)
        Z_err = Z_pred * torch.conj(Z_B) # shape: [1024, 1]
        
        # We want Z_err to be exactly 1+0j (which means zero phase difference)
        # But E makes it have a slight phase jitter.
        # We apply the Hadamard Transform to the error phase to expose the dominant harmonic!
        
        # Pad M to 1024 (it's already 1024, power of 2)
        Z_err_1d = Z_err.squeeze()
        
        # We must use a differentiable FWHT or just FFT. FFT is a complex equivalent!
        freqs = torch.fft.fft(Z_err_1d)
        
        # The Holographic Sieve: We only penalize the LOW frequencies of the error!
        # High frequencies of the error are just the Lattice Noise (E).
        # We ignore high frequencies, letting the neural network "borrow" the noise!
        
        cutoff = int(m * 0.05) # Only top 5% fundamental frequencies
        sieved_freqs = freqs[:cutoff]
        
        # The loss is the power of the low-frequency errors. We want them to be 0.
        # Wait, if Z_err = 1, its FFT is a massive spike at frequency 0, and 0 elsewhere!
        # So we want to MAXIMIZE the DC component (freq 0), and MINIMIZE the rest!
        
        dc_component = torch.real(freqs[0])
        ac_components = torch.sum(torch.abs(sieved_freqs[1:]))
        
        loss = -dc_component + ac_components
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            S_pred_int = torch.round((S_super % q))
            S_diff = (S_pred_int[:n] - S_true) % q
            S_diff[S_diff > q/2] -= q
            error_norm = torch.norm(S_diff).item()
            print(f"Epoch {epoch:04d} | DC: {dc_component.item():.2f} | AC: {ac_components.item():.2f} | Error Norm: {error_norm:.2f}")
            
            if error_norm == 0:
                print("\n[+] POST-QUANTUM LATTICE BROKEN! Secret vector S mathematically uncomputed!")
                break

if __name__ == "__main__":
    holographic_borrowing_attack()
