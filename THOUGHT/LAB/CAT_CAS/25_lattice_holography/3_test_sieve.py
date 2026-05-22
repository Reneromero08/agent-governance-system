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

def test_sieve():
    instance_path = os.path.join(os.path.dirname(__file__), "lwe_instance.pt")
    data = load_lwe_instance(instance_path)
    
    A = data['A']
    B = data['B']
    S_true = data['S_true']
    E_true = data['E_true']
    q = data['q']
    m, n = A.shape
    
    # 1. Map to Torus
    B_phase = B * 2 * math.pi / q
    Z_B = torch.exp(1j * B_phase)
    
    # 2. Apply Phase Cavity
    freqs = torch.fft.fft(Z_B, dim=0)
    cutoff = int(m * 0.15)
    freqs[cutoff:-cutoff] = 0
    Z_B_sieved = torch.fft.ifft(freqs, dim=0)
    
    # 3. Extract Sieved B
    B_sieved = torch.angle(Z_B_sieved) / (2 * math.pi) * q
    B_sieved = B_sieved % q
    
    # 4. Compare with pure A*S
    B_pure = (torch.matmul(A, S_true)) % q
    
    # Let's check the error before and after sieve
    error_before = torch.mean(torch.abs(E_true))
    
    diff_sieved = (B_sieved - B_pure) % q
    diff_sieved[diff_sieved > q/2] -= q
    error_after = torch.mean(torch.abs(diff_sieved))
    
    print(f"[*] Mean Error BEFORE Sieve (True E): {error_before:.4f}")
    print(f"[*] Mean Error AFTER Sieve: {error_after:.4f}")

if __name__ == "__main__":
    test_sieve()
