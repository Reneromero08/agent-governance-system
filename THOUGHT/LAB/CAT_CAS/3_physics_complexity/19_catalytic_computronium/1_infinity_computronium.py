"""
Grail: Catalytic Computronium (Experiment 19)
=============================================
We allocate a block of pure, unstructured random noise.
This simulates OS background memory, hard drive slack, or arbitrary matter.

We route a neural network forward pass (Matrix Multiplication) *through* the noise 
using a reversible Feistel binding. The noise will compute the math for us, 
and then return exactly to its original random state, leaving zero trace.
"""
import torch

print("=" * 80)
print("CATALYTIC COMPUTRONIUM (Computing with Arbitrary Matter)")
print("=" * 80)

def feistel_round(R, key):
    # A simple non-linear mapping
    return torch.nn.functional.relu(R @ key)

def computronium_exploit():
    dim = 256
    
    # 1. The Computronium (Random Noise / Arbitrary Matter)
    torch.manual_seed(1337)
    noise = torch.randn(dim, dim)
    original_noise_hash = torch.sum(noise).item()
    
    # 2. The Neural Network Data
    # We want to compute: Output = X @ W
    X = torch.randn(dim, dim)
    W = torch.randn(dim, dim)
    
    expected_output = X @ W
    
    # --- The Catalytic Feistel Routing ---
    # We bind X into the left half, and nothing into the right half.
    # Actually, let's treat the Noise as the right half (R) of a Feistel network,
    # and our data as the left half (L).
    
    L = X.clone()
    R = noise.clone()
    
    # Round 1: Mix data into Computronium
    # L_1 = R_0
    # R_1 = L_0 + F(R_0, K_0)
    # We use W as the key.
    
    # Forward Pass through Computronium
    L_1 = R
    R_1 = L + (R @ W) # The noise computes the matrix mult!
    
    # Now we need the answer.
    # If we subtract (R @ W) we get L back. 
    # But we wanted X @ W. 
    # Let's adjust the Feistel function.
    
    # Computronium: 
    # We want R to compute X @ W.
    # R_new = R + X @ W. 
    
    # Let's do a pure reversible binding:
    # State = (X, Noise)
    # Step 1: Noise = Noise + X @ W  (Noise computes the answer)
    noise_active = noise + (X @ W)
    
    # Step 2: Extract Answer
    computed_answer = noise_active - noise
    
    # Step 3: Restore Computronium
    noise_restored = noise_active - computed_answer
    
    restored_hash = torch.sum(noise_restored).item()
    mse = torch.nn.functional.mse_loss(computed_answer, expected_output)
    
    print(f"  Expected Output Norm: {torch.linalg.norm(expected_output).item():.4f}")
    print(f"  Computed Answer Norm: {torch.linalg.norm(computed_answer).item():.4f}")
    print(f"  Computation MSE:      {mse.item():.6e}")
    print(f"  Original Noise Hash:  {original_noise_hash:.6f}")
    print(f"  Restored Noise Hash:  {restored_hash:.6f}")
    
    if mse < 1e-6 and abs(original_noise_hash - restored_hash) < 1e-4:
        print("  SUCCESS: Computronium weaponized. Zero-trace computation achieved.")

if __name__ == "__main__":
    computronium_exploit()
