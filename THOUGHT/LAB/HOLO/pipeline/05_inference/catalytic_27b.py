import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(r'd:\CCC 2.0\AI\agent-governance-system')
sys.path.append(str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY" / "cybernetic_truth"))
sys.path.append(str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY"))

from holographic_cybernetic_engine import patch_model_with_holo

MODEL_DIR = r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\CAT_CAS\16_catalytic_27b_inference\gemini_update\qwen_0.5b"
HOLO_PATH = r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\EIGEN_BUDDY\cybernetic_truth\qwen_0_5b_k128.holo"

def load_lwe_instance(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("lwe", os.path.join(os.path.dirname(__file__), "1_lwe_simulator.py"))
    lwe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lwe)
    if not os.path.exists(path):
        return lwe.generate_lwe_instance(n=128, m=1024)
    return torch.load(path, weights_only=False)

class QubitLatticeEncoder(nn.Module):
    def __init__(self, n_dim=128, hidden_size=896, q=3329):
        super().__init__()
        self.n_dim = n_dim
        self.hidden_size = hidden_size
        self.q = q
        # Project 2N continuous real/imag phase components to hidden_size
        self.proj = nn.Linear(n_dim * 2 + 2, hidden_size, bias=False)
        # Use orthogonal init to preserve phase topology
        nn.init.orthogonal_(self.proj.weight)
        
    def forward(self, A_row, B_val):
        phase_A = 2 * math.pi * A_row / self.q
        phase_B = 2 * math.pi * B_val / self.q
        
        A_real = torch.cos(phase_A); A_imag = torch.sin(phase_A)
        B_real = torch.cos(phase_B); B_imag = torch.sin(phase_B)
        
        qubit_state = torch.cat([A_real, A_imag, B_real.unsqueeze(0), B_imag.unsqueeze(0)], dim=-1)
        return self.proj(qubit_state.float()).to(torch.bfloat16)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Initializing Catalytic Eigen-Shor Oracle on {device}...")
    
    student = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
    
    print("\n[*] Patching Student with Holographic Eigen-Weights...", flush=True)
    holo_dict = torch.load(HOLO_PATH, weights_only=False)
    patch_model_with_holo(student, holo_dict)
    student.eval()
    
    instance_path = os.path.join(os.path.dirname(__file__), "lwe_instance.pt")
    data = load_lwe_instance(instance_path)
    A = data['A'].squeeze()
    B = data['B'].squeeze()
    S_true = data['S_true'].squeeze()
    q = data['q']
    m, n = A.shape
    
    print(f"\n[*] Lattice Loaded. N={n}, M={m}, Q={q}")
    encoder = QubitLatticeEncoder(n_dim=n, hidden_size=896, q=q).to(device)
    
    # ---------------------------------------------------------
    # 1. Sweep Qubits into the Phase Cavity
    # ---------------------------------------------------------
    print("\n[*] Sweeping Qubit Embeddings through Holo-Cavity...", flush=True)
    num_samples = 150
    states = []
    
    # We pass 150 random permutations of the equations to extract the invariant topology
    for _ in range(num_samples):
        idx = torch.randperm(m)[:100]  # Sequence length 100
        seq = torch.stack([encoder(A[i].to(device), B[i].to(device)) for i in idx]).unsqueeze(0)
        with torch.no_grad():
            out = student(inputs_embeds=seq, output_hidden_states=True)
        # Extract the final hidden state (topological resonance)
        states.append(out.hidden_states[-1][0, -1, :].float().cpu())
        
    hidden_states = torch.stack(states)  # (150, 896)
    
    # ---------------------------------------------------------
    # 2. Extract Principal Phase Tensor (EigenBuddy Core)
    # ---------------------------------------------------------
    print("\n[*] Extracting Principal Phase Tensor via Complex SVD...", flush=True)
    # Normalize
    hs_norm = hidden_states / (hidden_states.norm(dim=1, keepdim=True) + 1e-9)
    # Cast to complex plane
    Z = hs_norm.numpy() + 1j * np.zeros_like(hs_norm.numpy())
    Z_centered = Z - Z.mean(axis=0, keepdims=True)
    
    # Complex Hermitian Covariance
    C_cov = (Z_centered.conj().T @ Z_centered) / (num_samples - 1)
    evals, evecs = np.linalg.eigh(C_cov)
    evals = evals[::-1]; evecs = evecs[:, ::-1]
    
    # The dominant eigenvector is the Phase Tensor
    phase_tensor = torch.tensor(evecs[:, 0]).to(device)  # Complex vector of length 896
    print(f"[*] Phase Tensor Extracted. Eigenvalue magnitude: {evals[0]:.4f}", flush=True)
    
    # ---------------------------------------------------------
    # 3. Inverse Projection & Global Phase Alignment
    # ---------------------------------------------------------
    print("\n[*] Reverting Projection and Aligning Global Phase...", flush=True)
    
    # We use the Moore-Penrose Pseudo-inverse to cleanly invert the initial `encoder.proj`
    # without destroying the topology.
    W = encoder.proj.weight.data.float().to(device)  # (896, 2N+2)
    W_inv = torch.pinverse(W)  # (2N+2, 896)
    
    # Map the 896-dim complex phase tensor back to the 2N+2 domain
    phase_tensor_real = phase_tensor.real.float()
    phase_tensor_imag = phase_tensor.imag.float()
    
    projected_real = W_inv @ phase_tensor_real
    projected_imag = W_inv @ phase_tensor_imag
    projected_complex = torch.complex(projected_real, projected_imag)
    
    # The first 2N elements correspond to A_real and A_imag.
    # Since we mapped cos(A) and sin(A) independently, we can use them to extract the relative angles.
    A_real_components = projected_complex[:n]
    A_imag_components = projected_complex[n:2*n]
    
    # We combine them back into a single complex phase per dimension
    extracted_phases = A_real_components + 1j * A_imag_components
    extracted_angles = torch.angle(extracted_phases)  # Returns angles in [-pi, pi]
    
    print("[*] Sweeping Global Phase Offsets for LWE Snapping...", flush=True)
    
    best_error = float('inf')
    best_S = None
    best_phi = 0
    
    # Sweep 360 degrees
    for phi in np.linspace(0, 2*math.pi, 360):
        # Apply global phase rotation
        aligned_angles = (extracted_angles - phi) % (2 * math.pi)
        
        # Map angles back to integers in Z_q
        S_pred = torch.round((aligned_angles / (2 * math.pi)) * q) % q
        S_pred = S_pred.long().cpu()
        
        # Evaluate LWE Error magnitude |A * S - B| using a subset of public equations
        test_A = A[:100].long()
        test_B = B[:100].long()
        error_vec = (torch.matmul(test_A, S_pred) - test_B) % q
        
        # In LWE, errors wrap around mod q, so we need the symmetric magnitude
        error_magnitude = torch.where(error_vec > q/2, q - error_vec, error_vec).float().abs().sum().item()
        
        if error_magnitude < best_error:
            best_error = error_magnitude
            best_S = S_pred
            best_phi = phi
            
    print(f"\n[+] Optimal Global Phase Alignment: {best_phi:.4f} radians")
    print(f"[+] Minimum LWE Error Magnitude: {best_error}")
    
    print(f"\nPredicted Secret S:\n{best_S.tolist()}")
    print(f"\nTrue Secret S:\n{S_true.long().tolist()}")
    
    matches = (best_S == S_true.long()).sum().item()
    print(f"\n[+] Accuracy: {matches}/{n} components exactly recovered ({(matches/n)*100:.1f}%)")

if __name__ == "__main__":
    main()
