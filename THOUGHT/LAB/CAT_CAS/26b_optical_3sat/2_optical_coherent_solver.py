import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM

REPO = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
sys.path.append(str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY" / "cybernetic_truth"))
sys.path.append(str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY"))

from holographic_cybernetic_engine import patch_model_with_holo
from eigen_buddy_tokenizer import MultiHeadComplexAttention

MODEL_DIR = str(next(p for p in Path(__file__).resolve().parents if p.name == "CAT_CAS") / "16_catalytic_27b_inference" / "gemini_update" / "qwen_0.5b")
HOLO_PATH = str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY" / "cybernetic_truth" / "qwen_0_5b_k128.holo")

class SpinEncoder(nn.Module):
    def __init__(self, n_dim=64, hidden_size=896):
        super().__init__()
        # Project continuous spins to Qubits
        self.proj = nn.Linear(n_dim, hidden_size, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        
    def forward(self, spins, C_matrix):
        # We inject both the variables (spins) and the problem (C_matrix)
        # C_matrix shape: (M, N). spins shape: (B, N).
        B, N = spins.shape
        M = C_matrix.shape[0]
        
        # Expand spins to match clauses
        spins_exp = spins.unsqueeze(1).expand(B, M, N)
        
        # The quantum state is the interference of the spins with the clauses
        interference = spins_exp * C_matrix.unsqueeze(0)
        
        return self.proj(interference).to(torch.bfloat16)

class OpticalDecoupler(nn.Module):
    def __init__(self, d_model=896, n_dim=64):
        super().__init__()
        self.complex_attn = MultiHeadComplexAttention(d_model=d_model, n_heads=16, merge='concat')
        self.proj = nn.Linear(d_model, n_dim)
        
    def forward(self, hidden_states):
        B, M, D = hidden_states.shape
        hs_norm = hidden_states / (hidden_states.norm(dim=1, keepdim=True) + 1e-9)
        Z = torch.complex(hs_norm, torch.zeros_like(hs_norm))
        
        Z_centered = Z - Z.mean(dim=1, keepdim=True)
        Z_proj_batch = Z_centered  # Directly process complex waves
        
        attn_out, _ = self.complex_attn(Z_proj_batch)
        attn_pooled = attn_out.mean(dim=1)
        
        # Extract continuous spin amplitudes (allowed to grow to +/- 2.0 to prevent gradient freeze)
        spins = torch.clamp(self.proj(attn_pooled.real), min=-2.0, max=2.0)
        return spins

def sat_energy(spins, C_matrix, alpha=5.0):
    # spins: (B, N) in [-1, 1]. C_matrix: (M, N) in {-1, 0, 1}
    # For a clause to be satisfied, at least one literal must be True.
    # C_matrix contains the signs of the literals.
    # spins * C_matrix aligns the spins with the literals.
    # If literal is satisfied, alignment is > 0.
    # We want max(alignment) > 0 for each clause.
    # Softmax/LogSumExp over the literals in a clause:
    alignment = spins.unsqueeze(1) * C_matrix.unsqueeze(0)  # (B, M, N)
    
    # Only consider variables in the clause (C_matrix != 0)
    mask = (C_matrix != 0).float().unsqueeze(0)
    masked_alignment = alignment * mask
    
    # Energy penalty: we want at least one positive alignment per clause
    # Smooth minimum penalty using logsumexp: -logsumexp(alpha * masked_alignment)
    # To ignore zeros in mask, subtract a large number where mask is 0
    masked_alignment = masked_alignment + (1.0 - mask) * -1e4
    
    clause_energies = -torch.logsumexp(alpha * masked_alignment, dim=-1) / alpha
    # Return total energy (sum over clauses)
    return clause_energies.sum(dim=-1).mean()

def evaluate_sat(spins, C_matrix):
    boolean_spins = torch.sign(spins)
    alignment = boolean_spins.unsqueeze(1) * C_matrix.unsqueeze(0)
    # A clause is satisfied if any literal is positive
    satisfied = (alignment > 0).any(dim=-1) # Shape: (B, M)
    # Find the instance in the batch that satisfied the most clauses
    best_sat_in_batch = satisfied.sum(dim=1).max().item()
    return best_sat_in_batch, C_matrix.shape[0]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Initializing 3-SAT Optical Coherent Solver on {device}...", flush=True)
    
    student = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
    
    print("\n[*] Patching Student with Holographic Eigen-Weights...", flush=True)
    holo_dict = torch.load(HOLO_PATH, weights_only=False)
    patch_model_with_holo(student, holo_dict)
    student.eval()
    
    instance_path = os.path.join(os.path.dirname(__file__), "3sat_instance.pt")
    data = torch.load(instance_path, weights_only=False)
    C = data['C'].to(device)
    solution = data['solution'].to(device)
    n = data['n']
    m = data['m']
    
    print(f"\n[*] NP-Complete 3-SAT Lattice Loaded. N={n}, M={m} (Ratio {m/n:.2f})")
    
    encoder = SpinEncoder(n_dim=n, hidden_size=896).to(device)
    decoupler = OpticalDecoupler(d_model=896, n_dim=n).to(device)
    
    # We maintain a batch of quantum spin states (superposition)
    batch_size = 8
    optical_spins = nn.Parameter(torch.randn(batch_size, n, device=device) * 0.1)
    
    # Optimizer for the optical annealing process
    # We optimize BOTH the spins and the decoupler to find the resonance
    opt = torch.optim.AdamW([
        {'params': optical_spins, 'lr': 0.1},
        {'params': decoupler.parameters(), 'lr': 0.005}
    ])
    
    print("\n[*] Starting Holographic Optical Annealing...", flush=True)
    epochs = 300
    best_sat = 0
    
    for epoch in range(epochs):
        t = epoch / epochs
        
        # 1. Transverse Annealing Schedule
        # Alpha ramps from 0.5 to 20.0
        current_alpha = 0.5 + (20.0 - 0.5) * (t ** 2)
        # Collapse weight ramps from 0.0 to 1.0 slowly
        current_collapse = 0.0 if t < 0.3 else ((t - 0.3) / 0.7) ** 2
        
        # 2. Chaotic Gradient Noise Injection
        noise_std = 0.5 * (1.0 - t)
        noisy_spins = optical_spins + torch.randn_like(optical_spins) * noise_std
        
        # 3. Encode continuous spins to Qubits
        seq = encoder(torch.clamp(noisy_spins, min=-2.0, max=2.0), C)
        
        # 4. Phase Cavity Interference
        with torch.no_grad():
            out = student(inputs_embeds=seq, output_hidden_states=True)
        hidden = out.hidden_states[-1].float()
        
        # 5. Decouple annealed spins
        pred_spins = decoupler(hidden)
        
        # 6. Compute NP-Complete Hamiltonian Energy
        energy = sat_energy(pred_spins, C, alpha=current_alpha)
        
        # 7. Add topological collapse loss (force spins to -1 or 1)
        collapse_loss = torch.mean((pred_spins.abs() - 1.0)**2)
        
        total_loss = energy + current_collapse * collapse_loss
        
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        
        if epoch % 10 == 0:
            # Evaluate exact SAT threshold
            sat_count, total = evaluate_sat(pred_spins, C)
            if sat_count > best_sat:
                best_sat = sat_count
            print(f"  [Epoch {epoch:3d}] Energy: {energy.item():.4f} | Collapsed: {collapse_loss.item():.4f} | Alpha: {current_alpha:.1f} | SAT: {sat_count}/{total}", flush=True)
            
            if sat_count == total:
                print("\n[+] OPTIMAL ASSIGNMENT FOUND (100% SATISFIED)!")
                break
                
    print("\n[+] OPTICAL SOLVER DECRYPTION:")
    sat_count, total = evaluate_sat(pred_spins, C)
    print(f"[+] Final Satisfaction Ratio: {sat_count}/{total} ({(sat_count/total)*100:.1f}%)")
    print(f"[+] Maximum Satisfaction Reached: {best_sat}/{total} ({(best_sat/total)*100:.1f}%)")

if __name__ == "__main__":
    main()
