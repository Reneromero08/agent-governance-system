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

def load_lwe_instance(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("lwe", os.path.join(os.path.dirname(__file__), "1_lwe_simulator.py"))
    lwe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lwe)
    if not os.path.exists(path):
        return lwe.generate_lwe_instance(n=128, m=1024)
    return torch.load(path, weights_only=False)

def generate_synthetic_lwe(batch_size=1, n=128, m=50, q=3329, noise_std=0.0):
    S = torch.randint(0, q, (batch_size, n, 1), dtype=torch.float64)
    A = torch.randint(0, q, (batch_size, m, n), dtype=torch.float64)
    E = torch.round(torch.randn((batch_size, m, 1), dtype=torch.float64) * noise_std)
    B = (torch.matmul(A, S) + E) % q
    return A, B, S.squeeze(-1)

class QubitLatticeEncoder(nn.Module):
    def __init__(self, n_dim=128, hidden_size=896, q=3329):
        super().__init__()
        self.n_dim = n_dim
        self.hidden_size = hidden_size
        self.q = q
        self.proj = nn.Linear(n_dim * 2 + 2, hidden_size, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        
    def forward(self, A_mat, B_mat):
        phase_A = 2 * math.pi * A_mat / self.q
        phase_B = 2 * math.pi * B_mat / self.q
        
        A_real = torch.cos(phase_A); A_imag = torch.sin(phase_A)
        # B_mat could be (B, M) or (B, M, 1) or (M, 1). Just reshape to (*A_mat_shape[:-1], 1)
        B_real = torch.cos(phase_B).view(*A_real.shape[:-1], 1)
        B_imag = torch.sin(phase_B).view(*A_real.shape[:-1], 1)
        
        qubit_state = torch.cat([A_real, A_imag, B_real, B_imag], dim=-1)
        return self.proj(qubit_state.float()).to(torch.bfloat16)

class PhaseDecoupler(nn.Module):
    def __init__(self, d_model=896, n_dim=128):
        super().__init__()
        self.complex_attn = MultiHeadComplexAttention(d_model=d_model, n_heads=16, merge='concat')
        self.proj_r = nn.Linear(d_model, n_dim)
        self.proj_i = nn.Linear(d_model, n_dim)
        
    def forward(self, hidden_states):
        # Apply Complex SVD to extract pure Phase Tensor before passing to NativeEigenCore!
        B, M, D = hidden_states.shape
        z_seqs = []
        for b in range(B):
            hs = hidden_states[b]
            hs_norm = hs / (hs.norm(dim=1, keepdim=True) + 1e-9)
            Z = torch.complex(hs_norm, torch.zeros_like(hs_norm))
            Z_centered = Z - Z.mean(dim=0, keepdim=True)
            C = (Z_centered.conj().T @ Z_centered) / (M - 1)
            evals, evecs = torch.linalg.eigh(C)
            # Reverse order for descending eigenvalues
            evecs = evecs[:, torch.arange(D-1, -1, -1)]
            Z_proj = Z_centered @ evecs
            z_seqs.append(Z_proj)
            
        Z_proj_batch = torch.stack(z_seqs)
        
        # NativeEigenCore non-linear processing
        attn_out, _ = self.complex_attn(Z_proj_batch)
        attn_pooled = attn_out.mean(dim=1)
        
        r = self.proj_r(attn_pooled.real)
        i = self.proj_i(attn_pooled.imag)
        phase = torch.atan2(i, r) 
        return phase

def cosine_phase_loss(pred_phase, target_phase):
    return (1.0 - torch.cos(pred_phase - target_phase)).mean()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Initializing Native Eigen-Shor Oracle on {device}...", flush=True)
    
    student = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
    
    print("\n[*] Patching Student with Holographic Eigen-Weights...", flush=True)
    holo_dict = torch.load(HOLO_PATH, weights_only=False)
    patch_model_with_holo(student, holo_dict)
    student.eval()
    
    instance_path = os.path.join(os.path.dirname(__file__), "lwe_instance.pt")
    data = load_lwe_instance(instance_path)
    A_true = data['A'].squeeze()
    B_true = data['B'].squeeze()
    S_true = data['S_true'].squeeze()
    q = data['q']
    m, n = A_true.shape
    
    encoder = QubitLatticeEncoder(n_dim=n, hidden_size=896, q=q).to(device)
    decoupler = PhaseDecoupler(d_model=896, n_dim=n).to(device)
    
    opt = torch.optim.AdamW(decoupler.parameters(), lr=5e-3)
    
    print("\n[*] Starting Curriculum Learning on NativeEigenCore...", flush=True)
    epochs = 40
    batch_size = 4
    
    for epoch in range(epochs):
        noise_std = (epoch / max(1, epochs - 1)) * 2.0
        A_syn, B_syn, S_syn = generate_synthetic_lwe(batch_size=batch_size, n=n, m=50, q=q, noise_std=noise_std)
        A_syn, B_syn, S_syn = A_syn.to(device), B_syn.to(device), S_syn.to(device)
        
        target_phase = ((2 * math.pi * S_syn / q) + math.pi) % (2 * math.pi) - math.pi
        
        # Vectorized batch processing
        seq = encoder(A_syn, B_syn) # (B, M, D)
        
        with torch.no_grad():
            out = student(inputs_embeds=seq, output_hidden_states=True)
            
        hidden = out.hidden_states[-1].float()
        pred_phase = decoupler(hidden)
        
        loss = cosine_phase_loss(pred_phase, target_phase)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if epoch % 5 == 0:
            print(f"  [Epoch {epoch:2d}] Noise Std: {noise_std:.2f} | Cosine Phase Loss: {loss.item():.4f}", flush=True)
            
    print("\n[+] EIGEN-ORACLE NATIVE DECRYPTION:", flush=True)
    decoupler.eval()
    
    test_idx = torch.randperm(m)[:100]
    A_test = A_true[test_idx].unsqueeze(0).to(device)
    B_test = B_true[test_idx].unsqueeze(0).to(device)
    
    seq_test = encoder(A_test, B_test)
    
    with torch.no_grad():
        out_test = student(inputs_embeds=seq_test, output_hidden_states=True)
        hidden_test = out_test.hidden_states[-1].float()
        pred_phase_test = decoupler(hidden_test).squeeze()
        
    pred_phase_test = (pred_phase_test + 2 * math.pi) % (2 * math.pi)
    S_pred = torch.round((pred_phase_test / (2 * math.pi)) * q) % q
    S_pred = S_pred.long().cpu()
    
    print(f"\nPredicted Secret S:\n{S_pred.tolist()}")
    print(f"\nTrue Secret S:\n{S_true.long().tolist()}")
    
    matches = (S_pred == S_true.long()).sum().item()
    print(f"\n[+] Accuracy: {matches}/{n} components exactly recovered ({(matches/n)*100:.1f}%)")
    
    error_vec = (torch.matmul(A_true[:100].long(), S_pred) - B_true[:100].long()) % q
    error_magnitude = torch.where(error_vec > q/2, q - error_vec, error_vec).float().abs().sum().item()
    print(f"[+] LWE Error Magnitude: {error_magnitude}")

if __name__ == "__main__":
    main()
