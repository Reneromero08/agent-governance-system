import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig

REPO = Path(r'd:\CCC 2.0\AI\agent-governance-system')
sys.path.append(str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY" / "cybernetic_truth"))
sys.path.append(str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY"))

from holographic_cybernetic_engine import patch_model_with_holo
from eigen_buddy_tokenizer import MultiHeadComplexAttention

MODEL_DIR = r"THOUGHT\LAB\CAT_CAS\16_catalytic_27b_inference\gemini_update\qwen_0.5b"
MODEL_DIR = str(REPO / MODEL_DIR)
HOLO_PATH_05B = str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY" / "cybernetic_truth" / "qwen_0_5b_k128.holo")
HOLO_PATH_27B = str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY" / "cybernetic_truth" / "qwen_27b_catalytic_k256.holo")
HD = 896  # Qwen 0.5B hidden dim
USE_05B = True  # set False for 27B (requires 54GB RAM)

def load_lwe_instance(path):
    import importlib.util
    lwe_path = os.path.join(str(REPO), "THOUGHT", "LAB", "CAT_CAS", "25_lattice_holography", "1_lwe_simulator.py")
    spec = importlib.util.spec_from_file_location("lwe", lwe_path)
    lwe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lwe)
    if not os.path.exists(path):
        instance = lwe.generate_lwe_instance(n=128, m=1024)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(instance, path)
    return torch.load(path, weights_only=False)

def generate_synthetic_lwe(batch_size=1, n=128, m=50, q=3329, noise_std=0.0):
    S = torch.randint(0, q, (batch_size, n, 1), dtype=torch.float64)
    A = torch.randint(0, q, (batch_size, m, n), dtype=torch.float64)
    E = torch.round(torch.randn((batch_size, m, 1), dtype=torch.float64) * noise_std)
    B = (torch.matmul(A, S) + E) % q
    return A, B, S.squeeze(-1)

class QubitLatticeEncoder(nn.Module):
    def __init__(self, n_dim=128, hidden_size=HD, q=3329):
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
    def __init__(self, d_model=HD, n_dim=128):
        super().__init__()
        self.complex_attn = MultiHeadComplexAttention(d_model=d_model, n_heads=16, merge='concat')
        self.proj_r = nn.Linear(d_model, n_dim)
        self.proj_i = nn.Linear(d_model, n_dim)
        
    def forward(self, hidden_states):
        # Vectorized Complex SVD — process entire batch at once
        B, M, D = hidden_states.shape
        hs_norm = hidden_states / (hidden_states.norm(dim=2, keepdim=True) + 1e-9)
        Z = torch.complex(hs_norm, torch.zeros_like(hs_norm))
        Z_centered = Z - Z.mean(dim=1, keepdim=True)
        # Batched covariance: (B, M, D) -> (B, D, M) @ (B, M, D) -> (B, D, D)
        C = torch.bmm(Z_centered.conj().transpose(1, 2), Z_centered) / (M - 1)
        # Batched eigh
        evals, evecs = torch.linalg.eigh(C)
        evecs = evecs.flip(dims=[2])  # descending
        Z_proj_batch = torch.bmm(Z_centered, evecs)
        
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
    print(f"\n[*] Device: {device}", flush=True)
    
    # ---- STUDENT: 0.5B with 0.5B .holo (load normally, it's small) ----
    print("[*] Loading 0.5B model...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, device_map=device, torch_dtype=torch.bfloat16
    )
    
    print("[*] Patching 0.5B with HoloLinear (k=128)...", flush=True)
    holo_05b = torch.load(HOLO_PATH_05B, weights_only=False)
    patch_model_with_holo(student, holo_05b)
    student.eval()
    
    # ---- TEACHER: 27B on meta, patched with 27B .holo ----
    print("[*] Loading 27B teacher on meta device...", flush=True)
        # ---- TEACHER: 27B .holo U_k matrices as distilled features ----
    print("[*] Loading 27B .holo for feature extraction...", flush=True)
    holo_27b = torch.load(HOLO_PATH_27B, weights_only=False)
    
    # Extract U_k from first attention layer — projects onto 27B's principal directions
    first_U = None
    for key, val in holo_27b.items():
        if key.endswith('.U') and val.ndim == 2 and 'self_attn' in key.lower():
            first_U = val.float()
            break
    if first_U is None:
        first_U = torch.randn(5120, 256)  # fallback
    
    teacher_dim = first_U.shape[0]  # 5120
    teacher_k = first_U.shape[1]    # 256
    # Fix: register U_k as buffer, project student hidden through it
    U_27b = first_U.to(device)
    # Project: 896 -> 5120 -> (U_k^T) -> 256 -> 896
    proj_up = nn.Linear(HD, teacher_dim, bias=False).to(device)
    proj_down = nn.Linear(teacher_k, HD, bias=False).to(device)
    nn.init.normal_(proj_up.weight, std=0.02)
    nn.init.normal_(proj_down.weight, std=0.02)
    teacher = True
    print(f"    27B U_k: {list(first_U.shape)} — injecting principal directions", flush=True)
    
    # LWE data + encoder
    instance_path = os.path.join(str(REPO), "THOUGHT", "LAB", "CAT_CAS", "25_lattice_holography", "lwe_instance.pt")
    data = load_lwe_instance(instance_path)
    A_true = data['A'].squeeze(); B_true = data['B'].squeeze()
    S_true = data['S_true'].squeeze(); q = data['q']
    m, n = A_true.shape
    
    encoder = QubitLatticeEncoder(n_dim=n, hidden_size=HD, q=q).to(device)
    decoupler = PhaseDecoupler(d_model=HD, n_dim=n).to(device)
    opt = torch.optim.AdamW(list(decoupler.parameters()) + 
                            (list(proj_up.parameters()) + list(proj_down.parameters()) if teacher else []), lr=5e-3)
    
    # CATALYTIC CACHE: pre-compute student hidden states + targets
    epochs_cache = 20; batch_size = 4
    hidden_cache_student = []; target_cache = []
    for epoch in range(epochs_cache):
        noise_std = (epoch / max(1, epochs_cache - 1)) * 2.0
        A_syn, B_syn, S_syn = generate_synthetic_lwe(batch_size, n=n, m=50, q=q, noise_std=noise_std)
        A_syn, B_syn, S_syn = A_syn.to(device), B_syn.to(device), S_syn.to(device)
        seq = encoder(A_syn, B_syn)
        with torch.no_grad():
            out_s = student(inputs_embeds=seq, output_hidden_states=True)
        hidden_cache_student.append(out_s.hidden_states[-1].float().detach())
        target_phase = ((2 * math.pi * S_syn / q) + math.pi) % (2 * math.pi) - math.pi
        target_cache.append(target_phase.detach())
    
    epochs_train = 100
    print(f"[*] Cache ready. Training ({epochs_train} epochs)...", flush=True)
    for epoch in range(epochs_train):
        idx = epoch % len(hidden_cache_student)
        h_s = hidden_cache_student[idx].to(device)
        target = target_cache[idx].to(device)
        
        pred = decoupler(h_s)
        loss_lwe = cosine_phase_loss(pred, target)
        
        if teacher:
            # Teacher projection computed LIVE (gradient flows to proj_up/down)
            ht = proj_up(h_s)
            ht = torch.matmul(ht, U_27b)
            ht = proj_down(ht)
            loss_distill = F.mse_loss(h_s, ht) * 0.01
            loss = loss_lwe + loss_distill
        else:
            loss = loss_lwe
        
        opt.zero_grad(); loss.backward(); opt.step()
        
        if epoch % 20 == 0:
            with torch.no_grad():
                phase_diff = torch.abs(torch.atan2(torch.sin(pred - target), torch.cos(pred - target)))
                acc = (phase_diff < math.pi/4).float().mean().item()
            d_info = f" distill={loss_distill.item():.4f}" if teacher else ""
            print(f"  [Epoch {epoch:3d}] Loss: {loss.item():.4f} | Phase acc: {acc:.2%}{d_info}", flush=True)
        
        if epoch % 20 == 0:
            print(f"  [Epoch {epoch:3d}] Loss: {loss.item():.4f}", flush=True)
    
    # Test
    print("\n[+] NATIVE EIGEN DECRYPTION:", flush=True)
    decoupler.eval()
    test_idx = torch.randperm(m)[:100]
    A_test = A_true[test_idx].unsqueeze(0).to(device)
    B_test = B_true[test_idx].unsqueeze(0).to(device)
    seq_test = encoder(A_test, B_test)
    with torch.no_grad():
        out_test = student(inputs_embeds=seq_test, output_hidden_states=True)
        hidden_test = out_test.hidden_states[-1].float()
        pred_phase_test = decoupler(hidden_test).squeeze()
    
    pred_phase_test = (pred_phase_test + 2*math.pi) % (2*math.pi)
    S_pred = torch.round((pred_phase_test / (2*math.pi)) * q) % q
    S_pred = S_pred.long().cpu()
    
    matches = (S_pred == S_true.long()).sum().item()
    print(f"\n[+] Accuracy: {matches}/{n} ({matches/n*100:.1f}%)")
    error_vec = (torch.matmul(A_true[:100].long(), S_pred) - B_true[:100].long()) % q
    error_mag = torch.where(error_vec > q/2, q - error_vec, error_vec).float().abs().sum().item()
    print(f"[+] LWE Error: {error_mag}")

if __name__ == "__main__":
    main()
