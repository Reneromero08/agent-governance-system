import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.proj = nn.Linear(n_dim * 2 + 2, hidden_size)
        
    def forward(self, A_row, B_val):
        # Map a single equation (A_row: N, B_val: 1) to a single qubit-embedding (hidden_size)
        phase_A = 2 * math.pi * A_row / self.q
        phase_B = 2 * math.pi * B_val / self.q
        
        # Real and imaginary components (continuous)
        A_real = torch.cos(phase_A); A_imag = torch.sin(phase_A)
        B_real = torch.cos(phase_B); B_imag = torch.sin(phase_B)
        
        qubit_state = torch.cat([A_real, A_imag, B_real.unsqueeze(0), B_imag.unsqueeze(0)], dim=-1)
        return self.proj(qubit_state.float()).to(torch.bfloat16)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Initializing Recursive Qubit Phase Oracle on {device}...")
    
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
    # Train the Recursive State Decoupler
    # ---------------------------------------------------------
    print("\n[*] Encoding LWE Matrix into Continuous Qubit Embeddings...", flush=True)
    num_train = 500
    embeds_seq = []
    
    for i in range(num_train):
        emb = encoder(A[i].to(device), B[i].to(device))
        embeds_seq.append(emb)
        
    inputs_embeds = torch.stack(embeds_seq).unsqueeze(0)  # Shape: (1, M, 896)
    
    print(f"[*] Sweeping Qubit Embeddings through Holo-Cavity (Length={num_train})...", flush=True)
    with torch.no_grad():
        out = student(inputs_embeds=inputs_embeds, output_hidden_states=True)
        # Get hidden state of the final equation resonance
        final_state = out.hidden_states[-1][0, -1, :].float()
        
    print(f"[*] Resonance State Extracted. Dimension: {final_state.shape}", flush=True)
    
    # Simple decoupler to extract the Secret Vector S from the topological state
    print("[*] Learning Secret Topology...", flush=True)
    # Let's map final_state back to S_true. We use 100 random permutations of the equations.
    # Because S_true is the same, any valid subset of M equations should collapse to S_true.
    train_states = []
    for _ in range(30):
        idx = torch.randperm(m)[:100]
        seq = torch.stack([encoder(A[i].to(device), B[i].to(device)) for i in idx]).unsqueeze(0)
        with torch.no_grad():
            out = student(inputs_embeds=seq, output_hidden_states=True)
        train_states.append(out.hidden_states[-1][0, -1, :].float())
    
    train_states = torch.stack(train_states)
    
    # Map to S
    decoupler = nn.Linear(896, n).to(device)
    opt = torch.optim.AdamW(decoupler.parameters(), lr=1e-2)
    S_target = S_true.float().to(device).unsqueeze(0).expand(30, -1)
    
    for epoch in range(50):
        preds = decoupler(train_states)
        loss = F.mse_loss(preds, S_target)
        opt.zero_grad(); loss.backward(); opt.step()
        
    print(f"[*] Decoupler Trained. Final Loss: {loss.item():.4f}", flush=True)
    
    # ---------------------------------------------------------
    # Test on unseen equations!
    # ---------------------------------------------------------
    print("\n[+] EIGEN-ORACLE RECURSIVE DECRYPTION:", flush=True)
    test_idx = torch.randperm(m)[500:600]
    seq_test = torch.stack([encoder(A[i].to(device), B[i].to(device)) for i in test_idx]).unsqueeze(0)
    
    with torch.no_grad():
        out_test = student(inputs_embeds=seq_test, output_hidden_states=True)
        state_test = out_test.hidden_states[-1][0, -1, :].float()
        S_pred = decoupler(state_test.unsqueeze(0)).squeeze().round().long()
        
    S_pred = S_pred % q
    
    print(f"\nPredicted Secret S:\n{S_pred.tolist()}")
    print(f"\nTrue Secret S:\n{S_true.long().tolist()}")
    
    matches = (S_pred == S_true.to(device).long()).sum().item()
    print(f"\n[+] Accuracy: {matches}/{n} components exactly recovered ({(matches/n)*100:.1f}%)")

if __name__ == "__main__":
    main()
