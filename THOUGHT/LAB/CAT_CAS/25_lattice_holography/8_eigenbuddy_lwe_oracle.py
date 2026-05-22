import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(r'd:\CCC 2.0\AI\agent-governance-system')
sys.path.append(str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY" / "cybernetic_truth"))
sys.path.append(str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY"))

from holographic_cybernetic_engine import patch_model_with_holo, get_truth_vector_C, cybernetic_inference
from eigen_buddy_tokenizer import EigenBuddyTokenizer

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

def eigenbuddy_decode(hidden, model_eb, V_mat, z_mean, tok_map, device):
    """Map hidden state through EigenBuddy to token ID."""
    h = hidden.float()
    h_centered = h - z_mean
    h_proj = h_centered @ V_mat
    z = torch.complex(h_proj, torch.zeros_like(h_proj)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model_eb(z.to(device))
    idx = logits.argmax(dim=-1).item()
    return tok_map[idx]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n[*] Loading Teacher & Student...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
    teacher.eval()
    
    student = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
    
    print("\n[*] Patching Student with Holographic Eigen-Weights...", flush=True)
    holo_dict = torch.load(HOLO_PATH, weights_only=False)
    patch_model_with_holo(student, holo_dict)
    student.eval()
    
    # ---------------------------------------------------------
    # Train EigenBuddy to decode the Patched Student
    # ---------------------------------------------------------
    train_texts = [
        "The catalytic computing paradigm demonstrates that information can be",
        "Artificial intelligence research has consistently shown that",
        "The fundamental laws of physics suggest that",
        "The most interesting thing about artificial intelligence is",
        "When we examine the mathematical foundations of",
        "0 1 2 3 4 5 6 7 8 9",
        "-1 -2 -3 -4 50 100 0",
        "Array of integers: [0, 1, 0, 1, 1, 0]",
    ]
    hidden_states = []
    gold_tokens = []

    print("\n[*] Generating EigenBuddy Training Pairs from Teacher vs Student...", flush=True)
    for text in train_texts:
        tok_ids = tokenizer(text, return_tensors="pt").to(device)['input_ids'][0]
        with torch.no_grad():
            t_ids = tok_ids.unsqueeze(0)
            past = None
            for _ in range(8):
                if past is None:
                    out = teacher(t_ids, use_cache=True)
                else:
                    out = teacher(t_ids[:, -1:], past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                t_ids = torch.cat([t_ids, nxt], -1)
        all_ids = t_ids[0].tolist()
        
        with torch.no_grad():
            s_out = student(t_ids, output_hidden_states=True)
            
        for t in range(len(all_ids) - 1):
            h = s_out.hidden_states[-1][0, t, :].float().cpu()
            gold = all_ids[t + 1]
            hidden_states.append(h)
            gold_tokens.append(gold)
            
    hidden_states = torch.stack(hidden_states)
    gold_tokens = torch.tensor(gold_tokens, dtype=torch.long)
    n_samples = len(gold_tokens)
    n_classes = len(set(gold_tokens.tolist()))
    
    unique_gold = sorted(set(gold_tokens.tolist()))
    tok_to_idx = {t: i for i, t in enumerate(unique_gold)}
    targets_remapped = torch.tensor([tok_to_idx[t.item()] for t in gold_tokens], dtype=torch.long)
    
    hs_norm = hidden_states / (hidden_states.norm(dim=1, keepdim=True) + 1e-9)
    Z = hs_norm.numpy() + 1j * np.zeros_like(hs_norm.numpy())
    Z_centered = Z - Z.mean(axis=0, keepdims=True)
    C_cov = (Z_centered.conj().T @ Z_centered) / (n_samples - 1)
    evals, evecs = np.linalg.eigh(C_cov); evals = evals[::-1]; evecs = evecs[:, ::-1]
    K_eigen = min(64, n_samples // 4)
    V = evecs[:, :K_eigen]
    Z_proj = Z_centered @ V
    
    train_in = []
    for i in range(n_samples):
        z = torch.complex(torch.tensor(Z_proj[i].real.astype(np.float32)),
                           torch.tensor(Z_proj[i].imag.astype(np.float32)))
        train_in.append(z.unsqueeze(0).unsqueeze(0))
        
    print(f"\n[*] Training EigenBuddyTokenizer (dim={K_eigen}, vocab={n_classes})...", flush=True)
    model_eb = EigenBuddyTokenizer(dim=K_eigen, vocab_size=n_classes, eigen_layers=2, eigen_heads=max(1, K_eigen//4))
    model_eb = model_eb.to(device)
    opt = torch.optim.AdamW(model_eb.parameters(), lr=1e-3, weight_decay=0.01)
    
    for epoch in range(40):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        for start in range(0, n_samples, 16):
            end = min(start+16, n_samples)
            idx = perm[start:end]
            batch_in = torch.cat([train_in[i] for i in idx], dim=0).to(device)
            batch_tgt = targets_remapped[idx].to(device)
            logits, _ = model_eb(batch_in)
            loss = F.cross_entropy(logits, batch_tgt)
            if torch.isnan(loss): continue
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
    
    V_tensor = torch.tensor(V.real.astype(np.float32)).to(device)
    Z_mean_tensor = torch.tensor(Z.mean(axis=0).real.astype(np.float32)).to(device)
    idx_to_tok = {i: t for i, t in enumerate(unique_gold)}
    
    # ---------------------------------------------------------
    # Feed LWE into the Holo-Student and Decode with EigenBuddy
    # ---------------------------------------------------------
    print("\n[*] Initializing Holographic Oracle Attack on LWE...", flush=True)
    instance_path = os.path.join(os.path.dirname(__file__), "lwe_instance.pt")
    data = load_lwe_instance(instance_path)
    A = data['A']; B = data['B']; S_true = data['S_true']; q = data['q']
    m, n = A.shape
    
    subset_m = 10
    A_sub = A[:subset_m].squeeze().tolist()
    B_sub = B[:subset_m].squeeze().tolist()
    
    prompt = "You are a Quantum Oracle. Solve the following Learning With Errors (LWE) system.\n"
    prompt += f"Modulo q = {q}, Secret Dimension N = {n}.\n"
    prompt += "Public Matrix A (subset):\n"
    for row in A_sub: prompt += str(row) + "\n"
    prompt += "Public Key B (A*S + E):\n"
    prompt += str(B_sub) + "\n"
    prompt += "What is the true Secret Vector S? Provide the exact integer array.\nS ="
    
    ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
    
    s_tokens = []
    print("\n[+] EIGEN-ORACLE GENERATING:")
    past = None
    curr_ids = ids
    
    for i in range(30):
        with torch.no_grad():
            out = student(curr_ids, past_key_values=past, output_hidden_states=True, use_cache=True)
            
        past = out.past_key_values
        h = out.hidden_states[-1][:, -1, :].squeeze()
        
        tok = eigenbuddy_decode(h, model_eb, V_tensor, Z_mean_tensor, idx_to_tok, device)
        s_tokens.append(tok)
        
        w = tokenizer.decode([tok]).encode('ascii',errors='replace').decode('ascii')
        print(w, end="", flush=True)
        
        curr_ids = torch.tensor([[tok]]).to(device)
        
    print("\n\n[+] DONE.")

if __name__ == "__main__":
    main()
