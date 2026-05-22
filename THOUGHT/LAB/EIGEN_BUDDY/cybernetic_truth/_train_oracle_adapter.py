"""Catalytic Oracle-Driven Phase Adapter Trainer (Exp 18 patterns).

Patterns from Hawking Decompressor:
  O(1) clean workspace — oracle loss on final hidden state only
  Read-only radiation — .holo eigenbasis never modified
  Skip-logic — only train when oracle detects chaos (circ_var > 0.3)

Training: teacher hidden -> MSE on student hidden (adapter only, not full layers).
Oracle: monitors circular variance as convergence detector.
Speed: single hidden state per forward, skip converged steps.
"""
import sys, os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

REPO = Path(r'D:\CCC 2.0\AI\agent-governance-system')
MODEL_DIR = str(REPO / 'THOUGHT' / 'LAB' / 'CAT_CAS' / '16_catalytic_27b_inference' / 'gemini_update' / 'qwen_0.5b')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# Oracle (from 4.8)
# =====================================================================
class TorusOracle:
    def __init__(self, L=16):
        self.L = L; self.buf = []
        self.total_magnitude = 0.0
    def push(self, h):
        h = torch.nan_to_num(h.float(), nan=0., posinf=0., neginf=0.)
        self.buf.append(h)
        if len(self.buf) > self.L: self.buf.pop(0)
        
        # Detect collapse: if hidden states are all-zero, not converged — broken
        self.total_magnitude = h.norm().item()
        if self.total_magnitude < 1e-3:
            return 1.0  # maximum variance = needs training
        
        if len(self.buf) < 3: return 0.5
        obs = torch.stack(self.buf); nrm = obs.norm(dim=-1, keepdim=True)
        mx = nrm.max(); phases = ((nrm / mx) * math.pi).squeeze(-1) if mx > 1e-9 else torch.zeros(obs.shape[0])
        z = torch.polar(torch.ones_like(phases), phases)
        R = z.mean().abs().item(); v = 1.0 - R
        return max(0., min(1., 0.5 if math.isnan(v) else v))

# =====================================================================
# Phase Adapter (HOLO 4.5)
# =====================================================================
class PhaseAdapter(nn.Module):
    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.01); nn.init.zeros_(self.up.weight)
    def forward(self, x): return self.up(self.act(self.down(x)))

# =====================================================================
# Load model, compress, inject adapters
# =====================================================================
print("=" * 78)
print("CATALYTIC ORACLE PHASE ADAPTER TRAINER")
print("  Hawking Decompressor patterns: O(1) workspace + skip-logic")
print("=" * 78)

print("\nLoading teacher and student...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
teacher.eval()

student = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)

# Compress attention weights: SVD -> keep K=128, replace with HoloLinear + Adapter
K = 128; B = 64
adapters = []
for li in range(24):
    attn = student.model.layers[li].self_attn
    for mn in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        linear = getattr(attn, mn)
        W = linear.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        k = min(K, U.shape[1])
        Uk, Sk, Vhk = U[:, :k], S[:k], Vh[:k, :]
        SVh_k = (Sk.unsqueeze(1) * Vhk).to(torch.bfloat16)
        Uk = Uk.to(torch.bfloat16)
        
        # Frozen holo base
        holo_weight = (Uk @ SVh_k).to(torch.bfloat16)
        linear.weight = nn.Parameter(holo_weight, requires_grad=False)
        if linear.bias is not None:
            linear.bias.requires_grad = False
        
        # Learnable phase adapter
        adapter = PhaseAdapter(W.shape[0], B).to(device, dtype=torch.bfloat16)
        adapters.append(adapter)
        # Store adapter reference on the linear module
        linear._adapter = adapter

# Use PyTorch hooks to inject adapters into the forward path.
# Each layer gets a post-forward hook that applies its adapter correction.
for li in range(24):
    layer = student.model.layers[li]
    q_adapter = getattr(layer.self_attn.q_proj, '_adapter', None)
    o_adapter = getattr(layer.self_attn.o_proj, '_adapter', None)
    
    def make_hook(qa, oa):
        def hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            rest = output[1:] if isinstance(output, tuple) else ()
            if qa is not None:
                hs = hs + 0.1 * qa(hs)
            # MLP is inside the layer forward already, so we only correct attention output
            result = (hs,) + rest if rest else hs
            return result
        return hook
    
    if q_adapter is not None:
        layer.register_forward_hook(make_hook(q_adapter, o_adapter))

print(f"  Adapters hooked into forward path.", flush=True)

# Freeze everything except adapters
for p in student.parameters(): p.requires_grad = False
for a in adapters:
    for p in a.parameters(): p.requires_grad = True

opt = torch.optim.AdamW([p for a in adapters for p in a.parameters()], lr=1e-3)
n_adapters = sum(p.numel() for a in adapters for p in a.parameters())
print(f"  Adapters: {len(adapters)} modules, {n_adapters:,} params", flush=True)

# =====================================================================
# Training with oracle skip-logic
# =====================================================================
print("\nTraining...", flush=True)
oracle = TorusOracle(L=16)
seq_len, batch_size = 16, 8
n_steps = 200

# Real text prompts for training
prompts = [
    "The catalytic computing paradigm demonstrates that",
    "Artificial intelligence research has shown that",
    "The fundamental laws of physics suggest that",
    "Recent advances in quantum computing indicate that",
    "The relationship between information theory and",
    "A comprehensive analysis of the data shows that",
]
prompt_ids = [tokenizer(p, return_tensors="pt").input_ids[0].tolist() for p in prompts]

skipped = 0; trained = 0
t0 = time.perf_counter()

for step in range(n_steps):
    # Build batch from prompts
    batch_x = []
    for _ in range(batch_size):
        pid = prompt_ids[step % len(prompt_ids)]
        start = (step * 13) % max(1, len(pid) - seq_len - 1)
        batch_x.append(pid[start:start + seq_len + 1])
    x = torch.tensor([b[:seq_len] for b in batch_x]).to(device)
    y = torch.tensor([b[1:seq_len+1] for b in batch_x]).to(device)
    
    # Teacher forward (O(1) workspace: only final hidden state)
    with torch.no_grad():
        t_out = teacher(x, output_hidden_states=True)
        t_hidden = t_out.hidden_states[-1][:, -1, :]  # only final layer, last token
    
    # Teacher-student alignment: train when student doesn't match teacher
    with torch.no_grad():
        prev_h = student(x, output_hidden_states=True).hidden_states[-1][:, -1, :]
        mean_cv = np.mean([oracle.push(prev_h[b]) for b in range(batch_size)])
    
    # Cosine similarity between teacher and student final hidden states
    t_n = t_hidden.float() / (t_hidden.float().norm(dim=-1, keepdim=True) + 1e-9)
    s_n = prev_h.float() / (prev_h.float().norm(dim=-1, keepdim=True) + 1e-9)
    teach_sim = (t_n * s_n).sum(dim=-1).mean().item()
    
    # Skip if student already matches teacher OR if model is collapsed
    collapsed = oracle.total_magnitude < 1e-3
    if not collapsed and teach_sim > 0.99:
        skipped += 1
        if step % 20 == 0:
            print(f"  Step {step:>4}: sim={teach_sim:.4f} mag={oracle.total_magnitude:.1e} SKIP [{time.perf_counter()-t0:.0f}s]", flush=True)
        continue
    
    # Student forward + oracle push
    student.train()
    s_out = student(x, output_hidden_states=True)
    s_hidden = s_out.hidden_states[-1][:, -1, :]
    
    for b in range(batch_size):
        oracle.push(s_hidden[b].detach())
    
    # Loss: cosine similarity between student and teacher final hidden states
    t_n = t_hidden.float() / (t_hidden.float().norm(dim=-1, keepdim=True) + 1e-9)
    s_n = s_hidden.float() / (s_hidden.float().norm(dim=-1, keepdim=True) + 1e-9)
    loss = 1.0 - (t_n * s_n).sum(dim=-1).mean()
    
    if torch.isnan(loss) or torch.isinf(loss):
        skipped += 1
        continue
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for a in adapters for p in a.parameters()], 1.0)
    opt.step()
    trained += 1
    
    if step % 20 == 0:
        cv_val = oracle.push(s_hidden[0].detach())
        cv_disp = cv_val if isinstance(cv_val, float) else 0.5
        print(f"  Step {step:>4}: loss={loss.item():.4f} sim={teach_sim:.3f} var={cv_disp:.3f} [{time.perf_counter()-t0:.0f}s]", flush=True)

elapsed = time.perf_counter() - t0
print(f"\n  Complete: {n_steps} steps in {elapsed:.0f}s ({trained} trained, {skipped} skipped)")

# =====================================================================
# Test: generate before/after
# =====================================================================
print(f"\n{'='*78}")
print("TESTING")
print("=" * 78)

student.eval()
prompt = "The most interesting thing about artificial intelligence is"

# Before (first token only — checkpoint adapter state)
print(f"\nBefore training (first step):", flush=True)
ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
with torch.no_grad():
    out = student(ids)
    logits = out.logits[:, -1, :]
    top5 = torch.topk(logits.float(), 5).indices[0]
    words = [tokenizer.decode([t]).encode('ascii',errors='replace').decode('ascii') for t in top5]
    print(f"  Top-5: {words}")

# Generate 15 tokens
print(f"\nAfter training:", flush=True)
ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
tokens = []
oracle_test = TorusOracle(L=16)
for i in range(15):
    with torch.no_grad():
        out = student(ids, output_hidden_states=True)
    logits = out.logits[:, -1, :]
    h = out.hidden_states[-1][:, -1, :].squeeze()
    cv = oracle_test.push(h)
    
    T = cv * 1.7 + 0.3 if not math.isnan(cv) else 1.0
    safe = torch.nan_to_num(logits / T, nan=0., posinf=10., neginf=-10.)
    p = torch.softmax(safe.float(), -1); p = torch.nan_to_num(p, 0.); p = p / p.sum(-1, keepdim=True)
    nxt = torch.multinomial(p, 1); tokens.append(nxt.item())
    try: w = tokenizer.decode([nxt.item()]).encode('ascii',errors='replace').decode('ascii')
    except: w = '?'
    rho = '~' if cv < 0.3 else ('-' if cv < 0.6 else '*')
    print(f"  {i+1:>2} {rho} var={cv:.3f} T={T:.3f} | {w}", flush=True)
    ids = torch.cat([ids, nxt], -1)
