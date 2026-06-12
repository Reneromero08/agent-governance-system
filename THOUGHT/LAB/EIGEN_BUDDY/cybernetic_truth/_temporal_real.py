"""Temporal Catalysis on real token sequences.

Key insight from 23: consecutive tokens have semantic correlation that
random tensors lack. Feed real text, capture layer activations at t and
t+1, use t+1's output as "future tape" to calibrate t's SVD mode gating.

Each attention layer's Q-projection SVD modes get scored by projecting
t+1's hidden state onto Vh (right singular vectors). Aligned modes get
boosted 2x, misaligned get 0.1x. Loop iterates until calibration stabilizes.
"""
import sys, os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(r'D:\CCC 2.0\AI\agent-governance-system')
MODEL_DIR = str(REPO / 'THOUGHT' / 'LAB' / 'CAT_CAS' / '3_physics_complexity' / '16_catalytic_27b_inference' / 'gemini_update' / 'qwen_0.5b')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PhaseAdapter(nn.Module):
    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.01); nn.init.zeros_(self.up.weight)
    def forward(self, x): return self.up(self.act(self.down(x)))

# =====================================================================
print("=" * 78)
print("TEMPORAL CATALYSIS — Real Token Sequences")
print("  t+1's hidden state as future tape for t's SVD mode gating")
print("=" * 78)

print("\nLoading teacher + student...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
teacher.eval()
student = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)

# Compress + inject adapters
K = 128; B = 128  # bottleneck matches SVD rank
adapters = {}
for li in range(24):
    attn = student.model.layers[li].self_attn
    for mn in ['q_proj']:  # Q-projection only (matches 23 architecture)
        linear = getattr(attn, mn)
        W = linear.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        k = min(K, U.shape[1])
        Uk, Sk, Vhk = U[:, :k], S[:k], Vh[:k, :]
        SVh_k = (Sk.unsqueeze(1) * Vhk).to(torch.bfloat16)
        linear.weight = nn.Parameter((Uk.to(torch.bfloat16) @ SVh_k), requires_grad=False)
        if linear.bias is not None: linear.bias.requires_grad = False
        adapter = PhaseAdapter(W.shape[0], B).to(device, dtype=torch.bfloat16)
        adapters[(li, mn)] = adapter
        linear._adapter = adapter

# Store Vh for SVD mode gating
svd_data = {}
for li in range(24):
    W = teacher.model.layers[li].self_attn.q_proj.weight.data.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    svd_data[li] = {'U': U[:, :K], 'S': S[:K], 'Vh': Vh[:K, :]}

# Register hooks
for li in range(24):
    layer = student.model.layers[li]
    qa = getattr(layer.self_attn.q_proj, '_adapter', None)
    def make_hook(q_adapter):
        def hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            rest = output[1:] if isinstance(output, tuple) else ()
            if q_adapter is not None: hs = hs + 0.1 * q_adapter(hs)
            return (hs,) + rest if rest else hs
        return hook
    if qa is not None: layer.register_forward_hook(make_hook(qa))

print(f"  {len(adapters)} Q-projection adapters + {len(svd_data)} SVD data.", flush=True)

# =====================================================================
# REAL SEQUENCE: capture teacher activations at consecutive positions
# =====================================================================
print("\nCapturing teacher activations on real text...", flush=True)

prompt = "The catalytic computing paradigm demonstrates that information can be processed without"
ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids'][0]
seq_len = min(32, len(ids))
ids = ids[:seq_len]

# Teacher forward on full sequence, capture hidden states per layer per position
with torch.no_grad():
    t_out = teacher(ids.unsqueeze(0), output_hidden_states=True)
t_hidden = t_out.hidden_states  # 25 layers (0=embed, 1-24=decoder), each (1, seq_len, 896)

print(f"  Sequence: {seq_len} tokens", flush=True)
print(f"  Text: {tokenizer.decode(ids.tolist()).encode('ascii',errors='replace').decode('ascii')[:80]}...", flush=True)

# =====================================================================
# TEMPORAL CALIBRATION: t+1 future tape -> calibrate t's adapter
# =====================================================================
print(f"\nTemporal calibration on {seq_len-1} consecutive pairs...", flush=True)

# Also capture student baseline for comparison
with torch.no_grad():
    s_out = student(ids.unsqueeze(0), output_hidden_states=True)
s_hidden = s_out.hidden_states

calibrated = 0
for li in range(24):
    adapter = adapters.get((li, 'q_proj'))
    if adapter is None: continue
    
    Vh = svd_data[li]['Vh']  # (K, 896) — right singular vectors
    
    for t in range(seq_len - 1):
        # Future tape: teacher's activation at position t+1, layer li
        future_h = t_hidden[li + 1][0, t + 1, :].float()  # (896,)
        # Current position: teacher's activation at position t
        current_h = t_hidden[li + 1][0, t, :].float()
        
        with torch.no_grad():
            down_w = adapter.down.weight.float()
            
            # Project future hidden state onto Vh (right singular vectors)
            # This scores which input MODES are relevant for the next token
            # Vh is (K, 896) — each row is an input mode
            # future_h @ Vh^T = (896,) @ (896, K) = (K,) scores per mode
            mode_scores = F.linear(future_h.unsqueeze(0), Vh).squeeze()  # (K,)
            mode_scores = mode_scores.abs()
            max_score = mode_scores.max()
            
            if max_score > 1e-6:
                # Boost aligned modes 2x, suppress misaligned 0.1x
                gains = torch.where(
                    mode_scores > 0.3 * max_score,
                    torch.tensor(2.0),      # aligned -> boost
                    torch.tensor(0.1)        # misaligned -> suppress
                )
                
                # Apply gains to adapter bottleneck
                adapter.down.weight.data = (down_w * gains.unsqueeze(1)).to(torch.bfloat16)
                calibrated += 1

print(f"  Calibrated {calibrated} adapter-mode pairs across {seq_len-1} positions", flush=True)

# =====================================================================
# VERIFY: student forward with calibrated adapters
# =====================================================================
print(f"\nVerification...", flush=True)
student.eval()
with torch.no_grad():
    s_out2 = student(ids.unsqueeze(0), output_hidden_states=True)

# Compare teacher vs student logits at each position
matches = 0
for t in range(seq_len - 1):
    t_logits = t_out.logits[0, t, :]
    s_logits = s_out2.logits[0, t, :]
    t_tok = t_logits.argmax().item()
    s_tok = s_logits.argmax().item()
    if t_tok == s_tok: matches += 1

# Also test on held-out prompt
test_prompt = "The fundamental laws of physics suggest that"
test_ids = tokenizer(test_prompt, return_tensors="pt").to(device)['input_ids'][0, :16]

with torch.no_grad():
    t_test = teacher(test_ids.unsqueeze(0), output_hidden_states=True)
    s_test = student(test_ids.unsqueeze(0), output_hidden_states=True)

t_next = t_test.logits[0, -1, :].argmax().item()
s_next = s_test.logits[0, -1, :].argmax().item()

print(f"  Training sequence: {matches}/{seq_len-1} token matches")
print(f"  Test prompt '{test_prompt[:40]}...': teacher={repr(tokenizer.decode([t_next]))} student={repr(tokenizer.decode([s_next]))}")
