"""Full inference: Temporal-calibrated adapters + Oracle temperature modulation.

Combines:
  - CAT_CAS 23 temporal catalysis (Vh mode gating on consecutive tokens)
  - CAT_CAS 20.10 oracle (torus circular variance as resonance measure)
  - HOLO 4.6 HoloLinear + Phase Adapters

Tests: generate 30 tokens, measure token match rate vs teacher,
oracle metrics per step, and output coherence.
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

# =====================================================================
# Oracle
# =====================================================================
class TorusOracle:
    def __init__(self, L=16): self.L = L; self.buf = []; self.mag = 0.0
    def push(self, h):
        h = torch.nan_to_num(h.float(), nan=0., posinf=0., neginf=0.)
        self.buf.append(h); self.mag = h.norm().item()
        if len(self.buf) > self.L: self.buf.pop(0)
        if self.mag < 1e-3: return 1.0
        if len(self.buf) < 3: return 0.5
        obs = torch.stack(self.buf); n = obs.norm(dim=-1, keepdim=True)
        mx = n.max(); ph = ((n/mx)*math.pi).squeeze(-1) if mx>1e-9 else torch.zeros(obs.shape[0])
        z = torch.polar(torch.ones_like(ph), ph)
        R = z.mean().abs().item(); v = 1.0 - R
        return max(0., min(1., 0.5 if math.isnan(v) else v))

class PhaseAdapter(nn.Module):
    def __init__(self, dim, bottleneck=128):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.01); nn.init.zeros_(self.up.weight)
    def forward(self, x): return self.up(self.act(self.down(x)))

# =====================================================================
print("=" * 78)
print("FULL INFERENCE — Temporal Adapters + Oracle Engine")
print("=" * 78)

print("\nLoading models...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
teacher.eval()
student = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)

# Compress Q-proj + inject adapters
K = 128
adapters = {}
for li in range(24):
    attn = student.model.layers[li].self_attn
    linear = attn.q_proj
    W = linear.weight.data.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    k = min(K, U.shape[1])
    Uk, Sk, Vhk = U[:, :k], S[:k], Vh[:k, :]
    SVh_k = (Sk.unsqueeze(1) * Vhk).to(torch.bfloat16)
    linear.weight = nn.Parameter((Uk.to(torch.bfloat16) @ SVh_k), requires_grad=False)
    if linear.bias is not None: linear.bias.requires_grad = False
    adapter = PhaseAdapter(W.shape[0], K).to(device, dtype=torch.bfloat16)
    adapters[li] = adapter
    linear._adapter = adapter

# Teacher SVD data for mode gating
svd_data = {}
for li in range(24):
    W = teacher.model.layers[li].self_attn.q_proj.weight.data.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    svd_data[li] = Vh[:K, :]  # (K, 896)

# Hooks
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

print(f"  {len(adapters)} Q-proj adapters. {len(svd_data)} Vh banks.", flush=True)

# =====================================================================
# TEMPORAL CALIBRATION on prompt tokens
# =====================================================================
calibration_prompt = "The catalytic computing paradigm demonstrates that information can be processed without"
cal_ids = tokenizer(calibration_prompt, return_tensors="pt").to(device)['input_ids'][0, :24]  # 24 tokens

print(f"\nTemporal calibration on '{calibration_prompt[:50]}...' ({len(cal_ids)} tokens)", flush=True)

with torch.no_grad():
    t_out = teacher(cal_ids.unsqueeze(0), output_hidden_states=True)
t_h = t_out.hidden_states

calibrated = 0
for li in range(24):
    adapter = adapters.get(li)
    if adapter is None: continue
    Vh = svd_data[li]  # (K, 896)
    
    for t in range(len(cal_ids) - 1):
        future_h = t_h[li + 1][0, t + 1, :].float()
        
        with torch.no_grad():
            down_w = adapter.down.weight.float()
            mode_scores = F.linear(future_h.unsqueeze(0), Vh).squeeze().abs()
            mx = mode_scores.max()
            if mx > 1e-6:
                gains = torch.where(mode_scores > 0.3*mx, torch.tensor(2.0), torch.tensor(0.1))
                adapter.down.weight.data = (down_w * gains.unsqueeze(1)).to(torch.bfloat16)
                calibrated += 1

print(f"  Calibrated {calibrated} adapter-mode pairs", flush=True)

# =====================================================================
# FULL INFERENCE TEST
# =====================================================================
student.eval()
oracle = TorusOracle(L=16)

prompt = "The most interesting thing about artificial intelligence is"
ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
print(f"\n{'='*78}")
print(f"INFERENCE: '{prompt}'")
print("=" * 78)

# Teacher baseline
with torch.no_grad():
    t_ids = ids.clone()
    t_tokens = []
    for _ in range(30):
        out = teacher(t_ids)
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        t_tokens.append(nxt.item())
        t_ids = torch.cat([t_ids, nxt], -1)
teacher_text = tokenizer.decode(t_tokens)
print(f"Teacher: {teacher_text.encode('ascii',errors='replace').decode('ascii')[:100]}...")

# Student with temporal adapters + oracle
s_tokens = []
s_matches = 0
print(f"\nStudent (temporal + oracle):")
print("-" * 60)

for i in range(30):
    with torch.no_grad():
        out = student(ids, output_hidden_states=True)
    
    logits = out.logits[:, -1, :]
    h = out.hidden_states[-1][:, -1, :].squeeze()
    cv = oracle.push(h)
    
    # Oracle temperature modulation
    T = cv * 1.7 + 0.3 if not math.isnan(cv) else 1.0
    safe = torch.nan_to_num(logits / T, nan=0., posinf=10., neginf=-10.)
    p = torch.softmax(safe.float(), -1); p = torch.nan_to_num(p, 0.); p = p/p.sum(-1, keepdim=True)
    nxt = torch.multinomial(p, 1)
    s_tokens.append(nxt.item())
    
    # Match check
    if i < len(t_tokens) and nxt.item() == t_tokens[i]:
        s_matches += 1
    
    try: w = tokenizer.decode([nxt.item()]).encode('ascii',errors='replace').decode('ascii')
    except: w = '?'
    
    tw = ''
    if i < len(t_tokens):
        try: tw = tokenizer.decode([t_tokens[i]]).encode('ascii',errors='replace').decode('ascii')
        except: tw = '?'
    
    rho = '~' if cv < 0.3 else ('-' if cv < 0.6 else '*')
    match = '=' if (i < len(t_tokens) and nxt.item() == t_tokens[i]) else 'x'
    if i < 15 or i % 5 == 0:
        print(f"  {i+1:>2} {rho} var={cv:.3f} T={T:.2f} | stu={w:<12} tea={tw:<12} {match}", flush=True)
    
    ids = torch.cat([ids, nxt], -1)

student_text = tokenizer.decode(s_tokens)
print("-" * 60)
print(f"Student: {student_text.encode('ascii',errors='replace').decode('ascii')[:150]}...")

print(f"\n{'='*78}")
print(f"RESULTS")
print(f"  Token match rate: {s_matches}/30 = {s_matches/30*100:.0f}%")
print(f"  Teacher: {teacher_text.encode('ascii',errors='replace').decode('ascii')[:80]}...")
print(f"  Student: {student_text.encode('ascii',errors='replace').decode('ascii')[:80]}...")
print("=" * 78)
