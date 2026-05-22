"""Q-only temporal calibration with 100+ tokens across diverse prompts.

Scale from 12 to 100+ calibration tokens. More consecutive pairs 
-> more temporal mode gating -> higher coherence and match rate.
"""
import sys, os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(r'D:\CCC 2.0\AI\agent-governance-system')
MODEL_DIR = str(REPO / 'THOUGHT' / 'LAB' / 'CAT_CAS' / '16_catalytic_27b_inference' / 'gemini_update' / 'qwen_0.5b')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        z = torch.polar(torch.ones_like(ph), ph); R = z.mean().abs().item()
        v = 1.0 - R; return max(0., min(1., 0.5 if math.isnan(v) else v))

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
print("SCALED TEMPORAL CALIBRATION — 100+ tokens, multi-prompt")
print("=" * 78)

print("\nLoading models...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
teacher.eval()
student = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)

K = 128
adapters = {}

for li in range(24):
    attn = student.model.layers[li].self_attn
    t_attn = teacher.model.layers[li].self_attn
    # Q-proj only
    linear = attn.q_proj
    W = linear.weight.data.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    k = min(K, U.shape[1])
    Uk, Sk, Vhk = U[:, :k], S[:k], Vh[:k, :]
    SVh_k = (Sk.unsqueeze(1) * Vhk).to(torch.bfloat16)
    linear.weight = nn.Parameter((Uk.to(torch.bfloat16) @ SVh_k), requires_grad=False)
    if linear.bias is not None: linear.bias.requires_grad = False
    adapter = PhaseAdapter(W.shape[0], K).to(device, dtype=torch.bfloat16)
    adapters[li] = {'adapter': adapter, 'Vh': Vh[:K, :]}
    linear._adapter = adapter

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

# =====================================================================
# MULTI-PROMPT CALIBRATION
# =====================================================================
prompts = [
    "The catalytic computing paradigm demonstrates that information can be processed without",
    "Artificial intelligence research has consistently shown that the most important factor",
    "The fundamental laws of physics suggest that the universe operates on principles of",
    "Recent advances in quantum computing indicate that we are approaching a threshold where",
    "The relationship between information theory and thermodynamics reveals that entropy is",
]

# Collect all tokens into one long sequence
all_ids = []
for p in prompts:
    ids = tokenizer(p, return_tensors="pt").to(device)['input_ids'][0].tolist()
    all_ids.extend(ids)
all_ids = all_ids[:128]  # cap at 128 tokens

print(f"\nCalibrating on {len(all_ids)} tokens across {len(prompts)} prompts...", flush=True)
ids_tensor = torch.tensor(all_ids).unsqueeze(0).to(device)

with torch.no_grad():
    t_out = teacher(ids_tensor, output_hidden_states=True)
t_h = t_out.hidden_states

calibrated = 0
for li in range(24):
    Vh = adapters[li]['Vh']
    adapter = adapters[li]['adapter']
    
    for t in range(len(all_ids) - 1):
        future_h = t_h[li + 1][0, t + 1, :].float()
        
        with torch.no_grad():
            down_w = adapter.down.weight.float()
            mode_scores = F.linear(future_h.unsqueeze(0), Vh).squeeze().abs()
            mx = mode_scores.max()
            if mx > 1e-6:
                gains = torch.where(mode_scores > 0.3*mx, torch.tensor(2.0), torch.tensor(0.1))
                adapter.down.weight.data = (down_w * gains.unsqueeze(1)).to(torch.bfloat16)
                calibrated += 1

    if li % 6 == 0:
        print(f"  Layer {li:>2}: calibrated {calibrated} total pairs", flush=True)

print(f"\n  Total: {calibrated} adapter-mode pairs calibrated", flush=True)

# =====================================================================
# INFERENCE
# =====================================================================
student.eval()
oracle = TorusOracle(L=16)

prompt = "The most interesting thing about artificial intelligence is"
ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']

with torch.no_grad():
    t_ids = ids.clone(); t_tokens = []
    for _ in range(25):
        out = teacher(t_ids)
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        t_tokens.append(nxt.item()); t_ids = torch.cat([t_ids, nxt], -1)

teacher_text = tokenizer.decode(t_tokens)
print(f"\n{'='*78}")
print(f"Teacher: {teacher_text.encode('ascii',errors='replace').decode('ascii')[:100]}...")

s_tokens = []; s_matches = 0; ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
print(f"\nStudent:", flush=True)

for i in range(25):
    with torch.no_grad():
        out = student(ids, output_hidden_states=True)
    logits = out.logits[:, -1, :]
    h = out.hidden_states[-1][:, -1, :].squeeze()
    cv = oracle.push(h)
    T = cv * 1.7 + 0.3 if not math.isnan(cv) else 1.0
    safe = torch.nan_to_num(logits / T, nan=0., posinf=10., neginf=-10.)
    p = torch.softmax(safe.float(), -1); p = torch.nan_to_num(p, 0.); p = p/p.sum(-1, keepdim=True)
    nxt = torch.multinomial(p, 1); s_tokens.append(nxt.item())
    if i < len(t_tokens) and nxt.item() == t_tokens[i]: s_matches += 1
    try: w = tokenizer.decode([nxt.item()]).encode('ascii',errors='replace').decode('ascii')
    except: w = '?'
    ids = torch.cat([ids, nxt], -1)

student_text = tokenizer.decode(s_tokens)
print(f"  {student_text.encode('ascii',errors='replace').decode('ascii')[:200]}...")
print(f"\n  Match rate: {s_matches}/25 = {s_matches/25*100:.0f}%")
