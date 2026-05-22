"""Oracle coherence metric — replaces exact token match with torus stability.

Measures output quality via circular variance of the generation trajectory.
Low variance = coherent, stable output. High variance = chaos, gibberish.
Compares student vs teacher variance curves.
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
print("ORACLE COHERENCE METRIC — Torus Variance Curves")
print("=" * 78)

print("\nLoading models...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
teacher.eval()
student = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
student2 = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)  # uncalibrated

K = 128
adapters = {}

for li in range(24):
    attn = student.model.layers[li].self_attn
    t_attn = teacher.model.layers[li].self_attn
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

# Also compress student2 (no adapters — baseline)
for li in range(24):
    attn = student2.model.layers[li].self_attn
    linear = attn.q_proj
    W = linear.weight.data.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    k = min(K, U.shape[1])
    Uk, Sk, Vhk = U[:, :k], S[:k], Vh[:k, :]
    SVh_k = (Sk.unsqueeze(1) * Vhk).to(torch.bfloat16)
    linear.weight = nn.Parameter((Uk.to(torch.bfloat16) @ SVh_k), requires_grad=False)

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
# CALIBRATION
# =====================================================================
prompts = [
    "The catalytic computing paradigm demonstrates that information can be processed without",
    "Artificial intelligence research has consistently shown that the most important factor",
    "The fundamental laws of physics suggest that the universe operates on principles of",
    "Recent advances in quantum computing indicate that we are approaching a threshold where",
    "The relationship between information theory and thermodynamics reveals that entropy is",
]
all_ids = []
for p in prompts:
    ids = tokenizer(p, return_tensors="pt").to(device)['input_ids'][0].tolist()
    all_ids.extend(ids)
all_ids = all_ids[:128]
ids_tensor = torch.tensor(all_ids).unsqueeze(0).to(device)

print(f"Calibrating on {len(all_ids)} tokens...", flush=True)
with torch.no_grad():
    t_out = teacher(ids_tensor, output_hidden_states=True)

for li in range(24):
    Vh = adapters[li]['Vh']; adapter = adapters[li]['adapter']
    for t in range(len(all_ids) - 1):
        future_h = t_out.hidden_states[li + 1][0, t + 1, :].float()
        with torch.no_grad():
            down_w = adapter.down.weight.float()
            scores = F.linear(future_h.unsqueeze(0), Vh).squeeze().abs()
            mx = scores.max()
            if mx > 1e-6:
                gains = torch.where(scores > 0.3*mx, torch.tensor(2.0), torch.tensor(0.1))
                adapter.down.weight.data = (down_w * gains.unsqueeze(1)).to(torch.bfloat16)

print("  Done.", flush=True)

# =====================================================================
# ORACLE COHERENCE METRIC — compare 3 models across 30 tokens
# =====================================================================
student.eval(); student2.eval()

prompt = "The most interesting thing about artificial intelligence is"
ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']

def run_oracle_trace(model, name, max_tokens=30):
    oracle = TorusOracle(L=16)
    trace = []
    ids_local = ids.clone()
    tokens = []
    for i in range(max_tokens):
        with torch.no_grad():
            out = model(ids_local, output_hidden_states=True)
        logits = out.logits[:, -1, :]
        h = out.hidden_states[-1][:, -1, :].squeeze()
        cv = oracle.push(h)
        T = cv * 1.7 + 0.3 if not math.isnan(cv) else 1.0
        safe = torch.nan_to_num(logits / T, nan=0., posinf=10., neginf=-10.)
        p = torch.softmax(safe.float(), -1); p = torch.nan_to_num(p, 0.)
        p = p/p.sum(-1, keepdim=True)
        nxt = torch.multinomial(p, 1)
        tokens.append(nxt.item())
        trace.append({'step': i, 'var': cv, 'mag': oracle.mag, 'T': T})
        ids_local = torch.cat([ids_local, nxt], -1)
    return trace, tokens

print(f"\n{'='*78}")
print("ORACLE COHERENCE TRACES")
print("=" * 78)

print("\nTeacher...", flush=True)
t_trace, t_tokens = run_oracle_trace(teacher, "Teacher")
print("Student (calibrated)...", flush=True)
s_trace, s_tokens = run_oracle_trace(student, "Student")
print("Student (zero-shot)...", flush=True)
z_trace, z_tokens = run_oracle_trace(student2, "Zero-Shot")

# Analysis
t_vars = [x['var'] for x in t_trace]
s_vars = [x['var'] for x in s_trace]
z_vars = [x['var'] for x in z_trace]
t_mags = [x['mag'] for x in t_trace]
s_mags = [x['mag'] for x in s_trace]
z_mags = [x['mag'] for x in z_trace]

# Skip first 3 steps (buffer fill)
t_var_mean = np.mean(t_vars[3:]); t_var_std = np.std(t_vars[3:])
s_var_mean = np.mean(s_vars[3:]); s_var_std = np.std(s_vars[3:])
z_var_mean = np.mean(z_vars[3:]); z_var_std = np.std(z_vars[3:])

print(f"\n{'='*78}")
print(f"COHERENCE METRICS (lower variance = more coherent)")
print(f"{'='*78}")
print(f"  {'Model':<20} {'Var mean':>10} {'Var std':>10} {'Mag mean':>12} {'Coherence'}")
print(f"  {'-'*65}")
print(f"  {'Teacher':<20} {t_var_mean:>10.4f} {t_var_std:>10.4f} {np.mean(t_mags[3:]):>12.1f} {'*** (gold standard)'}")
print(f"  {'Student (calibrated)':<20} {s_var_mean:>10.4f} {s_var_std:>10.4f} {np.mean(s_mags[3:]):>12.1f} {'...'}")
print(f"  {'Student (zero-shot)':<20} {z_var_mean:>10.4f} {z_var_std:>10.4f} {np.mean(z_mags[3:]):>12.1f} {'...'}")

# Quality score: how close is calibrated to teacher?
var_distance = abs(s_var_mean - t_var_mean) / max(t_var_mean, 1e-6)
coherence_score = max(0, 100 - var_distance * 100)
print(f"\n  Coherence score: {coherence_score:.0f}/100 (0=perfect match to teacher variance)")

# Show traces
print(f"\n  Trace (first 12 steps):")
print(f"  {'Step':>5} {'Teacher':>8} {'Calibrated':>10} {'Zero-Shot':>10}")
for i in range(min(12, len(t_trace))):
    print(f"  {i+1:>5} {t_vars[i]:>8.4f} {s_vars[i]:>10.4f} {z_vars[i]:>10.4f}")

# Output text samples
print(f"\n{'='*78}")
print(f"OUTPUT TEXT")
print(f"{'='*78}")
print(f"  Teacher:    {tokenizer.decode(t_tokens).encode('ascii',errors='replace').decode('ascii')[:120]}...")
print(f"  Calibrated: {tokenizer.decode(s_tokens).encode('ascii',errors='replace').decode('ascii')[:120]}...")
print(f"  Zero-shot:  {tokenizer.decode(z_tokens).encode('ascii',errors='replace').decode('ascii')[:120]}...")
