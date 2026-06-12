"""Iterative convergence — 2nd calibration pass tightens variance.

From CAT_CAS 23: 'All configurations converge in exactly 2 iterations
to float32 precision limits.' Each pass uses the student's own outputs
as the new baseline, tightening phase alignment toward the teacher.

Tracks variance convergence across iterations.
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

class TorusOracle:
    def __init__(self, L=16): self.L = L; self.buf = []; self.mag = 0.0
    def push(self, h):
        h = torch.nan_to_num(h.float().cpu(), nan=0., posinf=0., neginf=0.)
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
print("ITERATIVE CONVERGENCE — 2-Pass Temporal Calibration")
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

# Calibration data
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
N = len(all_ids)

# =====================================================================
# ITERATION 1: calibrate using teacher as future tape
# =====================================================================
print(f"\nIter 1: Teacher -> Student ({N} tokens)...", flush=True)
t0 = time.perf_counter()
with torch.no_grad():
    t_out = teacher(ids_tensor, output_hidden_states=True)

for li in range(24):
    Vh = adapters[li]['Vh']; adapter = adapters[li]['adapter']
    for t in range(N - 1):
        future_h = t_out.hidden_states[li + 1][0, t + 1, :].float()
        with torch.no_grad():
            down_w = adapter.down.weight.float()
            scores = F.linear(future_h.unsqueeze(0), Vh).squeeze().abs()
            mx = scores.max()
            if mx > 1e-6:
                gains = torch.where(scores > 0.3*mx, torch.tensor(2.0), torch.tensor(0.1))
                adapter.down.weight.data = (down_w * gains.unsqueeze(1)).to(torch.bfloat16)

# Measure iter-1 variance
student.eval()
oracle1 = TorusOracle(L=16)
test_ids = ids_tensor.clone()
with torch.no_grad():
    for i in range(N):
        out = student(test_ids[:, :i+1], output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :].squeeze()
        oracle1.push(h)
v1 = np.mean([oracle1.push(torch.zeros(896)) for _ in range(3)] + [oracle1.push(torch.zeros(896))])  # dummy
# Actually compute properly:
var1_trace = []
test_ids2 = ids_tensor
for i in range(min(30, N)):
    with torch.no_grad():
        out = student(test_ids2[:, :i+1] if i < N else test_ids2, output_hidden_states=True)
    h = out.hidden_states[-1][:, -1, :].squeeze()
    cv = oracle1.push(h)
    if i >= 3: var1_trace.append(cv)

var1_mean = np.mean(var1_trace) if var1_trace else 0

# =====================================================================
# ITERATION 2: calibrate using STUDENT's OWN outputs as baseline
# =====================================================================
print(f"Iter 2: Student self-consistency...", flush=True)
with torch.no_grad():
    s_out = student(ids_tensor, output_hidden_states=True)

for li in range(24):
    Vh = adapters[li]['Vh']; adapter = adapters[li]['adapter']
    for t in range(N - 1):
        # Student's own output at t+1
        future_h = s_out.hidden_states[li + 1][0, t + 1, :].float()
        with torch.no_grad():
            down_w = adapter.down.weight.float()
            scores = F.linear(future_h.unsqueeze(0), Vh).squeeze().abs()
            mx = scores.max()
            if mx > 1e-6:
                gains = torch.where(scores > 0.3*mx, torch.tensor(2.0), torch.tensor(0.1))
                adapter.down.weight.data = (down_w * gains.unsqueeze(1)).to(torch.bfloat16)

elapsed = time.perf_counter() - t0

# Measure iter-2 variance
oracle2 = TorusOracle(L=16)
var2_trace = []
for i in range(min(30, N)):
    with torch.no_grad():
        out = student(test_ids2[:, :i+1] if i < N else test_ids2, output_hidden_states=True)
    h = out.hidden_states[-1][:, -1, :].squeeze()
    cv = oracle2.push(h)
    if i >= 3: var2_trace.append(cv)
var2_mean = np.mean(var2_trace) if var2_trace else 0

print(f"  Done in {elapsed:.0f}s", flush=True)

# =====================================================================
# Teacher reference
# =====================================================================
oracle_t = TorusOracle(L=16)
var_t_trace = []
for i in range(min(30, N)):
    with torch.no_grad():
        out = teacher(test_ids2[:, :i+1] if i < N else test_ids2, output_hidden_states=True)
    h = out.hidden_states[-1][:, -1, :].squeeze()
    oracle_t.push(h)
for i in range(min(30, N)):
    with torch.no_grad():
        out = teacher(test_ids2[:, :i+1] if i < N else test_ids2, output_hidden_states=True)
    h = out.hidden_states[-1][:, -1, :].squeeze()
    cv = oracle_t.push(h)
    if i >= 3: var_t_trace.append(cv)
var_t_mean = np.mean(var_t_trace) if var_t_trace else 0

# =====================================================================
# RESULTS
# =====================================================================
target_variance = var_t_mean
iter1_gap = abs(var1_mean - target_variance)
iter2_gap = abs(var2_mean - target_variance)
convergence = (iter2_gap < iter1_gap * 0.5)

print(f"\n{'='*78}")
print(f"ITERATIVE CONVERGENCE RESULTS")
print(f"{'='*78}")
print(f"  Teacher variance:     {var_t_mean:.6f}")
print(f"  Iter 1 (teacher->student): {var1_mean:.6f} (gap: {iter1_gap:.6f})")
print(f"  Iter 2 (self-consistent):  {var2_mean:.6f} (gap: {iter2_gap:.6f})")
print(f"  Converged: {convergence} ({'gap shrank' if iter2_gap < iter1_gap else 'diverged'})")

# Inference test
prompt = "The most interesting thing about artificial intelligence is"
ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
oracle = TorusOracle(L=16)
tokens = []
for i in range(25):
    with torch.no_grad():
        out = student(ids, output_hidden_states=True)
    logits = out.logits[:, -1, :]; h = out.hidden_states[-1][:, -1, :].squeeze()
    cv = oracle.push(h); T = cv * 1.7 + 0.3
    safe = torch.nan_to_num(logits / T, nan=0., posinf=10., neginf=-10.)
    p = torch.softmax(safe.float(), -1); p = torch.nan_to_num(p, 0.); p = p/p.sum(-1, keepdim=True)
    nxt = torch.multinomial(p, 1); tokens.append(nxt.item()); ids = torch.cat([ids, nxt], -1)

student_text = tokenizer.decode(tokens)
print(f"\n  Output: {student_text.encode('ascii',errors='replace').decode('ascii')[:200]}...")
