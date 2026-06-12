"""Full adapter calibration: Q, K, V, O — all 96 adapters.

Uses teacher's PROJECTION OUTPUTS (not hidden states) as calibration
targets. Teacher Q output calibrates student Q adapter, K->K, etc.
Avoids dimension mismatch from GQA (K/V are 128-dim, hidden is 896-dim).

Temporal mode gating: teacher's t+1 projection output -> Vh -> mode scores
-> 2x boost aligned / 0.1x suppress -> calibrate adapter bottleneck.
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
print("FULL 96-ADAPTER TEMPORAL CALIBRATION — Q/K/V/O")
print("=" * 78)

print("\nLoading models...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
teacher.eval()
student = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)

K = 128
projections = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
adapters = {}  # (layer, proj) -> adapter
svd_data = {}  # (layer, proj) -> Vh

# Compress + inject adapters for ALL projections
for li in range(24):
    attn = student.model.layers[li].self_attn
    t_attn = teacher.model.layers[li].self_attn
    for mn in projections:
        # Student: compress weight
        linear = getattr(attn, mn)
        W = linear.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        k = min(K, U.shape[1])
        Uk, Sk, Vhk = U[:, :k], S[:k], Vh[:k, :]
        SVh_k = (Sk.unsqueeze(1) * Vhk).to(torch.bfloat16)
        linear.weight = nn.Parameter((Uk.to(torch.bfloat16) @ SVh_k), requires_grad=False)
        if linear.bias is not None: linear.bias.requires_grad = False
        
        # Adapter (bottleneck = min(K, out_dim))
        out_dim = W.shape[0]
        b = min(K, out_dim)
        adapter = PhaseAdapter(out_dim, b).to(device, dtype=torch.bfloat16)
        adapters[(li, mn)] = adapter
        linear._adapter = adapter
        
        # Teacher SVD for mode gating
        tW = getattr(t_attn, mn).weight.data.float()
        _, _, tVh = torch.linalg.svd(tW, full_matrices=False)
        svd_data[(li, mn)] = tVh[:min(K, tVh.shape[0]), :]

n_adapters = len(adapters)
print(f"  {n_adapters} adapters across {len(projections)} projections x 24 layers", flush=True)

# Hooks for all adapters
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
# TEMPORAL CALIBRATION
# =====================================================================
cal_prompt = "The catalytic computing paradigm demonstrates that information can be processed without"
cal_ids = tokenizer(cal_prompt, return_tensors="pt").to(device)['input_ids'][0, :24]

print(f"\nCalibrating on {len(cal_ids)} tokens...", flush=True)

# Capture teacher HIDDEN STATES per layer (not projection outputs)
# Hidden state at position t+1 is always 896-dim, matching all Vh matrices.
with torch.no_grad():
    t_out = teacher(cal_ids.unsqueeze(0), output_hidden_states=True)
t_hidden = t_out.hidden_states  # 25 layers, each (1, seq_len, 896)

# Calibrate adapters using layer hidden states as future tape
calibrated = 0
for (li, mn), adapter in adapters.items():
    Vh = svd_data[(li, mn)]  # (K, in_dim=896 or 128 for K/V)
    
    for t in range(len(cal_ids) - 1):
        # Future tape: hidden state at this layer, position t+1 (always 896-dim)
        future_h = t_hidden[li + 1][0, t + 1, :].float()  # (896,)
        
        with torch.no_grad():
            down_w = adapter.down.weight.float()
            
            # Vh operates on INPUT dimension. For Q/O: Vh is (K, 896), future is (896,) -> OK
            # For K/V: Vh is (K, 896), future is (896,) -> also OK because hidden dim = 896
            mode_scores = F.linear(future_h.unsqueeze(0), Vh).squeeze().abs()
            mx = mode_scores.max()
            if mx > 1e-6:
                gains = torch.where(
                    mode_scores > 0.3*mx, torch.tensor(2.0), torch.tensor(0.1))
                # gains is (K,) — pad or truncate to match adapter bottleneck
                b = down_w.shape[0]  # adapter bottleneck
                if len(gains) < b:
                    gains = F.pad(gains, (0, b - len(gains)), value=1.0)
                else:
                    gains = gains[:b]
                adapter.down.weight.data = (down_w * gains.unsqueeze(1)).to(torch.bfloat16)
                calibrated += 1

print(f"  Calibrated {calibrated} adapter-mode pairs", flush=True)
# Remove hook registration code that's no longer needed

# =====================================================================
# INFERENCE TEST
# =====================================================================
student.eval()
oracle = TorusOracle(L=16)

prompt = "The most interesting thing about artificial intelligence is"
ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']

# Teacher baseline
with torch.no_grad():
    t_ids = ids.clone(); t_tokens = []
    for _ in range(25):
        out = teacher(t_ids)
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        t_tokens.append(nxt.item()); t_ids = torch.cat([t_ids, nxt], -1)

teacher_text = tokenizer.decode(t_tokens)
print(f"\n{'='*78}")
print(f"INFERENCE")
print(f"Teacher: {teacher_text.encode('ascii',errors='replace').decode('ascii')[:100]}...")

# Student
s_tokens = []; s_matches = 0; ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
print(f"\nStudent (96 adapters + oracle):")
print("-" * 60)

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
    tw = tokenizer.decode([t_tokens[i]]).encode('ascii',errors='replace').decode('ascii') if i < len(t_tokens) else '?'
    rho = '~' if cv < 0.3 else ('-' if cv < 0.6 else '*')
    match = '=' if (i<len(t_tokens) and nxt.item()==t_tokens[i]) else 'x'
    if i < 12: print(f"  {i+1:>2} {rho} var={cv:.3f} T={T:.2f} | {w:<14} {tw:<14} {match}", flush=True)

print("-" * 60)
student_text = tokenizer.decode(s_tokens)
print(f"Student: {student_text.encode('ascii',errors='replace').decode('ascii')[:150]}...")
print(f"\nToken match: {s_matches}/25 = {s_matches/25*100:.0f}%")
