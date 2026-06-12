"""Gain profile sweep — find optimal coherence/diversity balance.

Tests: aligned_boost from 1.1x to 2.0x, misaligned_suppress from 0.1x to 0.9x.
Measures sentences (structure) and unique token ratio (diversity).
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
print("GAIN PROFILE SWEEP")
print("=" * 78)

print("\nLoading...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
teacher.eval()

def build_and_calibrate(boost, suppress):
    """Build student, calibrate with given gain profile, return model."""
    s = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
    K = 128
    for li in range(24):
        linear = s.model.layers[li].self_attn.q_proj
        W = linear.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        k = min(K, U.shape[1])
        Uk, Sk, Vhk = U[:, :k], S[:k], Vh[:k, :]
        SVh_k = (Sk.unsqueeze(1) * Vhk).to(torch.bfloat16)
        linear.weight = nn.Parameter((Uk.to(torch.bfloat16) @ SVh_k), requires_grad=False)
        adapter = PhaseAdapter(W.shape[0], K).to(device, dtype=torch.bfloat16)
        linear._adapter = adapter
        def make_hook(qa):
            def hook(m, i, o):
                hs = o[0] if isinstance(o, tuple) else o; r = o[1:] if isinstance(o, tuple) else ()
                if qa is not None: hs = hs + 0.1 * qa(hs)
                return (hs,) + r if r else hs
            return hook
        s.model.layers[li].register_forward_hook(make_hook(adapter))
    
    # Calibrate with custom gains
    cal_prompts = [
        "The catalytic computing paradigm demonstrates that information can be processed without",
        "Artificial intelligence research has consistently shown that the most important factor",
        "The fundamental laws of physics suggest that the universe operates on principles of",
        "Recent advances in quantum computing indicate that we are approaching a threshold where",
        "The relationship between information theory and thermodynamics reveals that entropy is",
    ]
    cal_ids = []
    for p in cal_prompts:
        cal_ids.extend(tokenizer(p, return_tensors="pt").to(device)['input_ids'][0].tolist())
    cal_ids = cal_ids[:128]
    ids_t = torch.tensor(cal_ids).unsqueeze(0).to(device)
    
    with torch.no_grad():
        t_out = teacher(ids_t, output_hidden_states=True)
    
    for li in range(24):
        linear = s.model.layers[li].self_attn.q_proj
        adapter = linear._adapter
        tW = teacher.model.layers[li].self_attn.q_proj.weight.data.float()
        _, _, Vh = torch.linalg.svd(tW, full_matrices=False)
        Vh = Vh[:K, :]
        
        for t in range(len(cal_ids)-1):
            fh = t_out.hidden_states[li+1][0, t+1, :].float()
            with torch.no_grad():
                dw = adapter.down.weight.float()
                sc = F.linear(fh.unsqueeze(0), Vh).squeeze().abs()
                mx = sc.max()
                if mx > 1e-6:
                    aligned = sc > 0.3*mx
                    g = torch.where(aligned, torch.tensor(boost), torch.tensor(suppress))
                    adapter.down.weight.data = (dw * g.unsqueeze(1)).to(torch.bfloat16)
    
    s.eval()
    return s

test_prompts = [
    "The most interesting thing about artificial intelligence is",
    "When we examine the mathematical foundations of",
    "The future of computing depends on our ability to",
]

# Gain profiles to test
profiles = [
    (2.0, 0.1, "2.0x/0.1x"),   # current
    (1.5, 0.5, "1.5x/0.5x"),
    (1.3, 0.7, "1.3x/0.7x"),
    (1.1, 0.9, "1.1x/0.9x"),
    (1.0, 1.0, "1.0x/1.0x"),   # no calibration (baseline)
]

results = []
for boost, suppress, label in profiles:
    print(f"\n{label}...", flush=True)
    t0 = time.perf_counter()
    # Teacher already loaded — reuse
    
    s = build_and_calibrate(boost, suppress)
    elapsed = time.perf_counter() - t0
    
    promp_results = []
    for prompt in test_prompts:
        oracle = TorusOracle(L=16)
        ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
        tokens = []; trace = []
        for i in range(30):
            with torch.no_grad():
                out = s(ids, output_hidden_states=True)
            logits = out.logits[:, -1, :]; h = out.hidden_states[-1][:, -1, :].squeeze()
            cv = oracle.push(h); T = cv * 1.7 + 0.3
            safe = torch.nan_to_num(logits/T, nan=0., posinf=10., neginf=-10.)
            p = torch.softmax(safe.float(), -1); p = torch.nan_to_num(p,0.); p = p/p.sum(-1,keepdim=True)
            nxt = torch.multinomial(p,1); tokens.append(nxt.item())
            trace.append(cv); ids = torch.cat([ids, nxt], -1)
        
        text = tokenizer.decode(tokens)
        var_mean = np.mean(trace[3:]) if len(trace)>3 else 0
        unique = len(set(tokens)) / max(len(tokens), 1)
        sentences = text.count('.') + text.count('!') + text.count('?')
        words = len(text.split())
        
        # Repetition detection: longest repeated n-gram
        words_list = text.lower().split()
        max_repeat = 0
        for n in [3, 4, 5]:
            for i in range(len(words_list) - n*2 + 1):
                gram = tuple(words_list[i:i+n])
                for j in range(i+n, len(words_list) - n + 1):
                    if tuple(words_list[j:j+n]) == gram:
                        max_repeat = max(max_repeat, n)
        
        promp_results.append({
            'var_mean': var_mean, 'unique': unique, 'sentences': sentences,
            'words': words, 'max_repeat': max_repeat, 'text': text[:120]
        })
    
    avg = {k: np.mean([r[k] for r in promp_results]) for k in ['var_mean','unique','sentences','words','max_repeat']}
    avg['label'] = label; avg['time'] = elapsed
    results.append(avg)
    
    # Show first prompt output
    print(f"  [{elapsed:.0f}s] sent={avg['sentences']:.1f} uniq={avg['unique']:.3f} repeat={avg['max_repeat']:.1f}")
    print(f"    {promp_results[0]['text'].encode('ascii',errors='replace').decode('ascii')}...")

# Summary
print(f"\n{'='*78}")
print(f"SWEEP RESULTS — higher sentences = better structure, higher unique = more diversity")
print(f"{'='*78}")
print(f"  {'Profile':<15} {'Sentences':>10} {'Unique':>8} {'MaxRepeat':>10} {'Time':>6}")
print(f"  {'-'*55}")
for r in results:
    print(f"  {r['label']:<15} {r['sentences']:>10.1f} {r['unique']:>8.3f} {r['max_repeat']:>10.1f} {r['time']:>5.0f}s")
