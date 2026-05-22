"""Benchmark: Temporal-calibrated vs Zero-shot across 10 prompts.

Measures: output coherence (oracle variance), token diversity (unique tokens),
sentence structure (% sentences ending with punctuation), and teacher similarity.
"""
import sys, os, math, time, re
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
print("BENCHMARK — Temporal Calibration vs Zero-Shot")
print("=" * 78)

print("\nLoading...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
teacher.eval()

def build_student():
    s = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
    K = 128; adapters_out = {}
    for li in range(24):
        linear = s.model.layers[li].self_attn.q_proj
        W = linear.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        k = min(K, U.shape[1])
        Uk, Sk, Vhk = U[:, :k], S[:k], Vh[:k, :]
        SVh_k = (Sk.unsqueeze(1) * Vhk).to(torch.bfloat16)
        linear.weight = nn.Parameter((Uk.to(torch.bfloat16) @ SVh_k), requires_grad=False)
        adapter = PhaseAdapter(W.shape[0], K).to(device, dtype=torch.bfloat16)
        adapters_out[li] = {'adapter': adapter, 'Vh': Vh[:K, :]}
        linear._adapter = adapter
        def make_hook(qa):
            def hook(m, i, o):
                hs = o[0] if isinstance(o, tuple) else o; r = o[1:] if isinstance(o, tuple) else ()
                if qa is not None: hs = hs + 0.1 * qa(hs)
                return (hs,) + r if r else hs
            return hook
        s.model.layers[li].register_forward_hook(make_hook(adapter))
    return s, adapters_out

def calibrate(s, adapters, cal_ids):
    ids_t = torch.tensor(cal_ids).unsqueeze(0).to(device)
    with torch.no_grad():
        t_out = teacher(ids_t, output_hidden_states=True)
    for li in range(24):
        Vh = adapters[li]['Vh']; a = adapters[li]['adapter']
        for t in range(len(cal_ids)-1):
            fh = t_out.hidden_states[li+1][0, t+1, :].float()
            with torch.no_grad():
                dw = a.down.weight.float()
                sc = F.linear(fh.unsqueeze(0), Vh).squeeze().abs()
                mx = sc.max()
                if mx > 1e-6:
                    g = torch.where(sc>0.3*mx, torch.tensor(2.0), torch.tensor(0.1))
                    a.down.weight.data = (dw * g.unsqueeze(1)).to(torch.bfloat16)

def generate(model, prompt, max_tokens=30):
    oracle = TorusOracle(L=16)
    ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
    tokens = []; trace = []
    for i in range(max_tokens):
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        logits = out.logits[:, -1, :]; h = out.hidden_states[-1][:, -1, :].squeeze()
        cv = oracle.push(h); T = cv * 1.7 + 0.3
        safe = torch.nan_to_num(logits/T, nan=0., posinf=10., neginf=-10.)
        p = torch.softmax(safe.float(), -1); p = torch.nan_to_num(p,0.); p = p/p.sum(-1,keepdim=True)
        nxt = torch.multinomial(p,1); tokens.append(nxt.item())
        trace.append(cv); ids = torch.cat([ids, nxt], -1)
    return tokens, trace

def score_output(tokens, trace):
    text = tokenizer.decode(tokens)
    # Oracle metrics
    var_mean = np.mean(trace[3:]) if len(trace)>3 else 0
    var_std = np.std(trace[3:]) if len(trace)>3 else 0
    # Token diversity
    unique = len(set(tokens)) / max(len(tokens), 1)
    # Sentence structure
    sentences = text.count('.') + text.count('!') + text.count('?')
    # Word count (rough)
    words = len(text.split())
    # Repetition check
    words_list = text.lower().split()
    repeats = sum(1 for i in range(1, len(words_list)) if words_list[i] == words_list[i-1])
    return {'var_mean': var_mean, 'var_std': var_std, 'unique_ratio': unique,
            'sentences': sentences, 'words': words, 'repeats': repeats, 'text': text}

# =====================================================================
# Calibration data
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

# Test prompts (different from calibration)
test_prompts = [
    "The most interesting thing about artificial intelligence is",
    "When we examine the mathematical foundations of",
    "Scientists have long hypothesized that the nature of",
    "The key insight that emerged from decades of research is that",
    "A comprehensive analysis of the data shows that",
    "The future of computing depends on our ability to",
    "What makes human intelligence fundamentally different from machine intelligence is",
    "The ethical implications of advanced AI systems include",
    "Recent breakthroughs in natural language processing demonstrate that",
    "The convergence of quantum mechanics and information theory suggests that",
]

# Build both models
print("Building student models...", flush=True)
student_cal, adapters = build_student()
student_zero, _ = build_student()

# Calibrate one, leave other zero-shot
print("Calibrating...", flush=True)
calibrate(student_cal, adapters, cal_ids)
student_cal.eval(); student_zero.eval()

# Run benchmark
print(f"\n{'='*78}")
print(f"BENCHMARK — 10 test prompts, 30 tokens each")
print(f"{'='*78}")

results_cal = []; results_zero = []

for i, prompt in enumerate(test_prompts):
    print(f"\n[{i+1:>2}] {prompt[:50]}...", flush=True)
    
    cal_tokens, cal_trace = generate(student_cal, prompt, 30)
    zero_tokens, zero_trace = generate(student_zero, prompt, 30)
    
    cal_score = score_output(cal_tokens, cal_trace)
    zero_score = score_output(zero_tokens, zero_trace)
    
    results_cal.append(cal_score); results_zero.append(zero_score)
    
    # Print first prompt in detail
    if i == 0:
        print(f"    Calibrated: {cal_score['text'].encode('ascii',errors='replace').decode('ascii')[:120]}...")
        print(f"    Zero-shot:  {zero_score['text'].encode('ascii',errors='replace').decode('ascii')[:120]}...")

# Aggregate
cal_vars = [r['var_mean'] for r in results_cal]
zero_vars = [r['var_mean'] for r in results_cal]
cal_std = [r['var_std'] for r in results_cal]
zero_std = [r['var_std'] for r in results_zero]
cal_unique = [r['unique_ratio'] for r in results_cal]
zero_unique = [r['unique_ratio'] for r in results_zero]
cal_words = [r['words'] for r in results_cal]
zero_words = [r['words'] for r in results_zero]
cal_repeats = [r['repeats'] for r in results_cal]
zero_repeats = [r['repeats'] for r in results_zero]
cal_sent = [r['sentences'] for r in results_cal]
zero_sent = [r['sentences'] for r in results_zero]

print(f"\n{'='*78}")
print(f"AGGREGATE RESULTS (10 prompts, 30 tokens each)")
print(f"{'='*78}")
print(f"  {'Metric':<25} {'Calibrated':>12} {'Zero-Shot':>12}")
print(f"  {'-'*50}")
print(f"  {'Variance mean':<25} {np.mean(cal_vars):>12.4f} {np.mean(zero_vars):>12.4f}")
print(f"  {'Variance std':<25} {np.mean(cal_std):>12.4f} {np.mean(zero_std):>12.4f}")
print(f"  {'Unique token ratio':<25} {np.mean(cal_unique):>12.3f} {np.mean(zero_unique):>12.3f}")
print(f"  {'Sentences':<25} {np.mean(cal_sent):>12.1f} {np.mean(zero_sent):>12.1f}")
print(f"  {'Words':<25} {np.mean(cal_words):>12.1f} {np.mean(zero_words):>12.1f}")
print(f"  {'Consecutive repeats':<25} {np.mean(cal_repeats):>12.1f} {np.mean(zero_repeats):>12.1f}")

# Print all outputs
print(f"\n{'='*78}")
print(f"ALL OUTPUTS")
print(f"{'='*78}")
for i in range(len(test_prompts)):
    c = results_cal[i]['text'].encode('ascii',errors='replace').decode('ascii')[:100]
    z = results_zero[i]['text'].encode('ascii',errors='replace').decode('ascii')[:100]
    print(f"\n[{i+1:>2}] {test_prompts[i][:40]}...")
    print(f"    CAL: {c}...")
    print(f"    ZRO: {z}...")
