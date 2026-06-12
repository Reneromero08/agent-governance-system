"""Phase Cavity + Temporal Calibration — combined pipeline.

1. SVD compress Q-proj to K=128
2. Phase Cavity sieve: test each eigenmode, discard dispersion (keep ~64)
3. Build adapters with bottleneck = K_kept
4. Temporal calibration on kept modes only
5. Measure PPL vs baselines
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
    def __init__(self, dim, bottleneck):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.01); nn.init.zeros_(self.up.weight)
    def forward(self, x): return self.up(self.act(self.down(x)))

print("=" * 78)
print("PHASE CAVITY + TEMPORAL CALIBRATION")
print("=" * 78)

print("\nLoading models...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
teacher.eval()

def ppl(model, ids_list):
    nll = 0.0; nt = 0; w = 16
    for i in range(0, len(ids_list)-w, w//2):
        ctx = ids_list[i:i+w].unsqueeze(0); tgt = ids_list[i+1:i+w+1]
        if len(tgt)<2: continue
        with torch.no_grad(): out = model(ctx)
        logits = out.logits[:, -len(tgt):, :]; sl = logits[:, :-1, :].contiguous()
        st = tgt[:-1].unsqueeze(0).to(device)
        nll += F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1), reduction='sum').item()
        nt += len(tgt)-1
    return math.exp(nll/max(nt,1))

def cosine_sim(Wo, Wr, n_t=20):
    X = torch.randn(n_t, Wo.shape[1], device=Wo.device)
    Yo = Wo.float()@X.T; Yr = Wr.float()@X.T
    d = (Yo*Yr).sum(dim=0)
    return (d/(Yo.norm(dim=0)*Yr.norm(dim=0)+1e-9)).mean().item()

test_text = "The catalytic computing paradigm demonstrates that information can be processed without permanent storage. By borrowing computational resources temporarily and restoring them to their original state catalytic algorithms achieve zero net information change."
test_ids = tokenizer(test_text, return_tensors="pt").to(device)['input_ids'][0]
teacher_ppl = ppl(teacher, test_ids)
print(f"Teacher PPL: {teacher_ppl:.2f}", flush=True)

# Calibration data
cal_text = ("The catalytic computing paradigm demonstrates that information can be processed without "
            "Artificial intelligence research has consistently shown that the most important factor "
            "The fundamental laws of physics suggest that the universe operates on principles of")
cal_ids = tokenizer(cal_text, return_tensors="pt").to(device)['input_ids'][0, :96]

# =====================================================================
# BUILD ALL THREE MODELS
# =====================================================================
# Model A: K=128 calibrated (baseline)
# Model B: Phase Cavity K_kept calibrated
# Model C: Phase Cavity K_kept uncalibrated

K = 128
cavity_kept = {}  # per-layer: list of kept eigenmodes

def build_model(label):
    s = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
    s.eval()
    return s

# Phase Cavity first pass: identify which modes to keep
print("\nPhase Cavity sieving...", flush=True)
for li in range(24):
    W = teacher.model.layers[li].self_attn.q_proj.weight.data.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    k = min(K, U.shape[1])
    Uk, Sk, Vhk = U[:, :k], S[:k], Vh[:k, :]
    
    # Cavity sieve: test bottom 64 modes (smallest eigenvalues first)
    kept = list(range(k))
    check_start = max(0, k - 64)
    for i in range(k-1, check_start-1, -1):
        keep = [j for j in kept if j != i]
        if not keep: continue
        Wt = (Uk[:, keep] * Sk[keep].unsqueeze(0)) @ Vhk[keep, :]
        if cosine_sim(W, Wt) > 0.99: kept.remove(i)
    
    cavity_kept[li] = sorted(kept)
    if li % 6 == 0:
        print(f"  Layer {li:>2}: kept {len(kept)}/{k} modes", flush=True)

total_k = sum(len(v) for v in cavity_kept.values())
avg_kept = total_k / 24
print(f"\n  Avg kept: {avg_kept:.0f}/{K} ({avg_kept/K*100:.0f}%)\n", flush=True)

# Now build and calibrate models
for model_name, use_cavity, do_calibrate in [
    ("K=128 calibrated", False, True),
    ("Cavity uncalibrated", True, False),
    ("Cavity calibrated", True, True),
]:
    print(f"Building {model_name}...", flush=True)
    t0 = time.perf_counter()
    s = build_model(model_name)
    
    # Compress and optionally cavity
    for li in range(24):
        linear = s.model.layers[li].self_attn.q_proj
        W = linear.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        
        if use_cavity:
            kept = cavity_kept[li]
            k_use = len(kept)
            Uk, Sk, Vhk = U[:, kept], S[kept], Vh[kept, :]
        else:
            k_use = min(K, U.shape[1])
            Uk, Sk, Vhk = U[:, :k_use], S[:k_use], Vh[:k_use, :]
        
        SVh_k = (Sk.unsqueeze(1) * Vhk).to(torch.bfloat16)
        linear.weight = nn.Parameter((Uk.to(torch.bfloat16) @ SVh_k), requires_grad=False)
        
        # Adapter with matching bottleneck
        adapter = PhaseAdapter(W.shape[0], k_use).to(device, dtype=torch.bfloat16)
        linear._adapter = adapter
        
        def make_hook(qa):
            def hook(m,i,o):
                if isinstance(o, tuple):
                    hs = o[0]; return ((hs + 0.1*qa(hs)),) + o[1:] if qa is not None else o
                return o + 0.1*qa(o) if qa is not None else o
            return hook
        s.model.layers[li].register_forward_hook(make_hook(adapter))
    
    s.eval()
    
    # Temporal calibration
    if do_calibrate:
        ids_t = cal_ids.unsqueeze(0)
        with torch.no_grad():
            t_out = teacher(ids_t, output_hidden_states=True)
        
        for li in range(24):
            adapter = s.model.layers[li].self_attn.q_proj._adapter
            tW = teacher.model.layers[li].self_attn.q_proj.weight.data.float()
            _, _, tVh = torch.linalg.svd(tW, full_matrices=False)
            tVh = tVh[:K, :]
            
            if use_cavity:
                tVh = tVh[cavity_kept[li], :]
            else:
                tVh = tVh[:min(K, tVh.shape[0]), :]
            
            for t in range(len(cal_ids)-1):
                fh = t_out.hidden_states[li+1][0, t+1, :].float()
                with torch.no_grad():
                    dw = adapter.down.weight.float()
                    sc = F.linear(fh.unsqueeze(0), tVh).squeeze().abs()
                    mx = sc.max()
                    if mx > 1e-6:
                        g = torch.where(sc>0.3*mx, torch.tensor(2.0), torch.tensor(0.1))
                        adapter.down.weight.data = (dw * g.unsqueeze(1)).to(torch.bfloat16)
    
    s.eval()
    p = ppl(s, test_ids)
    elapsed = time.perf_counter() - t0
    
    # Compression ratio
    if use_cavity:
        avg_k = avg_kept
    else:
        avg_k = K
    ratio = (896*896) / (avg_k*896 + avg_k + avg_k*896)
    
    retention = teacher_ppl / max(p, 0.01) * 100
    
    # Adapter params
    a_params = sum(p.numel() for li in range(24) 
                   for p in s.model.layers[li].self_attn.q_proj._adapter.parameters())
    
    print(f"  PPL={p:.1f} retention={retention:.1f}% ratio={ratio:.1f}x "
          f"adapter={a_params/1000:.0f}K [{elapsed:.0f}s]", flush=True)
    
    del s; torch.cuda.empty_cache()

print(f"\n{'='*78}")
print("SUMMARY")
print(f"{'='*78}")
print(f"  Teacher PPL: {teacher_ppl:.1f}")
