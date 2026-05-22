"""Quality gap: compressed K=128 vs uncompressed teacher.

Measures perplexity on WikiText-2 sample to quantify compression loss.
Also tests different K values (64, 128, 256) to find the compression/quality knee.
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

print("=" * 78)
print("COMPRESSION QUALITY GAP — K sweep on real text")
print("=" * 78)

print("\nLoading tokenizer + teacher...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
teacher.eval()

# Test text — real English paragraph
test_text = """The catalytic computing paradigm demonstrates that information can be processed without permanent storage. 
By borrowing computational resources temporarily and restoring them to their original state, catalytic algorithms achieve 
zero net information change. This approach has profound implications for energy efficiency, as Landauer's principle states 
that only irreversible operations dissipate heat. A fully catalytic computation would operate at the thermodynamic limit."""

token_ids = tokenizer(test_text, return_tensors="pt").to(device)['input_ids'][0]
print(f"Test text: {len(token_ids)} tokens", flush=True)

# Perplexity measurement
def compute_perplexity(model, ids):
    """Sliding window perplexity on sequence."""
    nll_sum = 0.0; n_tokens = 0
    window = 16
    for i in range(0, len(ids) - window, window // 2):
        ctx = ids[i:i+window].unsqueeze(0)
        tgt = ids[i+1:i+window+1]
        if len(tgt) < 2: continue
        with torch.no_grad():
            out = model(ctx)
        logits = out.logits[:, -len(tgt):, :]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tgt[:-1].unsqueeze(0).to(device)
        nll = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='sum')
        nll_sum += nll.item(); n_tokens += len(tgt) - 1
    return math.exp(nll_sum / max(n_tokens, 1))

# Teacher baseline
teacher_ppl = compute_perplexity(teacher, token_ids)
print(f"\nTeacher (uncompressed): PPL = {teacher_ppl:.2f}", flush=True)

# Test different K values
K_values = [None, 256, 128, 64, 32, 16]
results = []

for K in K_values:
    label = f"K={K}" if K else "uncompressed"
    print(f"\n{label}...", flush=True)
    t0 = time.perf_counter()
    
    student = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
    
    if K:
        for li in range(24):
            linear = student.model.layers[li].self_attn.q_proj
            W = linear.weight.data.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            k = min(K, U.shape[1])
            Uk, Sk, Vhk = U[:, :k], S[:k], Vh[:k, :]
            SVh_k = (Sk.unsqueeze(1) * Vhk).to(torch.bfloat16)
            linear.weight = nn.Parameter((Uk.to(torch.bfloat16) @ SVh_k), requires_grad=False)
            if linear.bias is not None: linear.bias.requires_grad = False
        
        # Also compress K/V/O for fair comparison
        for mn in ['k_proj', 'v_proj', 'o_proj']:
            for li in range(24):
                linear = getattr(student.model.layers[li].self_attn, mn)
                W = linear.weight.data.float()
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                k = min(K, U.shape[1])
                Uk, Sk, Vhk = U[:, :k], S[:k], Vh[:k, :]
                SVh_k = (Sk.unsqueeze(1) * Vhk).to(torch.bfloat16)
                linear.weight = nn.Parameter((Uk.to(torch.bfloat16) @ SVh_k), requires_grad=False)
    
    student.eval()
    ppl = compute_perplexity(student, token_ids)
    elapsed = time.perf_counter() - t0
    
    # Compression ratio
    if K:
        orig_params = 896 * 896  # single Q matrix
        compressed = K * 896 + K + K * 896  # U + S + Vh
        ratio = orig_params / compressed
    else:
        ratio = 1.0
    
    quality_retention = teacher_ppl / max(ppl, 0.01) * 100
    results.append({'K': K or 896, 'label': label, 'ppl': ppl, 'ratio': ratio, 
                     'retention': quality_retention, 'time': elapsed})
    print(f"  PPL={ppl:.2f} (retention={quality_retention:.0f}%) ratio={ratio:.1f}x [{elapsed:.0f}s]", flush=True)
    
    del student; torch.cuda.empty_cache()

# Summary
print(f"\n{'='*78}")
print(f"COMPRESSION QUALITY SWEEP — lower PPL = better, higher retention = closer to teacher")
print(f"{'='*78}")
print(f"  {'K':>8} {'PPL':>8} {'Retention':>10} {'Ratio':>8} {'Time':>6}")
print(f"  {'-'*45}")
for r in results:
    print(f"  {r['label']:>8} {r['ppl']:>8.2f} {r['retention']:>9.1f}% {r['ratio']:>7.1f}x {r['time']:>5.0f}s")
print(f"\n  Teacher PPL: {teacher_ppl:.2f}")
