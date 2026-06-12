"""EigenBuddy lm_head replacement for holo-compressed model.

Replaces Qwen's 151K-parameter lm_head with EigenBuddy Platonic Tokenizer.
Collects hidden states from calibrated K=128 model, trains EigenBuddy to
map hidden_state -> gold_token, then hooks into inference.

Architecture (4.7): compressed attention + EigenBuddy tokenizer = full
holo-compressed inference without the massive vocabulary projection.
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
EIGEN_DIR = REPO / 'THOUGHT' / 'LAB' / 'EIGEN_BUDDY'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(EIGEN_DIR))
from eigen_buddy_tokenizer import STABLE_32, EigenBuddyTokenizer

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
print("EIGENBUDDY LM_HEAD REPLACEMENT")
print("  Platonic Tokenizer replaces 151K-param vocabulary projection")
print("=" * 78)

print("\nLoading models...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
teacher.eval()
student = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)

K = 128
for li in range(24):
    linear = student.model.layers[li].self_attn.q_proj
    W = linear.weight.data.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    k = min(K, U.shape[1])
    Uk, Sk, Vhk = U[:, :k], S[:k], Vh[:k, :]
    SVh_k = (Sk.unsqueeze(1) * Vhk).to(torch.bfloat16)
    linear.weight = nn.Parameter((Uk.to(torch.bfloat16) @ SVh_k), requires_grad=False)
    adapter = PhaseAdapter(W.shape[0], K).to(device, dtype=torch.bfloat16)
    linear._adapter = adapter
    def make_hook(qa):
        def hook(m,i,o):
            if isinstance(o, tuple):
                hs = o[0]
                if qa is not None: hs = hs + 0.1 * qa(hs)
                return (hs,) + o[1:]
            elif qa is not None:
                return o + 0.1 * qa(o)
            return o
        return hook
    student.model.layers[li].register_forward_hook(make_hook(adapter))

# Calibrate
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

print("Calibrating...", flush=True)
with torch.no_grad():
    t_out = teacher(ids_t, output_hidden_states=True)

for li in range(24):
    adapter = student.model.layers[li].self_attn.q_proj._adapter
    tW = teacher.model.layers[li].self_attn.q_proj.weight.data.float()
    _, _, Vh = torch.linalg.svd(tW, full_matrices=False); Vh = Vh[:K, :]
    for t in range(len(cal_ids)-1):
        fh = t_out.hidden_states[li+1][0, t+1, :].float()
        with torch.no_grad():
            dw = adapter.down.weight.float(); sc = F.linear(fh.unsqueeze(0), Vh).squeeze().abs()
            mx = sc.max()
            if mx>1e-6:
                g = torch.where(sc>0.3*mx, torch.tensor(2.0), torch.tensor(0.1))
                adapter.down.weight.data = (dw*g.unsqueeze(1)).to(torch.bfloat16)
student.eval()

# =====================================================================
# COLLECT HIDDEN STATE / GOLD TOKEN PAIRS
# =====================================================================
print("\nCollecting training data for EigenBuddy...", flush=True)

# Use calibration prompts + generate more from teacher
train_texts = [
    "The catalytic computing paradigm demonstrates that information can be",
    "Artificial intelligence research has consistently shown that",
    "The fundamental laws of physics suggest that",
    "The most interesting thing about artificial intelligence is",
    "When we examine the mathematical foundations of",
    "The future of computing depends on our ability to",
    "Recent breakthroughs in natural language processing demonstrate",
    "The convergence of quantum mechanics and information theory",
]
hidden_states = []
gold_tokens = []

for text in train_texts:
    tok_ids = tokenizer(text, return_tensors="pt").to(device)['input_ids'][0]
    
    # Teacher generates gold tokens
    with torch.no_grad():
        t_ids = tok_ids.unsqueeze(0)
        for _ in range(8):
            out = teacher(t_ids)
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            t_ids = torch.cat([t_ids, nxt], -1)
    all_ids = t_ids[0].tolist()
    
    # Student forward — collect hidden states
    with torch.no_grad():
        s_out = student(t_ids, output_hidden_states=True)
    
    # Pairs: (hidden_state at position t, gold_token at position t+1)
    for t in range(len(all_ids) - 1):
        h = s_out.hidden_states[-1][0, t, :].float().cpu()
        gold = all_ids[t + 1]
        hidden_states.append(h)
        gold_tokens.append(gold)

hidden_states = torch.stack(hidden_states)  # (N, 896)
gold_tokens = torch.tensor(gold_tokens, dtype=torch.long)
n_samples = len(gold_tokens)
n_classes = len(set(gold_tokens.tolist()))
print(f"  Collected {n_samples} pairs, {n_classes} unique gold tokens", flush=True)

# =====================================================================
# TRAIN EIGENBUDDY
# =====================================================================
print("\nTraining EigenBuddy...", flush=True)

# Remap gold tokens to 0..n_classes-1
unique_gold = sorted(set(gold_tokens.tolist()))
tok_to_idx = {t: i for i, t in enumerate(unique_gold)}
targets_remapped = torch.tensor([tok_to_idx[t.item()] for t in gold_tokens], dtype=torch.long)

# Per-sample normalize hidden states
hs_norm = hidden_states / (hidden_states.norm(dim=1, keepdim=True) + 1e-9)

# Complex SVD compression to reduce dimensionality
Z = hs_norm.numpy() + 1j * np.zeros_like(hs_norm.numpy())
Z_centered = Z - Z.mean(axis=0, keepdims=True)
C = (Z_centered.conj().T @ Z_centered) / (n_samples - 1)
evals, evecs = np.linalg.eigh(C); evals = evals[::-1]; evecs = evecs[:, ::-1]
cum = np.cumsum(evals / evals.sum())
k95 = int(np.searchsorted(cum, 0.95) + 1)
K_eigen = max(8, min(k95, n_samples // 4, 64))
print(f"  D_pr (hidden states): {1.0/((evals/evals.sum())**2).sum():.1f}, K95={k95}, using K={K_eigen}", flush=True)

# Project to compressed dimension
V = evecs[:, :K_eigen]
Z_proj = Z_centered @ V  # (N, K_eigen)

# Split
split = int(n_samples * 0.8)
train_in = []; test_in = []
for i in range(n_samples):
    z = torch.complex(torch.tensor(Z_proj[i].real.astype(np.float32)),
                       torch.tensor(Z_proj[i].imag.astype(np.float32)))
    if i < split: train_in.append(z.unsqueeze(0).unsqueeze(0))
    else: test_in.append(z.unsqueeze(0).unsqueeze(0))
train_tgt = targets_remapped[:split]; test_tgt = targets_remapped[split:]

# Build EigenBuddy
model = EigenBuddyTokenizer(dim=K_eigen, vocab_size=n_classes, eigen_layers=2,
                             eigen_heads=max(1, K_eigen//4) if K_eigen >= 4 else 1)

# Manual training loop (avoids anchor bank mismatch)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
model = model.to(device)

print(f"  Training {len(train_in)} samples, {n_classes} classes, dim={K_eigen}", flush=True)
for epoch in range(200):
    perm = torch.randperm(len(train_in))
    epoch_loss = 0.0; epoch_acc = 0.0; n_batches = 0
    for start in range(0, len(train_in), 32):
        end = min(start+32, len(train_in))
        idx = perm[start:end]
        batch_in = torch.cat([train_in[i] for i in idx], dim=0).to(device)
        batch_tgt = train_tgt[idx].to(device)
        logits, _ = model(batch_in)
        loss = F.cross_entropy(logits, batch_tgt)
        if torch.isnan(loss) or torch.isinf(loss): continue
        opt.zero_grad(); loss.backward(); opt.step()
        preds = logits.argmax(dim=-1)
        epoch_loss += loss.item(); epoch_acc += (preds==batch_tgt).float().mean().item(); n_batches += 1
    if epoch % 40 == 0:
        print(f"  Epoch {epoch:>3}: loss={epoch_loss/max(n_batches,1):.4f} acc={epoch_acc/max(n_batches,1):.3f}", flush=True)

# Evaluate
model.eval()
test_correct = 0
with torch.no_grad():
    for i in range(len(test_in)):
        logits, _ = model(test_in[i].to(device))
        pred = logits.argmax(dim=-1).item()
        if pred == test_tgt[i].item(): test_correct += 1
test_acc = test_correct / max(len(test_tgt), 1)

train_correct = 0
with torch.no_grad():
    for i in range(len(train_in)):
        logits, _ = model(train_in[i].to(device))
        pred = logits.argmax(dim=-1).item()
        if pred == train_tgt[i].item(): train_correct += 1
train_acc = train_correct / max(len(train_tgt), 1)

print(f"\n  EigenBuddy: train acc={train_acc:.3f} test acc={test_acc:.3f}", flush=True)

# =====================================================================
# INFERENCE WITH EIGENBUDDY AS LM_HEAD
# =====================================================================
print(f"\n{'='*78}")
print("INFERENCE — EigenBuddy replaces lm_head")
print("=" * 78)

# EigenBuddy inference: hidden_state -> token
# We need the projection matrix for inference time
V_tensor = torch.tensor(V.real.astype(np.float32)).to(device)
Z_mean_tensor = torch.tensor(Z.mean(axis=0).real.astype(np.float32)).to(device)

def eigenbuddy_decode(hidden, model_eb, V_mat, z_mean, tok_map):
    """Map hidden state through EigenBuddy to token ID."""
    h = hidden.float()
    h_centered = h - z_mean
    h_proj = h_centered @ V_mat  # (K_eigen,)
    z = torch.complex(h_proj, torch.zeros_like(h_proj)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model_eb(z.to(device))
    idx = logits.argmax(dim=-1).item()
    return tok_map[idx]  # map back to original token ID

idx_to_tok = {i: t for i, t in enumerate(unique_gold)}

prompt = "The most interesting thing about artificial intelligence is"
ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']

# Teacher baseline
with torch.no_grad():
    t_ids = ids.clone(); t_tokens = []
    for _ in range(20):
        out = teacher(t_ids)
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        t_tokens.append(nxt.item()); t_ids = torch.cat([t_ids, nxt], -1)

print(f"\nTeacher: {tokenizer.decode(t_tokens).encode('ascii',errors='replace').decode('ascii')[:120]}...")

# Student with EigenBuddy
s_tokens = []; ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
matches = 0

print("\nStudent (EigenBuddy lm_head):")
for i in range(20):
    with torch.no_grad():
        out = student(ids, output_hidden_states=True)
    h = out.hidden_states[-1][:, -1, :].squeeze()
    
    tok = eigenbuddy_decode(h, model, V_tensor, Z_mean_tensor, idx_to_tok)
    s_tokens.append(tok)
    if i < len(t_tokens) and tok == t_tokens[i]: matches += 1
    
    try: w = tokenizer.decode([tok]).encode('ascii',errors='replace').decode('ascii')
    except: w = '?'
    tw = tokenizer.decode([t_tokens[i]]).encode('ascii',errors='replace').decode('ascii') if i<len(t_tokens) else '?'
    match = '=' if (i<len(t_tokens) and tok==t_tokens[i]) else 'x'
    if i < 12: print(f"  {i+1:>2} eig={w:<14} tea={tw:<14} {match}", flush=True)
    
    ids = torch.cat([ids, torch.tensor([[tok]]).to(device)], -1)

student_text = tokenizer.decode(s_tokens)
print(f"\nStudent: {student_text.encode('ascii',errors='replace').decode('ascii')[:150]}...")
print(f"Match rate: {matches}/20 = {matches/20*100:.0f}%")

# Compare: EigenBuddy params vs lm_head params
lm_head_params = 151936 * 896
eb_params = sum(p.numel() for p in model.parameters())
print(f"\n  lm_head params: {lm_head_params:,} -> EigenBuddy params: {eb_params:,} ({lm_head_params/eb_params:.0f}x reduction)")
