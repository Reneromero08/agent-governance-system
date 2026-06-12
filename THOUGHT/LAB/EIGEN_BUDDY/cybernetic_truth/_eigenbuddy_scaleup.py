"""EigenBuddy scale-up: generate 500+ training pairs, train, test.

Collects hidden_state -> gold_token pairs from calibrated K=128 model
across diverse prompts. Projects to compressed dimension via SVD.
Trains EigenBuddy with more data for generalization.
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
from eigen_buddy_tokenizer import EigenBuddyTokenizer

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
print("EIGENBUDDY SCALE-UP — 500+ training pairs")
print("=" * 78)

print("\nLoading...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
teacher.eval()

# Build calibrated student once
K = 128
student = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
for li in range(24):
    linear = student.model.layers[li].self_attn.q_proj
    W = linear.weight.data.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    k = min(K, U.shape[1]); Uk, Sk, Vhk = U[:,:k], S[:k], Vh[:k,:]
    SVh_k = (Sk.unsqueeze(1)*Vhk).to(torch.bfloat16)
    linear.weight = nn.Parameter((Uk.to(torch.bfloat16)@SVh_k), requires_grad=False)
    adapter = PhaseAdapter(W.shape[0], K).to(device, dtype=torch.bfloat16)
    linear._adapter = adapter
    def make_hook(qa):
        def hook(m,i,o):
            if isinstance(o,tuple): hs=o[0]; return ((hs+0.1*qa(hs)),)+o[1:] if qa else o
            return o+0.1*qa(o) if qa else o
        return hook
    student.model.layers[li].register_forward_hook(make_hook(adapter))

# Calibrate
cal_text = ("The catalytic computing paradigm demonstrates that information can be processed without "
            "Artificial intelligence research has consistently shown that the most important factor "
            "The fundamental laws of physics suggest that the universe operates on principles of "
            "Recent advances in quantum computing indicate that we are approaching a threshold where "
            "The relationship between information theory and thermodynamics reveals that entropy is")
cal_ids = tokenizer(cal_text, return_tensors="pt").to(device)['input_ids'][0, :128]
print("Calibrating...", flush=True)
ids_t = cal_ids.unsqueeze(0)
with torch.no_grad(): t_out = teacher(ids_t, output_hidden_states=True)
for li in range(24):
    adapter = student.model.layers[li].self_attn.q_proj._adapter
    tW = teacher.model.layers[li].self_attn.q_proj.weight.data.float()
    _, _, Vh = torch.linalg.svd(tW, full_matrices=False); Vh = Vh[:K,:]
    for t in range(len(cal_ids)-1):
        fh = t_out.hidden_states[li+1][0, t+1, :].float()
        with torch.no_grad():
            dw = adapter.down.weight.float(); sc = F.linear(fh.unsqueeze(0), Vh).squeeze().abs()
            if sc.max()>1e-6:
                g = torch.where(sc>0.3*sc.max(), torch.tensor(2.0), torch.tensor(0.1))
                adapter.down.weight.data = (dw*g.unsqueeze(1)).to(torch.bfloat16)
student.eval()

# =====================================================================
# COLLECT 500+ PAIRS
# =====================================================================
print("\nCollecting pairs from teacher generation...", flush=True)
prompts = [
    "The catalytic computing paradigm demonstrates",
    "Artificial intelligence research has shown",
    "The fundamental laws of physics suggest",
    "Recent advances in quantum computing indicate",
    "The relationship between information theory and thermodynamics",
    "When we examine the mathematical foundations of",
    "Scientists have long hypothesized that",
    "The future of computing depends on",
    "What makes human intelligence different from",
    "The ethical implications of advanced AI include",
    "Recent breakthroughs in natural language processing",
    "The convergence of quantum mechanics and information",
    "A comprehensive analysis of the data reveals",
    "The key insight from decades of research is",
    "Modern machine learning systems demonstrate that",
    "The most interesting thing about artificial intelligence is",
]

all_hidden = []
all_gold = []

for prompt in prompts:
    tok_ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids'][0]
    
    # Teacher autogressive generation
    with torch.no_grad():
        t_ids = tok_ids.unsqueeze(0)
        for _ in range(20):  # more tokens per prompt
            out = teacher(t_ids)
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            t_ids = torch.cat([t_ids, nxt], -1)
    full_ids = t_ids[0]
    
    # Student forward — collect hidden states at teacher's token positions
    with torch.no_grad():
        s_out = student(t_ids, output_hidden_states=True)
    
    for t in range(len(full_ids)-1):
        h = s_out.hidden_states[-1][0, t, :].float().cpu()
        gold = full_ids[t+1].item()
        all_hidden.append(h)
        all_gold.append(gold)

n_pairs = len(all_gold)
n_classes = len(set(all_gold))
print(f"  Collected {n_pairs} pairs, {n_classes} unique gold tokens", flush=True)

# =====================================================================
# COMPRESS + TRAIN EIGENBUDDY
# =====================================================================
print("\nCompressing hidden states...", flush=True)
hs = torch.stack(all_hidden)
hs_norm = hs / (hs.norm(dim=1, keepdim=True) + 1e-9)

# Complex SVD for dimension reduction
Z = hs_norm.numpy() + 1j * np.zeros_like(hs_norm.numpy())
Zc = Z - Z.mean(axis=0, keepdims=True)
C = (Zc.conj().T @ Zc) / max(n_pairs-1, 1)
evals, evecs = np.linalg.eigh(C); evals = evals[::-1]; evecs = evecs[:, ::-1]
cum = np.cumsum(evals / evals.sum())
df = 1.0/((evals/evals.sum())**2).sum(); k95 = int(np.searchsorted(cum, 0.95)+1)
K_eb = max(16, min(k95, n_pairs//8, 128))
V = evecs[:, :K_eb]
Z_proj = Zc @ V
print(f"  Df={df:.1f} K95={k95} using K={K_eb}", flush=True)

# Remap targets
unique_tokens = sorted(set(all_gold))
tok_to_idx = {t:i for i,t in enumerate(unique_tokens)}
targets = torch.tensor([tok_to_idx[t] for t in all_gold], dtype=torch.long)

# Split
split = int(n_pairs * 0.8)
inputs = []
for i in range(n_pairs):
    z = torch.complex(torch.tensor(Z_proj[i].real.astype(np.float32)),
                       torch.tensor(Z_proj[i].imag.astype(np.float32)))
    inputs.append(z.unsqueeze(0).unsqueeze(0))

train_in = inputs[:split]; train_tgt = targets[:split]
test_in = inputs[split:]; test_tgt = targets[split:]
print(f"  Train: {len(train_in)} Test: {len(test_in)} Classes: {n_classes}", flush=True)

# Train EigenBuddy
print("\nTraining...", flush=True)
model = EigenBuddyTokenizer(dim=K_eb, vocab_size=n_classes, eigen_layers=2,
                             eigen_heads=max(1, K_eb//4) if K_eb>=4 else 1)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
model = model.to(device)
batch_size = min(32, len(train_in))

for epoch in range(400):
    perm = torch.randperm(len(train_in))
    epoch_loss = 0.0; epoch_acc = 0.0; n_batches = 0
    for start in range(0, len(train_in), batch_size):
        end = min(start+batch_size, len(train_in)); idx = perm[start:end]
        b_in = torch.cat([train_in[i] for i in idx], dim=0).to(device)
        b_tgt = train_tgt[idx].to(device)
        logits, _ = model(b_in)
        loss = F.cross_entropy(logits, b_tgt)
        if torch.isnan(loss) or torch.isinf(loss): continue
        opt.zero_grad(); loss.backward(); opt.step()
        preds = logits.argmax(dim=-1)
        epoch_loss += loss.item(); epoch_acc += (preds==b_tgt).float().mean().item(); n_batches += 1
    if epoch % 60 == 0:
        print(f"  Epoch {epoch:>3}: loss={epoch_loss/max(n_batches,1):.4f} acc={epoch_acc/max(n_batches,1):.3f}", flush=True)

# Evaluate
model.eval()
train_correct = 0; test_correct = 0
with torch.no_grad():
    for i in range(len(train_in)):
        logits, _ = model(train_in[i].to(device))
        if logits.argmax().item() == train_tgt[i].item(): train_correct += 1
    for i in range(len(test_in)):
        logits, _ = model(test_in[i].to(device))
        if logits.argmax().item() == test_tgt[i].item(): test_correct += 1

train_acc = train_correct/max(len(train_tgt),1); test_acc = test_correct/max(len(test_tgt),1)
eb_params = sum(p.numel() for p in model.parameters())
lm_head_params = 151936 * 896

print(f"\n{'='*78}")
print(f"RESULTS")
print(f"{'='*78}")
print(f"  Pairs: {n_pairs}  Classes: {n_classes}  Dim: {K_eb}")
print(f"  Train acc: {train_acc:.3f}  Test acc: {test_acc:.3f}")
print(f"  EigenBuddy params: {eb_params:,}  lm_head: {lm_head_params:,}  ({lm_head_params/eb_params:.0f}x)")
