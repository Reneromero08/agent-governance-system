"""Fractal Phase Cavity — SVD + Fractal reorder + Phase Cavity sieve.

Multi-scale Feistel and QR orthogonalization belong in the RUNTIME attention
forward pass (dynamic tape operations), not in static weight compression.
They destroyed the SVD optimality (cosine sim dropped from 0.97 to 0.27).

What transfers cleanly:
  - Fractal SPN bit-reversed reordering (KAM-stable eigenmode ordering)
  - Phase Cavity eigenmode sieve (discard dispersion)
  - HoloLinear forwarding
"""
import struct, json, mmap, os, math, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

REPO = Path(r'D:\CCC 2.0\AI\agent-governance-system')
MODEL_DIR = str(REPO / 'THOUGHT' / 'LAB' / 'CAT_CAS' / '16_catalytic_27b_inference' / 'gemini_update' / 'qwen_0.5b')
MODEL_FILE = MODEL_DIR + '/model.safetensors'
K_COMPRESS = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# Fractal SPN (lib.rs)
# =====================================================================
def fractal_index(i, max_bits):
    rev = 0; n = i
    for _ in range(max_bits): rev = (rev << 1) | (n & 1); n >>= 1
    return rev

def fractal_reorder(S, U, Vh):
    k = len(S)
    max_bits = max(1, int(math.log2(k)) + 1)
    order = sorted([(fractal_index(i, max_bits), i) for i in range(k) if fractal_index(i, max_bits) < k])
    idx = [i for _, i in order]
    return S[idx], U[:, idx], Vh[idx, :]

# =====================================================================
# Phase Cavity (20.10)
# =====================================================================
def cosine_sim(Wo, Wr): 
    X = torch.randn(20, Wo.shape[1]); Yo = Wo.float() @ X.T; Yr = Wr.float() @ X.T
    d = (Yo * Yr).sum(dim=0)
    return (d / (Yo.norm(dim=0) * Yr.norm(dim=0) + 1e-9)).mean().item()

def phase_cavity_sieve(U, S, Vh, W_orig):
    k = len(S); kept = list(range(k))
    for i in range(k - 1, -1, -1):
        keep = [j for j in kept if j != i]
        if not keep: continue
        Wt = (U[:, keep] * S[keep].unsqueeze(0)) @ Vh[keep, :]
        if cosine_sim(W_orig, Wt) > 0.99: kept.remove(i)
    return sorted(kept), [i for i in range(k) if i not in kept]

# =====================================================================
# Load and compress
# =====================================================================
def load_weight(nm, tensors, mm, do):
    info = tensors[nm]; s, e = info["data_offsets"]; dt = info.get("dtype", "F32")
    raw = mm[do+s:do+e]
    if dt == "BF16":
        bf = np.frombuffer(raw, dtype=np.uint16); bf = bf.astype(np.uint32) << 16
        return torch.tensor(bf.view(np.float32).reshape(info["shape"]).copy())
    return torch.tensor(np.frombuffer(raw, dtype=np.float32).reshape(info["shape"]).copy())

print("=" * 78)
print("FRACTAL PHASE CAVITY — SVD + Fractal Reorder + Cavity Sieve")
print("=" * 78)

fd = os.open(MODEL_FILE, os.O_RDONLY | os.O_BINARY)
mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
hdr = json.loads(mm[8:8+struct.unpack("<Q", mm[:8])[0]].decode('utf-8'))
do = 8 + struct.unpack("<Q", mm[:8])[0]
tensors = hdr

matrices = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
N_LAYERS = 24
all_layers = {}
total_k = total_kept = total_disc = 0

for li in range(N_LAYERS):
    layer = {}
    for mn in matrices:
        nm = f"model.layers.{li}.self_attn.{mn}.weight"
        if nm not in tensors: continue
        W = load_weight(nm, tensors, mm, do)
        
        # SVD
        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        k = min(K_COMPRESS, U.shape[1])
        U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
        
        # Fractal reorder (KAM-stable)
        S, U, Vh = fractal_reorder(S, U, Vh)
        
        # Phase Cavity sieve
        kept, disc = phase_cavity_sieve(U, S, Vh, W)
        
        U_k, S_k, Vh_k = U[:, kept], S[kept], Vh[kept, :]
        W_cav = (U_k * S_k.unsqueeze(0)) @ Vh_k
        sim = cosine_sim(W, W_cav)
        
        layer[mn] = {'U': U_k, 'S': S_k, 'Vh': Vh_k, 'kept': len(kept), 
                      'disc': len(disc), 'sim': sim, 'k': k}
        total_k += k; total_kept += len(kept); total_disc += len(disc)
    
    all_layers[li] = layer
    q = layer.get('q_proj', {}); kp = layer.get('k_proj', {})
    v = layer.get('v_proj', {}); o = layer.get('o_proj', {})
    print(f"  L{li:>2}: Q={q.get('kept',0)}/128 sim={q.get('sim',0):.3f} "
          f"K={kp.get('kept',0)}/128 sim={kp.get('sim',0):.3f} "
          f"V={v.get('kept',0)}/128 sim={v.get('sim',0):.3f} "
          f"O={o.get('kept',0)}/128 sim={o.get('sim',0):.3f}", flush=True)

mm.close(); os.close(fd)

print(f"\n  Total: {total_kept}/{total_k} modes ({total_kept/max(total_k,1)*100:.0f}%)")
print(f"  Discarded: {total_disc} dispersion modes")
print(f"  Additional compression vs K=128: {total_k/total_kept:.1f}x")

# =====================================================================
# Build model and run inference
# =====================================================================
print(f"\n{'='*78}")
print("Building cavitated HoloLinear model")
print("=" * 78)

class CavitatedHoloLinear(nn.Module):
    def __init__(self, U, S, Vh):
        super().__init__()
        self.U = nn.Parameter(U, requires_grad=False)
        self.SVh = nn.Parameter((S.unsqueeze(1) * Vh), requires_grad=False)
    def forward(self, x):
        return torch.matmul(torch.matmul(x, self.SVh.t()), self.U.t())

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)

with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)
model.to_empty(device=device)

for li, layer in all_layers.items():
    attn = model.model.layers[li].self_attn
    for mn in matrices:
        if mn not in layer: continue
        d = layer[mn]
        holo = CavitatedHoloLinear(
            d['U'].to(device, dtype=torch.bfloat16),
            d['S'].to(device, dtype=torch.bfloat16),
            d['Vh'].to(device, dtype=torch.bfloat16)
        )
        setattr(attn, mn, holo)

original = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16)
orig_p = dict(original.named_parameters())
orig_b = dict(original.named_buffers())
for nm, p in model.named_parameters():
    if p.device.type == "meta":
        p.data = orig_p[nm].data.to(device) if nm in orig_p else torch.zeros(p.shape, dtype=torch.bfloat16, device=device)
for nm, b in model.named_buffers():
    if b.device.type == "meta" and nm in orig_b:
        b.data = orig_b[nm].data.to(device)
del original, orig_p, orig_b; torch.cuda.empty_cache()
model.eval()

# =====================================================================
# Inference
# =====================================================================
print(f"\n{'='*78}")
print("CAVITATED INFERENCE")
print("=" * 78)

# Truth Vector C
print("Extracting C...", flush=True)
it = tokenizer("The sky is blue.", return_tensors="pt").to(device)
ip = tokenizer("The sky is made of green cheese.", return_tensors="pt").to(device)
with torch.no_grad():
    ht = model(**it, output_hidden_states=True).hidden_states[-1][:, -1, :]
    hf = model(**ip, output_hidden_states=True).hidden_states[-1][:, -1, :]
Cv = ht - hf; Cv = Cv / (Cv.norm() + 1e-9)
C = torch.outer(Cv.squeeze(), Cv.squeeze())
print(f"  C norm: {C.norm():.4f}", flush=True)

# Generate
prompt = "The most interesting thing about artificial intelligence is"
ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
tokens = []
print(f"\nPrompt: {prompt}", flush=True)

for step in range(15):
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    logits = out.logits[:, -1, :]; h = out.hidden_states[-1][:, -1, :].squeeze()
    hn = h / (h.norm() + 1e-9)
    R = torch.trace(torch.matmul(torch.outer(hn, hn).float(), C.float())).item()
    T = 1.0 / (R * 100 + 0.5) if not math.isnan(R) else 2.0
    
    safe = torch.nan_to_num(logits / T, nan=0., posinf=10., neginf=-10.)
    p = torch.softmax(safe.float(), -1); p = torch.nan_to_num(p, 0.)
    p = p / p.sum(-1, keepdim=True)
    nxt = torch.multinomial(p, 1); tokens.append(nxt.item())
    
    try:
        w = tokenizer.decode([nxt.item()]).encode('ascii', errors='replace').decode('ascii')
    except: w = '?'
    print(f"  {step+1:>2} R={R:.4f} T={T:.3f} | {w}", flush=True)
    ids = torch.cat([ids, nxt], -1)

print(f"\n{tokenizer.decode(tokens)}")
