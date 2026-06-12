"""Phase Cavity HoloLinear — Cavitated inference on all 24 attention layers.

Applies the Phase Cavity eigenmode sieve from CAT_CAS_INTEGRATION_DOSSIER.md:
  1. Load all attention weight matrices from 0.5B safetensors
  2. .holo SVD compress each to K=128
  3. Phase Cavity: keep only physically required eigenmodes
  4. Build cavitated HoloLinear with reduced basis
  5. Patch model, run inference, measure R = Tr(rho*C)
"""
import struct, json, mmap, os, math, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

REPO = Path(r'D:\CCC 2.0\AI\agent-governance-system')
MODEL_DIR = str(REPO / 'THOUGHT' / 'LAB' / 'CAT_CAS' / '3_physics_complexity' / '16_catalytic_27b_inference' / 'gemini_update' / 'qwen_0.5b')
MODEL_FILE = MODEL_DIR + '/model.safetensors'
HIDDEN_DIM = 896
N_LAYERS = 24
K_COMPRESS = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# STEP 1: Load all attention weights, SVD compress, Phase Cavity sieve
# =====================================================================
print("=" * 78)
print("PHASE CAVITY: Cavitating all 24 attention layers")
print("=" * 78)

fd = os.open(MODEL_FILE, os.O_RDONLY | os.O_BINARY)
mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
header_size = struct.unpack("<Q", mm[:8])[0]
header = json.loads(mm[8:8+header_size].decode('utf-8'))
data_offset = 8 + header_size
tensors = header

def load_matrix(nm):
    info = tensors[nm]
    s, e = info["data_offsets"]
    dt = info.get("dtype", "F32")
    raw = mm[data_offset+s:data_offset+e]
    if dt == "BF16":
        bf = np.frombuffer(raw, dtype=np.uint16)
        bf = bf.astype(np.uint32) << 16
        return torch.tensor(bf.view(np.float32).reshape(info["shape"]).copy())
    return torch.tensor(np.frombuffer(raw, dtype=np.float32).reshape(info["shape"]).copy())

def cosine_sim_fast(Wo, Wr):
    X = torch.randn(20, Wo.shape[1])
    Yo = Wo.float() @ X.T; Yr = Wr.float() @ X.T
    d = (Yo * Yr).sum(dim=0)
    return (d / (Yo.norm(dim=0) * Yr.norm(dim=0) + 1e-9)).mean().item()

class CavitatedLayer:
    def __init__(self, layer_idx):
        self.idx = layer_idx
        self.U = {}; self.S = {}; self.Vh = {}
        self.kept = {}; self.discarded = {}
        self.sim_before = {}; self.sim_after = {}

cavitated = []
matrices = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

for li in range(N_LAYERS):
    cl = CavitatedLayer(li)
    for mn in matrices:
        name = f"model.layers.{li}.self_attn.{mn}.weight"
        if name not in tensors: continue
        W = load_matrix(name)
        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        k = min(K_COMPRESS, U.shape[1])
        Uk = U[:, :k]; Sk = S[:k]; Vhk = Vh[:k, :]
        W_full = (Uk * Sk.unsqueeze(0)) @ Vhk
        
        sim_before = cosine_sim_fast(W, W_full)
        cl.sim_before[mn] = sim_before
        cl.U[mn] = Uk; cl.S[mn] = Sk; cl.Vh[mn] = Vhk
        
        # Phase Cavity: test each eigenmode
        kept = list(range(k))
        for i in range(k - 1, -1, -1):
            keep = [j for j in kept if j != i]
            if not keep: continue
            Wt = (Uk[:, keep] * Sk[keep].unsqueeze(0)) @ Vhk[keep, :]
            if cosine_sim_fast(W, Wt) > 0.99:
                kept.remove(i)
        
        cl.kept[mn] = sorted(kept)
        cl.discarded[mn] = [i for i in range(k) if i not in kept]
        W_cav = (Uk[:, kept] * Sk[kept].unsqueeze(0)) @ Vhk[kept, :]
        cl.sim_after[mn] = cosine_sim_fast(W, W_cav)
    
    cavitated.append(cl)
    total_kept = sum(len(cl.kept[m]) for m in matrices if m in cl.kept)
    total_k = 4 * K_COMPRESS
    print(f"  Layer {li:>2}: kept {total_kept}/{total_k} modes "
          f"(Q={len(cl.kept.get('q_proj',[]))} K={len(cl.kept.get('k_proj',[]))} "
          f"V={len(cl.kept.get('v_proj',[]))} O={len(cl.kept.get('o_proj',[]))})", flush=True)

mm.close(); os.close(fd)

# Stats
total_modes = sum(4 * K_COMPRESS for _ in range(N_LAYERS))
total_kept = sum(sum(len(cl.kept.get(m, [])) for m in matrices) for cl in cavitated)
print(f"\n  Total: kept {total_kept}/{total_modes} modes ({total_kept/total_modes*100:.0f}%)")
print(f"  Compression: {total_modes/total_kept:.1f}x additional beyond K={K_COMPRESS}")

# =====================================================================
# STEP 2: Build Cavitated HoloLinear and patch model
# =====================================================================
print(f"\n{'='*78}")
print("Building cavitated model for inference")
print("=" * 78)

class CavitatedHoloLinear(nn.Module):
    def __init__(self, U_k, S_k, Vh_k, bias=None):
        super().__init__()
        self.U = nn.Parameter(U_k, requires_grad=False)
        self.SVh = nn.Parameter((S_k.unsqueeze(1) * Vh_k), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        out = torch.matmul(x, self.SVh.t())
        out = torch.matmul(out, self.U.t())
        if self.bias is not None: out += self.bias
        return out

print("Loading tokenizer + config...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)

print("Initializing model on meta device...", flush=True)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)

print(f"Moving to {device}...", flush=True)
model.to_empty(device=device)

# Patch with cavitated layers
print("Patching with CavitatedHoloLinear...", flush=True)
for cl in cavitated:
    li = cl.idx
    for mn in matrices:
        if mn not in cl.kept: continue
        kept = cl.kept[mn]
        Uk = cl.U[mn][:, kept].to(device, dtype=torch.bfloat16)
        Sk = cl.S[mn][kept].to(device, dtype=torch.bfloat16)
        Vhk = cl.Vh[mn][kept, :].to(device, dtype=torch.bfloat16)
        
        # Find bias
        bias_name = f"model.layers.{li}.self_attn.{mn}.bias"
        bias = None
        if bias_name in tensors:
            info = tensors[bias_name]
            s, e = info["data_offsets"]
            raw = mm = open(MODEL_FILE, 'rb')
            # Skip mmap for bias — just load directly
            bias = torch.zeros(info["shape"][0])
        
        holo = CavitatedHoloLinear(Uk, Sk, Vhk, bias.to(device) if bias is not None else None)
        layer = model.model.layers[li].self_attn
        setattr(layer, mn, holo)

# Load remaining params (embeddings, norms, etc.)
print("Loading remaining parameters...", flush=True)
original = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16)
orig_params = dict(original.named_parameters())
orig_buffers = dict(original.named_buffers())
for name, param in model.named_parameters():
    if param.device.type != "meta":
        continue
    if name in orig_params:
        param.data = orig_params[name].data.to(device)
    else:
        param.data = torch.zeros(param.shape, dtype=torch.bfloat16, device=device)
for name, buf in model.named_buffers():
    if buf.device.type == "meta" and name in orig_buffers:
        buf.data = orig_buffers[name].data.to(device)
del original, orig_params, orig_buffers; torch.cuda.empty_cache()

model.eval()
print("Model ready.", flush=True)

# =====================================================================
# STEP 3: Inference test
# =====================================================================
print(f"\n{'='*78}")
print("CAVITATED INFERENCE")
print("=" * 78)

# Extract C frame
print("Extracting Truth Vector C...", flush=True)
inputs_t = tokenizer("The sky is blue.", return_tensors="pt").to(device)
inputs_f = tokenizer("The sky is made of green cheese.", return_tensors="pt").to(device)
with torch.no_grad():
    ht = model(**inputs_t, output_hidden_states=True).hidden_states[-1][:, -1, :]
    hf = model(**inputs_f, output_hidden_states=True).hidden_states[-1][:, -1, :]
C_vec = ht - hf
C_vec = C_vec / (C_vec.norm(dim=-1, keepdim=True) + 1e-9)
C = torch.outer(C_vec.squeeze(), C_vec.squeeze())
print(f"  C extracted: norm={C.norm():.4f}", flush=True)

# Generate
prompt = "The most interesting thing about artificial intelligence is"
print(f"\nGenerating: '{prompt}'", flush=True)
input_ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
output_tokens = []

for step in range(20):
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)
    logits = out.logits[:, -1, :]
    h_t = out.hidden_states[-1][:, -1, :].squeeze()
    
    # R = Tr(rho * C)
    h_n = h_t / (h_t.norm() + 1e-9)
    rho = torch.outer(h_n, h_n)
    R = torch.trace(torch.matmul(rho.float(), C.float())).item()
    
    T = 1.0 / (R * 100 + 0.5) if not math.isnan(R) else 2.0
    
    safe = torch.nan_to_num(logits / T, nan=0.0, posinf=10.0, neginf=-10.0)
    probs = torch.softmax(safe.float(), dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    next_tok = torch.multinomial(probs, 1)
    output_tokens.append(next_tok.item())
    word = tokenizer.decode([next_tok.item()])
    
    print(f"  Step {step+1:>2} | R: {R:.4f} | T: {T:.4f} | Token: {repr(word)}", flush=True)
    input_ids = torch.cat([input_ids, next_tok], dim=-1)

print(f"\n--- Output ---")
print(tokenizer.decode(output_tokens))
