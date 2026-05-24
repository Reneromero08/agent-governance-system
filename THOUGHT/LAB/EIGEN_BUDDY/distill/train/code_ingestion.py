"""
Catalytic Code Ingestion — HDC phase-correction
=================================================
Streams Python code tokens through Eigen Buddy, maps each token to
an HDC complex phase vector, and applies one-shot geometric rotations
to align attention head phases with code token frequencies.

No backprop. No AdamW. Pure phase geometry.
"""
import sys, math, json, hashlib, ast, tokenize, io, numpy as np, torch
from pathlib import Path
import safetensors.torch as st
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.attention import MultiHeadComplexAttention
from transformers import AutoTokenizer

MODEL_DIR = Path(r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B")
HOLO = np.load(Path(__file__).parent / "distilled" / "eigenbuddy_distilled.holo.npz")
META = json.load(open(Path(__file__).parent / "distilled" / "eigenbuddy_distilled.json"))
D_MODEL = 1024; N_HEADS = 8; DH = D_MODEL // N_HEADS

def hdc_encode(token_str, dim=D_MODEL):
    """Encode any string as a complex HDC vector on S^1."""
    h = hashlib.sha256(token_str.encode()).digest()
    vec = np.zeros(dim, dtype=np.complex64)
    for i in range(min(len(h), dim)):
        angle = 2 * math.pi * h[i] / 256.0
        vec[i] = np.exp(1j * angle)
    return torch.tensor(vec)

def hdc_phase_correct(attn, token_vecs, learning_rate=0.01):
    """One-shot geometric rotation: align each head's phase to token HDC vectors.
    
    For each attention head h:
      1. Compute current phase: phase_h = angle(mean(Q_h output))
      2. Compute target phase: target_h = angle(mean(token HDC vectors))
      3. Phase error: delta = target_h - phase_h
      4. Rotate Q_h and K_h weights by delta
    """
    with torch.no_grad():
        # Compute per-head target phase from HDC token vectors
        target_phases = torch.angle(token_vecs.mean(dim=0))  # (D_MODEL,)
        target_phases_h = target_phases.view(N_HEADS, DH).mean(dim=1)  # (N_HEADS,)
        
        for h in range(N_HEADS):
            s, e = h * DH, (h + 1) * DH
            
            # Current phase from Q projection weights
            w_q = attn.qr.weight.data[s:e]  # (DH, D_MODEL) real part
            w_qi = attn.qi.weight.data[s:e]  # imag part
            q_complex = torch.complex(w_q, w_qi)  # (DH, D_MODEL)
            current_phase = torch.angle(q_complex.mean())
            
            # Phase error
            delta = target_phases_h[h] - current_phase
            # Wrap to (-pi, pi]
            delta = torch.atan2(torch.sin(delta), torch.cos(delta))
            
            # Apply geometric rotation to Q and K weight blocks
            rot = complex(math.cos(delta.item() * learning_rate), 
                         math.sin(delta.item() * learning_rate))
            
            # Rotate weights by multiplying by rot (complex rotation)
            for pn in ['qr', 'qi', 'kr', 'ki']:
                w = getattr(attn, pn).weight
                block = w.data[s:e]  # (DH, D_MODEL)
                # Apply rotation: for small delta, this is a simple mix
                cos_a = math.cos(delta.item() * learning_rate)
                sin_a = math.sin(delta.item() * learning_rate)
                if 'i' in pn:
                    w.data[s:e] = block * cos_a
                else:
                    w.data[s:e] = block * cos_a

class CatalyticLM(torch.nn.Module):
    def __init__(self, V, D, H):
        super().__init__()
        self.embed_r = torch.nn.Embedding(V, D); self.embed_r.weight.data.copy_(embed.float())
        self.embed_i = torch.nn.Embedding(V, D)
        self.attn = MultiHeadComplexAttention(D, H, geo_init=False)
        self.out = torch.nn.Linear(D, V, bias=False); self.out.weight.data.copy_(lm_head.float())
    
    def forward(self, ids):
        x = torch.complex(self.embed_r(ids), self.embed_i(ids))
        z, _ = self.attn(x)
        return self.out(z.real)

def inject_eigenbasis(attn):
    k_gratings = []; v_gratings = []
    for key in HOLO.files:
        g = torch.tensor(HOLO[key])
        if g.shape[1] != D_MODEL: continue
        sv = META.get(key, {}).get('singular_values', [1.0] * g.shape[0])
        if 'k_proj' in key: k_gratings.append((g, sv))
        elif 'v_proj' in key: v_gratings.append((g, sv))
    
    with torch.no_grad():
        for h in range(N_HEADS):
            s, e = h * DH, (h + 1) * DH
            k_idx = h % len(k_gratings); v_idx = h % len(v_gratings)
            k_gr, k_sv = k_gratings[k_idx]; v_gr, v_sv = v_gratings[v_idx]
            for pn in ['qr', 'qi', 'kr', 'ki', 'vr', 'vi']:
                w = getattr(attn, pn).weight
                gr, sv_vals = (k_gr, k_sv) if 'v' not in pn else (v_gr, v_sv)
                k_dim = gr.shape[0]; em = gr[h % k_dim, :].numpy()
                sv_scale = math.sqrt(float(sv_vals[h % len(sv_vals)]) / max(sum(sv_vals), 1e-12) + 1e-8)
                ha = 2 * math.pi * h / N_HEADS; rot = em * np.exp(1j * ha)
                val = rot.real.astype(np.float32) * sv_scale
                w.data[s:e] = torch.tensor(val).unsqueeze(0).expand(DH, -1)

# Python code corpus for HDC ingestion
PYTHON_CODE = """
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y

def factorial(n):
    if n <= 1: return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n-1) + fibonacci(n-2)

def is_even(n):
    return n % 2 == 0

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def sum_list(lst):
    total = 0
    for x in lst: total += x
    return total

def reverse_string(s):
    return s[::-1]

def count_vowels(s):
    return sum(1 for c in s if c.lower() in 'aeiou')

def gcd(a, b):
    while b: a, b = b, a % b
    return a

def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target: return mid
        elif arr[mid] < target: left = mid + 1
        else: right = mid - 1
    return -1

def merge_sort(arr):
    if len(arr) <= 1: return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []; i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]: result.append(left[i]); i += 1
        else: result.append(right[j]); j += 1
    result.extend(left[i:]); result.extend(right[j:])
    return result

def quicksort(arr):
    if len(arr) <= 1: return arr
    pivot = arr[0]
    less = [x for x in arr[1:] if x <= pivot]
    greater = [x for x in arr[1:] if x > pivot]
    return quicksort(less) + [pivot] + quicksort(greater)
"""

def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    V = tokenizer.vocab_size
    
    global embed, lm_head
    embed = None; lm_head = None
    for sp in sorted(MODEL_DIR.glob('model-*.safetensors')):
        t = st.load_file(str(sp))
        for k in t:
            if 'embed_tokens' in k: embed = t[k][:V, :D_MODEL]
            if 'lm_head' in k: lm_head = t[k][:V, :D_MODEL]
        if embed is not None and lm_head is not None: break
    
    model = CatalyticLM(V, D_MODEL, N_HEADS)
    inject_eigenbasis(model.attn)
    DEV = torch.device('cuda'); model = model.to(DEV); model.eval()
    n_par = int(sum(p.numel() for p in model.parameters()) / 1e6)
    
    print(f"Catalytic Code Ingestion: {n_par}M params")
    
    # Tokenize Python code
    code_tokens = tokenizer.encode(PYTHON_CODE)
    print(f"Code tokens: {len(code_tokens)}")
    
    # HDC ingestion: for each token, encode as HDC vector and phase-correct
    total_delta = 0.0
    batch_size = 32
    for i in range(0, min(len(code_tokens) - 1, 512), batch_size):
        batch = code_tokens[i:i+batch_size]
        ids = torch.tensor([batch], device=DEV)
        
        # Forward pass to get attention output
        x = torch.complex(model.embed_r(ids), model.embed_i(ids))
        
        # HDC encode each token and phase-correct
        token_strs = [tokenizer.decode([t]) for t in batch]
        hdc_vecs = torch.stack([hdc_encode(ts, D_MODEL) for ts in token_strs]).to(DEV)
        target_phase = torch.angle(hdc_vecs.mean(dim=0))
        
        # Measure current head phases and compute correction
        hdc_phase_correct(model.attn, hdc_vecs, learning_rate=0.05)
        
        # Forward again to measure change
        r = model.attn.kuramoto_order(x)
        
        if i == 0:
            print(f"  Initial Kuramoto r: {r:.4f}")
    
    # Final measurement
    x = torch.complex(model.embed_r(torch.tensor([code_tokens[:128]], device=DEV)), 
                       model.embed_i(torch.tensor([code_tokens[:128]], device=DEV)))
    r_final = model.attn.kuramoto_order(x)
    print(f"  Final Kuramoto r: {r_final:.4f}")
    
    # Generate code completion test
    print(f"\nCode completion test:")
    prompts = [
        "def add(a, b):",
        "def fibonacci(n):",
        "def is_prime(n):",
        "def factorial(n):",
    ]
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(DEV)
        with torch.no_grad():
            for _ in range(15):
                logits = model(ids)
                probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
                nxt = torch.multinomial(probs, 1)
                ids = torch.cat([ids, nxt], dim=1)
        out = tokenizer.decode(ids[0], skip_special_tokens=True)
        clean = ''.join(c for c in out if ord(c) < 128)
        print(f"  {prompt}")
        print(f"  -> {clean[len(prompt):][:80]}")
    
    # Save HDC-ingested model
    out_path = Path(__file__).parent / "distilled" / "eigenbuddy_hdc_code.pt"
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved HDC-ingested model: {out_path}")

if __name__ == "__main__":
    main()
