"""
Eigen Buddy v2 — Multi-layer with Kuramoto + CE, scaled training
==================================================================
Proof-of-concept scaled: 2 attention layers, 64 training pairs,
temperature sampling, CurvatureModulator integration.
"""
import sys, math, json, hashlib, numpy as np, torch
from pathlib import Path
import safetensors.torch as st
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.attention import MultiHeadComplexAttention
from core.curvature import CurvatureModulator
from core.phase import PhaseAccumulator
from transformers import AutoTokenizer

MODEL_DIR = Path(r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B")
HOLO = np.load(Path(__file__).parent / "distilled" / "eigenbuddy_distilled.holo.npz")
META = json.load(open(Path(__file__).parent / "distilled" / "eigenbuddy_distilled.json"))
D_MODEL = 1024; N_HEADS = 8; N_LAYERS = 2; DH = D_MODEL // N_HEADS

TRAINING_DATA = [
    ("The capital of France is", " Paris"),
    ("The answer to life is", " forty two"),
    ("Hello, my name is", " Eigen"),
    ("The sky is", " blue"),
    ("One plus one equals", " two"),
    ("Water is", " wet"),
    ("Fire is", " hot"),
    ("The opposite of up is", " down"),
    ("The largest ocean is the", " Pacific"),
    ("The smallest planet is", " Mercury"),
    ("Dogs say", " woof"),
    ("Cats say", " meow"),
    ("The sun rises in the", " east"),
    ("The sun sets in the", " west"),
    ("Ice is", " cold"),
    ("The color of grass is", " green"),
    ("A group of lions is called a", " pride"),
    ("The king of the jungle is the", " lion"),
    ("The fastest land animal is the", " cheetah"),
    ("The largest land animal is the", " elephant"),
    ("Humans have", " two eyes"),
    ("A triangle has", " three sides"),
    ("A square has", " four sides"),
    ("The first month is", " January"),
    ("The last month is", " December"),
    ("Birds can", " fly"),
    ("Fish can", " swim"),
    ("The moon orbits the", " Earth"),
    ("The Earth orbits the", " Sun"),
    ("A year has", " twelve months"),
    ("A day has", " twenty four hours"),
    ("An hour has", " sixty minutes"),
]

class EigenBuddyLayer(torch.nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.attn = MultiHeadComplexAttention(d, heads, geo_init=False)
        self.curve = CurvatureModulator(d)
        self.phase = PhaseAccumulator(d)
    def forward(self, x, si):
        z, si_out = self.attn(x)
        z = self.curve(z, si_out)
        z = self.phase(z)
        return z, si_out

class CatalyticLM(torch.nn.Module):
    def __init__(self, V, D, H, L):
        super().__init__()
        self.embed_r = torch.nn.Embedding(V, D); self.embed_r.weight.data.copy_(embed.float())
        self.embed_i = torch.nn.Embedding(V, D)
        self.layers = torch.nn.ModuleList([EigenBuddyLayer(D, H) for _ in range(L)])
        self.out = torch.nn.Linear(D, V, bias=False); self.out.weight.data.copy_(lm_head.float())
    
    def forward(self, ids):
        xr = self.embed_r(ids); xi = self.embed_i(ids)
        x = torch.complex(xr, xi)
        si = None
        for layer in self.layers:
            x, si = layer(x, si)
        return self.out(x.real)

def inject_eigenbasis(model):
    k_gratings = []; v_gratings = []
    for key in HOLO.files:
        g = torch.tensor(HOLO[key])
        if g.shape[1] != D_MODEL: continue
        sv = META.get(key, {}).get('singular_values', [1.0] * g.shape[0])
        if 'k_proj' in key: k_gratings.append((g, sv))
        elif 'v_proj' in key: v_gratings.append((g, sv))
    
    for layer in model.layers:
        attn = layer.attn
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

def generate(model, tokenizer, prompt, max_tokens=20, temperature=0.8):
    ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(DEV)
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(ids)
            next_logits = logits[:, -1, :] / max(temperature, 0.1)
            probs = torch.softmax(next_logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
    out = tokenizer.decode(ids[0], skip_special_tokens=True)
    return ''.join(c for c in out if ord(c) < 128)

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
    
    model = CatalyticLM(V, D_MODEL, N_HEADS, N_LAYERS)
    inject_eigenbasis(model)
    global DEV; DEV = torch.device('cuda'); model = model.to(DEV); model.train()
    n_par = int(sum(p.numel() for p in model.parameters()) / 1e6)
    print(f"Eigen Buddy v2: {n_par}M params  layers={N_LAYERS}  d={D_MODEL}  heads={N_HEADS}")
    
    # Train on phrase pairs + code line completions
    training = TRAINING_DATA + [
        ("def add(a, b): return a + b", ""),
        ("def factorial(n):", " return 1 if n <= 1 else n * factorial(n-1)"),
        ("def fibonacci(n):", " return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"),
        ("def is_even(n):", " return n % 2 == 0"),
        ("def gcd(a, b):", " while b: a, b = b, a % b; return a"),
        ("def is_prime(n):", " if n < 2: return False"),
    ]
    
    print(f"  Training on {len(training)} pairs")
    
    for epoch in range(100):
        total_r, total_ce = 0.0, 0.0
        for prompt, target in training:
            full = prompt + target
            ids = tokenizer(full, return_tensors='pt')['input_ids'].to(DEV)
            prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(DEV)
            pl = min(prompt_ids.shape[1], ids.shape[1] - 1)
            
            x = torch.complex(model.embed_r(ids), model.embed_i(ids))
            z, _ = model.attn(x)
            logits = model.out(z.real)
            
            # Next-token CE
            shift_logits = logits[:, pl-1:-1, :]
            shift_labels = ids[:, pl:]
            if shift_labels.numel() == 0: continue
            ce_loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, V), shift_labels.reshape(-1))
            
            # Kuramoto coherence
            k_loss = model.attn.kuramoto_loss(x, target_r=0.8, coupling=0.2)
            loss = ce_loss + 0.05 * k_loss
            
            for p in model.parameters():
                if p.grad is not None: p.grad.zero_()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data -= 0.003 * torch.clamp(p.grad, -0.01, 0.01)
            
            with torch.no_grad():
                total_r += model.attn.kuramoto_order(x)
            total_ce += ce_loss.item()
        
        avg_r = total_r / len(training)
        avg_ce = total_ce / len(training)
        if epoch % 20 == 0 or epoch < 5:
            print(f"  E{epoch+1:>3}: r={avg_r:.4f}  ce={avg_ce:.3f}")
    
    model.eval()
    print(f"\nGeneration (temp=0.8):")
    test_prompts = [
        "The capital of France is", "The answer to life is",
        "Hello, my name is", "Fire is", "Dogs say",
        "def add(a, b): return a +", "def factorial(n):",
        "def fibonacci(n):", "def is_even(n):",
    ]
    for prompt in test_prompts:
        out = generate(model, tokenizer, prompt, max_tokens=15, temperature=0.8)
        print(f"  '{prompt}' -> '{out[len(prompt):].strip()}'")

if __name__ == "__main__":
    main()
