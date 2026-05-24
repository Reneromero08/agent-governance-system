"""
Eigen Buddy Code Training — CE + Kuramoto, manual SGD
========================================================
Single-layer attention, proven working config. Manual SGD with
gradient clipping. Trains on phrase pairs + short code completions.
No Adam. No AdamW. Phase coherence regularizes.
"""
import sys, math, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.attention import MultiHeadComplexAttention
from transformers import AutoTokenizer

MODEL_DIR = Path(r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B")
HOLO = np.load(Path(__file__).parent / "distilled" / "eigenbuddy_distilled.holo.npz")
META = json.load(open(Path(__file__).parent / "distilled" / "eigenbuddy_distilled.json"))
D_MODEL = 1024; N_HEADS = 8; DH = D_MODEL // N_HEADS

TRAINING = [
    ("The capital of France is", " Paris"),
    ("The answer to life is", " forty two"),
    ("Hello my name is", " Eigen"),
    ("The sky is", " blue"),
    ("Fire is", " hot"),
    ("Water is", " wet"),
    ("Ice is", " cold"),
    ("Dogs say", " woof"),
    ("Cats say", " meow"),
    ("One plus one equals", " two"),
    ("The opposite of up is", " down"),
    ("def add(a, b): return a + ", "b"),
    ("def multiply(x, y): return x * ", "y"),
    ("def is_even(n): return n % 2 ==", " 0"),
    ("def factorial(n): return 1 if n <= 1 else n * factorial(n - ", "1)"),
    ("def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-", "2)"),
    ("def gcd(a, b): while b: a, b = b, a % ", "b; return a"),
    ("def is_prime(n): if n < 2: return ", "False"),
    ("for i in range(", "10):"),
    ("if __name__ == '__main__", "':"),
]

class CatalyticLM(torch.nn.Module):
    def __init__(self, V, D, H):
        super().__init__()
        self.er = torch.nn.Embedding(V, D); self.ei = torch.nn.Embedding(V, D)
        self.attn = MultiHeadComplexAttention(D, H, geo_init=False)
        self.out = torch.nn.Linear(D, V, bias=False)
    def forward(self, ids):
        x = torch.complex(self.er(ids), self.ei(ids)); z, _ = self.attn(x)
        return self.out(z.real)

def inject_eigenbasis(attn):
    k_gr = []; v_gr = []
    for key in HOLO.files:
        g = torch.tensor(HOLO[key])
        if g.shape[1] != D_MODEL: continue
        sv = META.get(key, {}).get('singular_values', [1.0] * g.shape[0])
        if 'k_proj' in key: k_gr.append((g, sv))
        elif 'v_proj' in key: v_gr.append((g, sv))
    with torch.no_grad():
        for h in range(N_HEADS):
            s, e = h * DH, (h + 1) * DH
            ki = h % len(k_gr); vi = h % len(v_gr)
            kg, ks = k_gr[ki]; vg, vs = v_gr[vi]
            for pn in ['qr', 'qi', 'kr', 'ki', 'vr', 'vi']:
                w = getattr(attn, pn).weight
                gr, svs = (kg, ks) if 'v' not in pn else (vg, vs)
                em = gr[h % gr.shape[0], :].numpy()
                scale = math.sqrt(float(svs[h % len(svs)]) / max(sum(svs), 1e-12) + 1e-8)
                rot = em * np.exp(1j * 2 * math.pi * h / N_HEADS)
                w.data[s:e] = torch.tensor(rot.real.astype(np.float32) * scale).unsqueeze(0).expand(DH, -1)

def main():
    import safetensors.torch as _st
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    V = tokenizer.vocab_size
    
    embed = None; lm_h = None
    for sp in sorted(MODEL_DIR.glob('model-*.safetensors')):
        tensors = _st.load_file(str(sp))
        for k in tensors:
            if 'embed_tokens' in k: embed = tensors[k][:V, :D_MODEL]
            if 'lm_head' in k: lm_h = tensors[k][:V, :D_MODEL]
        if embed is not None and lm_h is not None:
            break
    
    model = CatalyticLM(V, D_MODEL, N_HEADS)
    model.er.weight.data.copy_(embed.float()); model.out.weight.data.copy_(lm_h.float())
    inject_eigenbasis(model.attn)
    DEV = torch.device('cuda'); model = model.to(DEV); model.train()
    n_par = int(sum(p.numel() for p in model.parameters()) / 1e6)
    print(f"Eigen Buddy Code: {n_par}M params  {len(TRAINING)} training pairs")
    
    for epoch in range(200):
        total_r, total_ce = 0.0, 0.0
        for prompt, target in TRAINING:
            full = prompt + target
            ids = tokenizer(full, return_tensors='pt')['input_ids'].to(DEV)
            pl = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
            if pl >= ids.shape[1]: continue
            
            x = torch.complex(model.er(ids), model.ei(ids)); z, _ = model.attn(x)
            logits = model.out(z.real)
            
            sl = logits[:, pl-1:-1, :]; st = ids[:, pl:]
            if st.numel() == 0: continue
            ce = torch.nn.functional.cross_entropy(sl.reshape(-1, V), st.reshape(-1))
            k = model.attn.kuramoto_loss(x, target_r=0.8, coupling=0.2)
            loss = ce + 0.05 * k
            
            for p in model.parameters():
                if p.grad is not None: p.grad.zero_()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        g = torch.clamp(p.grad, -0.01, 0.01)
                        p.data -= 0.005 * g
            
            total_r += model.attn.kuramoto_order(x)
            total_ce += ce.item()
        
        avg_r = total_r / len(TRAINING)
        avg_ce = total_ce / len(TRAINING)
        if epoch % 50 == 0 or epoch < 5:
            print(f"  E{epoch+1:>3}: r={avg_r:.4f}  ce={avg_ce:.3f}")
    
    model.eval()
    print(f"\nGeneration:")
    tests = [
        "The capital of France is", "def add(a, b): return a + ",
        "def factorial(n): return 1 if n <= 1 else n * factorial(n - ",
        "def is_even(n): return n % 2 ==", "for i in range(",
        "The answer to life is", "Hello my name is",
    ]
    for prompt in tests:
        ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(DEV)
        with torch.no_grad():
            for _ in range(10):
                logits = model(ids)
                probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
                ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=1)
        out = tokenizer.decode(ids[0], skip_special_tokens=True)
        clean = ''.join(c for c in out if ord(c) < 128)
        print(f"  {prompt} -> {clean[len(prompt):].strip()}")
    
    torch.save(model.state_dict(), Path(__file__).parent / "distilled" / "eigenbuddy_code.pt")
    print(f"\nSaved.")

if __name__ == "__main__":
    main()
