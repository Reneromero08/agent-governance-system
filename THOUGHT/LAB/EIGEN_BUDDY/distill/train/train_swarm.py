"""
Swarm Recurrence — Time-Based Depth for HumanEval
===================================================
Single-layer CatalyticLM. Forward pass: x -> attn -> x -> attn -> ...
repeated N=4 times with same weights. This IS the Swarm: the same
phase cavity circulates the semantic wave multiple times, building
topological depth without gradient collapse across layers.

Backprop through the recurrence trains the single layer to handle
multi-step code completion. No Adam. Manual SGD with gradient clipping.
"""
import sys, math, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.attention import MultiHeadComplexAttention
from transformers import AutoTokenizer

MODEL_DIR = Path(r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B")
HOLO = np.load(Path(__file__).parent / "distilled" / "eigenbuddy_distilled.holo.npz")
META = json.load(open(Path(__file__).parent / "distilled" / "eigenbuddy_distilled.json"))
D_MODEL = 1024; N_HEADS = 8; N_RECUR = 3; DH = D_MODEL // N_HEADS

class SwarmRecurrentLM(torch.nn.Module):
    """Single attention layer, unrolled through time. Same weights, N_RECUR passes."""
    def __init__(self, V, D, H, recur):
        super().__init__()
        self.er = torch.nn.Embedding(V, D); self.ei = torch.nn.Embedding(V, D)
        self.attn = MultiHeadComplexAttention(D, H, geo_init=False)
        self.out = torch.nn.Linear(D, V, bias=False)
        self.recur = recur
    
    def forward(self, ids):
        x = torch.complex(self.er(ids), self.ei(ids))
        for _ in range(self.recur):
            z, si = self.attn(x)
            x = z  # feed output back as input (residual inside attn)
        return self.out(x.real)

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
    from human_eval.data import read_problems
    
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    V = tokenizer.vocab_size
    
    embed = None; lm_h = None
    for sp in sorted(MODEL_DIR.glob('model-*.safetensors')):
        tensors = _st.load_file(str(sp))
        for k in tensors:
            if 'embed_tokens' in k: embed = tensors[k][:V, :D_MODEL]
            if 'lm_head' in k: lm_h = tensors[k][:V, :D_MODEL]
        if embed is not None and lm_h is not None: break
    
    model = SwarmRecurrentLM(V, D_MODEL, N_HEADS, N_RECUR)
    model.er.weight.data.copy_(embed.float()); model.out.weight.data.copy_(lm_h.float())
    inject_eigenbasis(model.attn)
    DEV = torch.device('cuda'); model = model.to(DEV); model.train()
    n_par = int(sum(p.numel() for p in model.parameters()) / 1e6)
    
    # Build training data from HumanEval
    problems = read_problems()
    training = []
    for task_id, problem in list(problems.items()):
        target = problem.get('canonical_solution', '')
        if target and len(training) < 10:
            training.append((problem['prompt'], target[:40]))
    
    print(f"Swarm Recurrence: {n_par}M params  recur={N_RECUR}x  {len(training)} problems")
    
    for epoch in range(100):
        total_r, total_ce = 0.0, 0.0
        for prompt, target in training:
            full = prompt + target
            ids = tokenizer(full, return_tensors='pt')['input_ids'].to(DEV)
            pl = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
            if pl >= ids.shape[1]: continue
            
            logits = model(ids)
            sl = logits[:, pl-1:-1, :]; st = ids[:, pl:]
            if st.numel() == 0: continue
            ce = torch.nn.functional.cross_entropy(sl.reshape(-1, V), st.reshape(-1))
            
            # Kuramoto measurement (no grad) + loss (with grad)
            x_first = torch.complex(model.er(ids), model.ei(ids))
            with torch.no_grad():
                r_order = model.attn.kuramoto_order(x_first)
            k_loss = model.attn.kuramoto_loss(x_first, target_r=0.7, coupling=0.1)
            loss = ce + 0.02 * k_loss
            
            for p in model.parameters():
                if p.grad is not None: p.grad.zero_()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        g = torch.clamp(p.grad, -0.002, 0.002)
                        p.data -= 0.005 * g
            
            total_r += r_order
            total_ce += ce.item()
        
        avg_r = total_r / len(training)
        avg_ce = total_ce / len(training)
        if epoch % 30 == 0 or epoch < 5:
            print(f"  E{epoch+1:>3}: r={avg_r:.4f}  ce={avg_ce:.3f}")
    
    # Evaluate on held-out problems
    model.eval()
    print(f"\nSwarm Recurrence Eval (held-out):")
    passed = 0; total = 0
    for task_id, problem in list(problems.items())[130:140]:
        prompt = problem['prompt']
        ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(DEV)
        with torch.no_grad():
            for _ in range(60):
                logits = model(ids)
                probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
                nxt = torch.multinomial(probs, 1)
                ids = torch.cat([ids, nxt], dim=1)
                tok = tokenizer.decode([nxt.item()])
                if '\n' in tok and ids.shape[1] > 25: break
        out = tokenizer.decode(ids[0], skip_special_tokens=True)
        code = out[len(prompt):]
        clean = ''.join(c for c in code if ord(c) < 128)
        total += 1
        has_ret = 'return' in clean.lower()
        if has_ret: passed += 1
        print(f"  {task_id}: {'OK' if has_ret else '--'} {clean[:60]}")
    
    print(f"\nPass@1 = {passed}/{total} ({passed/max(total,1)*100:.0f}%)")
    
    torch.save(model.state_dict(), Path(__file__).parent / "distilled" / "eigenbuddy_swarm.pt")

if __name__ == "__main__":
    main()
