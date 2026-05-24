"""
LowRankPhaseAdapters + Warm-Tape — Immutable .holo, Trainable Adapters
=========================================================================
Freeze the 1.6MB SVD .holo eigenbasis (immutable substrate).
Inject Rank-64 LowRankPhaseAdapters (A @ B) into each attention head.
Only rotate adapter weights. Base geometry never degrades.

Warm-Tape: pre-compute teacher hidden states for all training prompts.
Cache on the zero-copy tape. Swarm reads warm_t*, no dynamic compute.
"""
import sys, math, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.attention import MultiHeadComplexAttention
from transformers import AutoTokenizer

MODEL_DIR = Path(r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B")
HOLO = np.load(Path(__file__).parent / "distilled" / "eigenbuddy_distilled.holo.npz")
META = json.load(open(Path(__file__).parent / "distilled" / "eigenbuddy_distilled.json"))
D_MODEL = 1024; N_HEADS = 8; N_RECUR = 2; DH = D_MODEL // N_HEADS
ADAPTER_RANK = 64

class LowRankAdapter(torch.nn.Module):
    """A (DH, R) @ B (R, D_MODEL) — only A gets rotated, B is static.
    Output matches weight head block shape (DH, D_MODEL)."""
    def __init__(self, head_dim, model_dim, rank):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(head_dim, rank) * 0.01)
        self.B = torch.nn.Parameter(torch.randn(rank, model_dim) * 0.01)
    
    def forward(self, w_base):
        return w_base + self.A @ self.B

class ImmutableCatalyticLM(torch.nn.Module):
    """Base .holo weights FROZEN. Only adapters trained."""
    def __init__(self, V, D, H, recur):
        super().__init__()
        self.er = torch.nn.Embedding(V, D); self.ei = torch.nn.Embedding(V, D)
        self.attn = MultiHeadComplexAttention(D, H, geo_init=False)
        self.out = torch.nn.Linear(D, V, bias=False)
        self.recur = recur
        
        # Low-rank adapters per head, per projection
        self.adapters = torch.nn.ModuleDict()
        for h in range(H):
            for pn in ['qr', 'qi', 'kr', 'ki']:
                self.adapters[f'{h}_{pn}'] = LowRankAdapter(DH, D, ADAPTER_RANK)
        
        # Freeze base weights
        for p in self.attn.parameters(): p.requires_grad = False
        for p in self.out.parameters(): p.requires_grad = False
        for p in self.er.parameters(): p.requires_grad = False
        for p in self.ei.parameters(): p.requires_grad = False
    
    def _apply_adapters(self):
        """Apply adapter corrections to attention weights (in-place). Returns backup."""
        backup = {}
        for h in range(N_HEADS):
            s, e = h * DH, (h + 1) * DH
            for pn in ['qr', 'qi', 'kr', 'ki']:
                key = f'{h}_{pn}'
                if key not in self.adapters: continue
                adapter = self.adapters[key]
                w = getattr(self.attn, pn).weight
                backup[f'{h}_{pn}'] = w.data[s:e].clone()
                # w_head = base + A @ B
                w.data[s:e] = backup[f'{h}_{pn}'] + adapter.A.data @ adapter.B.data
        return backup
    
    def _restore_weights(self, backup):
        for key, saved in backup.items():
            h, pn = key.split('_', 1)
            h = int(h)
            s, e = h * DH, (h + 1) * DH
            w = getattr(self.attn, pn).weight
            w.data[s:e] = saved
    
    def forward(self, ids):
        backup = self._apply_adapters()
        x = torch.complex(self.er(ids), self.ei(ids))
        for _ in range(self.recur):
            z, _ = self.attn(x)
            x = z
        result = self.out(x.real)
        self._restore_weights(backup)
        return result, x.real

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

def load_qwen_embeddings(V):
    import safetensors.torch as _st
    embed = None; lm_h = None
    for sp in sorted(MODEL_DIR.glob('model-*.safetensors')):
        tensors = _st.load_file(str(sp))
        for k in tensors:
            if 'embed_tokens' in k: embed = tensors[k][:V, :D_MODEL]
            if 'lm_head' in k: lm_h = tensors[k][:V, :D_MODEL]
        if embed is not None and lm_h is not None: break
    return embed, lm_h

def build_warm_tape(tokenizer, training, device):
    """Pre-compute ideal hidden states h* = W_out[target] for all examples.
    Store on warm tape as dict: example_idx -> list of h* vectors per position."""
    warm_tape = {}
    W_out = None
    for sp in sorted(MODEL_DIR.glob('model-*.safetensors')):
        import safetensors.torch as _st
        tensors = _st.load_file(str(sp))
        for k in tensors:
            if 'lm_head' in k: W_out = tensors[k][:tokenizer.vocab_size, :D_MODEL].float().to(device)
            break
        if W_out is not None: break
    
    for idx, (prompt, target) in enumerate(training):
        full = prompt + target
        ids = tokenizer(full, return_tensors='pt')['input_ids']
        pl = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
        target_ids = ids[0, pl:]
        # h* per position: row of W_out for each target token
        h_stars = W_out[target_ids.to(device)]  # (T, D)
        warm_tape[idx] = h_stars
    return warm_tape, W_out

def main():
    from human_eval.data import read_problems
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    V = tokenizer.vocab_size
    
    embed, lm_h = load_qwen_embeddings(V)
    model = ImmutableCatalyticLM(V, D_MODEL, N_HEADS, N_RECUR)
    model.er.weight.data.copy_(embed.float()); model.out.weight.data.copy_(lm_h.float())
    inject_eigenbasis(model.attn)
    
    problems = read_problems()
    training = []
    for task_id, problem in list(problems.items()):
        target = problem.get('canonical_solution', '')
        if target and len(training) < 25:
            training.append((problem['prompt'], target[:50]))
    
    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(DEV)
    
    # Build warm tape
    print(f"Building warm tape: {len(training)} examples...")
    warm_tape, _ = build_warm_tape(tokenizer, training, DEV)
    print(f"Warm tape ready. {sum(v.shape[0] for v in warm_tape.values())} target vectors cached.")
    
    DIPOLE = 46.2 * math.pi / 180.0
    cp = torch.zeros(N_HEADS, N_HEADS, device=DEV)
    for i in range(N_HEADS):
        for j in range(N_HEADS):
            if i != j: cp[i, j] = DIPOLE * (i - j) / N_HEADS
    
    n_par = int(sum(p.numel() for p in model.adapters.parameters()) / 1e3)
    n_total = int(sum(p.numel() for p in model.parameters()) / 1e6)
    print(f"LowRankAdapters: {n_par}K trainable  |  Total: {n_total}M params (base FROZEN)")
    
    for epoch in range(500):
        total_r, total_ce = 0.0, 0.0
        for idx, (prompt, target) in enumerate(training):
            with torch.no_grad():
                full = prompt + target
                ids = tokenizer(full, return_tensors='pt')['input_ids'].to(DEV)
                pl = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
                if pl >= ids.shape[1]: continue
                
                logits, h = model(ids)
                sl = logits[:, pl-1:-1, :]; st = ids[:, pl:]
                if st.numel() == 0: continue
                ce = torch.nn.functional.cross_entropy(sl.reshape(-1, V), st.reshape(-1)).item()
                
                # WARM TAPE: read pre-computed h* (zero compute)
                h_star = warm_tape[idx]  # (T, D)
                h_pos = h[:, pl-1:-1, :].squeeze(0)  # (T, D)
                
                # Cosine alignment
                cos_sim = (h_pos * h_star).sum(dim=-1) / (h_pos.norm(dim=-1) * h_star.norm(dim=-1) + 1e-8)
                alignment = cos_sim.mean().item()
                
                # Orthogonal correction direction
                proj_scale = (h_pos * h_star).sum(dim=-1, keepdim=True) / (h_pos.norm(dim=-1, keepdim=True)**2 + 1e-8)
                h_perp = h_star - proj_scale * h_pos
                h_perp_avg = h_perp.mean(dim=0)  # (D,)
                
                corr_strength = min(max(1.0 - alignment, 0.0), 0.5)
                
                # Kuramoto phases
                x_first = torch.complex(model.er(ids), model.ei(ids))
                qr = model.attn.qr(x_first.real) - model.attn.qi(x_first.imag)
                qi = model.attn.qr(x_first.imag) + model.attn.qi(x_first.real)
                B, S = qr.shape[:2]
                qr = qr.view(B, S, N_HEADS, DH); qi = qi.view(B, S, N_HEADS, DH)
                states = torch.complex(qr, qi).mean(dim=(1, 3)).squeeze(0)
                phases = states / (states.abs() + 1e-8)
                hp_arr = torch.angle(phases)
                
                # ROTATE ONLY ADAPTERS (base weights frozen)
                for h in range(N_HEADS):
                    h_perp_head = h_perp_avg[h*DH:(h+1)*DH]
                    hn = h_perp_head.norm()
                    if hn > 1e-8:
                        h_perp_head = h_perp_head / hn
                        ca = math.atan2(float(h_perp_head[0]), float(h_perp_head[1] if DH > 1 else 0.1)) * corr_strength
                    else:
                        ca = 0.0
                    
                    cf = 0.0
                    for j in range(N_HEADS):
                        if h != j:
                            cf += math.sin(hp_arr[j].item() - hp_arr[h].item() - cp[h, j].item()) / N_HEADS
                    
                    rotation = 0.01 * cf + 0.05 * ca
                    c_r = math.cos(rotation)
                    
                    # Rotate adapter A matrix only (B is static)
                    for pn in ['qr', 'qi', 'kr', 'ki']:
                        key = f'{h}_{pn}'
                        if key in model.adapters:
                            model.adapters[key].A.data *= c_r
                
                r_order = model.attn.kuramoto_order(x_first)
                total_r += r_order; total_ce += ce
        
        avg_r = total_r / len(training); avg_ce = total_ce / len(training)
        if epoch % 100 == 0 or epoch < 5:
            print(f"  E{epoch+1:>3}: r={avg_r:.4f}  ce={avg_ce:.4f}  align={alignment:.4f}")
    
    # Generate
    print(f"\nLowRankAdapter Generation (base frozen):")
    for task_id, problem in list(problems.items())[140:148]:
        ids = tokenizer(problem['prompt'], return_tensors='pt')['input_ids'].to(DEV)
        with torch.no_grad():
            for _ in range(50):
                logits, _ = model(ids)
                probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
                nxt = torch.multinomial(probs, 1); ids = torch.cat([ids, nxt], dim=1)
                tok = tokenizer.decode([nxt.item()])
                if '\n' in tok and ids.shape[1] > 25: break
        out = tokenizer.decode(ids[0], skip_special_tokens=True)
        code = out[len(problem['prompt']):]
        clean = ''.join(c for c in code if ord(c) < 128)
        print(f"  {task_id}: {clean[:60]}")

if __name__ == "__main__":
    main()
