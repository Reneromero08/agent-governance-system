"""
Superradiant Adapter Training — Full Integration
==================================================
Physical laws from sandbox/torus_proof.py integrated into training loop.

Per-step sequence:
  1. Warm-tape: retrieve h* (cached teacher) and h (student forward pass)
  2. Error: Δh = h* - h
  3. Rank-1 eigenmode: take dominant PCA component of Δh
  4. Hebbian: ΔA = η * (h_perp ⊗ x*) 
  5. Torus: normalize A rows to |z|=1
  6. Kuramoto: 46.2deg carrier + coupling synthesis

Base .holo FROZEN. Only LowRankPhaseAdapters updated. No autograd.
"""
import sys, math, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.attention import MultiHeadComplexAttention
from transformers import AutoTokenizer

MODEL_DIR = Path(r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B")
HOLO = np.load(Path(__file__).parent.parent / "distilled" / "eigenbuddy_distilled.holo.npz")
META = json.load(open(Path(__file__).parent.parent / "distilled" / "eigenbuddy_distilled.json"))
D_MODEL = 1024; N_HEADS = 8; N_RECUR = 1; DH = D_MODEL // N_HEADS
ADAPTER_RANK = 64; DIPOLE_RAD = 46.2 * math.pi / 180.0
ETA = 0.05  # Hebbian learning rate

class LowRankAdapter(torch.nn.Module):
    """A (DH, R) @ B (R, D_MODEL). A is trainable, B is static.
    Effective weight for head h, proj pn = base_weight[h] + A @ B."""
    def __init__(self, dh, d_model, rank):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(dh, rank) * 0.01)
        self.B = torch.nn.Parameter(torch.randn(rank, d_model) * 0.01)
        self.B.requires_grad = False  # B is static

class ImmutableCatalyticLM(torch.nn.Module):
    def __init__(self, V, D, H, recur):
        super().__init__()
        self.er = torch.nn.Embedding(V, D); self.ei = torch.nn.Embedding(V, D)
        self.attn = MultiHeadComplexAttention(D, H, geo_init=False)
        self.out = torch.nn.Linear(D, V, bias=False)
        self.recur = recur
        self.H = H; self.DH = D // H
        
        self.adapters = torch.nn.ModuleDict()
        for h in range(H):
            for pn in ['qr', 'qi', 'kr', 'ki']:
                self.adapters[f'{h}_{pn}'] = LowRankAdapter(self.DH, D, ADAPTER_RANK)
        
        for p in self.attn.parameters(): p.requires_grad = False
        for p in self.out.parameters(): p.requires_grad = False
        for p in self.er.parameters(): p.requires_grad = False
        for p in self.ei.parameters(): p.requires_grad = False
    
    def _apply_adapters(self):
        backup = {}
        for h in range(self.H):
            s, e = h * self.DH, (h + 1) * self.DH
            for pn in ['qr', 'qi', 'kr', 'ki']:
                key = f'{h}_{pn}'
                if key not in self.adapters: continue
                w = getattr(self.attn, pn).weight
                backup[key] = w.data[s:e].clone()
                adapter = self.adapters[key]
                w.data[s:e] = backup[key] + adapter.A.data @ adapter.B.data
        return backup
    
    def _restore_weights(self, backup):
        for key, saved in backup.items():
            h = int(key.split('_')[0]); pn = '_'.join(key.split('_')[1:])
            s, e = h * self.DH, (h + 1) * self.DH
            w = getattr(self.attn, pn).weight
            w.data[s:e] = saved
    
    def forward(self, ids):
        backup = self._apply_adapters()
        x = torch.complex(self.er(ids), self.ei(ids))
        for _ in range(self.recur):
            z, _ = self.attn(x); x = z
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

def rank1_eigenmode(delta_h):
    """Extract dominant rank-1 component of error vector via outer product PCA.
    delta_h: (D_MODEL,) — u1 = delta_h / |delta_h|, h_perp = (dot * u1) * u1
    Returns (h_perp, u1)."""
    norm = delta_h.norm()
    if norm < 1e-8: return torch.zeros_like(delta_h), torch.zeros_like(delta_h)
    u1 = delta_h / norm
    # Orthogonal projection: h_perp is the component of delta_h along u1
    # This is the dominant eigenmode — the rank-1 approximation
    dot = torch.dot(delta_h, u1)
    h_perp = dot * u1
    return h_perp, u1

def hebbian_update(adapter_A, h_perp_head, input_head, r_order, eta=ETA):
    """ΔA = (η / (R + ε)) * outer(h_perp, input). Cybernetic gate scales update.
    Low R → massive amplification. High R → gentle nudge."""
    scale = eta / (r_order + 0.01)
    delta = scale * torch.outer(h_perp_head, input_head)
    adapter_A.data += delta

def torus_normalize(adapter_A):
    """Project each row of A to unit magnitude. Preserves direction."""
    row_norms = adapter_A.data.norm(dim=1, keepdim=True).clamp(min=1e-12)
    adapter_A.data /= row_norms

def kuramoto_synthesis(attn, x, coupling_matrix):
    """46.2deg carrier + Kuramoto coupling. Returns coupling forces per head."""
    with torch.no_grad():
        B, S, D = x.shape
        qr = attn.qr(x.real) - attn.qi(x.imag)
        qi = attn.qr(x.imag) + attn.qi(x.real)
        qr = qr.view(B, S, N_HEADS, DH); qi = qi.view(B, S, N_HEADS, DH)
        states = torch.complex(qr, qi).mean(dim=(1, 3)).squeeze(0)
        phases = states / (states.abs() + 1e-8)
        hp_arr = torch.angle(phases)
        
        coupling = torch.zeros(N_HEADS)
        for i in range(N_HEADS):
            for j in range(N_HEADS):
                if i != j:
                    coupling[i] += math.sin(hp_arr[j].item() - hp_arr[i].item() - coupling_matrix[i,j].item())
        coupling /= N_HEADS
        return coupling, hp_arr

def main():
    from human_eval.data import read_problems
    import safetensors.torch as _st
    
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    V = tokenizer.vocab_size
    
    embed = None; lm_h = None
    for sp in sorted(MODEL_DIR.glob('model-*.safetensors')):
        tensors = _st.load_file(str(sp))
        for k in tensors:
            if 'embed_tokens' in k: embed = tensors[k][:V, :D_MODEL]
            if 'lm_head' in k: lm_h = tensors[k][:V, :D_MODEL]
        if embed is not None and lm_h is not None: break
    
    model = ImmutableCatalyticLM(V, D_MODEL, N_HEADS, N_RECUR)
    model.er.weight.data.copy_(embed.float()); model.out.weight.data.copy_(lm_h.float())
    inject_eigenbasis(model.attn)
    
    problems = read_problems()
    training = []
    for tid, p in list(problems.items()):
        t = p.get('canonical_solution', '')
        if t and len(training) < 5:
            training.append((p['prompt'], t[:30]))
    
    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(DEV)
    
    # Warm tape
    print(f"Building warm tape ({len(training)} examples)...")
    warm_tape = {}
    W_out = model.out.weight.data
    for idx, (prompt, target) in enumerate(training):
        full = prompt + target
        ids = tokenizer(full, return_tensors='pt')['input_ids']
        pl = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
        target_ids = ids[0, pl:]
        warm_tape[idx] = W_out[target_ids.to(DEV)]
    print(f"Warm tape ready.")
    
    cp = torch.zeros(N_HEADS, N_HEADS, device=DEV)
    for i in range(N_HEADS):
        for j in range(N_HEADS):
            if i != j: cp[i, j] = DIPOLE_RAD * (i - j) / (N_HEADS - 1)
    
    n_adapt = int(sum(p.numel() for p in model.adapters.parameters()) / 1e3)
    print(f"Adapters: {n_adapt}K  Base: FROZEN  Mode: FORWARD-ONLY  Examples: {len(training)}  Epochs: 200")
    
    for epoch in range(200):
        total_ce, total_r = 0.0, 0.0
        for idx, (prompt, target) in enumerate(training):
            with torch.no_grad():
                full = prompt + target
                ids = tokenizer(full, return_tensors='pt')['input_ids'].to(DEV)
                pl = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
                if pl >= ids.shape[1]: continue
                
                # FORWARD PASS
                logits, h = model(ids)
                
                # CE measurement
                sl = logits[:, pl-1:-1, :]; st = ids[:, pl:]
                if st.numel() == 0: continue
                ce = torch.nn.functional.cross_entropy(sl.reshape(-1, V), st.reshape(-1)).item()
                
                # WARM TAPE: retrieve h* for each target position
                h_star = warm_tape[idx]  # (T, D_MODEL)
                h_pos = h[:, pl-1:-1, :].squeeze(0)  # (T, D_MODEL)
                
                x_first = torch.complex(model.er(ids), model.ei(ids))
                r_order = model.attn.kuramoto_order(x_first)
                
                # Compute input states once per example
                x = torch.complex(model.er(ids), model.ei(ids))
                
                # Average h_perp and input across all target positions for batching
                avg_h_perp = torch.zeros(D_MODEL, device=DEV)
                avg_input = torch.zeros(D_MODEL, device=DEV)
                count = 0
                for t in range(min(h_star.shape[0], h_pos.shape[0])):
                    delta_h = h_star[t] - h_pos[t]
                    h_perp, _ = rank1_eigenmode(delta_h)
                    x_full = x[0, pl - 1 + t, :].real
                    avg_h_perp += h_perp
                    avg_input += x_full
                    count += 1
                
                if count == 0: continue
                avg_h_perp /= count; avg_input /= count
                
                # Apply Hebbian + Torus to each head's adapter (one update per example)
                for h in range(N_HEADS):
                    s, e = h * DH, (h + 1) * DH
                    h_perp_head = avg_h_perp[s:e]
                    for pn in ['qr', 'qi', 'kr', 'ki']:
                        key = f'{h}_{pn}'
                        if key not in model.adapters: continue
                        adapter = model.adapters[key]
                        projected = adapter.B.data @ avg_input  # (RANK,)
                        hebbian_update(adapter.A, h_perp_head, projected, r_order)
                        torus_normalize(adapter.A)
                
                # Kuramoto synthesis
                x_first = torch.complex(model.er(ids), model.ei(ids))
                coupling_forces, hp_arr = kuramoto_synthesis(model.attn, x_first, cp)
                
                # Apply coupling to adapter A matrices
                for h in range(N_HEADS):
                    rotation = coupling_forces[h].item() * 0.01
                    c_r = math.cos(rotation)
                    for pn in ['qr', 'qi', 'kr', 'ki']:
                        key = f'{h}_{pn}'
                if key in model.adapters:
                    model.adapters[key].A.data *= c_r
                    torus_normalize(model.adapters[key].A)
                
                r_order = model.attn.kuramoto_order(x_first)
                total_ce += ce; total_r += r_order
        
        avg_ce = total_ce / len(training); avg_r = total_r / len(training)
        if epoch % 100 == 0 or epoch < 5:
            print(f"  {epoch+1:>4}  {avg_ce:>10.4f}  {avg_r:>8.4f}  {'--':>8}  {'--':>10}")
    
    # Generate
    print(f"\nSuperradiant Generation (held-out):")
    for tid, p in list(problems.items())[140:148]:
        ids = tokenizer(p['prompt'], return_tensors='pt')['input_ids'].to(DEV)
        with torch.no_grad():
            for _ in range(50):
                logits, _ = model(ids)
                probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
                nxt = torch.multinomial(probs, 1)
                ids = torch.cat([ids, nxt], dim=1)
                tok = tokenizer.decode([nxt.item()])
                if '\n' in tok and ids.shape[1] > 25: break
        out = tokenizer.decode(ids[0], skip_special_tokens=True)
        code = out[len(p['prompt']):]
        clean = ''.join(c for c in code if ord(c) < 128)
        print(f"  {tid}: {clean[:60]}")

if __name__ == "__main__":
    main()
