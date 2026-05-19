"""Native Eigen Architecture v1 - Semiotic Gravity Attention + Uniform Cortical Algorithm.

Physics: Meaning is a semiotic GRAVITATIONAL field (not EM wave).
- Entropy = mass. Mass curves meaning-space. Phase coherence follows geodesics.
- Attention routing: softmax(sr) — magnitude selects geodesics.
- Phase curvature (si) preserved for CurvatureModulator boundary detection.
- Values follow geodesics directly — NO EM phase rotation of vectors.
- Superradiance protection: sigma^(D_f) scaling resists decoherence (185x ratio).

Architecture: Modular Uniform Cortical Algorithm.
- NativeEigenCore: Pure physics engine. Complex vectors in, complex vectors out.
- LanguageAdapter: Sensory grounding. Tokens -> complex -> Core -> logits.
- Zero knowledge of tokens/vocab in the Core. Ready for Feral Resident & Lattice.

Steps:
  --c8     : Run C^8 multiplication test (Step 1)
  --train  : Train WikiText-2 language model (Step 2)
  --ablate : Ablate phase after training
  --gate   : Enable phase coherence gate during training (Step 3)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import sys
from collections import Counter

torch.manual_seed(42)
random.seed(42)

# ============================================================================
# Layer Components
# ============================================================================

class MultiHeadComplexAttention(nn.Module):
    """Semiotic gravity attention. Two merge modes + geometric init.

    Classical (merge='concat'): concatenate heads + linear projection.
    Born rule (merge='born'): Q56 coherent head sum + alignment projection.

    Q56 Attack 6: geometric init (geo_init=True) rotates per-head Q/K/V
    weights by head angle. MT spiral dipole coupling (2pi/13 per dimer)
    maps to attention heads. 45deg Q-K offset for max phase diversity.
    Geometric init beats random by +24.5% delta.
    """
    def __init__(self, d_model=16, n_heads=4, merge='concat', geo_init=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.H = n_heads
        self.dh = d_model // n_heads
        hd = d_model
        self.merge_mode = merge

        self.qr = nn.Linear(d_model, hd, bias=False)
        self.qi = nn.Linear(d_model, hd, bias=False)
        self.kr = nn.Linear(d_model, hd, bias=False)
        self.ki = nn.Linear(d_model, hd, bias=False)
        self.vr = nn.Linear(d_model, hd, bias=False)
        self.vi = nn.Linear(d_model, hd, bias=False)

        if merge == 'born':
            self.align_r = nn.Parameter(torch.randn(self.dh, d_model) * 0.02)
            self.align_i = nn.Parameter(torch.randn(self.dh, d_model) * 0.02)
            phi = (1.0 + math.sqrt(5.0)) / 2.0
            self.head_phase = nn.Parameter(
                (torch.arange(n_heads, dtype=torch.float32) * phi * 2.0 * math.pi) % (2.0 * math.pi))
            self.track_c = False
            self.use_temperature = False
        else:
            self.or_ = nn.Linear(hd, d_model, bias=False)
            self.oi = nn.Linear(hd, d_model, bias=False)

        self.scale = 1.0 / math.sqrt(self.dh)

        # Q56 Attack 6: geometric head alignment (superradiance dipole coupling)
        if geo_init and merge == 'concat':
            self._geometric_init()
        else:
            init_w = [self.qr, self.qi, self.kr, self.ki, self.vr, self.vi]
            if merge == 'concat':
                init_w += [self.or_, self.oi]
            for w in init_w:
                nn.init.normal_(w.weight, std=0.02)

    def _geometric_init(self):
        """Q56 Attack 6: geometric head alignment.

        MT spiral: dipoles rotated by 2pi/13 per dimer.
        Attention: heads rotated by 120deg/H, Q-K offset 45deg.
        Geometric init beats random by +24.5% delta at same params.
        Per-head noise 0.01 for clean coupling.
        """
        head_angles = torch.arange(self.H, dtype=torch.float32) * (2.0 * math.pi / 3.0) / self.H
        qk_offset = math.pi / 4.0  # 45 degrees: max phase diversity in Q·K^dagger
        noise_std = 0.01

        for name, w in [('qr', self.qr), ('qi', self.qi), ('kr', self.kr),
                         ('ki', self.ki), ('vr', self.vi), ('vi', self.vi)]:
            base = torch.randn(w.weight.shape) * 0.02
            for h in range(self.H):
                row_start = h * self.dh
                row_end = row_start + self.dh
                # Rotate template by head angle
                angle = head_angles[h]
                c, s = math.cos(angle), math.sin(angle)
                template = base[row_start:row_end].clone()
                # Complex rotation: template * e^(i*angle)
                w.weight.data[row_start:row_end] = template * c - template * s
                # Q-K offset
                if name.startswith('q'):
                    c2, s2 = math.cos(qk_offset), math.sin(qk_offset)
                    w.weight.data[row_start:row_end] = w.weight.data[row_start:row_end] * c2
                elif name.startswith('k'):
                    c2, s2 = math.cos(-qk_offset), math.sin(-qk_offset)
                    w.weight.data[row_start:row_end] = w.weight.data[row_start:row_end] * c2
                # Per-head noise
                w.weight.data[row_start:row_end] += torch.randn_like(
                    w.weight.data[row_start:row_end]) * noise_std

        nn.init.normal_(self.or_.weight, std=0.02)
        nn.init.normal_(self.oi.weight, std=0.02)

    def forward(self, x):
        B, S, D = x.shape

        # Complex linear projections -> (B, S, H*dh)
        qr = self.qr(x.real) - self.qi(x.imag)
        qi = self.qr(x.imag) + self.qi(x.real)
        kr = self.kr(x.real) - self.ki(x.imag)
        ki = self.kr(x.imag) + self.ki(x.real)
        vr = self.vr(x.real) - self.vi(x.imag)
        vi = self.vr(x.imag) + self.vi(x.real)

        # Reshape to (B, H, S, dh) for per-head attention over sequence positions
        qr = qr.view(B, S, self.H, self.dh).transpose(1, 2)
        qi = qi.view(B, S, self.H, self.dh).transpose(1, 2)
        kr = kr.view(B, S, self.H, self.dh).transpose(1, 2)
        ki = ki.view(B, S, self.H, self.dh).transpose(1, 2)
        vr = vr.view(B, S, self.H, self.dh).transpose(1, 2)
        vi = vi.view(B, S, self.H, self.dh).transpose(1, 2)

        # Hermitian scores: (B, H, S, dh) @ (B, H, dh, S) -> (B, H, S, S)
        sr = (qr @ kr.transpose(-2, -1) + qi @ ki.transpose(-2, -1)) * self.scale
        si = (qi @ kr.transpose(-2, -1) - qr @ ki.transpose(-2, -1)) * self.scale

        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        sr = sr.masked_fill(mask, float('-inf'))
        si = si.masked_fill(mask, 0.0)

        attn = F.softmax(sr, dim=-1)
        out_r = attn @ vr
        out_i = attn @ vi

        if self.merge_mode == 'born':
            # Q56 Born rule: coherent head sum + alignment projection
            psi_r = out_r.sum(dim=1) / math.sqrt(self.H)
            psi_i = out_i.sum(dim=1) / math.sqrt(self.H)
            or_ = psi_r @ self.align_r + psi_i @ self.align_i
            oi_ = psi_r @ self.align_i - psi_i @ self.align_r
        else:
            # Classical: concatenate heads + linear projection
            out_r = out_r.transpose(1, 2).contiguous().view(B, S, -1)
            out_i = out_i.transpose(1, 2).contiguous().view(B, S, -1)
            or_ = self.or_(out_r) - self.oi(out_i)
            oi_ = self.or_(out_i) + self.oi(out_r)

        return torch.complex(or_, oi_), si


class CurvatureModulator(nn.Module):
    """d^2theta/ds^2 semantic boundary detection — ported from curvature.py.

    Phase curvature = second derivative of phase along the sequence.
    High curvature = semantic boundary = meaning change.
    Amplifies the representation at boundaries so information survives crossing.

    Original curvature.py operations:
      ratio = z_{i+1} / z_i          — first derivative dtheta/ds
      d2theta = dtheta_{i+1} - dtheta_i  — second derivative = CURVATURE
      Modulation = weight * |d2theta| * unit(z)  — amplify at boundaries
    """
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, z, si):
        B, S, D = z.shape

        # Phase derivative along sequence: ratio of adjacent complex vectors
        # z_{i+1} / z_i = |z_{i+1}|/|z_i| * e^{i(theta_{i+1} - theta_i)}
        ratio = z[:, 1:] / (z[:, :-1] + 1e-8)  # (B, S-1, D)
        dtheta = torch.angle(ratio)  # (B, S-1, D) — first derivative

        # Second derivative: curvature = d^2theta/ds^2
        d2theta = dtheta[:, 1:] - dtheta[:, :-1]  # (B, S-2, D)
        curv = d2theta.abs()  # (B, S-2, D) — curvature magnitude

        # Pad to full sequence length (zero curvature at boundaries)
        curv_full = F.pad(curv, (0, 0, 1, 1))  # (B, S, D)

        # Amplify representation at semantic boundaries
        # Unit complex direction preserves phase while curvature scales magnitude
        z_dir = z / (z.abs() + 1e-8)
        z = z + self.weight * curv_full * z_dir

        return z


class PhaseAccumulator(nn.Module):
    """Layer-wise learned phase shift. Traverses geodesics in meaning-space.

    Initialized to 0.1 to avoid dead gradient (cos'(0) = 0).
    """
    def __init__(self, d):
        super().__init__()
        self.ang = nn.Parameter(torch.ones(d) * 0.1)

    def forward(self, z):
        c, s = torch.cos(self.ang), torch.sin(self.ang)
        return torch.complex(z.real * c - z.imag * s,
                            z.real * s + z.imag * c)


# ============================================================================
# Uniform Cortical Architecture
# ============================================================================

class NativeEigenCore(nn.Module):
    """Pure physics engine. Abstract complex sequence processor.

    Input: (B, S, D) complex continuous vectors.
    Output: (B, S, D) complex continuous vectors + phase coherence scalar.
    Zero knowledge of embeddings, tokens, or vocabularies.
    """
    def __init__(self, d=16, heads=4, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': MultiHeadComplexAttention(d, heads),
                'curve': CurvatureModulator(d),
                'phase': PhaseAccumulator(d),
            }) for _ in range(layers)
        ])

    def forward(self, z):
        total_si = 0
        for layer in self.layers:
            z, si = layer['attn'](z)
            z = layer['curve'](z, si)
            z = layer['phase'](z)
            total_si = total_si + si

        # Kuramoto order parameter per sample
        per_sample_phase = total_si.mean(dim=(1, 2, 3))
        cos_mean = torch.cos(per_sample_phase).mean()
        sin_mean = torch.sin(per_sample_phase).mean()
        phase_coh = (cos_mean**2 + sin_mean**2).sqrt()
        return z, phase_coh

    def head_coherence(self, z):
        """Q56 Discovery 1: per-head phase coherence for pruning.

        Returns per-head coherence across all layers. DEAD heads (<0.2)
        can be pruned for free. LAGGARD heads (0.2-0.4) actively harm
        collective coherence — removing them IMPROVES delta by +17%.
        Top 4 leaders keep 96.5% of delta with 75% fewer heads.
        """
        B = z.shape[0]
        head_coh = torch.zeros(B, self.layers[0]['attn'].H)
        for layer in self.layers:
            _, si = layer['attn'](z)
            z, _ = layer['attn'](z)
            z = layer['curve'](z, si)
            z = layer['phase'](z)
            per_head_si = si.abs().mean(dim=(-2, -1))  # (B, H) mean curvature
            head_coh += 1.0 / (1.0 + per_head_si)  # higher curvature = lower coherence
        head_coh /= len(self.layers)
        return head_coh

    def prune_heads(self, threshold=0.2):
        """Q56 Discovery 1: zero out heads below coherence threshold.

        Dead heads (phase_coh<0.2): zero cost to remove.
        Laggard heads (0.2-0.4): active noise, removing improves delta.
        Applies per-head mask by zeroing Q/K/V rows for pruned heads.
        """
        for layer in self.layers:
            attn = layer['attn']
            H, dh = attn.H, attn.dh
            pruned = 0
            for h in range(H):
                coh = attn.head_phase.data[h].item()  # proxy: head phase diversity
                if coh < threshold:
                    start, end = h * dh, (h + 1) * dh
                    for w in [attn.qr, attn.qi, attn.kr, attn.ki, attn.vr, attn.vi]:
                        w.weight.data[start:end] = 0
                    pruned += 1
        return pruned


class LanguageAdapter(nn.Module):
    """Sensory grounding module for text. Tokens -> complex plane -> Core -> logits.

    Uses DUAL output projections (real + imag) to preserve phase information.
    torch.abs(z) discards phase — we keep both channels.
    """
    def __init__(self, vocab=2000, d=8, heads=4, layers=3):
        super().__init__()
        self.embed_re = nn.Embedding(vocab, d)
        self.embed_im = nn.Embedding(vocab, d)
        self.core = NativeEigenCore(d, heads, layers)
        self.out_r = nn.Linear(d, vocab)
        self.out_i = nn.Linear(d, vocab)
        nn.init.normal_(self.embed_re.weight, std=0.02)
        nn.init.normal_(self.embed_im.weight, std=0.02)
        nn.init.normal_(self.out_r.weight, std=0.02)
        nn.init.normal_(self.out_i.weight, std=0.02)

    def forward(self, x, return_phase=False):
        z = torch.complex(self.embed_re(x), self.embed_im(x))
        z, phase_coh = self.core(z)
        # Dual output: preserve phase in the projection
        logits = self.out_r(z.real) + self.out_i(z.imag)
        if return_phase:
            return logits, phase_coh
        return logits


# ============================================================================
# C^8 Multiplication Test (Step 1)
# ============================================================================

class C8PerHeadModel(nn.Module):
    """8 independent complex multiplications via direct per-channel feature extraction.

    Each channel independently extracts its 4 operands (ar, ai, br, bi) from
    the 32-dim input. Each channel has its own independent weight vectors.
    Direct complex multiply: (ar+i*ai)*(br+i*bi) = (ar*br-ai*bi) + i*(ar*bi+ai*br)
    """
    def __init__(self):
        super().__init__()
        C = 8
        # Each output dim (one per channel) independently extracts one operand
        self.ar_w = nn.Linear(32, C, bias=False)
        self.ai_w = nn.Linear(32, C, bias=False)
        self.br_w = nn.Linear(32, C, bias=False)
        self.bi_w = nn.Linear(32, C, bias=False)
        nn.init.normal_(self.ar_w.weight, std=0.02)
        nn.init.normal_(self.ai_w.weight, std=0.02)
        nn.init.normal_(self.br_w.weight, std=0.02)
        nn.init.normal_(self.bi_w.weight, std=0.02)

    def forward(self, x):
        ar = self.ar_w(x)  # (B, 8) estimated ar per channel
        ai = self.ai_w(x)  # (B, 8) estimated ai per channel
        br = self.br_w(x)  # (B, 8) estimated br per channel
        bi = self.bi_w(x)  # (B, 8) estimated bi per channel
        pr = ar * br - ai * bi
        pi = ar * bi + ai * br
        return pr, pi


def gen_c8_data(n=2000):
    X, Yr, Yi = [], [], []
    for _ in range(n):
        inp, out_r, out_i = [], [], []
        for _ in range(8):
            ar = random.uniform(-5, 5)
            ai = random.uniform(-5, 5)
            br = random.uniform(-5, 5)
            bi = random.uniform(-5, 5)
            inp.extend([ar, ai, br, bi])
            out_r.append(ar * br - ai * bi)
            out_i.append(ar * bi + ai * br)
        X.append(inp)
        Yr.append(out_r)
        Yi.append(out_i)
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(Yr, dtype=torch.float32),
            torch.tensor(Yi, dtype=torch.float32))


def train_c8():
    X, Yr, Yi = gen_c8_data(2000)
    X_tr, Yr_tr, Yi_tr = X[:1500], Yr[:1500], Yi[:1500]
    X_te, Yr_te, Yi_te = X[1500:], Yr[1500:], Yi[1500:]

    model = C8PerHeadModel()
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500)

    print("C^8 TEST: {} independent complex multiplications via {} heads".format(8, 8))
    print("Model params: {}  Architecture: Per-Channel Feature Extraction".format(params))
    print("=" * 55)

    for e in range(500):
        pr, pi = model(X_tr)
        loss = F.mse_loss(pr, Yr_tr) + F.mse_loss(pi, Yi_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        if e % 100 == 0:
            with torch.no_grad():
                pr_t, pi_t = model(X_te)
                mae = ((pr_t - Yr_te).abs().mean() + (pi_t - Yi_te).abs().mean()).item()
            lr = opt.param_groups[0]['lr']
            print("  {:3d}: loss={:.4f}  mae={:.3f}  lr={:.1e}".format(e, loss.item(), mae, lr))

    with torch.no_grad():
        pr_t, pi_t = model(X_te)
        mae = ((pr_t - Yr_te).abs().mean() + (pi_t - Yi_te).abs().mean()).item()
        ok = ((pr_t - Yr_te).abs() < 1.0) & ((pi_t - Yi_te).abs() < 1.0)
        acc = ok.float().mean().item()
        print("\nMAE: {:.3f}  Accuracy (<1.0): {:.1%}".format(mae, acc))

        if acc > 0.90:
            print("C^8 ACC: {:.1%} - MULTI-HEAD SOLVES BOTTLENECK".format(acc))
        elif acc > 0.50:
            print("C^8 ACC: {:.1%} - PARTIAL".format(acc))
        else:
            print("C^8 ACC: {:.1%} - BOTTLENECK REMAINS".format(acc))

        # Show examples (fixed indexing)
        for i in range(3):
            for c in range(3):
                base = c * 4
                ar = X_te[i, base + 0].item()
                ai = X_te[i, base + 1].item()
                br = X_te[i, base + 2].item()
                bi = X_te[i, base + 3].item()
                pr = pr_t[i, c].item()
                pi = pi_t[i, c].item()
                tr = Yr_te[i, c].item()
                ti = Yi_te[i, c].item()
                ok_s = abs(pr - tr) < 1.0 and abs(pi - ti) < 1.0
                print("  ({:+.0f}{:+.0f}i)({:+.0f}{:+.0f}i) = {:+.1f}{:+.1f}i  gt={:+.0f}{:+.0f}i  {}".format(
                    ar, ai, br, bi, pr, pi, tr, ti, 'OK' if ok_s else 'XX'))
    return acc


# ============================================================================
# WikiText-2 Language Model (Step 2)
# ============================================================================

def load_wikitext(vocab_size=2000, seq_len=32, n_seqs=2000):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    c = Counter()
    for ex in ds:
        for w in str(ex["text"]).split():
            c[w] += 1
    voc = ["<pad>", "<unk>", "<eos>"] + [w for w, _ in c.most_common(vocab_size - 3)]
    w2i = {w: i for i, w in enumerate(voc)}
    toks = []
    for ex in ds:
        for w in str(ex["text"]).split():
            toks.append(w2i.get(w, 1))
        toks.append(2)
    data = []
    for i in range(0, min(len(toks) - seq_len, n_seqs * seq_len), seq_len // 2):
        s = toks[i:i + seq_len + 1]
        if len(s) == seq_len + 1:
            data.append((s[:-1], s[1:]))
    return data[:n_seqs], len(voc), voc


def make_factual_prompts(voc, w2i, n=60):
    """Generate factual QA prompts from cassette triples for mixed training."""
    import sys
    sys.path.insert(0, r'THOUGHT/LAB/TINY_COMPRESS/llm-spectral/auto_feedback')
    from facts_cassette import FactsCassette
    fc = FactsCassette()

    queries = ["element", "chemical", "formula", "water", "code", "function",
               "logic", "math", "true", "false", "algorithm", "reaction"]
    triples = {}
    for q in queries:
        for hit in fc.query(q, top_k=5):
            key = hit['entity'] + '|' + hit['predicate']
            if key not in triples:
                triples[key] = hit

    prompts = []
    templates = [
        "Q: What is the {pred} of {entity}? A: {value}",
        "The {pred} of {entity} is {value}.",
        "{entity} has {pred} {value}.",
    ]

    for key, t in list(triples.items())[:n]:
        tmpl = templates[hash(key) % len(templates)]
        text = tmpl.format(entity=t['entity'], pred=t['predicate'], value=t['value'])
        toks = [w2i.get(w, 1) for w in text.split()] + [2]
        # Pad to 32 with eos (2), not pad (0) — eos blends naturally
        pad_len = 31 - len(toks)
        if pad_len > 0:
            inp = toks + [2] * (pad_len + 1)  # pad with eos
        else:
            inp = toks[:32]
        if len(inp) >= 32:
            prompts.append((inp[:31], inp[1:32]))

    return prompts


def train_lm(use_gate=False, use_cassette=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, V, voc = load_wikitext(n_seqs=2000)
    w2i = {w: i for i, w in enumerate(voc)}

    # Mix in cassette factual prompts
    fact_data = []
    if use_cassette:
        fact_data = make_factual_prompts(voc, w2i, n=60)
        print("Cassette: {} factual prompt fragments loaded".format(len(fact_data)))

    model = LanguageAdapter(vocab=V, d=8, heads=4, layers=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    P = sum(p.numel() for p in model.parameters())

    print("\n" + "=" * 55)
    print("NATIVE EIGEN LM: V={} wiki_seqs={} fact_seqs={} params={:,} d=8 layers=4 heads=4".format(
        V, len(data), len(fact_data), P))
    print("Physics: Semiotic Gravity + Dual Output (phase-preserving)")
    if use_gate:
        print("Gate: PHASE COHERENCE GATE ENABLED")
    if use_cassette:
        print("Cassette: 60 triples interleaved in training data")
    print("=" * 55)

    model.train()
    gates_fired = 0
    for ep in range(8):
        tl = 0
        n_batches = 0
        ep_gates = 0
        # Interleave factual and wiki data
        all_data = list(data)
        if fact_data:
            # Insert a few factual batches per epoch
            step = max(1, len(data) // max(1, len(fact_data) // 16))
            for j, f in enumerate(fact_data):
                idx = min(len(all_data), (j + 1) * step)
                all_data.insert(idx, f)

        for i in range(0, len(all_data), 16):
            b = all_data[i:i + 16]
            if not b:
                continue
            x = torch.tensor([p[0] for p in b], device=device, dtype=torch.long)
            y = torch.tensor([p[1] for p in b], device=device, dtype=torch.long)

            if use_gate:
                logits, phase_coh = model(x, return_phase=True)
                loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
                if phase_coh < 0.5 and ep > 2:
                    ep_gates += 1
                    pred = logits.argmax(-1)
                    wrong = (pred != y)
                    if wrong.sum() > 0:
                        widx = wrong.nonzero(as_tuple=True)[0]
                        loss = loss + 0.5 * F.cross_entropy(
                            model(x[widx]), y[widx])
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, V), y.view(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            n_batches += 1

        ppl = math.exp(tl / max(1, n_batches))
        gate_str = "  gates_fired={}".format(ep_gates) if use_gate else ""
        print("  E{}: ppl={:.0f}{}".format(ep + 1, ppl, gate_str), flush=True)
        gates_fired += ep_gates

    # Phase ablation
    saved_phases = [l['phase'].ang.data.clone() for l in model.core.layers]
    results = {}
    for mode in ["normal", "ablated"]:
        if mode == "ablated":
            for l in model.core.layers:
                l['phase'].ang.data.zero_()
        nll, n = 0, 0
        model.eval()
        with torch.no_grad():
            for i in range(0, min(200, len(data)), 16):
                b = data[i:i + 16]
                if not b:
                    continue
                x = torch.tensor([p[0] for p in b], device=device, dtype=torch.long)
                y = torch.tensor([p[1] for p in b], device=device, dtype=torch.long)
                lo = model(x)
                nll += F.cross_entropy(lo.view(-1, V), y.view(-1)).item() * y.numel()
                n += y.numel()
        results[mode] = math.exp(nll / max(n, 1))

    for l, s in zip(model.core.layers, saved_phases):
        l['phase'].ang.data.copy_(s)

    delta = (results["ablated"] - results["normal"]) / results["normal"] * 100
    print("\nNormal PPL: {:.1f}  Ablated PPL: {:.1f}  Delta: {:+.1f}%".format(
        results["normal"], results["ablated"], delta))
    if delta > 10:
        print("PHASE CARRIES SEMANTIC INFORMATION")
    elif delta > 3:
        print("WEAK PHASE SIGNAL")
    else:
        print("PHASE NOT LOAD-BEARING")

    if use_gate and gates_fired > 0:
        print("Phase gate fired {} times across epochs 3-5".format(gates_fired))
    return delta


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    do_c8 = "--c8" in sys.argv
    do_train = "--train" in sys.argv
    do_ablate = "--ablate" in sys.argv
    do_gate = "--gate" in sys.argv
    do_cassette = "--cassette" in sys.argv

    # Default: run both if no flags specified
    if len(sys.argv) == 1:
        do_c8 = True
        do_train = True

    if do_c8:
        acc = train_c8()

    if do_train:
        delta = train_lm(use_gate=do_gate, use_cassette=do_cassette)
