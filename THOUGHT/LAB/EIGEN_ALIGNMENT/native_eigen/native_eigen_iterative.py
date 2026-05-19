"""Native Eigen HRM — iterative geodesic processing.

Inspired by sapientinc/HRM-Text-1B: dual-timescale recurrent architecture.
H_cycles x L_cycles iterations over same input with additive state injection.
Replaces fixed-depth forward with variable-depth iterative processing.
The gate decides how many cycles to run — unbounded compute at bounded params.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
torch.manual_seed(42); random.seed(42)
import sys
sys.path.insert(0, r'THOUGHT/LAB/EIGEN_ALIGNMENT/native_eigen')
from native_eigen_core import NativeEigenCore

# ---- Data: token sequences from WikiText-2 ----
from collections import Counter
from datasets import load_dataset

def load_data(vocab_size=2000, seq_len=32, n_seqs=1000):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    c = Counter()
    for ex in ds:
        for w in str(ex["text"]).split(): c[w] += 1
    voc = ["<pad>","<unk>","<eos>"] + [w for w,_ in c.most_common(vocab_size-3)]
    w2i = {w:i for i,w in enumerate(voc)}
    toks = []
    for ex in ds:
        for w in str(ex["text"]).split(): toks.append(w2i.get(w,1))
        toks.append(2)
    data = []
    for i in range(0, min(len(toks)-seq_len, n_seqs*seq_len), seq_len//2):
        s = toks[i:i+seq_len+1]
        if len(s)==seq_len+1: data.append((s[:-1], s[1:]))
    return data[:n_seqs], len(voc)

# ---- HRM-inspired iterative Core ----
class IterativeCore(nn.Module):
    """Iterative geodesic processing. Multiple cycles with state injection.

    Pattern from HRM-Text-1B:
      for each H_cycle:
        for each L_cycle:
          z = L_module(z + z_init)
        z = H_module(z + z_init)

    Our adaptation:
      z_current = embed(x)
      for cycle in range(max_cycles):
        z_current, pc = core(z_current + z_init)  # additive state injection
        if gate.should_stop(pc): break
      return z_current
    """
    def __init__(self, d=16, heads=4, layers=2, max_cycles=3):
        super().__init__()
        self.d = d
        self.max_cycles = max_cycles
        self.core = NativeEigenCore(d=d, heads=heads, layers=layers, merge='concat', geo_init=True)
        # Learn when to stop iterating
        self.stop_threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, z, return_cycles=False):
        """Iterative processing with state injection from initial input."""
        z_init = z  # the original input is the persistent state
        z_current = z
        cycles_used = 1

        for cycle in range(self.max_cycles):
            z_next, pc = self.core(z_current + z_init)  # HRM-style additive injection
            z_current = z_next
            cycles_used = cycle + 1
            # Stop if phase coherence exceeds threshold (converged on a geodesic)
            if pc > self.stop_threshold:
                break

        if return_cycles:
            return z_current, cycles_used
        return z_current

# ---- Language Adapter with iterative processing ----
class IterativeLM(nn.Module):
    def __init__(self, vocab, d=16, heads=4, layers=2, max_cycles=3):
        super().__init__()
        self.embed_re = nn.Embedding(vocab, d)
        self.embed_im = nn.Embedding(vocab, d)
        self.iterative = IterativeCore(d=d, heads=heads, layers=layers, max_cycles=max_cycles)
        self.out_r = nn.Linear(d, vocab)
        self.out_i = nn.Linear(d, vocab)
        nn.init.normal_(self.embed_re.weight, std=0.02)
        nn.init.normal_(self.embed_im.weight, std=0.02)
        nn.init.normal_(self.out_r.weight, std=0.02)
        nn.init.normal_(self.out_i.weight, std=0.02)

    def forward(self, x):
        z = torch.complex(self.embed_re(x), self.embed_im(x))
        z = self.iterative(z)
        return self.out_r(z.real) + self.out_i(z.imag)

# ---- Test ----
print("=" * 55)
print("ITERATIVE GEODESIC CORE (HRM-inspired)")
print("=" * 55)

data, V = load_data(n_seqs=1000)

for cycles in [2, 3, 4]:
    t0 = time.time()
    model = IterativeLM(V, d=16, heads=4, layers=2, max_cycles=cycles)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    P = sum(p.numel() for p in model.parameters())

    model.train()
    for ep in range(5):
        tl = n = 0
        for i in range(0, len(data), 16):
            b = data[i:i+16]
            if not b: continue
            x = torch.tensor([p[0] for p in b], dtype=torch.long)
            y = torch.tensor([p[1] for p in b], dtype=torch.long)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item(); n += 1

    saved = [l['phase'].ang.data.clone() for l in model.iterative.core.layers]
    results = {}
    for mode in ['normal','ablated']:
        if mode == 'ablated':
            for l in model.iterative.core.layers: l['phase'].ang.data.zero_()
        nll = n = 0
        model.eval()
        with torch.no_grad():
            for i in range(0, min(200, len(data)), 16):
                b = data[i:i+16]
                if not b: continue
                x = torch.tensor([p[0] for p in b], dtype=torch.long)
                y = torch.tensor([p[1] for p in b], dtype=torch.long)
                lo = model(x)
                nll += F.cross_entropy(lo.view(-1, V), y.view(-1)).item() * y.numel()
                n += y.numel()
        results[mode] = math.exp(nll/max(n,1))
    for l,s in zip(model.iterative.core.layers, saved): l['phase'].ang.data.copy_(s)

    delta = (results['ablated']-results['normal'])/results['normal']*100
    t = time.time()-t0
    print(f"cycles={cycles} P={P:>7,} norm={results['normal']:.0f} abl={results['ablated']:.0f} delta={delta:+.1f}% time={t:.0f}s")
    print(f"  {'PHASE CARRIES' if delta>10 else 'WEAK' if delta>3 else 'NOT LOAD-BEARING'}")
    del model, opt
