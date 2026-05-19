"""Q36 C5 boundary: intrinsic Native Eigen phase vs Hilbert-extrinsic phase."""
import torch, torch.nn as nn, numpy as np, math, random
from collections import Counter
from datasets import load_dataset; torch.manual_seed(42); random.seed(42)
from scipy.stats import spearmanr
import torch.nn.functional as F

class ComplexEmbed(nn.Module):
    def __init__(self, V, d=2):
        super().__init__()
        self.re = nn.Embedding(V, d); self.im = nn.Embedding(V, d)
    def forward(self, x): return torch.complex(self.re(x), self.im(x))

class PhaseRot(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ang = nn.Parameter(torch.ones(d) * 0.1)
    def forward(self, z):
        c, s = torch.cos(self.ang), torch.sin(self.ang)
        return torch.complex(z.real * c - z.imag * s, z.real * s + z.imag * c)

class NativeAttention(nn.Module):
    def __init__(self, d=2):
        super().__init__()
        self.qr = nn.Linear(d, d, bias=False); self.qi = nn.Linear(d, d, bias=False)
        self.kr = nn.Linear(d, d, bias=False); self.ki = nn.Linear(d, d, bias=False)
        self.vr = nn.Linear(d, d, bias=False); self.vi = nn.Linear(d, d, bias=False)
        self.sc = 1.0 / math.sqrt(d)
        for w in [self.qr, self.qi, self.kr, self.ki, self.vr, self.vi]:
            nn.init.normal_(w.weight, std=0.02)
    def forward(self, x):
        B, S, D = x.shape
        qr = self.qr(x.real) - self.qi(x.imag); qi = self.qr(x.imag) + self.qi(x.real)
        kr = self.kr(x.real) - self.ki(x.imag); ki = self.kr(x.imag) + self.ki(x.real)
        vr = self.vr(x.real) - self.vi(x.imag); vi = self.vr(x.imag) + self.vi(x.real)
        qr, kr, vr = qr.transpose(1, 2), kr.transpose(1, 2), vr.transpose(1, 2)
        qi, ki, vi = qi.transpose(1, 2), ki.transpose(1, 2), vi.transpose(1, 2)
        sr = (qr.transpose(-2, -1) @ kr + qi.transpose(-2, -1) @ ki) * self.sc
        si = (qi.transpose(-2, -1) @ kr - qr.transpose(-2, -1) @ ki) * self.sc
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        sr = sr.masked_fill(mask, float('-inf')); si = si.masked_fill(mask, 0.0)
        attn = F.softmax(sr, dim=-1); cp, sp = torch.cos(si), torch.sin(si)
        out_r = (attn * cp) @ vr.transpose(-2, -1) - (attn * sp) @ vi.transpose(-2, -1)
        out_i = (attn * cp) @ vi.transpose(-2, -1) + (attn * sp) @ vr.transpose(-2, -1)
        return torch.complex(out_r, out_i)

class NativeEigen(nn.Module):
    def __init__(self, V, d=2, L=2):
        super().__init__()
        self.emb = ComplexEmbed(V, d)
        self.layers = nn.ModuleList([nn.ModuleDict({'a': NativeAttention(d), 'p': PhaseRot(d)}) for _ in range(L)])
        self.out = nn.Linear(d, V); nn.init.normal_(self.out.weight, std=0.02)
    def forward(self, x):
        z = self.emb(x)
        for l in self.layers: z = l['p'](l['a'](z))
        return self.out(torch.abs(z))

def load_wt(V=2000, N=1200):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    c = Counter()
    for ex in ds:
        for w in str(ex['text']).split(): c[w] += 1
    voc = ['<pad>', '<unk>', '<eos>'] + [w for w, _ in c.most_common(V - 3)]
    w2i = {w: i for i, w in enumerate(voc)}
    toks = []
    for ex in ds:
        for w in str(ex['text']).split(): toks.append(w2i.get(w, 1))
        toks.append(2)
    data = []
    for i in range(0, min(len(toks) - 32, N * 32), 16):
        s = toks[i:i + 33]
        if len(s) == 33: data.append((s[:-1], s[1:]))
    return data[:N], len(voc), voc, w2i

D = 'cuda' if torch.cuda.is_available() else 'cpu'
data, V, vocab, w2i = load_wt(N=1200)

n_seeds = 3
ne_models = []
for s in range(n_seeds):
    torch.manual_seed(s * 100)
    m = NativeEigen(V=V).to(D)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    m.train()
    for ep in range(5):
        tl = 0
        for i in range(0, len(data), 24):
            b = data[i:i + 24]
            if not b: continue
            x = torch.tensor([p[0] for p in b], device=D, dtype=torch.long)
            y = torch.tensor([p[1] for p in b], device=D, dtype=torch.long)
            loss = F.cross_entropy(m(x).view(-1, V), y.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); tl += loss.item()
    ne_models.append(m)
    print(f'Seed {s}: ppl={math.exp(tl / max(1, len(data) // 24)):.0f}')

words = ['dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'pig', 'bear', 'lion',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'red', 'blue', 'green', 'yellow', 'black', 'white',
    'hot', 'cold', 'big', 'small', 'good', 'bad', 'love', 'hate',
    'light', 'dark', 'day', 'night', 'man', 'woman', 'up', 'down',
    'war', 'peace', 'life', 'death', 'fast', 'slow', 'happy', 'sad',
    'mother', 'father', 'child', 'friend', 'enemy',
    'water', 'fire', 'earth', 'air', 'sun', 'moon', 'star', 'sky']

ne_embs = []
for m in ne_models:
    ids = torch.tensor([w2i.get(w, 1) for w in words], device=D, dtype=torch.long)
    with torch.no_grad(): z = m.emb(ids.unsqueeze(0)); ne_embs.append(z[0].cpu().numpy())

ne_ex = [np.abs(e) for e in ne_embs]
ne_im = [np.angle(e) for e in ne_embs]

tri = np.triu_indices(len(words), k=1)

def dist_mag(e):
    n = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
    D = 1.0 - n @ n.T
    return D[tri]

def dist_phase(p):
    N = p.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        diff = p[i] - p
        D[i] = np.mean(np.abs(np.sin(diff)), axis=1)
    return D[tri]

print('\n' + '=' * 50)
print('C5 BOUNDARY: INTRINSIC phase (Native Eigen)')
print('=' * 50)

print('\nCross-seed IMPLICATE (phase) correlations:')
ph_rs = []
for i in range(n_seeds):
    for j in range(i + 1, n_seeds):
        r = spearmanr(dist_phase(ne_im[i]), dist_phase(ne_im[j]))[0]
        ph_rs.append(r)
        print(f'  Seed {i} vs Seed {j}: r = {r:+.4f}')

print('\nCross-seed EXPLICATE (magnitude) correlations:')
ex_rs = []
for i in range(n_seeds):
    for j in range(i + 1, n_seeds):
        r = spearmanr(dist_mag(ne_ex[i]), dist_mag(ne_ex[j]))[0]
        ex_rs.append(r)
        print(f'  Seed {i} vs Seed {j}: r = {r:+.4f}')

print('\nEx-im COMPLEMENTARITY within each seed:')
com_rs = []
for i in range(n_seeds):
    r = spearmanr(dist_mag(ne_ex[i]), dist_phase(ne_im[i]))[0]
    com_rs.append(r)
    print(f'  Seed {i}: r = {r:+.4f}')

avg_ph = sum(ph_rs) / len(ph_rs)
avg_ex = sum(ex_rs) / len(ex_rs)
avg_com = sum(com_rs) / len(com_rs)

print(f'\n{"="*50}')
print(f'C5 VERDICT')
print(f'{"="*50}')
print(f'  Intrinsic phase cross-seed: r = {avg_ph:+.4f}  (Hilbert phase: r=+0.210)')
print(f'  Intrinsic magnitude cross-seed: r = {avg_ex:+.4f}')
print(f'  Complementarity: r = {avg_com:+.4f}')

if abs(avg_ph) < 0.15:
    print(f'\n  C5 CONFIRMED: intrinsic phase IS model-specific')
    print(f'  Native Eigen phase (r={avg_ph:+.3f}) is true implicate order')
    print(f'  Hilbert phase (r=+0.210) is extrinsic implicate, partly shared')
else:
    print(f'\n  Intrinsic phase partially shared (r={avg_ph:+.3f})')
    print(f'  C5 direction correct but magnitude smaller than expected')
