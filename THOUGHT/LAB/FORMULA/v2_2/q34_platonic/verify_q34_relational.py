"""Q34: Relational structure convergence — distance matrix comparison."""
import torch, torch.nn as nn, numpy as np, math
from collections import Counter
from datasets import load_dataset; torch.manual_seed(42)
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from scipy.signal import hilbert

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
        attn = F.softmax(sr, dim=-1)
        cp, sp = torch.cos(si), torch.sin(si)
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
model = NativeEigen(V=V).to(D)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
model.train()
print('Training Native Eigen d=2 (8 epochs)...')
for ep in range(8):
    tl = 0
    for i in range(0, len(data), 24):
        b = data[i:i + 24]
        if not b: continue
        x = torch.tensor([p[0] for p in b], device=D, dtype=torch.long)
        y = torch.tensor([p[1] for p in b], device=D, dtype=torch.long)
        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item()
    print(f'  E{ep + 1}: ppl={math.exp(tl / max(1, len(data) // 24)):.0f}', flush=True)

minilm = SentenceTransformer('all-MiniLM-L6-v2')
mpnet = SentenceTransformer('all-mpnet-base-v2')
all_words = list(set(vocab[:500]) - {'<pad>', '<unk>', '<eos>'})
sorted_words = sorted(all_words, key=lambda w: all_words.index(w))

def get_ne(words):
    ids = torch.tensor([w2i.get(w, 1) for w in words], device=D, dtype=torch.long)
    with torch.no_grad(): z = model.emb(ids.unsqueeze(0))
    return z[0].cpu().numpy()

def hilbert_cpx(embeds):
    N, D = embeds.shape; cpx = np.zeros((N, D), dtype=np.complex128)
    for d in range(D): cpx[:, d] = hilbert(embeds[:, d])
    return cpx

def dist_matrix(embeds, complex_input=False):
    if complex_input:
        D = np.zeros((len(embeds), len(embeds)))
        for i in range(len(embeds)):
            for j in range(len(embeds)):
                diff = embeds[i] - embeds[j]
                D[i, j] = np.abs(np.vdot(diff, diff))
    else:
        n = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
        D = 1.0 - n @ n.T
    return D

print('\n' + '=' * 60)
print('RELATIONAL STRUCTURE: Distance matrix correlation')
print('(invariant to rotation/scale -- pure semantic topology)')
print('=' * 60)

for band_label, (start, end) in [('TOP-50', (0, 50)), ('TOP-100', (0, 100)), ('TOP-200', (0, 200))]:
    words = sorted_words[start:end]
    ne = get_ne(words)
    ml = minilm.encode(words, show_progress_bar=False)
    mp = mpnet.encode(words, show_progress_bar=False)
    ml_cpx = hilbert_cpx(ml); mp_cpx = hilbert_cpx(mp)

    # Distance matrices
    D_ne = dist_matrix(ne, True)
    D_ml = dist_matrix(ml, False)
    D_mp = dist_matrix(mp, False)
    D_ml_c = dist_matrix(ml_cpx, True)
    D_mp_c = dist_matrix(mp_cpx, True)

    tri = np.triu_indices(len(words), k=1)
    r_nm = spearmanr(D_ne[tri], D_ml_c[tri])[0]
    r_np = spearmanr(D_ne[tri], D_mp_c[tri])[0]
    r_mm = spearmanr(D_ml[tri], D_mp[tri])[0]
    print(f'  {band_label:>12}: NE-ML r={r_nm:+.4f}  NE-MP r={r_np:+.4f}  ML-MP r={r_mm:+.4f}')

print()
print(f'  ML-MP relational baseline: r={r_mm:.3f}')
print(f'  NativeEigen relational to ML: r={r_nm:.3f}')
if abs(r_nm) > 0.5:
    print('  PLATONIC FORM DETECTED in relational structure at d=2')
elif abs(r_nm) > 0.3:
    print('  Partial relational convergence at d=2')
else:
    print('  No relational convergence at d=2')
