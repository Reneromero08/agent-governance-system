"""Q34 PUSH: Statistical rigor, Hilbert vs random phase, vocab sweep."""
import torch, torch.nn as nn, numpy as np, math, random
from collections import Counter
from datasets import load_dataset; torch.manual_seed(42); random.seed(42)

class ComplexEmbed(nn.Module):
    def __init__(self, V, d=2):
        super().__init__()
        self.re = nn.Embedding(V, d); self.im = nn.Embedding(V, d)
    def forward(self, x):
        return torch.complex(self.re(x), self.im(x))

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
        attn = torch.nn.functional.softmax(sr, dim=-1)
        cp, sp = torch.cos(si), torch.sin(si)
        out_r = (attn * cp) @ vr.transpose(-2, -1) - (attn * sp) @ vi.transpose(-2, -1)
        out_i = (attn * cp) @ vi.transpose(-2, -1) + (attn * sp) @ vr.transpose(-2, -1)
        return torch.complex(out_r, out_i)

class NativeEigen(nn.Module):
    def __init__(self, V, d=2, L=2):
        super().__init__()
        self.emb = ComplexEmbed(V, d)
        self.layers = nn.ModuleList([nn.ModuleDict({'a': NativeAttention(d), 'p': PhaseRot(d)}) for _ in range(L)])
        self.out = nn.Linear(d, V)
        nn.init.normal_(self.out.weight, std=0.02)
    def forward(self, x):
        z = self.emb(x)
        for l in self.layers:
            z = l['p'](l['a'](z))
        return self.out(torch.abs(z))

def load_wt(V=2000, N=1000):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    c = Counter()
    for ex in ds:
        for w in str(ex['text']).split():
            c[w] += 1
    voc = ['<pad>', '<unk>', '<eos>'] + [w for w, _ in c.most_common(V - 3)]
    w2i = {w: i for i, w in enumerate(voc)}
    toks = []
    for ex in ds:
        for w in str(ex['text']).split():
            toks.append(w2i.get(w, 1))
        toks.append(2)
    data = []
    for i in range(0, min(len(toks) - 32, N * 32), 16):
        s = toks[i:i + 33]
        if len(s) == 33:
            data.append((s[:-1], s[1:]))
    return data[:N], len(voc), voc, w2i

D = 'cuda' if torch.cuda.is_available() else 'cpu'
data, V, vocab, w2i = load_wt(N=1000)
model = NativeEigen(V=V).to(D)
import torch.nn.functional as F
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
model.train()
print('Training Native Eigen d=2...')
print('Training Native Eigen d=2...')
for ep in range(5):
    tl = 0
    for i in range(0, len(data), 16):
        b = data[i:i + 16]
        if not b:
            continue
        x = torch.tensor([p[0] for p in b], device=D, dtype=torch.long)
        y = torch.tensor([p[1] for p in b], device=D, dtype=torch.long)
        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item()
    print(f'  E{ep + 1}: ppl={math.exp(tl / max(1, len(data) // 16)):.0f}', flush=True)

from sentence_transformers import SentenceTransformer
minilm = SentenceTransformer('all-MiniLM-L6-v2')
mpnet = SentenceTransformer('all-mpnet-base-v2')
from scipy.signal import hilbert
from scipy.stats import spearmanr

def get_ne(words):
    ids = torch.tensor([w2i.get(w, 1) for w in words], device=D, dtype=torch.long)
    with torch.no_grad():
        z = model.emb(ids.unsqueeze(0))
    return z[0].cpu().numpy()


def hilbert_cpx(embeds):
    N, D = embeds.shape
    cpx = np.zeros((N, D), dtype=np.complex128)
    for d in range(D):
        cpx[:, d] = hilbert(embeds[:, d])
    return cpx


def random_cpx(embeds):
    N, D = embeds.shape
    phases = np.random.uniform(0, 2 * np.pi, (N, D))
    return embeds.astype(np.complex128) * np.exp(1j * phases)


def cumvar_corr(emb_a, emb_b, ca=False, cb=False):
    if ca:
        G_a = np.array([[np.vdot(emb_a[i], emb_a[j]) for j in range(len(emb_a))] for i in range(len(emb_a))])
    else:
        n = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-8)
        G_a = n @ n.T
    if cb:
        G_b = np.array([[np.vdot(emb_b[i], emb_b[j]) for j in range(len(emb_b))] for i in range(len(emb_b))])
    else:
        n = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-8)
        G_b = n @ n.T
    ev_a = np.linalg.eigvalsh(G_a)[::-1]
    ev_a = np.maximum(ev_a, 0)
    ev_a = ev_a / np.sum(ev_a)
    ev_b = np.linalg.eigvalsh(G_b)[::-1]
    ev_b = np.maximum(ev_b, 0)
    ev_b = ev_b / np.sum(ev_b)
    ca_ = np.cumsum(ev_a)
    cb_ = np.cumsum(ev_b)
    n = min(len(ca_), len(cb_))
    return spearmanr(ca_[:n], cb_[:n])[0]

all_words = list(set(vocab[:400]) - {'<pad>', '<unk>', '<eos>'})
print(f'\nShared vocab: {len(all_words)} words')

print('\n' + '=' * 60)
print('PUSH 1: Statistical rigor (10 random vocab subsets, N=150)')
print('=' * 60)
n_runs = 10
N_w = 150
cors = {}
for k in ['ne_ml_cpx', 'ne_ml_real', 'ml_mp_cpx', 'ne_ml_rand']:
    cors[k] = []
for run in range(n_runs):
    np.random.seed(run)
    words = np.random.choice(all_words, N_w, replace=False).tolist()
    ne = get_ne(words)
    ml = minilm.encode(words, show_progress_bar=False)
    mp = mpnet.encode(words, show_progress_bar=False)
    ml_cpx = hilbert_cpx(ml)
    mp_cpx = hilbert_cpx(mp)
    ml_rand = random_cpx(ml)
    cors['ne_ml_cpx'].append(cumvar_corr(ne, ml_cpx, True, True))
    cors['ne_ml_real'].append(cumvar_corr(ne.real, ml, False, False))
    cors['ml_mp_cpx'].append(cumvar_corr(ml_cpx, mp_cpx, True, True))
    cors['ne_ml_rand'].append(cumvar_corr(ne, ml_rand, True, True))
for k in cors:
    v = cors[k]
    print(f'  {k:>15}: r={np.mean(v):.4f}+/-{np.std(v):.4f} [{np.min(v):.4f},{np.max(v):.4f}]')

print('\n' + '=' * 60)
print('PUSH 2: Hilbert vs Random Phase (causal control)')
print('=' * 60)
hc = np.mean(cors['ne_ml_cpx'])
rc = np.mean(cors['ne_ml_rand'])
print(f'  Hilbert: r={hc:.4f}  Random: r={rc:.4f}')
gap = hc - rc
if gap > 0.05:
    print(f'  HILBERT IS CAUSAL — structured phase adds +{gap:.3f} convergence signal')
elif abs(gap) < 0.03:
    print(f'  No difference — any complexification works')
else:
    print(f'  Random beats Hilbert by {abs(gap):.3f}')

print('\n' + '=' * 60)
print('PUSH 3: Vocabulary size sweep')
print('=' * 60)
for n_w in [25, 50, 100, 150, 200, 300]:
    np.random.seed(0)
    words = np.random.choice(all_words, n_w, replace=False).tolist()
    ne = get_ne(words)
    ml = minilm.encode(words, show_progress_bar=False)
    r = cumvar_corr(ne, hilbert_cpx(ml), True, True)
    print(f'  N={n_w:>3}: r={r:+.4f}')

print(f'\nQ34 PUSH VERDICT:')
print(f'  Convergence r={np.mean(cors["ne_ml_cpx"]):.3f}+/-{np.std(cors["ne_ml_cpx"]):.3f} (10 runs)')
print(f'  Complex > Real: +{(np.mean(cors["ne_ml_cpx"]) - np.mean(cors["ne_ml_real"])):.3f}')
print(f'  Hilbert > Random phase: +{gap:.3f}')
