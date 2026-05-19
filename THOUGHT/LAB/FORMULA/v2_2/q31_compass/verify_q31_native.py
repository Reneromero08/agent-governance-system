"""Q31: Native complex-plane compass — standalone with copied architecture."""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
import numpy as np
from collections import Counter
from datasets import load_dataset
torch.manual_seed(42); random.seed(42)

# ---- Architecture copied from native_eigen.py ----
class ComplexEmbed(nn.Module):
    def __init__(self, V, d=2):
        super().__init__()
        self.re = nn.Embedding(V, d)
        self.im = nn.Embedding(V, d)
        nn.init.normal_(self.re.weight, std=0.02)
        nn.init.normal_(self.im.weight, std=0.02)
    def forward(self, x):
        return torch.complex(self.re(x), self.im(x))

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
        qr, kr, vr = qr.transpose(1,2), kr.transpose(1,2), vr.transpose(1,2)
        qi, ki, vi = qi.transpose(1,2), ki.transpose(1,2), vi.transpose(1,2)
        sr = (qr.transpose(-2,-1) @ kr + qi.transpose(-2,-1) @ ki) * self.sc
        si = (qi.transpose(-2,-1) @ kr - qr.transpose(-2,-1) @ ki) * self.sc
        dtheta = si.diagonal(offset=1, dim1=-2, dim2=-1)
        curv = (dtheta[:, 1:] - dtheta[:, :-1]).abs()
        curv_pad = F.pad(curv, (1, 1)).unsqueeze(1)
        sr = sr + 1.0 * curv_pad; si = si + 0.5 * curv_pad * torch.sign(si)
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        sr = sr.masked_fill(mask, float('-inf')); si = si.masked_fill(mask, 0.0)
        attn = F.softmax(sr, dim=-1); cp, sp = torch.cos(si), torch.sin(si)
        out_r = (attn*cp)@vr.transpose(-2,-1) - (attn*sp)@vi.transpose(-2,-1)
        out_i = (attn*cp)@vi.transpose(-2,-1) + (attn*sp)@vr.transpose(-2,-1)
        return torch.complex(out_r, out_i)

class PhaseRot(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ang = nn.Parameter(torch.ones(d)*0.1)
    def forward(self, z):
        c, s = torch.cos(self.ang), torch.sin(self.ang)
        return torch.complex(z.real*c - z.imag*s, z.real*s + z.imag*c)

class NativeEigen(nn.Module):
    def __init__(self, V=2000, d=2, L=2):
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

def load_corpus(V=2000, seq=32, N=2000):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    c = Counter()
    for ex in ds:
        for w in str(ex["text"]).split():
            c[w] += 1
    voc = ["<pad>", "<unk>", "<eos>"] + [w for w, _ in c.most_common(V-3)]
    w2i = {w: i for i, w in enumerate(voc)}
    toks = []
    for ex in ds:
        for w in str(ex["text"]).split():
            toks.append(w2i.get(w, 1))
        toks.append(2)
    data = []
    for i in range(0, min(len(toks)-seq, N*seq), seq//2):
        s = toks[i:i+seq+1]
        if len(s) == seq+1:
            data.append((s[:-1], s[1:]))
    return data[:N], len(voc), voc, w2i

def phase_coh_set(z_np):
    z_np = z_np / (np.linalg.norm(z_np, axis=1, keepdims=True) + 1e-12)
    z_np = z_np / np.abs(z_np + 1e-15)
    n = len(z_np)
    H = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(i, n):
            v = np.conj(z_np[i]).dot(z_np[j])
            H[i,j] = v; H[j,i] = np.conj(v)
    ev = np.linalg.eigvalsh(H)
    ev = np.maximum(ev, 1e-15); ev /= ev.sum()
    return 1.0 - (-np.sum(ev * np.log(ev + 1e-15))) / math.log(n)

# ---- Train ----
D = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training Native Eigen on {D}...")
data, V, voc, w2i = load_corpus(N=2000)
model = NativeEigen(V=V, d=2, L=2).to(D)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
model.train()
for ep in range(5):
    tl = 0; batches = 0
    for i in range(0, len(data), 16):
        b = data[i:i+16]
        if not b: continue
        x = torch.tensor([p[0] for p in b], device=D, dtype=torch.long)
        y = torch.tensor([p[1] for p in b], device=D, dtype=torch.long)
        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); tl += loss.item(); batches += 1
    print(f"  E{ep+1}: ppl={math.exp(tl/max(batches,1)):.0f}", flush=True)

# ---- Extract native complex embeddings ----
re_w = model.emb.re.weight.detach().cpu().numpy()
im_w = model.emb.im.weight.detach().cpu().numpy()
z = re_w + 1j * im_w  # native complex, NO Hilbert

print(f"\n{'='*60}")
print(f"Native complex embeddings: {z.shape} words, 2D complex")
print(f"Top-20 phase_coh: {phase_coh_set(z[:20]):.4f}")
np.random.seed(42)
print(f"Rand-20 phase_coh: {phase_coh_set(z[np.random.choice(range(20,V),20,replace=False)]):.4f}")

# ---- Phase vs Cosine NN ----
re_norm = re_w / (np.linalg.norm(re_w, axis=1, keepdims=True) + 1e-12)
cos_mat = re_norm @ re_norm.T

def pc_pair(z1, z2):
    return phase_coh_set(np.vstack([z1.reshape(1,-1), z2.reshape(1,-1)]))

pc_acc = 0; cos_acc = 0; total = 0
for i in range(100):
    cands = list(range(100))
    if i in cands: cands.remove(i)
    pc_vals = [pc_pair(z[i], z[j]) for j in cands]
    best_pc = cands[np.argmax(pc_vals)]
    best_cos = cands[np.argmax(cos_mat[i, cands])]
    # Same frequency band = correct
    pc_acc += 1 if (best_pc < 50) == (i < 50) else 0
    cos_acc += 1 if (best_cos < 50) == (i < 50) else 0
    total += 1

print(f"\nPhase NN same-band: {pc_acc}/{total} ({pc_acc/total:.1%})")
print(f"Cosine NN same-band: {cos_acc}/{total} ({cos_acc/total:.1%})")
winner = "PHASE" if pc_acc > cos_acc else "COSINE"
print(f"Winner: {winner}")

# Phase coherence between specific word pairs
print(f"\nSemantic pairs:")
pairs = [("the","a"),("is","was"),("he","she"),("of","in"),("and","or"),("to","for")]
for w1, w2 in pairs:
    if w1 in w2i and w2 in w2i:
        i1, i2 = w2i[w1], w2i[w2]
        pc = pc_pair(z[i1], z[i2])
        cos_s = cos_mat[i1, i2]
        print(f"  {w1}-{w2}: PC={pc:.4f}  Cos={cos_s:+.4f}")
