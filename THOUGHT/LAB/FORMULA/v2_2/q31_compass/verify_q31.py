"""Q31: Phase coherence as a compass — simplified class-level test."""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
import numpy as np
from scipy import stats
torch.manual_seed(42); random.seed(42)

def gen_transforms(n=200):
    X, Y = [], []
    for _ in range(n):
        t = random.randint(0, 3)
        zx = torch.randn(8)*2.0; zy = torch.randn(8)*2.0
        th = random.random()*2*math.pi
        if t==0: c,s=math.cos(th),math.sin(th); ox=zx*c-zy*s; oy=zx*s+zy*c
        elif t==1: c,s=math.cos(th),math.sin(th); ox=zx*c+zy*s; oy=zx*s-zy*c
        elif t==2: sc=0.2+random.random()*3.0; ox=zx*sc; oy=zy*sc
        else: k=random.random()*2-1; ox=zx+k*zy; oy=zy
        X.append((zx,zy,ox,oy)); Y.append(t)
    return X, torch.tensor(Y)

class RealMLP(nn.Module):
    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(6,16), nn.ReLU(), nn.Linear(16,4))
    def forward(self,x): return self.net(x)

    def get_hidden(self, x):
        out = x
        for layer in self.net:
            out = layer(out)
            if isinstance(layer, nn.ReLU): return out.detach().cpu()
        return out.detach().cpu()

def phase_coh_set(embeddings):
    """Phase coherence for a set of embeddings (returns scalar)."""
    x = embeddings; x = x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)
    n = len(x); H = np.zeros((n,n), dtype=np.complex128); z = x + 0j
    for i in range(n):
        for j in range(i,n): v=np.conj(z[i]).dot(z[j]); H[i,j]=v; H[j,i]=np.conj(v)
    ev = np.linalg.eigvalsh(H); ev = np.maximum(ev,1e-15); ev/=ev.sum()
    return 1.0 - (-np.sum(ev*np.log(ev+1e-15)))/math.log(n)

raw, Y = gen_transforms(400)
X_feats = []
for zx,zy,ox,oy in raw:
    z=torch.complex(zx,zy); zp=torch.complex(ox,oy); ratio=zp/(z+1e-8)
    X_feats.append(torch.stack([torch.cos(torch.angle(ratio)).mean(),torch.sin(torch.angle(ratio)).mean(),torch.cos(torch.angle(ratio)).std(),torch.sin(torch.angle(ratio)).std(),torch.abs(ratio).mean(),torch.abs(ratio).std()]))
X = torch.stack(X_feats)

m = RealMLP(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
for e in range(200):
    loss = F.cross_entropy(m(X), Y); opt.zero_grad(); loss.backward(); opt.step()
print(f"Trained: acc={(m(X).argmax(-1)==Y).float().mean():.1%}")
with torch.no_grad(): hidden = m.get_hidden(X).numpy()

# Group by class
class_hiddens = {c: hidden[Y==c] for c in range(4)}

print("\n=== ANGLE 1: Within-class vs cross-class phase coherence ===")
np.random.seed(42)
same_pc = []; diff_pc = []
for _ in range(200):
    c1 = random.randint(0,3); c2 = random.randint(0,3)
    if c1 == c2:
        i1 = random.randint(0, len(class_hiddens[c1])-1)
        i2 = random.randint(0, len(class_hiddens[c1])-1)
        if i1 != i2:
            same_pc.append(phase_coh_set(np.vstack([class_hiddens[c1][i1], class_hiddens[c1][i2]])))
    else:
        i1 = random.randint(0, len(class_hiddens[c1])-1)
        i2 = random.randint(0, len(class_hiddens[c2])-1)
        diff_pc.append(phase_coh_set(np.vstack([class_hiddens[c1][i1], class_hiddens[c2][i2]])))

sa = np.array(same_pc); da = np.array(diff_pc)
print(f"  Same class: {sa.mean():.4f}+/-{sa.std():.4f}")
print(f"  Diff class: {da.mean():.4f}+/-{da.std():.4f}")
t, p = stats.ttest_ind(sa, da)
print(f"  t={t:.1f} p={p:.2e} {'SAME > DIFF (phase compass works!)' if sa.mean()>da.mean() else 'DIFF > SAME'}")

# ANGLE 2: Phase coherence for 5-example sets
print("\n=== ANGLE 2: 5-sample class coherence ===")
for c in range(4):
    h = class_hiddens[c]
    for _ in range(5):
        idx = np.random.choice(len(h), 5, replace=False)
        pc = phase_coh_set(h[idx])
        print(f"  Class {c}: {pc:.4f}")

# ANGLE 3: Cosine similarity baseline at class level
print("\n=== ANGLE 3: Cosine similarity baseline ===")
h_norm = hidden / (np.linalg.norm(hidden, axis=1, keepdims=True) + 1e-12)
for c in range(4):
    h_c = h_norm[Y==c]; cos_mat = h_c @ h_c.T
    tril = cos_mat[np.tril_indices(len(h_c), k=-1)]
    print(f"  Class {c}: mean_cos={tril.mean():.4f}+/-{tril.std():.4f}")

print(f"\n{'='*60}")
print("VERDICT:")
print(f"  Phase coherence compass: {'WORKS' if sa.mean()>da.mean() and p<0.01 else 'WEAK' if sa.mean()>da.mean() else 'FAILS'}")
print(f"  Phase coherence groups same-class pairs better than cross-class? {sa.mean()>da.mean()}")
