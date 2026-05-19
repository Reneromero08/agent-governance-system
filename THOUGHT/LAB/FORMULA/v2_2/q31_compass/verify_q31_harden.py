"""Q31: Hilbert-complexified 5-sample phase coherence compass."""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
import numpy as np
from scipy.signal import hilbert
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
    def get_relu(self, x):
        out = x
        for layer in self.net:
            out = layer(out)
            if isinstance(layer, nn.ReLU): return out.detach().cpu()
        return out.detach().cpu()
        out = x
        for layer in self.net:
            out = layer(out)
            if isinstance(layer, nn.ReLU): return out.detach().cpu()
        return out.detach().cpu()

def phase_coh_complex(x_real):
    """Hilbert-complexified phase coherence for a set of embeddings."""
    x = x_real - x_real.mean(axis=0)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    z = hilbert(x, axis=0).astype(np.complex128)
    zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)
    z = z / np.abs(z + 1e-15)
    n = len(z); H = np.zeros((n,n), dtype=np.complex128)
    for i in range(n):
        for j in range(i,n): v=np.conj(z[i]).dot(z[j]); H[i,j]=v; H[j,i]=np.conj(v)
    ev = np.linalg.eigvalsh(H); ev = np.maximum(ev,1e-15); ev/=ev.sum()
    return 1.0 - (-np.sum(ev*np.log(ev+1e-15)))/math.log(n)

def train_model(seed):
    torch.manual_seed(seed); random.seed(seed)
    raw, Y = gen_transforms(300)
    X_feats = []
    for zx,zy,ox,oy in raw:
        z=torch.complex(zx,zy); zp=torch.complex(ox,oy); ratio=zp/(z+1e-8)
        X_feats.append(torch.stack([torch.cos(torch.angle(ratio)).mean(),torch.sin(torch.angle(ratio)).mean(),torch.cos(torch.angle(ratio)).std(),torch.sin(torch.angle(ratio)).std(),torch.abs(ratio).mean(),torch.abs(ratio).std()]))
    X = torch.stack(X_feats)
    m = RealMLP(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for e in range(200):
        loss = F.cross_entropy(m(X), Y); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): hidden = m.get_relu(X).numpy()
    return hidden, Y.numpy()

print("Q31: 5-sample Hilbert phase coherence compass")
print("=" * 60)

hidden, Y = train_model(0)
h_norm = hidden / (np.linalg.norm(hidden, axis=1, keepdims=True) + 1e-12)
cos_mat = h_norm @ h_norm.T
n = len(hidden)
class_h = {c: hidden[Y==c] for c in range(4)}

# ANGLE 1: 5-sample same vs mixed class
print("\n=== ANGLE 1: 5-sample class coherence ===")
np.random.seed(42)
same, diff = [], []
for _ in range(50):
    c1 = random.randint(0,3)
    if len(class_h[c1]) >= 5:
        idx = np.random.choice(len(class_h[c1]), 5, replace=False)
        same.append(phase_coh_complex(class_h[c1][idx]))
    c2 = (c1 + random.randint(1,3)) % 4
    if len(class_h[c1]) >= 3 and len(class_h[c2]) >= 2:
        idx1 = np.random.choice(len(class_h[c1]), 3, replace=False)
        idx2 = np.random.choice(len(class_h[c2]), 2, replace=False)
        diff.append(phase_coh_complex(np.vstack([class_h[c1][idx1], class_h[c2][idx2]])))
sa = np.mean(same); da = np.mean(diff)
print(f"  Same class: {sa:.4f}, Mixed class: {da:.4f}, Ratio: {sa/da:.1f}x")

# ANGLE 2: Anchor classification
print("\n=== ANGLE 2: 5-sample anchor classification (200 trials) ===")
anchors = {c: class_h[c][np.random.choice(len(class_h[c]),5,replace=False)] for c in range(4) if len(class_h[c])>=5}
pc_accs = []; cos_accs = []
for trial in range(200):
    i = random.randint(0, n-1)
    pc_scores = []
    for c in range(4):
        if c in anchors:
            qs = np.vstack([hidden[i].reshape(1,-1), anchors[c]])
            pc_scores.append(phase_coh_complex(qs))
        else: pc_scores.append(0)
    pc_accs.append(1 if Y[i] == np.argmax(pc_scores) else 0)

    cos_scores = []
    for c in range(4):
        c_idx = np.where(Y == c)[0]
        if len(c_idx) >= 5:
            cos_scores.append(np.mean(cos_mat[i, c_idx[:5]]))
        else: cos_scores.append(0)
    cos_accs.append(1 if Y[i] == np.argmax(cos_scores) else 0)

pa = np.mean(pc_accs); ca = np.mean(cos_accs)
print(f"  Phase anchor acc: {pa:.1%}")
print(f"  Cosine anchor acc: {ca:.1%}")
print(f"  {'PHASE BEATS COSINE' if pa > ca else 'COSINE BEATS PHASE' if ca > pa else 'TIE'}")

print(f"\n{'='*60}")
print(f"Phase: {pa:.1%} | Cosine: {ca:.1%} | Same/Mixed ratio: {sa/da:.1f}x")
print(f"Winner: {'PHASE' if pa > ca else 'COSINE'}")
