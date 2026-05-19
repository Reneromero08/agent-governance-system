"""Geometry: complex ratio classifies transformations.

The cybernetic principle: the representation does the work.
zp/z gives the transform directly. A 2-layer MLP reads the result.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
torch.manual_seed(42); random.seed(42)

def gen(n=800, n_pts=8):
    X, Y = [], []
    for _ in range(n):
        t = random.randint(0, 3)
        zx = torch.randn(n_pts) * 2.0; zy = torch.randn(n_pts) * 2.0
        th = random.random() * 2 * math.pi
        
        if t == 0:  # Rotation
            c, s = math.cos(th), math.sin(th)
            ox = zx*c - zy*s; oy = zx*s + zy*c
        elif t == 1:  # Reflection (conjugate + rotate)
            c, s = math.cos(th), math.sin(th)
            rx, ry = zx, -zy  # conjugate
            ox = rx*c - ry*s; oy = rx*s + ry*c
        elif t == 2:  # Scaling
            sc = 0.2 + random.random() * 3.0
            ox = zx*sc; oy = zy*sc
        else:  # Shear
            k = random.random() * 2 - 1
            ox = zx + k*zy; oy = zy
        
        # Compute complex ratio features per point
        z = torch.complex(zx, zy); zp = torch.complex(ox, oy)
        ratio = zp / (z + 1e-8)
        # Features: per-point [magnitude, real phase, imag phase] -> (n_pts, 3)
        feats = torch.stack([
            torch.abs(ratio),
            torch.cos(torch.angle(ratio)),
            torch.sin(torch.angle(ratio)),
        ], dim=-1)  # (n_pts, 3)
        # Pool: mean and std across points
        feat_vec = torch.cat([feats.mean(0), feats.std(0)])  # (6,)
        X.append(feat_vec)
        Y.append(t)
    return torch.stack(X), torch.tensor(Y)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 4))
    def forward(self, x): return self.net(x)

m = SimpleMLP(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
X, Y = gen(800); vX, vY = gen(200)
names = ["ROTATE","REFLECT","SCALE","SHEAR"]

for e in range(100):
    loss = F.cross_entropy(m(X), Y); opt.zero_grad(); loss.backward(); opt.step()
    if e % 20 == 0:
        with torch.no_grad():
            acc = (m(vX).argmax(-1)==vY).float().mean()
        print("  {:3d}: loss={:.4f}  acc={:.1%}".format(e, loss.item(), acc))

with torch.no_grad():
    vp = m(vX); acc = (vp.argmax(-1)==vY).float().mean()
    print("\nAccuracy: {:.1%}".format(acc))
    for t in range(4):
        mask = vY==t
        if mask.any(): print("  {}: {:.1%}".format(names[t], (vp[mask].argmax(-1)==t).float().mean()))
    print("Verdict: {}".format("COMPLEX RATIO CLASSIFIES GEOMETRY" if acc>0.9 else "not learned"))
