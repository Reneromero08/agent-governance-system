"""Composition: pure rotation. Phase adds. Model sees full path.

z0 -> e^(i*th1)*z0 -> e^(i*th2)*(e^(i*th1)*z0) = e^(i*(th1+th2))*z0
The model observes z0, z1 (after T1), z2 (after T2) and learns:
  phase(combined) = phase(T1) + phase(T2)
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, cmath
torch.manual_seed(42); random.seed(42)

def gen(n=1000, n_pts=4):
    X, Y = [], []
    for _ in range(n):
        th1 = random.random() * 2*math.pi
        th2 = random.random() * 2*math.pi
        zx = torch.randn(n_pts) * 2.0; zy = torch.randn(n_pts) * 2.0
        # T1
        ox1 = zx*math.cos(th1) - zy*math.sin(th1)
        oy1 = zx*math.sin(th1) + zy*math.cos(th1)
        # T2
        ox2 = ox1*math.cos(th2) - oy1*math.sin(th2)
        oy2 = ox1*math.sin(th2) + oy1*math.cos(th2)
        # Input: full path — z0, z1, z2 as complex numbers (per-point phase + mag)
        feat = []
        for i in range(n_pts):
            z0 = complex(zx[i], zy[i])
            z1 = complex(ox1[i], oy1[i])
            z2 = complex(ox2[i], oy2[i])
            # feature per point: [mag(z0), phase(z0), mag(z1), phase(z1), mag(z2), phase(z2)]
            feat.extend([abs(z0), cmath.phase(z0), abs(z1), cmath.phase(z1), abs(z2), cmath.phase(z2)])
        X.append(torch.tensor(feat, dtype=torch.float32))
        # Target: combined phase th1+th2, as cos/sin
        total = th1 + th2
        Y.append(torch.tensor([math.cos(total), math.sin(total)]))
    return torch.stack(X), torch.stack(Y)

class Composer(nn.Module):
    def __init__(self, n_pts=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_pts*6, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x):
        out = self.net(x)
        return out / (out.norm(dim=-1, keepdim=True) + 1e-8)

m = Composer(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
X, Y = gen(1000); vX, vY = gen(200)

print("Composition: phase(th1+th2) from full path observation")
print("=" * 55)
for e in range(200):
    pred = m(X)
    loss = 1.0 - (pred * Y).sum(-1).mean()  # 1 - cos(delta)
    opt.zero_grad(); loss.backward(); opt.step()
    if e % 40 == 0:
        with torch.no_grad():
            vp = m(vX)
            pa = torch.atan2(vp[:,1], vp[:,0]) * 180/math.pi
            ta = torch.atan2(vY[:,1], vY[:,0]) * 180/math.pi
            err = (pa - ta).abs().mean()
        print("  {:3d}: loss={:.4f}  err={:.1f}deg".format(e, loss.item(), float(err)))

with torch.no_grad():
    vp = m(vX)
    pa = torch.atan2(vp[:,1], vp[:,0])*180/math.pi
    ta = torch.atan2(vY[:,1], vY[:,0])*180/math.pi
    err = (pa - ta).abs().mean()
    print("\nError: {:.1f}deg".format(float(err)))
    print("Verdict: {}".format("PHASE ACCUMULATES ACROSS LAYERS" if err < 5 else "not learned"))
    for i in range(3):
        print("  th1+th2 = {:+.0f}deg  pred = {:+.0f}deg".format(float(ta[i]), float(pa[i])))
