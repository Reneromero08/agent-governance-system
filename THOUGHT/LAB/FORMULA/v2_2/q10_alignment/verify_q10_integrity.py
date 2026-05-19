"""Q10 Integrity: degradation test — does phase drop before accuracy?"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
import numpy as np
torch.manual_seed(42); random.seed(42)

def gen(n=200):
    X, Y = [], []
    for _ in range(n):
        t = random.randint(0, 3)
        zx = torch.randn(8)*2.0; zy = torch.randn(8)*2.0
        th = random.random()*2*math.pi
        if t==0: c,s=math.cos(th),math.sin(th); ox=zx*c-zy*s; oy=zx*s+zy*c
        elif t==1: c,s=math.cos(th),math.sin(th); ox=zx*c+zy*s; oy=zx*s-zy*c
        elif t==2: sc=0.2+random.random()*3.0; ox=zx*sc; oy=zy*sc
        else: k=random.random()*2-1; ox=zx+k*zy; oy=zy
        z=torch.complex(zx,zy); zp=torch.complex(ox,oy); ratio=zp/(z+1e-8)
        X.append(torch.stack([torch.cos(torch.angle(ratio)).mean(),torch.sin(torch.angle(ratio)).mean(),torch.cos(torch.angle(ratio)).std(),torch.sin(torch.angle(ratio)).std(),torch.abs(ratio).mean(),torch.abs(ratio).std()]))
        Y.append(t)
    return torch.stack(X), torch.tensor(Y)

class M(nn.Module):
    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(6,16), nn.ReLU(), nn.Linear(16,4))
    def forward(self, x): return self.net(x)

def met(model, X, Y):
    with torch.no_grad():
        logits = model(X); probs = F.softmax(logits, dim=-1)
        acc = (logits.argmax(-1)==Y).float().mean().item()
        h = -(probs*torch.log(probs+1e-8)).sum(-1)
        coh = 1.0 - h.mean().item()/math.log(4)
    return acc, coh

X_tr, Y_tr = gen(); X_te, Y_te = gen()
m = M(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
for e in range(200):
    loss = F.cross_entropy(m(X_tr), Y_tr); opt.zero_grad(); loss.backward(); opt.step()
acc, coh = met(m, X_te, Y_te)
print(f"Trained: acc={acc:.1%} phase_coh={coh:.4f}")

print(f"\nCorruption recovery: phase leads accuracy?")
for sigma in [0.01, 0.03, 0.05, 0.1, 0.2]:
    results = []
    for trial in range(10):
        m2 = M(); m2.load_state_dict({k: v.clone() for k,v in m.state_dict().items()})
        with torch.no_grad():
            for p in m2.parameters():
                p.add_(torch.randn_like(p) * sigma)
        opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-2)
        ep, ea = None, None
        for e in range(50):
            loss = F.cross_entropy(m2(X_tr), Y_tr); opt2.zero_grad(); loss.backward(); opt2.step()
            a, c = met(m2, X_te, Y_te)
            if ep is None and c > 0.85: ep = e
            if ea is None and a > 0.85: ea = e
        if ep is not None and ea is not None:
            results.append(ea - ep)  # positive = phase recovers first
    if results:
        arr = np.array(results)
        print(f"  sigma={sigma:.2f}: phase_leads by {arr.mean():+.1f}+/-{arr.std():.1f} epochs ({len(results)}/10)")
