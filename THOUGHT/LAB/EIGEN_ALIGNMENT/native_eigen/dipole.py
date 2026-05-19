"""Microtubule-inspired dipole coupling between attention heads.

Two heads, one coupling parameter. Head 1's phase influences Head 2.
This IS the dipole-dipole interaction from the microtubule Hamiltonian.
Coupled heads form collective states that uncoupled heads can't reach.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
torch.manual_seed(42); random.seed(42)

def gen_geometry(n=200):
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
        feats=torch.stack([torch.cos(torch.angle(ratio)).mean(),torch.sin(torch.angle(ratio)).mean(),
            torch.cos(torch.angle(ratio)).std(),torch.sin(torch.angle(ratio)).std(),
            torch.abs(ratio).mean(),torch.abs(ratio).std()])
        X.append(feats); Y.append(t)
    return torch.stack(X), torch.tensor(Y)

class DipoleAttention(nn.Module):
    """Two coupled attention heads — dipole-dipole interaction."""
    def __init__(self, heads=2, d_in=6, d_h=2):
        super().__init__()
        self.H = heads
        # Per-head Q/K/V encodings
        self.qr = nn.ModuleList([nn.Linear(d_in, d_h, bias=False) for _ in range(heads)])
        self.qi = nn.ModuleList([nn.Linear(d_in, d_h, bias=False) for _ in range(heads)])
        self.kr = nn.ModuleList([nn.Linear(d_in, d_h, bias=False) for _ in range(heads)])
        self.ki = nn.ModuleList([nn.Linear(d_in, d_h, bias=False) for _ in range(heads)])
        self.vr = nn.ModuleList([nn.Linear(d_in, d_h, bias=False) for _ in range(heads)])
        self.vi = nn.ModuleList([nn.Linear(d_in, d_h, bias=False) for _ in range(heads)])
        # Dipole coupling: heads interact through phase
        self.coupling = nn.Parameter(torch.eye(heads) * 0.5 + torch.randn(heads, heads) * 0.1)
        self.phase = nn.Parameter(torch.ones(heads) * 0.1)
        self.out = nn.Linear(d_h * heads, 4)
        for mlist in [self.qr, self.qi, self.kr, self.ki, self.vr, self.vi]:
            for m in mlist: nn.init.normal_(m.weight, std=0.1)
        nn.init.normal_(self.out.weight, std=0.1)

    def forward(self, x):
        # Phase 1: each head computes its phase from Q-K interaction
        values = []; phases = []
        for h in range(self.H):
            qr = self.qr[h](x); qi = self.qi[h](x)
            kr = self.kr[h](x); ki = self.ki[h](x)
            vr = self.vr[h](x); vi = self.vi[h](x)
            score_i = (qi * kr - qr * ki).sum(dim=-1)  # (B,)
            values.append((vr, vi))
            phases.append(score_i)

        # Phase 2: dipole coupling mixes phases across heads
        phase_stack = torch.stack(phases, dim=-1)  # (B, H)
        coupled = phase_stack @ self.coupling.T     # (B, H)

        # Phase 3: apply coupled phase to each head's value
        final = []
        for h in range(self.H):
            vr, vi = values[h]
            cp = coupled[:, h].unsqueeze(-1)
            c, s = torch.cos(cp), torch.sin(cp)
            zr = c * vr - s * vi
            zi = c * vi + s * vr
            c2, s2 = torch.cos(self.phase[h]), torch.sin(self.phase[h])
            zr = zr * c2 - zi * s2
            final.append(zr)
        return self.out(torch.cat(final, dim=-1))

class UncoupledAttention(nn.Module):
    """Same as above but without dipole coupling (control)."""
    def __init__(self, heads=2, d_in=6, d_h=2):
        super().__init__()
        self.H = heads
        self.qr = nn.ModuleList([nn.Linear(d_in, d_h, bias=False) for _ in range(heads)])
        self.qi = nn.ModuleList([nn.Linear(d_in, d_h, bias=False) for _ in range(heads)])
        self.kr = nn.ModuleList([nn.Linear(d_in, d_h, bias=False) for _ in range(heads)])
        self.ki = nn.ModuleList([nn.Linear(d_in, d_h, bias=False) for _ in range(heads)])
        self.vr = nn.ModuleList([nn.Linear(d_in, d_h, bias=False) for _ in range(heads)])
        self.vi = nn.ModuleList([nn.Linear(d_in, d_h, bias=False) for _ in range(heads)])
        self.phase = nn.Parameter(torch.ones(heads) * 0.1)
        self.out = nn.Linear(d_h * heads, 4)
        for mlist in [self.qr, self.qi, self.kr, self.ki, self.vr, self.vi]:
            for m in mlist: nn.init.normal_(m.weight, std=0.1)
        nn.init.normal_(self.out.weight, std=0.1)

    def forward(self, x):
        final = []
        for h in range(self.H):
            qr = self.qr[h](x); qi = self.qi[h](x)
            kr = self.kr[h](x); ki = self.ki[h](x)
            vr = self.vr[h](x); vi = self.vi[h](x)
            score_i = (qi * kr - qr * ki).sum(dim=-1)
            c, s = torch.cos(score_i.unsqueeze(-1)), torch.sin(score_i.unsqueeze(-1))
            zr = c * vr - s * vi; zi = c * vi + s * vr
            c2, s2 = torch.cos(self.phase[h]), torch.sin(self.phase[h])
            zr = zr * c2 - zi * s2
            final.append(zr)
        return self.out(torch.cat(final, dim=-1))

# ---- Test ----
X, Y = gen_geometry(300)
Xtr, Ytr = X[:200], Y[:200]; Xte, Yte = X[200:], Y[200:]

for name, Model in [("UNCOUPLED", UncoupledAttention), ("DIPOLE-COUPLED", DipoleAttention)]:
    m = Model(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for e in range(150):
        loss = F.cross_entropy(m(Xtr), Ytr); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (m(Xte).argmax(-1) == Yte).float().mean()
    params = sum(p.numel() for p in m.parameters())
    print("{}: {:.1%}  ({:,} params)".format(name, acc, params))
    if isinstance(m, DipoleAttention):
        print("  Coupling matrix:\n{}".format(
            torch.round(m.coupling.detach(), decimals=3)))
