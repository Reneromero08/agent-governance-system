"""C^8 scaling: 8 independent complex multiplications via phase attention.

Does the phase architecture scale from 1 channel (C^1, proven) to 8 channels?
Each channel independently computes (a_i + i*b_i) * (c_i + i*d_i).
The model must route 8 parallel computations through shared complex attention.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
torch.manual_seed(42); random.seed(42)

class C8Multiplier(nn.Module):
    def __init__(self):
        super().__init__()
        C = 8
        self.enc_qr = nn.Linear(32, C, bias=False); self.enc_qi = nn.Linear(32, C, bias=False)
        self.enc_kr = nn.Linear(32, C, bias=False); self.enc_ki = nn.Linear(32, C, bias=False)
        self.enc_vr = nn.Linear(32, C, bias=False); self.enc_vi = nn.Linear(32, C, bias=False)
        # Per-channel output scaling (diagonal — no cross-channel mixing)
        self.scale_r = nn.Parameter(torch.ones(C))
        self.scale_i = nn.Parameter(torch.zeros(C))
        self.phase = nn.Parameter(torch.ones(C) * 0.1)
        for w in [self.enc_qr, self.enc_qi, self.enc_kr, self.enc_ki, self.enc_vr, self.enc_vi]:
            nn.init.normal_(w.weight, std=0.1)

    def forward(self, x):
        qr = self.enc_qr(x); qi = self.enc_qi(x)
        kr = self.enc_kr(x); ki = self.enc_ki(x)
        vr = self.enc_vr(x); vi = self.enc_vi(x)
        score_i = qi * kr - qr * ki
        c, s = torch.cos(score_i), torch.sin(score_i)
        zr = c * vr - s * vi
        zi = c * vi + s * vr
        c2, s2 = torch.cos(self.phase), torch.sin(self.phase)
        zr = zr * c2 - zi * s2
        zi = zr * s2 + zi * c2
        # Per-channel scaling (no cross-channel mixing)
        return zr * self.scale_r - zi * self.scale_i, zr * self.scale_i + zi * self.scale_r

# Data: 8 independent complex multiplications
def gen(n=2000):
    X, Yr, Yi = [], [], []
    for _ in range(n):
        inp, out_r, out_i = [], [], []
        for _ in range(8):
            ar = random.uniform(-5, 5); ai = random.uniform(-5, 5)
            br = random.uniform(-5, 5); bi = random.uniform(-5, 5)
            inp.extend([ar, ai, br, bi])
            out_r.append(ar*br - ai*bi)
            out_i.append(ar*bi + ai*br)
        X.append(inp); Yr.append(out_r); Yi.append(out_i)
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(Yr, dtype=torch.float32),
            torch.tensor(Yi, dtype=torch.float32))

X, Yr, Yi = gen(2000)
X_tr, Yr_tr, Yi_tr = X[:1500], Yr[:1500], Yi[:1500]
X_te, Yr_te, Yi_te = X[1500:], Yr[1500:], Yi[1500:]

m = C8Multiplier(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
print("C^8 Scaling: {} params, 8 independent complex multiplications".format(
    sum(p.numel() for p in m.parameters())))
print("=" * 55)

for e in range(300):
    pr, pi = m(X_tr)
    loss = F.mse_loss(pr, Yr_tr) + F.mse_loss(pi, Yi_tr)
    opt.zero_grad(); loss.backward(); opt.step()
    if e % 60 == 0:
        with torch.no_grad():
            pr_t, pi_t = m(X_te)
            mae = ((pr_t - Yr_te).abs().mean() + (pi_t - Yi_te).abs().mean()).item()
        print("  {:3d}: loss={:.4f}  mae={:.3f}".format(e, loss.item(), mae))

with torch.no_grad():
    pr_t, pi_t = m(X_te)
    mae = ((pr_t - Yr_te).abs().mean() + (pi_t - Yi_te).abs().mean()).item()
    ok = ((pr_t - Yr_te).abs() < 1.0) & ((pi_t - Yi_te).abs() < 1.0)
    acc = ok.float().mean().item()
    print("\nMAE: {:.3f}  Accuracy (<1.0): {:.1%}".format(mae, acc))
    print("Verdict: {}".format("C^8 PHASE SCALES" if acc > 0.9 else
          "PARTIAL" if acc > 0.5 else "phase doesn't scale"))
    for i in range(3):
        for c in range(3):
            idx = i*16 + c*4
            ar, ai = X_te[i, idx], X_te[i, idx+1]
            br, bi = X_te[i, idx+2], X_te[i, idx+3]
            pr, pi = float(pr_t[i, c]), float(pi_t[i, c])
            tr, ti = float(Yr_te[i, c]), float(Yi_te[i, c])
            ok_s = abs(pr-tr) < 1.0 and abs(pi-ti) < 1.0
            print("  ({:+.0f}{:+.0f}i)({:+.0f}{:+.0f}i) = {:+.1f}{:+.1f}i  gt={:+.0f}{:+.0f}i  {}".format(
                ar,ai,br,bi,pr,pi,tr,ti,'OK' if ok_s else 'XX'))
