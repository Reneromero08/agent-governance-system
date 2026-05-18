"""Phase learns multiplication: 2D complex attention + complex FFN.

(a+bi)(c+di) = (ac-bd) + (ad+bc)i requires cross-terms between
real and imaginary components. A complex FFN with nonlinearity
should learn this. If it does, complex operations are learnable.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
torch.manual_seed(42)

class PhaseMultiplier(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.enc = nn.Linear(4, hidden)
        hh = hidden // 2  # 8
        self.w1_r = nn.Linear(hh, hh, bias=False)
        self.w1_i = nn.Linear(hh, hh, bias=False)
        self.w2_r = nn.Linear(hh, hh, bias=False)
        self.w2_i = nn.Linear(hh, hh, bias=False)
        self.out_r = nn.Linear(hh, 1)
        self.out_i = nn.Linear(hh, 1)
        for w in [self.enc, self.w1_r, self.w1_i, self.w2_r, self.w2_i, self.out_r, self.out_i]:
            nn.init.normal_(w.weight, std=0.1)

    def forward(self, a, b):
        # a, b: (B, 2) real+imag
        x = torch.cat([a, b], dim=-1)  # (B, 4)
        h = self.enc(x)  # (B, 16)
        # Split into real/imag channels
        hr, hi = h[:, :8], h[:, 8:]  # (B, 8), (B, 8)
        # Complex FFN: modReLU
        r1 = self.w1_r(hr) - self.w1_i(hi)
        i1 = self.w1_r(hi) + self.w1_i(hr)
        mag = torch.sqrt(r1**2 + i1**2 + 1e-8)
        gate = F.relu(mag)
        r1g, i1g = r1 * gate / (mag + 1e-8), i1 * gate / (mag + 1e-8)
        # Second complex linear
        r2 = self.w2_r(r1g) - self.w2_i(i1g)
        i2 = self.w2_r(i1g) + self.w2_i(r1g)
        # Output: from complex state (both real and imaginary)
        return torch.cat([self.out_r(r2), self.out_i(i2)], dim=-1)

# Data: integer complex numbers
def gen(n=1000):
    ar = torch.randint(-5, 6, (n,)).float()
    ai = torch.randint(-5, 6, (n,)).float()
    br = torch.randint(-5, 6, (n,)).float()
    bi = torch.randint(-5, 6, (n,)).float()
    a = torch.stack([ar, ai], dim=-1)
    b = torch.stack([br, bi], dim=-1)
    y = torch.stack([ar*br - ai*bi, ar*bi + ai*br], dim=-1)
    return a, b, y

model = PhaseMultiplier()
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
ta, tb, ty = gen(800)
va, vb, vy = gen(200)

print("Phase multiplication — complex FFN")
print("=" * 45)
for epoch in range(100):
    model.train()
    loss = F.mse_loss(model(ta, tb), ty)
    opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(va, vb), vy)
        print("  {:3d}: train={:.4f}  test={:.4f}".format(epoch, loss.item(), vl.item()))

model.eval()
with torch.no_grad():
    pred = model(va, vb)
    acc = ((pred - vy).abs() < 0.5).all(dim=-1).float().mean().item()
    print("\nAccuracy: {:.0%}".format(acc))
    for i in range(4):
        ar,ai,br,bi = float(va[i,0]),float(va[i,1]),float(vb[i,0]),float(vb[i,1])
        pr,pi,tr,ti = float(pred[i,0]),float(pred[i,1]),float(vy[i,0]),float(vy[i,1])
        print("  ({:+.0f}{:+.0f}i)({:+.0f}{:+.0f}i) = {:+.1f}{:+.1f}i  gt={:+.0f}{:+.0f}i  {}".format(
            ar,ai,br,bi,pr,pi,tr,ti,'OK' if abs(pr-tr)<.5 and abs(pi-ti)<.5 else 'XX'))
    print("Verdict: {}".format("COMPLEX OPERATIONS LEARNABLE" if acc > 0.9 else "not learned"))
