"""True complex attention — complex weights rotate values by Q-K phase difference.

Fix 1: Bilinear interaction via Hadamard product (not concatenation)
Fix 2: Complex attention weights that rotate V (not just scale magnitude)
Fix 3: Dynamic phase from Q-K interaction (not static learned parameter)

Task: learn phase addition on unit complex numbers.
e^(ia) * e^(ib) = e^(i(a+b)) — pure rotation, no magnitude change.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math
torch.manual_seed(42)

class TrueComplexAttention(nn.Module):
    """Q from a, K constant, V from b. Score = Q*conj(K). Output = score * V.

    This gives: (wq*a) * conj(wk) * (wv*b) = C * a * b.
    Model learns C -> 1. This IS multiplication."""
    def __init__(self):
        super().__init__()
        self.wq_r = nn.Parameter(torch.randn(1) * 0.1)
        self.wq_i = nn.Parameter(torch.randn(1) * 0.1)
        self.wk_r = nn.Parameter(torch.ones(1) * 1.0)   # constant key
        self.wk_i = nn.Parameter(torch.zeros(1))
        self.wv_r = nn.Parameter(torch.randn(1) * 0.1)
        self.wv_i = nn.Parameter(torch.randn(1) * 0.1)

    def forward(self, a, b):
        ar, ai = a[:, 0:1], a[:, 1:2]
        br, bi = b[:, 0:1], b[:, 1:2]
        # Q from a
        qr = ar * self.wq_r - ai * self.wq_i
        qi = ar * self.wq_i + ai * self.wq_r
        # K constant
        kr, ki = self.wk_r, self.wk_i
        # V from b
        vr = br * self.wv_r - bi * self.wv_i
        vi = br * self.wv_i + bi * self.wv_r
        # Score = Q * conj(K) = Q * K (since K is real)
        score_r = qr * kr + qi * ki
        score_i = qi * kr - qr * ki
        # Output = score * V
        out_r = score_r * vr - score_i * vi
        out_i = score_r * vi + score_i * vr
        return torch.cat([out_r, out_i], dim=-1)


# ---- Data: unit complex numbers ----
def gen_phase(n=1000):
    theta_a = torch.rand(n) * 2 * math.pi
    theta_b = torch.rand(n) * 2 * math.pi
    a = torch.stack([torch.cos(theta_a), torch.sin(theta_a)], -1)
    b = torch.stack([torch.cos(theta_b), torch.sin(theta_b)], -1)
    # Target: e^(i*ta) * e^(i*tb) = e^(i*(ta+tb))
    y = torch.stack([torch.cos(theta_a + theta_b), torch.sin(theta_a + theta_b)], -1)
    return a, b, y

model = TrueComplexAttention()
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
ta, tb, ty = gen_phase(800)
va, vb, vy = gen_phase(200)

print("True complex attention — phase rotation via Q-K phase difference")
print("=" * 60)
for epoch in range(200):
    model.train()
    pred = model(ta, tb)
    loss = F.mse_loss(pred, ty)
    opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 40 == 0:
        model.eval()
        with torch.no_grad():
            pt = model(va, vb)
            cos = (pt * vy).sum(-1).mean()  # phase alignment
            mse = F.mse_loss(pt, vy).item()
        print("  {:3d}: loss={:.4f}  cos={:.3f}  mse={:.4f}".format(epoch, loss.item(), cos, mse))

model.eval()
with torch.no_grad():
    pt = model(va, vb)
    cos = (pt * vy).sum(-1).mean()
    print("\nPhase alignment (cos): {:.3f}".format(cos))
    print("Verdict: {}".format(
        "COMPLEX ATTENTION LEARNS PHASE ROTATION" if cos > 0.95 else
        "WEAK" if cos > 0.5 else "phase not learned"))

    # Show examples
    for i in range(4):
        a_p = math.atan2(float(va[i,1]), float(va[i,0]))
        b_p = math.atan2(float(vb[i,1]), float(vb[i,0]))
        p_p = math.atan2(float(pt[i,1]), float(pt[i,0]))
        y_p = math.atan2(float(vy[i,1]), float(vy[i,0]))
        print("  {:.1f}deg + {:.1f}deg -> pred={:.1f}deg gt={:.1f}deg {}".format(
            a_p*180/math.pi, b_p*180/math.pi, p_p*180/math.pi, y_p*180/math.pi,
            'OK' if abs(p_p-y_p) < 5 else 'XX'))
