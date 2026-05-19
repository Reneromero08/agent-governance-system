"""Q56 Geometric Coupling v2: Independent Q/K/V per head, geometrically initialized.

Key fix from superradiance: each head has its OWN dipole (independent weights).
The geometric coupling is in INITIALIZATION — heads start with structured
angular relationships, then training adds diversity.

Superradiance analog:
  Base dipole = one TuD's tryptophan orientation
  Per-chromophore dipole = base rotated by its position + protein environment
  Heads initialized with correlated Q matrices, maintain independence during training
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
from math import pi, sqrt, cos, sin
torch.manual_seed(42); random.seed(42)


def gen_geometry(n=300, n_pts=8):
    X, Y = [], []
    for _ in range(n):
        t = random.randint(0, 3)
        zx = torch.randn(n_pts) * 2.0; zy = torch.randn(n_pts) * 2.0
        th = random.random() * 2 * pi
        if t == 0:
            c, s = math.cos(th), math.sin(th)
            ox, oy = zx*c - zy*s, zx*s + zy*c
        elif t == 1:
            c, s = math.cos(th), math.sin(th)
            ox, oy = zx*c + zy*s, zx*s - zy*c
        elif t == 2:
            sc = 0.2 + random.random() * 3.0
            ox, oy = zx*sc, zy*sc
        else:
            k = random.random() * 2 - 1
            ox, oy = zx + k*zy, zy
        z = torch.complex(zx, zy); zp = torch.complex(ox, oy)
        ratio = zp / (z + 1e-8)
        feats = torch.stack([
            torch.cos(torch.angle(ratio)).mean(), torch.sin(torch.angle(ratio)).mean(),
            torch.cos(torch.angle(ratio)).std(), torch.sin(torch.angle(ratio)).std(),
            torch.abs(ratio).mean(), torch.abs(ratio).std(),
        ])
        X.append(feats); Y.append(t)
    return torch.stack(X), torch.tensor(Y)


def rotation_matrix_2d(dim, angle):
    R = torch.eye(dim)
    c, s = cos(angle), sin(angle)
    R[0,0]=c; R[0,1]=-s; R[1,0]=s; R[1,1]=c
    return R


class GeomInitBorn(nn.Module):
    """Independent Q/K/V per head, initialized with geometric coupling structure."""
    def __init__(self, in_dim=6, mid_dim=8, total_heads=8, n_classes=4,
                 align_angle=pi/3, noise=0.1):
        super().__init__()
        self.total_heads = total_heads; self.mid_dim = mid_dim
        self.qr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.qi = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.kr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.ki = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.vr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.vi = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])

        # Initialize with geometric coupling: base + per-head rotation + noise
        Qr_base = torch.randn(mid_dim, in_dim) * 0.1
        Qi_base = torch.randn(mid_dim, in_dim) * 0.1
        Kr_base = torch.randn(mid_dim, in_dim) * 0.1
        Ki_base = torch.randn(mid_dim, in_dim) * 0.1
        Vr_base = torch.randn(mid_dim, in_dim) * 0.1
        Vi_base = torch.randn(mid_dim, in_dim) * 0.1

        for h in range(total_heads):
            angle = h * align_angle
            R_mat = rotation_matrix_2d(mid_dim, angle)
            self.qr[h].weight.data = (R_mat @ Qr_base + torch.randn_like(Qr_base) * noise)
            self.qi[h].weight.data = (R_mat @ Qi_base + torch.randn_like(Qi_base) * noise)
            angle_k = h * align_angle + pi/6
            Rk = rotation_matrix_2d(mid_dim, angle_k)
            self.kr[h].weight.data = (Rk @ Kr_base + torch.randn_like(Kr_base) * noise)
            self.ki[h].weight.data = (Rk @ Ki_base + torch.randn_like(Ki_base) * noise)
            self.vr[h].weight.data = Vr_base + torch.randn_like(Vr_base) * noise
            self.vi[h].weight.data = Vi_base + torch.randn_like(Vi_base) * noise

        angles = torch.linspace(0, 2*pi, total_heads+1)[:total_heads]
        self.phases = nn.ParameterList([nn.Parameter(a.unsqueeze(0).expand(1)) for a in angles])
        self.align = nn.Parameter(torch.randn(mid_dim, n_classes)*0.1)

    def forward(self, x):
        B = x.shape[0]
        sum_r = torch.zeros(B, self.mid_dim, device=x.device)
        sum_i = torch.zeros_like(sum_r)
        for h in range(self.total_heads):
            qr, qi = self.qr[h](x), self.qi[h](x)
            kr, ki = self.kr[h](x), self.ki[h](x)
            vr, vi = self.vr[h](x), self.vi[h](x)
            score_i = (qi*kr - qr*ki).sum(-1)
            c, s = torch.cos(score_i.unsqueeze(-1)), torch.sin(score_i.unsqueeze(-1))
            zr, zi = c*vr - s*vi, c*vi + s*vr
            c2, s2 = torch.cos(self.phases[h]), torch.sin(self.phases[h])
            sum_r += (zr*c2 - zi*s2) / sqrt(self.total_heads)
            sum_i += (zr*s2 + zi*c2) / sqrt(self.total_heads)
        return (sum_r @ self.align)**2 + (sum_i @ self.align)**2


class RandomBorn(nn.Module):
    """Standard independent heads with fully random initialization (baseline)."""
    def __init__(self, in_dim=6, mid_dim=8, total_heads=8, n_classes=4):
        super().__init__()
        self.total_heads = total_heads; self.mid_dim = mid_dim
        self.qr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.qi = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.kr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.ki = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.vr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.vi = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        angles = torch.linspace(0, 2*pi, total_heads+1)[:total_heads]
        self.phases = nn.ParameterList([nn.Parameter(a.unsqueeze(0).expand(1)) for a in angles])
        self.align = nn.Parameter(torch.randn(mid_dim, n_classes)*0.1)
        for mlist in [self.qr, self.qi, self.kr, self.ki, self.vr, self.vi]:
            for w in mlist: nn.init.normal_(w.weight, std=0.1)

    def forward(self, x):
        B = x.shape[0]
        sum_r = torch.zeros(B, self.mid_dim, device=x.device)
        sum_i = torch.zeros_like(sum_r)
        for h in range(self.total_heads):
            qr, qi = self.qr[h](x), self.qi[h](x)
            kr, ki = self.kr[h](x), self.ki[h](x)
            vr, vi = self.vr[h](x), self.vi[h](x)
            score_i = (qi*kr - qr*ki).sum(-1)
            c, s = torch.cos(score_i.unsqueeze(-1)), torch.sin(score_i.unsqueeze(-1))
            zr, zi = c*vr - s*vi, c*vi + s*vr
            c2, s2 = torch.cos(self.phases[h]), torch.sin(self.phases[h])
            sum_r += (zr*c2 - zi*s2) / sqrt(self.total_heads)
            sum_i += (zr*s2 + zi*c2) / sqrt(self.total_heads)
        return (sum_r @ self.align)**2 + (sum_i @ self.align)**2


def train_eval(model, X_tr, Y_tr, X_te, Y_te, epochs=120):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    n_p = sum(p.numel() for p in model.parameters())
    for e in range(epochs):
        loss = F.cross_entropy(model(X_tr), Y_tr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        logits = model(X_te)
        acc = (logits.argmax(-1) == Y_te).float().mean().item()
        saved = {}
        for n, p in model.named_parameters():
            if 'qi' in n or 'ki' in n or 'phases' in n:
                saved[n] = p.data.clone(); p.data.zero_()
        ab_logits = model(X_te)
        ab_acc = (ab_logits.argmax(-1) == Y_te).float().mean().item()
        for n, p in model.named_parameters():
            if n in saved: p.data.copy_(saved[n])
    return acc, ab_acc, acc-ab_acc, n_p


X_train, Y_train = gen_geometry(400)
X_test, Y_test = gen_geometry(200)
mid_dim = 8
h = 8

print("=" * 80)
print("GEOMETRIC INITIALIZATION: Independent heads, correlated starts")
print("=" * 80)

# ---- Angle sweep with independent heads ----
print(f"\nATTACK 1: Geometric init angle sweep (h={h}, independent Q/K/V)")
print(f"  Superradiance optimal: ~60 deg")
print(f"  {'angle':>8}  {'acc':>7}  {'ab_acc':>7}  {'delta':>7}  {'params':>8}")
print("  " + "-" * 50)

best_delta, best_angle = 0, 0
for angle_deg in [0, 30, 60, 90, 120]:
    angle = pi * angle_deg / 180
    m = GeomInitBorn(mid_dim=mid_dim, total_heads=h, align_angle=angle, noise=0.05)
    acc, ab_acc, delta, n_p = train_eval(m, X_train, Y_train, X_test, Y_test)
    if delta > best_delta:
        best_delta, best_angle = delta, angle_deg
    marker = " <-- BEST" if delta == best_delta else ""
    print(f"  {angle_deg:>6}deg  {acc:>7.1%}  {ab_acc:>7.1%}  {delta:>+6.1%}  {n_p:>8}{marker}")

m_r = RandomBorn(mid_dim=mid_dim, total_heads=h)
acc_r, ab_acc_r, delta_r, n_p_r = train_eval(m_r, X_train, Y_train, X_test, Y_test)
print(f"  {'random':>8}  {acc_r:>7.1%}  {ab_acc_r:>7.1%}  {delta_r:>+6.1%}  {n_p_r:>8}  <-- BASELINE")

# ---- Noise level in geometric init ----
print(f"\nATTACK 2: Init noise level (angle={best_angle}deg)")
print(f"  {'noise':>8}  {'acc':>7}  {'ab_acc':>7}  {'delta':>7}")
print("  " + "-" * 40)
for noise in [0.0, 0.01, 0.05, 0.1, 0.5]:
    m = GeomInitBorn(mid_dim=mid_dim, total_heads=h, align_angle=pi*best_angle/180, noise=noise)
    acc, ab_acc, delta, _ = train_eval(m, X_train, Y_train, X_test, Y_test)
    marker = " <-- ZERO NOISE" if noise == 0 else (" <-- SUPER CLEAN" if noise <= 0.01 else "")
    print(f"  {noise:>8.3f}  {acc:>7.1%}  {ab_acc:>7.1%}  {delta:>+6.1%}{marker}")

# ---- Q-K angular offset sweep ----
print(f"\nATTACK 3: Q-K offset angle (geom init, angle={best_angle}deg)")
offsets = [0, pi/6, pi/4, pi/3, pi/2]
print(f"  {'offset':>8}  {'acc':>7}  {'ab_acc':>7}  {'delta':>7}")
print("  " + "-" * 40)
for off in offsets:
    m = GeomInitBorn(mid_dim=mid_dim, total_heads=h, align_angle=pi*best_angle/180, noise=0.05)
    # Override K initialization with Q-K offset
    Kr_base = torch.randn(mid_dim, 6) * 0.1
    Ki_base = torch.randn(mid_dim, 6) * 0.1
    for hh in range(h):
        Rk = rotation_matrix_2d(mid_dim, hh * pi*best_angle/180 + off)
        m.kr[hh].weight.data = Rk @ Kr_base + torch.randn_like(Kr_base) * 0.05
        m.ki[hh].weight.data = Rk @ Ki_base + torch.randn_like(Ki_base) * 0.05
    acc, ab_acc, delta, _ = train_eval(m, X_train, Y_train, X_test, Y_test)
    off_deg = int(off * 180 / pi)
    print(f"  {off_deg:>6}deg  {acc:>7.1%}  {ab_acc:>7.1%}  {delta:>+6.1%}")

# ---- Per-head decay with best config ----
print(f"\nATTACK 4: Per-head decay (geom init, angle={best_angle}deg, noise=0.01)")
print(f"  Superradiance target: per-chr drops 4.3x (single MT -> centriole)")
print(f"  {'h':>4}  {'geom-init':>10}  {'g/h':>7}  {'random':>10}  {'r/h':>7}")
print("  " + "-" * 50)
g_deltas = {}; r_deltas = {}
for hh in [2, 4, 8, 16]:
    m_g = GeomInitBorn(mid_dim=mid_dim, total_heads=hh, align_angle=pi*best_angle/180, noise=0.01)
    acc_g, ab_acc_g, dg, _ = train_eval(m_g, X_train, Y_train, X_test, Y_test, epochs=80)
    g_deltas[hh] = dg
    m_rr = RandomBorn(mid_dim=mid_dim, total_heads=hh)
    acc_rr, ab_acc_rr, dr, _ = train_eval(m_rr, X_train, Y_train, X_test, Y_test, epochs=80)
    r_deltas[hh] = dr
    print(f"  {hh:>4}  {dg:>+9.1%}  {dg/hh:>6.3f}  {dr:>+9.1%}  {dr/hh:>6.3f}")

g_decay = (g_deltas[2]/2) / max(g_deltas[16]/16, 1e-6) if g_deltas[16] > 0 else float('inf')
r_decay = (r_deltas[2]/2) / max(r_deltas[16]/16, 1e-6) if r_deltas[16] > 0 else float('inf')
print(f"\n  Per-head decay: geom={g_decay:.1f}x  rand={r_decay:.1f}x  target=4.3x")

# ---- Final: best config summary ----
print(f"\nATTACK 5: Best config (angle={best_angle}deg, noise=0.01, {h} heads)")
m_best = GeomInitBorn(mid_dim=mid_dim, total_heads=h, align_angle=pi*best_angle/180, noise=0.01)
acc_b, ab_acc_b, delta_b, n_pb = train_eval(m_best, X_train, Y_train, X_test, Y_test, epochs=150)
m_rand = RandomBorn(mid_dim=mid_dim, total_heads=h)
acc_r2, ab_acc_r2, delta_r2, n_pr2 = train_eval(m_rand, X_train, Y_train, X_test, Y_test, epochs=150)
print(f"  Geom init: acc={acc_b:.1%}  delta={delta_b:+.1%}")
print(f"  Random:    acc={acc_r2:.1%}  delta={delta_r2:+.1%}")
delta_gap = delta_b - delta_r2
if delta_gap > 0.02:
    print(f"  GEOMETRIC INIT WINS: +{delta_gap:+.1%} delta")
elif delta_gap > 0:
    print(f"  Weak geometric advantage: +{delta_gap:+.1%}")
else:
    print(f"  Random init wins at this scale")
