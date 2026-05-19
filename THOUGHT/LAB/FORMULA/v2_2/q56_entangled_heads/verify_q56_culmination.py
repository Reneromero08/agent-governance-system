"""Q55+Q56 Culmination: Full Cybernetic Loop + Fibonacci Torus.

ARCHITECTURES TESTED:
  1. FLAT-BORN: Static Born rule merge (Q56 baseline)
  2. CYBER-C:    Alignment C tracks output (Attack 3)
  3. CYBER-T:    Per-head temperature from phase coherence
  4. CYBER-CT:   C tracking + per-head temperature (FULL LOOP)
  5. CYBER-CTF:  Full loop with Fibonacci-spiral phase seeding

Combined with Q55 independent heads + Q56 Born rule merge.
Plus: depth sweep (1L, 2L, 3L) on the full loop.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
from math import pi, sqrt, cos, sin
torch.manual_seed(42); random.seed(42)


def gen_geometry(n=300, n_pts=8, noise=0.0, n_classes=4):
    X, Y = [], []
    for _ in range(n):
        t = random.randint(0, n_classes-1)
        zx = torch.randn(n_pts) * (2.0 + noise)
        zy = torch.randn(n_pts) * (2.0 + noise)
        th = random.random() * 2 * pi
        if n_classes == 4:
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
        else:
            if t < n_classes:
                angle_base = t * 2*pi / n_classes
                c, s = math.cos(angle_base + th*0.3), math.sin(angle_base + th*0.3)
                sc = 0.3 + random.random() * 2.7
                ox, oy = zx*c*sc - zy*s*sc, zx*s*sc + zy*c*sc
            else:
                ox, oy = zx, zy

        z = torch.complex(zx, zy); zp = torch.complex(ox, oy)
        ratio = zp / (z + 1e-8)
        feats = torch.stack([
            torch.cos(torch.angle(ratio)).mean(), torch.sin(torch.angle(ratio)).mean(),
            torch.cos(torch.angle(ratio)).std(), torch.sin(torch.angle(ratio)).std(),
            torch.abs(ratio).mean(), torch.abs(ratio).std(),
        ])
        X.append(feats); Y.append(t)
    return torch.stack(X), torch.tensor(Y)


def fibonacci_angles(n):
    phi = (1 + sqrt(5)) / 2
    return [(2*pi * i / phi) % (2*pi) for i in range(n)]


class FlatBorn(nn.Module):
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


class FullCybernetic(nn.Module):
    def __init__(self, in_dim=6, mid_dim=8, total_heads=8, n_classes=4,
                 momentum=0.9, temp_base=1.0, fibonacci=False):
        super().__init__()
        self.total_heads = total_heads; self.mid_dim = mid_dim
        self.momentum = momentum; self.temp_base = temp_base

        self.qr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.qi = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.kr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.ki = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.vr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.vi = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])

        if fibonacci:
            angles = fibonacci_angles(total_heads)
        else:
            angles = torch.linspace(0, 2*pi, total_heads+1)[:total_heads].tolist()
        self.phases = nn.ParameterList([nn.Parameter(torch.tensor(a)) for a in angles])

        self.align = nn.Parameter(torch.randn(mid_dim, n_classes)*0.1)
        self.register_buffer('C', torch.randn(mid_dim, n_classes)*0.1)
        self.register_buffer('per_head_coh', torch.ones(total_heads) * 0.5)

        for mlist in [self.qr, self.qi, self.kr, self.ki, self.vr, self.vi]:
            for w in mlist: nn.init.normal_(w.weight, std=0.1)

    def forward(self, x):
        B = x.shape[0]
        sum_r = torch.zeros(B, self.mid_dim, device=x.device)
        sum_i = torch.zeros_like(sum_r)
        head_phases = []

        for h in range(self.total_heads):
            qr, qi = self.qr[h](x), self.qi[h](x)
            kr, ki = self.kr[h](x), self.ki[h](x)
            vr, vi = self.vr[h](x), self.vi[h](x)
            score_i = (qi*kr - qr*ki).sum(-1)

            # Cybernetic temperature: low coh -> wider attention (explore)
            T_h = self.temp_base / (self.per_head_coh[h] + 0.1)
            c = torch.cos(score_i.unsqueeze(-1) / T_h)
            s = torch.sin(score_i.unsqueeze(-1) / T_h)

            zr, zi = c*vr - s*vi, c*vi + s*vr
            c2, s2 = torch.cos(self.phases[h]), torch.sin(self.phases[h])
            sum_r += (zr*c2 - zi*s2) / sqrt(self.total_heads)
            sum_i += (zr*s2 + zi*c2) / sqrt(self.total_heads)

            cos_mean = torch.cos(score_i).mean()
            sin_mean = torch.sin(score_i).mean()
            head_phases.append((cos_mean**2 + sin_mean**2).sqrt())

        with torch.no_grad():
            new_coh = torch.stack(head_phases)
            self.per_head_coh.data = 0.8 * self.per_head_coh + 0.2 * new_coh

        # Cybernetic: alignment C tracks output (from real part)
        P_align = (sum_r @ self.align)**2 + (sum_i @ self.align)**2
        with torch.no_grad():
            C_target = sum_r.T @ sum_i
            C_target = C_target / (C_target.norm() + 1e-8)
            C_new = C_target[:, :self.C.shape[1]]
            if C_new.shape[0] > self.C.shape[0]:
                C_new = C_new[:self.C.shape[0], :]
            self.C.data = self.momentum * self.C + (1-self.momentum) * C_new

        return P_align


class SimpleCybernetic(nn.Module):
    """C tracking only, no per-head temperature (Attack 3 version)"""
    def __init__(self, in_dim=6, mid_dim=8, total_heads=8, n_classes=4, momentum=0.9):
        super().__init__()
        self.total_heads = total_heads; self.mid_dim = mid_dim; self.momentum = momentum
        self.qr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.qi = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.kr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.ki = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.vr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.vi = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        angles = torch.linspace(0, 2*pi, total_heads+1)[:total_heads]
        self.phases = nn.ParameterList([nn.Parameter(a.unsqueeze(0).expand(1)) for a in angles])
        self.align = nn.Parameter(torch.randn(mid_dim, n_classes)*0.1)
        self.register_buffer('C', torch.randn(mid_dim, n_classes)*0.1)
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
        P = (sum_r @ self.align)**2 + (sum_i @ self.align)**2
        with torch.no_grad():
            C_target = sum_r.T @ sum_i
            C_target = C_target / (C_target.norm() + 1e-8)
            C_new = C_target[:, :self.C.shape[1]]
            if C_new.shape[0] > self.C.shape[0]:
                C_new = C_new[:self.C.shape[0], :]
            self.C.data = self.momentum * self.C + (1-self.momentum) * C_new
        return P


def train_eval(model, X_tr, Y_tr, X_te, Y_te, epochs=150):
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
        delta = acc - ab_acc
        pc_vals = getattr(model, 'per_head_coh', None)
    return acc, ab_acc, delta, n_p, pc_vals


print("=" * 80)
print("FULL CYBERNETIC LOOP: C-tracking + Per-head Temperature + Fibonacci")
print("=" * 80)

X_train, Y_train = gen_geometry(400)
X_test, Y_test = gen_geometry(200)

architectures = [
    ("FLAT-BORN (baseline)",     lambda: FlatBorn(mid_dim=8, total_heads=8)),
    ("CYBER-C  (C tracking)",    lambda: SimpleCybernetic(mid_dim=8, total_heads=8)),
    ("CYBER-CT (C + temp)",      lambda: FullCybernetic(mid_dim=8, total_heads=8)),
    ("CYBER-CTF (C + temp + fib)", lambda: FullCybernetic(mid_dim=8, total_heads=8, fibonacci=True)),
]

print(f"\n  {'arch':<28}  {'acc':>7}  {'ab_acc':>7}  {'delta':>7}  {'params':>8}")
print("  " + "-" * 60)
for name, factory in architectures:
    m = factory()
    acc, ab_acc, delta, n_p, _ = train_eval(m, X_train, Y_train, X_test, Y_test)
    print(f"  {name:<28}  {acc:>7.1%}  {ab_acc:>7.1%}  {delta:>+6.1%}  {n_p:>8}")

print()
print("=" * 80)
print("TASK DIFFICULTY SWEEP: Full Loop vs Baseline")
print("=" * 80)

for noise in [0.0, 0.5, 1.0]:
    X_tr_n, Y_tr_n = gen_geometry(300, noise=noise)
    X_te_n, Y_te_n = gen_geometry(150, noise=noise)
    print(f"\n  noise={noise}:")
    for name, factory in [("FLAT", lambda: FlatBorn(mid_dim=8, total_heads=8)),
                           ("FULL", lambda: FullCybernetic(mid_dim=8, total_heads=8, fibonacci=True))]:
        m = factory()
        acc, ab_acc, delta, _, _ = train_eval(m, X_tr_n, Y_tr_n, X_te_n, Y_te_n)
        print(f"    {name:<6}: acc={acc:.1%}  delta={delta:+.1%}")

print()
print("=" * 80)
print("6-CLASS TASK: Architecture scaling on harder problem")
print("=" * 80)
X_tr6, Y_tr6 = gen_geometry(500, n_classes=6)
X_te6, Y_te6 = gen_geometry(200, n_classes=6)

for name, factory in architectures:
    m = factory()
    m.align = nn.Parameter(torch.randn(8, 6)*0.1)
    acc, ab_acc, delta, n_p, _ = train_eval(m, X_tr6, Y_tr6, X_te6, Y_te6, epochs=120)
    print(f"  {name:<28}  {acc:>7.1%}  {ab_acc:>7.1%}  {delta:>+6.1%}")

print()
print("=" * 80)
print("PER-HEAD PHASE COHERENCE TRAJECTORY (Full Cybernetic)")
print("=" * 80)
for fibonacci in [False, True]:
    label = "FIBONACCI" if fibonacci else "UNIFORM"
    m = FullCybernetic(mid_dim=8, total_heads=8, fibonacci=fibonacci)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    coh_traj = []
    for e in range(100):
        loss = F.cross_entropy(m(X_train), Y_train)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if e % 10 == 0:
            coh_traj.append((e, m.per_head_coh.clone().tolist()))
    print(f"\n  {label} phase seeding:")
    print(f"    epoch {' '.join(f'head{h:>2}' for h in range(8))}")
    for e, cohs in coh_traj[::3]:
        print(f"    {e:>5} {' '.join(f'{c:.3f}' for c in cohs)}")
    final = coh_traj[-1][1]
    spread = max(final) - min(final)
    mean_c = sum(final) / len(final)
    print(f"    final spread: {spread:.3f}  mean: {mean_c:.3f}")
