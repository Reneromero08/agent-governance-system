"""Q33: R shows emergent properties at macro scale.

Tests: does phase_coh stabilize as dataset size increases?
  1. Variance sweep: 10 seeds at 6 dataset sizes (25 to 800)
  2. CV analysis: does CV → 0 as N → ∞?
  3. Attractor convergence: do seeds converge to Q28's fixed point?
  4. Emergence threshold: what N produces macro-scale stability?
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
from math import pi, sqrt
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


class FlatBorn(nn.Module):
    def __init__(self, mid_dim=8, total_heads=8, n_classes=4):
        super().__init__()
        self.H = total_heads; self.D = mid_dim
        self.qr = nn.ModuleList([nn.Linear(6, mid_dim, False) for _ in range(total_heads)])
        self.qi = nn.ModuleList([nn.Linear(6, mid_dim, False) for _ in range(total_heads)])
        self.kr = nn.ModuleList([nn.Linear(6, mid_dim, False) for _ in range(total_heads)])
        self.ki = nn.ModuleList([nn.Linear(6, mid_dim, False) for _ in range(total_heads)])
        self.vr = nn.ModuleList([nn.Linear(6, mid_dim, False) for _ in range(total_heads)])
        self.vi = nn.ModuleList([nn.Linear(6, mid_dim, False) for _ in range(total_heads)])
        angles = torch.linspace(0, 2*pi, total_heads+1)[:total_heads]
        self.phases = nn.ParameterList([nn.Parameter(a.unsqueeze(0).expand(1)) for a in angles])
        self.align = nn.Parameter(torch.randn(mid_dim, n_classes)*0.1)
        for mlist in [self.qr, self.qi, self.kr, self.ki, self.vr, self.vi]:
            for w in mlist: nn.init.normal_(w.weight, std=0.1)

    def forward(self, x):
        B = x.shape[0]
        sum_r = torch.zeros(B, self.D, device=x.device)
        sum_i = torch.zeros_like(sum_r)
        phase_coh = []
        for h in range(self.H):
            qr, qi = self.qr[h](x), self.qi[h](x)
            kr, ki = self.kr[h](x), self.ki[h](x)
            vr, vi = self.vr[h](x), self.vi[h](x)
            score_i = (qi*kr - qr*ki).sum(-1)
            c, s = torch.cos(score_i.unsqueeze(-1)), torch.sin(score_i.unsqueeze(-1))
            zr, zi = c*vr - s*vi, c*vi + s*vr
            c2, s2 = torch.cos(self.phases[h]), torch.sin(self.phases[h])
            sum_r += (zr*c2 - zi*s2) / sqrt(self.H)
            sum_i += (zr*s2 + zi*c2) / sqrt(self.H)
            cm = torch.cos(score_i).mean(); sm = torch.sin(score_i).mean()
            phase_coh.append((cm**2 + sm**2).sqrt().item())
        return (sum_r @ self.align)**2 + (sum_i @ self.align)**2, phase_coh


print("=" * 70)
print("Q33: R EMERGENT PROPERTIES AT MACRO SCALE")
print("=" * 70)

sizes = [25, 50, 100, 200, 400, 800]
n_seeds = 10
h, d = 8, 8

X_pool, Y_pool = gen_geometry(1200)
X_te, Y_te = gen_geometry(200)

all_results = {}

for N in sizes:
    print(f"\n  N={N} ({n_seeds} seeds):")
    final_pcs = []
    final_deltas = []
    final_accs = []
    pc_trajectories = []

    for seed in range(n_seeds):
        torch.manual_seed(10000 + seed); random.seed(10000 + seed)
        X_tr = X_pool[:N]; Y_tr = Y_pool[:N]

        m = FlatBorn(mid_dim=d, total_heads=h)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        pc_traj = []

        for e in range(80):
            logits, pc = m(X_tr)
            loss = F.cross_entropy(logits, Y_tr)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if e % 5 == 0:
                with torch.no_grad():
                    _, pcv = m(X_te)
                    pc_traj.append(sum(pcv) / len(pcv))

        with torch.no_grad():
            logits, pcv = m(X_te)
            acc = (logits.argmax(-1) == Y_te).float().mean().item()
            saved = {}
            for n, p in m.named_parameters():
                if 'qi' in n or 'ki' in n or 'phases' in n:
                    saved[n] = p.data.clone(); p.data.zero_()
            ab, _ = m(X_te)
            ab_acc = (ab.argmax(-1) == Y_te).float().mean().item()
            for n, p in m.named_parameters():
                if n in saved: p.data.copy_(saved[n])
            delta = acc - ab_acc

        final_pc = pc_traj[-1]
        final_pcs.append(final_pc)
        final_deltas.append(delta)
        final_accs.append(acc)
        pc_trajectories.append(pc_traj)

    mean_pc = sum(final_pcs) / len(final_pcs)
    std_pc = (sum((x - mean_pc)**2 for x in final_pcs) / len(final_pcs)) ** 0.5
    cv_pc = std_pc / max(mean_pc, 1e-6)
    mean_delta = sum(final_deltas) / len(final_deltas)
    std_acc = (sum((x - sum(final_accs)/len(final_accs))**2 for x in final_accs) / len(final_accs)) ** 0.5

    all_results[N] = {
        'mean_pc': mean_pc, 'std_pc': std_pc, 'cv_pc': cv_pc,
        'mean_delta': mean_delta, 'std_acc': std_acc,
        'values': final_pcs
    }

    print(f"    pc: {mean_pc:.4f} +/- {std_pc:.4f}  CV={cv_pc:.4f}  delta={mean_delta:+.1%}  acc_std={std_acc:.4f}")

print(f"\n{'='*70}")
print(f"EMERGENCE ANALYSIS")
print(f"{'='*70}")

sizes_arr = list(all_results.keys())
cvs = [all_results[N]['cv_pc'] for N in sizes_arr]
means = [all_results[N]['mean_pc'] for N in sizes_arr]
stds = [all_results[N]['std_pc'] for N in sizes_arr]
mean_deltas = [all_results[N]['mean_delta'] for N in sizes_arr]

print(f"\n  {'N':>5} {'mean pc':>8} {'std pc':>8} {'CV':>8} {'mean delta':>10} {'acc std':>8}")
print("  " + "-" * 55)
for N in sizes_arr:
    r = all_results[N]
    print(f"  {N:>5} {r['mean_pc']:>8.4f} {r['std_pc']:>8.4f} {r['cv_pc']:>8.4f} {r['mean_delta']:>9.1%} {r['std_acc']:>8.4f}")

# Convergence test: does variance drop with N?
if len(cvs) >= 4:
    first_half = sum(cvs[:len(cvs)//2]) / (len(cvs)//2)
    second_half = sum(cvs[len(cvs)//2:]) / (len(cvs) - len(cvs)//2)
    cv_reduction = first_half - second_half
    print(f"\n  CV trend: early={first_half:.4f} late={second_half:.4f}")
    print(f"  CV reduction: {cv_reduction:+.4f}")

    # Power-law fit: std ~ N^(-alpha)
    import numpy as np
    logN = np.log(np.array(sizes_arr))
    logS = np.log(np.array(stds))
    alpha, intercept = np.polyfit(logN, logS, 1)
    print(f"  Power-law: std ~ N^{alpha:+.3f} (alpha < 0 means stabilization)")
    if alpha < -0.1:
        print(f"  EMERGENCE CONFIRMED — phase_coh stabilizes with scale (alpha={alpha:+.3f})")
    elif alpha < 0:
        print(f"  WEAK stabilization (alpha={alpha:+.3f})")
    else:
        print(f"  No stabilization — variance increases with scale")

# Cross-seed trajectory convergence
print(f"\n  CROSS-SEED TRAJECTORY CONVERGENCE:")
print(f"  {'N':>5} {'final_range':>12} {'attractor?':>12}")
print("  " + "-" * 35)
for N in sizes_arr:
    vals = all_results[N]['values']
    pc_range = max(vals) - min(vals)
    stable = pc_range < 0.05
    print(f"  {N:>5} {pc_range:>12.4f} {'YES' if stable else 'no':>12}")

# Q28 connection: is there a fixed-point attractor?
largest_N = sizes_arr[-1]
largest_vals = all_results[largest_N]['values']
largest_std = all_results[largest_N]['std_pc']
print(f"\n  ATTRACTOR (Q28 connection):")
print(f"  At N={largest_N} (largest): mean={all_results[largest_N]['mean_pc']:.4f} +/- {largest_std:.4f}")
print(f"  Q28 found CV=0.39% at 10 seeds — we find CV={all_results[largest_N]['cv_pc']*100:.2f}% at {largest_N} samples")
if all_results[largest_N]['cv_pc'] < 0.05:
    print(f"  FIXED-POINT ATTRACTOR CONFIRMED — phase_coh converges to single value at macro scale")
elif all_results[largest_N]['cv_pc'] < 0.1:
    print(f"  WEAK attractor — CV < 10% at macro scale")
else:
    print(f"  No fixed point detected at this scale")
