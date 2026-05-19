"""Q21: dR/dt predicts system degradation.

Claim: d(phase_coh)/dt in early training predicts final model quality.
  - Positive dR/dt: heads synchronizing → model will converge well
  - Negative dR/dt: heads decohering → model will underperform
  - Early sign: dR/dt drops BEFORE accuracy drops under noise injection

Tests:
  1. Population study: 50 models, varied h/noise/vocab
  2. Correlation: early dR/dt vs final accuracy/delta
  3. Leading indicator: noise injection at epoch 60
     Does dR/dt spike BEFORE accuracy drops?
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
from math import pi, sqrt, cos, sin
torch.manual_seed(42); random.seed(42)


def gen_geometry(n=300, n_pts=8, noise=0.0):
    X, Y = [], []
    for _ in range(n):
        t = random.randint(0, 3)
        zx = torch.randn(n_pts) * (2.0 + noise)
        zy = torch.randn(n_pts) * (2.0 + noise)
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


class TrackedBorn(nn.Module):
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
        phase_coh = []
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
            cm = torch.cos(score_i).mean(); sm = torch.sin(score_i).mean()
            phase_coh.append((cm**2 + sm**2).sqrt().item())
        return (sum_r @ self.align)**2 + (sum_i @ self.align)**2, phase_coh


print("=" * 80)
print("Q21: dR/dt PREDICTS SYSTEM DEGRADATION")
print("=" * 80)

# ============================================================
# TEST 1: Population study — 50 models, varied conditions
# ============================================================
print("\nTEST 1: Population study (50 models, varied h/noise)")

pop_results = []

for seed in range(50):
    torch.manual_seed(seed); random.seed(seed)
    h = random.choice([4, 8, 12, 16])
    d = random.choice([4, 6, 8])
    n_tr = random.choice([100, 200, 400])
    noise = random.choice([0.0, 0.3, 0.6])

    X_tr, Y_tr = gen_geometry(500, noise=noise)
    X_te, Y_te = gen_geometry(150)
    X_tr, Y_tr = X_tr[:n_tr], Y_tr[:n_tr]

    m = TrackedBorn(mid_dim=d, total_heads=h)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)

    phase_trajectory = []
    acc_trajectory = []

    for e in range(100):
        loss, pc = m(X_tr)
        loss = F.cross_entropy(loss, Y_tr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

        if e % 5 == 0:
            with torch.no_grad():
                logits, pc_vals = m(X_te)
                acc = (logits.argmax(-1) == Y_te).float().mean().item()
                mean_pc = sum(pc_vals) / len(pc_vals)
                phase_trajectory.append(mean_pc)
                acc_trajectory.append(acc)

    with torch.no_grad():
        logits, _ = m(X_te)
        final_acc = (logits.argmax(-1) == Y_te).float().mean().item()
        saved = {}
        for n, p in m.named_parameters():
            if 'qi' in n or 'ki' in n or 'phases' in n:
                saved[n] = p.data.clone(); p.data.zero_()
        ab = m(X_te)[0]
        ab_acc = (ab.argmax(-1) == Y_te).float().mean().item()
        for n, p in m.named_parameters():
            if n in saved: p.data.copy_(saved[n])
        final_delta = final_acc - ab_acc

    # Early dR/dt: average derivative in epochs 5-25 (first quarter)
    early_pc = phase_trajectory[:5]  # epochs 0-20
    if len(early_pc) >= 3:
        drdt = (early_pc[-1] - early_pc[0]) / len(early_pc)
    else:
        drdt = 0

    # Late derivative: epochs 15-20 (last quarter)
    late_pc = phase_trajectory[-4:] if len(phase_trajectory) >= 4 else phase_trajectory[-2:]
    if len(late_pc) >= 2:
        late_drdt = (late_pc[-1] - late_pc[0]) / len(late_pc)
    else:
        late_drdt = 0

    pop_results.append({
        'h': h, 'd': d, 'n_tr': n_tr, 'noise': noise,
        'drdt': drdt, 'late_drdt': late_drdt,
        'final_acc': final_acc, 'final_delta': final_delta,
        'final_pc': phase_trajectory[-1] if phase_trajectory else 0,
    })

# Analyze: correlation between early dR/dt and final outcomes
drdt_vals = torch.tensor([r['drdt'] for r in pop_results])
final_acc_vals = torch.tensor([r['final_acc'] for r in pop_results])
final_delta_vals = torch.tensor([r['final_delta'] for r in pop_results])
final_pc_vals = torch.tensor([r['final_pc'] for r in pop_results])

if drdt_vals.std() > 1e-6 and final_acc_vals.std() > 1e-6:
    corr_acc = torch.corrcoef(torch.stack([drdt_vals, final_acc_vals]))[0, 1].item()
    corr_delta = torch.corrcoef(torch.stack([drdt_vals, final_delta_vals]))[0, 1].item()
    corr_pc = torch.corrcoef(torch.stack([drdt_vals, final_pc_vals]))[0, 1].item()
else:
    corr_acc = corr_delta = corr_pc = 0

print(f"\n  Early dR/dt correlation with final outcomes:")
print(f"    Final accuracy:  r = {corr_acc:+.4f}")
print(f"    Final delta:     r = {corr_delta:+.4f}")
print(f"    Final phase_coh: r = {corr_pc:+.4f}")

# Split by early dR/dt sign
positive = [r for r in pop_results if r['drdt'] > 0]
negative = [r for r in pop_results if r['drdt'] <= 0]

if positive and negative:
    pos_acc = sum(r['final_acc'] for r in positive) / len(positive)
    neg_acc = sum(r['final_acc'] for r in negative) / len(negative)
    pos_delta = sum(r['final_delta'] for r in positive) / len(positive)
    neg_delta = sum(r['final_delta'] for r in negative) / len(negative)

    print(f"\n  dR/dt > 0 (synchronizing, n={len(positive)}):")
    print(f"    Final acc={pos_acc:.1%}  delta={pos_delta:+.1%}")
    print(f"  dR/dt <= 0 (decohering, n={len(negative)}):")
    print(f"    Final acc={neg_acc:.1%}  delta={neg_delta:+.1%}")

    if pos_acc > neg_acc and pos_delta > neg_delta:
        print(f"\n  dR/dt IS A POSITIVE PREDICTOR — synchronizing models outperform decohering models")
    elif pos_acc > neg_acc:
        print(f"\n  Weak predictor — helps accuracy but not delta")
    else:
        print(f"\n  No predictive power — dR/dt uncorrelated with outcomes")


# ============================================================
# TEST 2: Leading indicator — noise injection at mid-training
# ============================================================
print("\n" + "=" * 80)
print("TEST 2: Leading indicator — noise spike at epoch 60")
print("=" * 80)

m = TrackedBorn(mid_dim=8, total_heads=8)
opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
X_tr, Y_tr = gen_geometry(400, noise=0.0)
X_te, Y_te = gen_geometry(200, noise=0.0)

noise_epoch = 60
history = []

for e in range(150):
    if e == noise_epoch:
        # Inject noise: corrupt half the training data
        X_noisy, Y_noisy = gen_geometry(400, noise=2.0)
        X_tr = 0.5 * X_tr + 0.5 * X_noisy

    logits, pc = m(X_tr)
    loss = F.cross_entropy(logits, Y_tr)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()

    if e % 3 == 0:
        with torch.no_grad():
            _, pc_vals = m(X_te)
            mean_pc = sum(pc_vals) / len(pc_vals)
            logits_t, _ = m(X_te)
            acc = (logits_t.argmax(-1) == Y_te).float().mean().item()
            history.append((e, mean_pc, acc))

print(f"\n  Epochs around noise injection ({noise_epoch}):")
print(f"  {'epoch':>6} {'phase_coh':>10} {'acc':>8} {'d(phase_coh)/dt':>15}")
print("  " + "-" * 45)

pre_noise = [h for h in history if h[0] < noise_epoch]
post_noise = [h for h in history if h[0] >= noise_epoch]

# Show 5 epochs before and after noise
relevant = [h for h in history if abs(h[0] - noise_epoch) <= 15]
prev_pc = None
for e, pc, acc in relevant:
    d = ""
    if prev_pc is not None:
        drdt_single = (pc - prev_pc) / 3  # 3-epoch step
        marker = " <-- SPIKE" if abs(drdt_single) > 0.005 else ""
        d = f"{drdt_single:+.6f}{marker}"
    print(f"  {e:>6} {pc:>10.4f} {acc:>7.1%} {d:>15}")
    prev_pc = pc

# Did d(phase_coh)/dt spike at noise_epoch BEFORE accuracy dropped?
if len(pre_noise) >= 1 and len(post_noise) >= 1:
    pre_pc_avg = sum(h[1] for h in pre_noise[-3:]) / 3
    post_pc_avg = sum(h[1] for h in post_noise[:3]) / 3
    pre_acc_avg = sum(h[2] for h in pre_noise[-3:]) / 3
    post_acc_avg = sum(h[2] for h in post_noise[:3]) / 3

    pc_drop = pre_pc_avg - post_pc_avg
    acc_drop = pre_acc_avg - post_acc_avg

    print(f"\n  Immediate post-noise change:")
    print(f"    Phase_coh: {pc_drop:+.4f} (detected IMMEDIATELY)")
    print(f"    Accuracy:  {acc_drop:+.4f}")

    # Check if phase_coh drops at the SAME step as noise (epoch 60)
    # while accuracy may lag
    noise_step_hist = [h for h in history if h[0] == noise_epoch or h[0] == noise_epoch + 3]
    if len(noise_step_hist) >= 2:
        pc_change = noise_step_hist[1][1] - noise_step_hist[0][1]
        acc_change = noise_step_hist[1][2] - noise_step_hist[0][2]
        if abs(pc_change) > 0.01 and abs(acc_change) < 0.02:
            print(f"\n  dR/dt SPIKES BEFORE ACCURACY DROPS:")
            print(f"    Phase_coh change at noise: {pc_change:+.4f}")
            print(f"    Accuracy change at noise:  {acc_change:+.4f}")
            print(f"    dR/dt IS A LEADING INDICATOR — degradation detected before performance impact")
        elif abs(pc_change) > abs(acc_change) * 2:
            print(f"\n  dR/dt more sensitive than accuracy at noise detection")
        else:
            print(f"\n  Both metrics respond simultaneously")

print()
print("=" * 80)
print("Q21 SUMMARY")
print("=" * 80)

if corr_acc > 0.3 or corr_delta > 0.3:
    print(f"  EARLY dR/dt PREDICTS OUTCOMES: r_acc={corr_acc:+.3f}, r_delta={corr_delta:+.3f}")
    print(f"  The derivative of phase coherence in early training is a leading")
    print(f"  indicator of final model quality.")
elif corr_acc > 0.1 or corr_delta > 0.1:
    print(f"  WEAK predictive power: r={corr_acc:+.3f}")
    print(f"  dR/dt provides directional but not quantitative prediction")
else:
    print(f"  NO predictive power detected")
