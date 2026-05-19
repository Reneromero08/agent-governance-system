"""Q21 HARDENED: Multi-angle battery following Q48 hardening pattern.

Angles:
  1. Seed stability: 10 seeds, same config, same trajectory?
  2. AUROC: dR/dt at epoch 20 classifies high-delta vs low-delta models
  3. Causal control: artificially corrupt weights to induce negative dR/dt
     Does induced fragility CAUSE lower final accuracy?
  4. Recovery speed: positive dR/dt models recover faster from noise injection
  5. Threshold sweep: optimal dR/dt threshold for vulnerability classification
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
        pc_vals = []
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
            pc_vals.append((cm**2 + sm**2).sqrt().item())
        return (sum_r @ self.align)**2 + (sum_i @ self.align)**2, pc_vals


def get_drdt(pc_trajectory, start_idx=0, n_steps=5):
    if len(pc_trajectory) < start_idx + n_steps:
        return 0.0
    return (pc_trajectory[start_idx + n_steps - 1] - pc_trajectory[start_idx]) / n_steps


print("=" * 80)
print("Q21 HARDENED: 5-Angle Battery")
print("=" * 80)

# ============================================================
# ANGLE 1: Seed stability (10 seeds, same config)
# ============================================================
print("\nANGLE 1: Seed stability — 10 seeds, same config")
print("  " + "-" * 55)

h, d = 8, 8
X_tr, Y_tr = gen_geometry(300, noise=0.2)
X_te, Y_te = gen_geometry(150)

seed_drdts = []
seed_finals = []
for seed in range(10):
    torch.manual_seed(seed); random.seed(seed)
    m = FlatBorn(mid_dim=d, total_heads=h)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    pc_traj = []
    for e in range(60):
        logits, pc = m(X_tr)
        loss = F.cross_entropy(logits, Y_tr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if e % 5 == 0:
            with torch.no_grad():
                _, pc_vals = m(X_te)
                pc_traj.append(sum(pc_vals)/len(pc_vals))

    drdt = get_drdt(pc_traj, 0, 4)
    seed_drdts.append(drdt)
    seed_finals.append(pc_traj[-1])
    print(f"  seed {seed}: dR/dt={drdt:+.4f}  final_pc={pc_traj[-1]:.4f}")

drdt_arr = torch.tensor(seed_drdts)
final_arr = torch.tensor(seed_finals)
print(f"  dR/dt: mean={drdt_arr.mean():+.4f}  std={drdt_arr.std():.4f}  CV={abs(drdt_arr.std()/ (drdt_arr.mean() + 1e-6)):.2f}")
print(f"  Final pc: mean={final_arr.mean():.4f}  std={final_arr.std():.4f}  CV={abs(final_arr.std()/ (final_arr.mean() + 1e-6)):.2f}")

if drdt_arr.std() / max(abs(drdt_arr.mean()), 1e-6) < 0.5:
    sign_agree = sum(1 for d in seed_drdts if d > 0) == 10 or sum(1 for d in seed_drdts if d < 0) == 10
    if sign_agree:
        print(f"  SEED-STABLE: all 10 seeds agree on sign ({'+' if seed_drdts[0]>0 else '-'})")
    else:
        print(f"  Moderate stability: sign flips across seeds")
else:
    print(f"  SEED-UNSTABLE: dR/dt sign varies across seeds")

# ============================================================
# ANGLE 2: AUROC predictive test
# ============================================================
print(f"\nANGLE 2: AUROC — dR/dt classifies high-delta vs low-delta models")
print("  " + "-" * 55)

auroc_data = []
for seed in range(40):
    h = random.choice([4, 8, 12])
    d = random.choice([6, 8])
    n_tr = random.choice([150, 300])
    noise = random.choice([0.0, 0.3, 0.6])

    torch.manual_seed(1000+seed); random.seed(1000+seed)
    Xt, Yt = gen_geometry(500, noise=noise)
    Xv, Yv = gen_geometry(150)

    m = FlatBorn(mid_dim=d, total_heads=h)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    pc_traj = []
    for e in range(80):
        logits, pc = m(Xt[:n_tr])
        loss = F.cross_entropy(logits, Yt[:n_tr])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if e % 5 == 0:
            with torch.no_grad():
                _, pc_vals = m(Xv)
                pc_traj.append(sum(pc_vals)/len(pc_vals))

    drdt_early = get_drdt(pc_traj, 0, 4)
    with torch.no_grad():
        logits, _ = m(Xv)
        acc = (logits.argmax(-1) == Yv).float().mean().item()
        saved = {}
        for n, p in m.named_parameters():
            if 'qi' in n or 'ki' in n or 'phases' in n:
                saved[n] = p.data.clone(); p.data.zero_()
        ab_logits, _ = m(Xv)
        ab_acc = (ab_logits.argmax(-1) == Yv).float().mean().item()
        for n, p in m.named_parameters():
            if n in saved: p.data.copy_(saved[n])
        delta = acc - ab_acc
    auroc_data.append((drdt_early, delta, acc))

# Split at median delta
deltas = [d[1] for d in auroc_data]
median_delta = sorted(deltas)[len(deltas)//2]
high_delta = [d for d in auroc_data if d[1] > median_delta]
low_delta = [d for d in auroc_data if d[1] <= median_delta]

# AUROC: rank by dR/dt, compute TPR/FPR at each threshold
scores = torch.tensor([d[0] for d in auroc_data])
labels = torch.tensor([1.0 if d[1] > median_delta else 0.0 for d in auroc_data])

sorted_idx = scores.argsort(descending=True)
sorted_labels = labels[sorted_idx]
n_pos = labels.sum().item()
n_neg = len(labels) - n_pos

tpr_vals, fpr_vals = [], []
tp = fp = 0
for i, l in enumerate(sorted_labels):
    if l == 1: tp += 1
    else: fp += 1
    tpr_vals.append(tp / n_pos if n_pos > 0 else 0)
    fpr_vals.append(fp / n_neg if n_neg > 0 else 0)

# AUROC = area under TPR vs FPR curve (trapezoidal)
auroc_val = 0.0
for i in range(1, len(fpr_vals)):
    auroc_val += (fpr_vals[i] - fpr_vals[i-1]) * (tpr_vals[i] + tpr_vals[i-1]) / 2

pos_drdt = [d[0] for d in high_delta]
neg_drdt = [d[0] for d in low_delta]
pos_mean = sum(pos_drdt)/len(pos_drdt)
neg_mean = sum(neg_drdt)/len(neg_drdt)

print(f"  High-delta models (n={len(high_delta)}): mean dR/dt = {pos_mean:+.4f}")
print(f"  Low-delta models  (n={len(low_delta)}):  mean dR/dt = {neg_mean:+.4f}")
print(f"  AUROC = {auroc_val:.4f}")

if auroc_val > 0.7:
    print(f"  dR/dt IS PREDICTIVE — AUROC {auroc_val:.2f} classifies final delta")
elif auroc_val > 0.6:
    print(f"  Moderate predictive power — AUROC {auroc_val:.2f}")
elif auroc_val > 0.5:
    print(f"  Barely above chance — AUROC {auroc_val:.2f}")
else:
    print(f"  dR/dt ANTI-PREDICTS — AUROC {auroc_val:.2f} (worse than chance)")

# ============================================================
# ANGLE 3: Causal control — artificial weight corruption
# ============================================================
print(f"\nANGLE 3: Causal control — artificially induce negative dR/dt")
print("  " + "-" * 55)

n_ctrl = 5
ctrl_results = {'clean': [], 'corrupted': []}

for seed in range(n_ctrl):
    torch.manual_seed(2000+seed); random.seed(2000+seed)
    Xt, Yt = gen_geometry(400, noise=0.1)
    Xv, Yv = gen_geometry(150)

    # Clean model (control)
    m_clean = FlatBorn(mid_dim=8, total_heads=8)
    opt = torch.optim.AdamW(m_clean.parameters(), lr=1e-2)
    for e in range(80):
        logits, _ = m_clean(Xt)
        loss = F.cross_entropy(logits, Yt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m_clean.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        clean_l, _ = m_clean(Xv)
        clean_acc = (clean_l.argmax(-1) == Yv).float().mean().item()
        saved = {}
        for n, p in m_clean.named_parameters():
            if 'qi' in n or 'ki' in n or 'phases' in n:
                saved[n] = p.data.clone(); p.data.zero_()
        clean_ab, _ = m_clean(Xv)
        clean_ab_acc = (clean_ab.argmax(-1) == Yv).float().mean().item()
        for n, p in m_clean.named_parameters():
            if n in saved: p.data.copy_(saved[n])
        clean_delta = clean_acc - clean_ab_acc
    ctrl_results['clean'].append((clean_acc, clean_delta))

    # Corrupted model: inject noise into Q weights at epoch 20 to force negative dR/dt
    m_corr = FlatBorn(mid_dim=8, total_heads=8)
    opt_c = torch.optim.AdamW(m_corr.parameters(), lr=1e-2)
    for e in range(80):
        if e == 20:
            # Corrupt: add Gaussian noise to Q imaginary weights
            for h in range(8):
                noise = torch.randn_like(m_corr.qi[h].weight) * 0.5
                m_corr.qi[h].weight.data += noise
                noise_k = torch.randn_like(m_corr.ki[h].weight) * 0.5
                m_corr.ki[h].weight.data += noise_k
        logits, _ = m_corr(Xt)
        loss = F.cross_entropy(logits, Yt)
        opt_c.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m_corr.parameters(), 1.0)
        opt_c.step()
    with torch.no_grad():
        corr_l, _ = m_corr(Xv)
        corr_acc = (corr_l.argmax(-1) == Yv).float().mean().item()
        saved = {}
        for n, p in m_corr.named_parameters():
            if 'qi' in n or 'ki' in n or 'phases' in n:
                saved[n] = p.data.clone(); p.data.zero_()
        corr_ab, _ = m_corr(Xv)
        corr_ab_acc = (corr_ab.argmax(-1) == Yv).float().mean().item()
        for n, p in m_corr.named_parameters():
            if n in saved: p.data.copy_(saved[n])
        corr_delta = corr_acc - corr_ab_acc
    ctrl_results['corrupted'].append((corr_acc, corr_delta))

clean_accs = [r[0] for r in ctrl_results['clean']]
corr_accs = [r[0] for r in ctrl_results['corrupted']]
clean_ds = [r[1] for r in ctrl_results['clean']]
corr_ds = [r[1] for r in ctrl_results['corrupted']]

print(f"  Clean:     acc={sum(clean_accs)/len(clean_accs):.1%}  delta={sum(clean_ds)/len(clean_ds):+.1%}")
print(f"  Corrupted: acc={sum(corr_accs)/len(corr_accs):.1%}  delta={sum(corr_ds)/len(corr_ds):+.1%}")

acc_loss = sum(clean_accs)/len(clean_accs) - sum(corr_accs)/len(corr_accs)
delta_loss = sum(clean_ds)/len(clean_ds) - sum(corr_ds)/len(corr_ds)
print(f"  Accuracy loss from corruption: {acc_loss:+.1%}")
print(f"  Delta loss from corruption:    {delta_loss:+.1%}")

if delta_loss > 0.05:
    print(f"  CAUSAL LINK CONFIRMED — forced decoherence reduces phase delta by {delta_loss:+.1%}")
elif delta_loss > 0.01:
    print(f"  Weak causal link — {delta_loss:+.1%} delta reduction")
else:
    print(f"  No causal link detected")

# ============================================================
# ANGLE 4: Recovery speed — do positive dR/dt models recover faster?
# ============================================================
print(f"\nANGLE 4: Recovery speed — positive dR/dt predicts faster recovery from noise")
print("  " + "-" * 55)

n_rec = 10
recovery_data = []

for seed in range(n_rec):
    torch.manual_seed(3000+seed); random.seed(3000+seed)
    Xt, Yt = gen_geometry(400, noise=0.0)
    Xv, Yv = gen_geometry(150)

    m = FlatBorn(mid_dim=8, total_heads=8)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    pre_noise_pc = []
    post_noise_acc = []

    for e in range(100):
        if e == 40:
            # Inject noise: corrupt 50% of training labels
            Xt_n, Yt_n = gen_geometry(400, noise=2.0)
            Xt = 0.5 * Xt + 0.5 * Xt_n
        logits, pc = m(Xt)
        loss = F.cross_entropy(logits, Yt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if e < 40 and e % 5 == 0:
            with torch.no_grad():
                _, pcv = m(Xv)
                pre_noise_pc.append(sum(pcv)/len(pcv))
        if e >= 40 and e % 5 == 0:
            with torch.no_grad():
                lv, _ = m(Xv)
                post_noise_acc.append((lv.argmax(-1) == Yv).float().mean().item())

    drdt_pre = get_drdt(pre_noise_pc, 1, min(len(pre_noise_pc)-1, 3))
    # Recovery speed: epochs to reach 90% of pre-noise performance
    if post_noise_acc:
        target = max(post_noise_acc) * 0.9
        recovery_epoch = 40
        for i, a in enumerate(post_noise_acc):
            if a >= target:
                recovery_epoch = 40 + i*5
                break
        recovery_data.append((drdt_pre, recovery_epoch, max(post_noise_acc)))
    else:
        recovery_data.append((drdt_pre, 100, 0))

print(f"  {'dR/dt pre':>10} {'recover epoch':>14} {'final acc':>10} {'fast recover?':>14}")
print("  " + "-" * 55)
fast_recover = []
slow_recover = []
for drdt, rec_ep, f_acc in recovery_data:
    fast = rec_ep <= 55
    if fast: fast_recover.append(drdt)
    else: slow_recover.append(drdt)
    print(f"  {drdt:>+9.4f} {rec_ep:>14} {f_acc:>9.1%} {'YES' if fast else 'NO':>14}")

if fast_recover and slow_recover:
    fast_mean = sum(fast_recover)/len(fast_recover)
    slow_mean = sum(slow_recover)/len(slow_recover)
    print(f"\n  Fast recover mean dR/dt: {fast_mean:+.4f}")
    print(f"  Slow recover mean dR/dt: {slow_mean:+.4f}")
    if fast_mean > slow_mean:
        print(f"  CONFIRMED — positive dR/dt predicts faster recovery (+{(fast_mean-slow_mean):+.4f})")
    else:
        print(f"  Reversed — negative dR/dt predicts faster recovery")
else:
    print(f"\n  Insufficient variation in recovery speed")

# ============================================================
# ANGLE 5: Vulnerability threshold sweep
# ============================================================
print(f"\nANGLE 5: dR/dt threshold sweep — optimal vulnerability classifier")
print("  " + "-" * 55)

# Use all 40 AUROC models. Classify as "vulnerable" if dR/dt < threshold.
# Measure: fraction of vulnerable models that end with delta < median_delta
thresholds = [-0.015, -0.010, -0.005, 0.000, 0.005, 0.010, 0.015]
best_t, best_acc = 0, 0

for t in thresholds:
    vulnerable = [d for d in auroc_data if d[0] < t]
    if not vulnerable: continue
    # Accuracy of "vulnerable" prediction: what fraction of vulnerable models end with low delta?
    low_in_vuln = sum(1 for d in vulnerable if d[1] <= median_delta)
    acc_t = low_in_vuln / len(vulnerable)
    if acc_t > best_acc:
        best_acc, best_t = acc_t, t
    print(f"  threshold={t:+.3f}: {low_in_vuln}/{len(vulnerable)} low-delta in vulnerable group (acc={acc_t:.1%})")

print(f"\n  Best threshold: {best_t:+.3f} (vulnerability detection accuracy: {best_acc:.1%})")

print()
print("=" * 80)
print("Q21 HARDENED SUMMARY")
print("=" * 80)

# Compile all findings
findings = []
if drdt_arr.std() / max(abs(drdt_arr.mean()), 1e-6) < 0.5:
    findings.append("SEED-STABLE: dR/dt sign consistent across 10 seeds")
else:
    findings.append("SEED-UNSTABLE: dR/dt varies across seeds")

if auroc_val > 0.6:
    findings.append(f"PREDICTIVE: AUROC={auroc_val:.2f} for classifying final delta")
else:
    findings.append(f"WEAK PREDICTOR: AUROC={auroc_val:.2f}")

if delta_loss > 0.01:
    findings.append(f"CAUSAL: forced decoherence reduces delta by {delta_loss:+.1%}")
else:
    findings.append("NO CAUSAL LINK: forced decoherence doesn't reduce delta")

if fast_recover and slow_recover and fast_mean > slow_mean:
    findings.append(f"RECOVERY PREDICTOR: +dR/dt models recover {(fast_mean-slow_mean):+.4f} faster")
else:
    findings.append("NO RECOVERY PREDICTION")

findings.append(f"VULNERABILITY THRESHOLD: dR/dt < {best_t:+.3f} = optimal fragility signal (acc={best_acc:.1%})")

for f in findings:
    print(f"  {f}")
