"""Q55+Q56 Toroidal Measurement: Track phase on T^h across training.

Measures per-head phase trajectories θ_h(t), frequency ratios, 
mode-locking detection, winding numbers, and the Arnold tongue boundary.

Hypothesis: Born rule merge induces rational frequency ratios (mode-locking)
between heads, explaining anti-saturation — stable phase relationships without
collapse to a single frequency.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
from math import pi, sqrt, gcd
torch.manual_seed(42); random.seed(42)


def gen_geometry(n=400, n_pts=8):
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
            torch.cos(torch.angle(ratio)).std(),  torch.sin(torch.angle(ratio)).std(),
            torch.abs(ratio).mean(), torch.abs(ratio).std(),
        ])
        X.append(feats); Y.append(t)
    return torch.stack(X), torch.tensor(Y)


class FlatBornWithTracking(nn.Module):
    def __init__(self, in_dim=6, mid_dim=8, total_heads=8, n_classes=4):
        super().__init__()
        self.total_heads = total_heads
        self.mid_dim = mid_dim
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
            for w in mlist:
                nn.init.normal_(w.weight, std=0.1)

    def forward(self, x, return_toroidal=False):
        B = x.shape[0]
        sum_r = torch.zeros(B, self.mid_dim, device=x.device)
        sum_i = torch.zeros_like(sum_r)
        per_head_phases = []

        for h in range(self.total_heads):
            qr, qi = self.qr[h](x), self.qi[h](x)
            kr, ki = self.kr[h](x), self.ki[h](x)
            vr, vi = self.vr[h](x), self.vi[h](x)
            score_i = (qi * kr - qr * ki).sum(dim=-1)
            c, s = torch.cos(score_i.unsqueeze(-1)), torch.sin(score_i.unsqueeze(-1))
            zr, zi = c*vr - s*vi, c*vi + s*vr
            c2, s2 = torch.cos(self.phases[h]), torch.sin(self.phases[h])
            zr, zi = zr*c2 - zi*s2, zr*s2 + zi*c2
            sum_r = sum_r + zr / sqrt(self.total_heads)
            sum_i = sum_i + zi / sqrt(self.total_heads)
            if return_toroidal:
                per_head_phases.append(self.phases[h].detach().item() % (2*pi))

        probs = (sum_r @ self.align)**2 + (sum_i @ self.align)**2
        if return_toroidal:
            return probs, per_head_phases
        return probs


def compute_toroidal_metrics(phase_history, heads_list):
    """phase_history[epoch][head] = phase angle in [0, 2*pi)"""
    n_epochs = len(phase_history)
    if n_epochs < 2:
        return {}

    phases = {}
    for h in heads_list:
        phases[h] = [epoch_phases[h] for epoch_phases in phase_history]

    # 1. Phase velocity dtheta/dt per head
    velocities = {}
    for h in heads_list:
        dtheta = []
        for t in range(1, n_epochs):
            delta = phases[h][t] - phases[h][t-1]
            delta = delta % (2*pi)
            if delta > pi:
                delta -= 2*pi
            dtheta.append(delta)
        velocities[h] = sum(dtheta) / len(dtheta) if dtheta else 0

    # 2. Frequency ratios between head pairs
    ratios = {}
    for i in heads_list:
        for j in heads_list:
            if i < j:
                vi, vj = velocities[i], velocities[j]
                denom = max(abs(vj), 1e-8)
                ratio = abs(vi) / denom
                ratios[(i, j)] = ratio

    # 3. Mode-locking detection: how close are ratios to small rationals?
    small_rationals = [(1,1), (1,2), (2,1), (1,3), (3,1), (2,3), (3,2), (3,4), (4,3), (1,4), (4,1), (2,5), (5,2)]
    locking_scores = {}
    for (i, j), ratio in ratios.items():
        if ratio == 0:
            locking_scores[(i, j)] = (0, 0, 0)
            continue
        best_error = float('inf')
        best_rational = (0, 0)
        for num, den in small_rationals:
            target = num / den
            err = abs(ratio - target)
            if err < best_error:
                best_error = err
                best_rational = (num, den)
        locking_scores[(i, j)] = (ratio, best_rational, best_error)

    # 4. Aggregate mode-locking score (fraction of pairs within rational tolerance)
    locked_count = 0
    total_pairs = 0
    for (i, j), (ratio, (num, den), err) in locking_scores.items():
        if num > 0 and err < 0.05:  # within 5% of a simple rational
            locked_count += 1
        total_pairs += 1
    mode_lock_fraction = locked_count / max(total_pairs, 1)

    # 5. Winding numbers: accumulated phase / (2*pi * epochs)
    windings = {}
    for h in heads_list:
        total_wrap = 0.0
        prev = phases[h][0]
        for t in range(1, n_epochs):
            delta = phases[h][t] - prev
            total_wrap += delta
            prev = phases[h][t]
        windings[h] = total_wrap / (2 * pi)

    # 6. Phase dispersion (std of instantaneous phases across heads)
    dispersion = []
    for epoch_phases in phase_history:
        pvals = [epoch_phases[h] for h in heads_list]
        mean_sin = sum(math.sin(p) for p in pvals) / len(pvals)
        mean_cos = sum(math.cos(p) for p in pvals) / len(pvals)
        r = sqrt(mean_sin**2 + mean_cos**2)
        dispersion.append(1.0 - r)
    mean_dispersion = sum(dispersion) / len(dispersion) if dispersion else 0

    return {
        'velocities': velocities,
        'ratios': ratios,
        'locking': locking_scores,
        'mode_lock_fraction': mode_lock_fraction,
        'windings': windings,
        'mean_dispersion': mean_dispersion,
        'dispersion_trajectory': dispersion,
    }


print("=" * 80)
print("Q55+Q56 TOROIDAL TOPOLOGY MEASUREMENT")
print("=" * 80)
print()

X_raw, Y_raw = gen_geometry(600)
X_tr, Y_tr = X_raw[:400], Y_raw[:400]
X_te, Y_te = X_raw[400:], Y_raw[400:]

heads_configs = [4, 8, 16]
mid_dim = 8
epochs = 120
track_every = 3

all_toroidal = {}

for h in heads_configs:
    print(f"--- h={h} heads, mid_dim={mid_dim}, {epochs} epochs ---")
    model = FlatBornWithTracking(mid_dim=mid_dim, total_heads=h)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)

    phase_history = []
    for e in range(epochs):
        loss = F.cross_entropy(model(X_tr), Y_tr)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if e % track_every == 0:
            with torch.no_grad():
                _, per_head = model(X_te, return_toroidal=True)
                phase_history.append(per_head)

    # Final metrics
    with torch.no_grad():
        final_phases_rad = [model.phases[i].item() % (2*pi) for i in range(h)]
        probs = model(X_te)
        acc = (probs.argmax(-1) == Y_te).float().mean().item()

        saved = {}
        for n, p in model.named_parameters():
            if 'qi' in n or 'ki' in n or 'phases' in n:
                saved[n] = p.data.clone()
                p.data.zero_()
        ab_logits = model(X_te)
        ab_acc = (ab_logits.argmax(-1) == Y_te).float().mean().item()
        for n, p in model.named_parameters():
            if n in saved:
                p.data.copy_(saved[n])

    delta = acc - ab_acc
    n_params = sum(p.numel() for p in model.parameters())

    metrics = compute_toroidal_metrics(phase_history, list(range(h)))
    all_toroidal[h] = metrics

    print(f"  Accuracy: {acc:.1%}  Phase delta: {delta:+.1%}  Params: {n_params}")
    print(f"  Mode-lock fraction: {metrics['mode_lock_fraction']:.1%}")
    print(f"  Mean dispersion: {metrics['mean_dispersion']:.4f}")
    vstr = ", ".join(f"{metrics['velocities'][hi]:.4f}" for hi in range(h))
    wstr = ", ".join(f"{metrics['windings'][hi]:.3f}" for hi in range(h))
    print(f"  Velocities: [{vstr}]")
    print(f"  Windings: [{wstr}]")

    print(f"  Frequency ratios (locked pairs):")
    locked_pairs = []
    unlocked_pairs = []
    for (i, j), (ratio, (num, den), err) in metrics['locking'].items():
        if num > 0 and err < 0.05:
            locked_pairs.append((i, j, f"{num}/{den}", ratio, err))
        elif ratio > 0.01:
            unlocked_pairs.append((i, j, ratio, err))

    for i, j, frac, ratio, err in locked_pairs:
        print(f"    head {i:>2} / head {j:>2}: {ratio:.4f} ~ {frac}  (err={err:.4f})")
    if unlocked_pairs:
        unlocked_str = ", ".join(f"({i},{j}):{r:.3f}" for i, j, r, err in unlocked_pairs[:4])
        print(f"    unlocked: {unlocked_str}..." if len(unlocked_pairs) > 4 else f"    unlocked: {unlocked_str}")

    print()

print("=" * 80)
print("TOROIDAL TOPOLOGY ANALYSIS")
print("=" * 80)

print(f"\n{'h':>4}  {'mode-lock':>10}  {'dispersion':>10}  {'delta':>8}  {'params':>7}")
print("-" * 50)
for h in heads_configs:
    m = all_toroidal[h]
    mlf = m['mode_lock_fraction']
    disp = m['mean_dispersion']
    print(f"  {h:>4}  {mlf:>9.1%}  {disp:>9.4f}         ...")

print()
print("ARNOLD TONGUE TEST:")
print("  Mode-locking fraction should grow with h (more head pairs = more locking opportunities)")
print("  But locking should be rational-ratio (mode-locked), not full-lock (all same frequency)")
print()

ml_vals = [all_toroidal[h]['mode_lock_fraction'] for h in heads_configs]
disp_vals = [all_toroidal[h]['mean_dispersion'] for h in heads_configs]

for i, h in enumerate(heads_configs):
    label = ""
    if i > 0:
        ml_growth = ml_vals[i] - ml_vals[i-1]
        disp_growth = disp_vals[i] - disp_vals[i-1]
        if ml_vals[i] > 0.5 and disp_vals[i] > 0.3:
            label = " <- MODE-LOCKED TORUS (rational ratios, high dispersion)"
        elif ml_vals[i] < 0.2 and disp_vals[i] < 0.2:
            label = " <- FULLY LOCKED (all same freq, D_f collapses)"
        elif ml_vals[i] > 0.3 and disp_vals[i] > 0.2:
            label = " <- PARTIAL MODE-LOCK"
        else:
            label = " <- INCOHERENT (no stable ratios)"
    print(f"  h={h}: ml={ml_vals[i]:.1%}  disp={disp_vals[i]:.3f}{label}")

print()

# Arnold tongue hypothesis test
if len(ml_vals) >= 3 and ml_vals[-1] > ml_vals[0]:
    print("MODE-LOCKING GROWS WITH HEAD COUNT:")
    print("  The Born rule merge induces rational frequency ratios between heads.")
    print("  Heads do NOT collapse to one frequency (D_f preserved).")
    print("  Heads DO establish stable phase relationships (cross-terms preserved).")
    print("  This is the structural explanation for anti-saturation.")
elif ml_vals[-1] < 0.2:
    print("NO MODE-LOCKING DETECTED:")
    print("  Heads are either fully locked (same frequency, D_f collapses to 1)")
    print("  or incoherent (no stable ratios, cross-terms random).")
    print("  Neither supports the Born rule anti-saturation claim.")

print()
print("DISPERSION TRAJECTORY (first 5 and last 5 epochs):")
for h in heads_configs:
    traj = all_toroidal[h]['dispersion_trajectory']
    start = traj[:5]
    end = traj[-5:]
    print(f"  h={h:>2}: start={[f'{d:.3f}' for d in start]}  end={[f'{d:.3f}' for d in end]}")
