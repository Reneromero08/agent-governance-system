"""Q56 Discovery 4-6: Head utility score, C transfer, re-init vs prune.

  4. HEAD UTILITY SCORE: phase_coh + entropy -> predict keep/drop BEFORE training ends
  5. C CROSS-TASK TRANSFER: freeze C from task A, test on task B
  6. RE-INIT vs PRUNE: can dead heads be awakened mid-training?
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, copy
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


def gen_alt_task(n=300):
    """Different task: classify magnitude patterns instead of geometric transforms."""
    X, Y = [], []
    for _ in range(n):
        t = random.randint(0, 3)
        zx = torch.randn(8) * 2.0
        if t == 0:
            zy = zx * 0.5  # shrink
        elif t == 1:
            zy = zx * 2.0  # expand
        elif t == 2:
            zy = zx**2  # quadratic
        else:
            zy = torch.sin(zx * pi)  # wave
        ratio = torch.complex(zx, zy) / (torch.complex(zx, zy) + 1e-8)
        feats = torch.stack([
            torch.cos(torch.angle(ratio)).mean(), torch.sin(torch.angle(ratio)).mean(),
            torch.cos(torch.angle(ratio)).std(), torch.sin(torch.angle(ratio)).std(),
            torch.abs(ratio).mean(), torch.abs(ratio).std(),
        ])
        X.append(feats); Y.append(t)
    return torch.stack(X), torch.tensor(Y)


class BornWithInstruments(nn.Module):
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

    def forward(self, x, instrument=False):
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
            if instrument:
                cm = torch.cos(score_i).mean(); sm = torch.sin(score_i).mean()
                phase_coh.append((cm**2 + sm**2).sqrt().item())
        probs = (sum_r @ self.align)**2 + (sum_i @ self.align)**2
        return (probs, phase_coh) if instrument else probs


def phase_ablate(model, X_te, Y_te):
    with torch.no_grad():
        logits = model(X_te)
        acc = (logits.argmax(-1) == Y_te).float().mean().item()
        saved = {}
        for n, p in model.named_parameters():
            if 'qi' in n or 'ki' in n or 'phases' in n:
                saved[n] = p.data.clone(); p.data.zero_()
        ab = model(X_te)
        ab_acc = (ab.argmax(-1) == Y_te).float().mean().item()
        for n, p in model.named_parameters():
            if n in saved: p.data.copy_(saved[n])
    return acc, ab_acc, acc - ab_acc


print("=" * 80)
print("DISCOVERY 4: HEAD UTILITY SCORE")
print("=" * 80)

X_tr, Y_tr = gen_geometry(400)
X_te, Y_te = gen_geometry(200)
h = 16

# Train with tracking at epochs 10, 30, 60, 100
model = BornWithInstruments(mid_dim=8, total_heads=h)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
utility_at_epoch = {}

for epoch in range(120):
    loss = F.cross_entropy(model(X_tr), Y_tr)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if epoch in [10, 30, 60, 100]:
        with torch.no_grad():
            _, phase_coh = model(X_te, instrument=True)
            utility_at_epoch[epoch] = phase_coh

# Compute utility: phase_coh at epoch e as predictor of final leader status
final_pc = utility_at_epoch[100]  # final values
is_leader = [pc > 0.5 for pc in final_pc]

print(f"\n  Can early phase_coh predict final leader status?")
print(f"  {'epoch':>8} {'pred acc':>9} {'precision':>10} {'recall':>10}")
print("  " + "-" * 45)
for e in [10, 30, 60]:
    early_pc = utility_at_epoch[e]
    early_predict = [pc > 0.5 for pc in early_pc]
    correct = sum(1 for ep, fl in zip(early_predict, is_leader) if ep == fl)
    acc = correct / h
    tp = sum(1 for ep, fl in zip(early_predict, is_leader) if ep and fl)
    fp = sum(1 for ep, fl in zip(early_predict, is_leader) if ep and not fl)
    fn = sum(1 for ep, fl in zip(early_predict, is_leader) if not ep and fl)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"  {e:>8} {acc:>8.1%} {prec:>9.1%} {rec:>9.1%}")

# Utility heuristic: U = phase_coh - lambda * (1 - phase_coh_rank)
# Higher U = keep. Sort by U, keep top-k, measure delta.
final_rank = sorted(range(h), key=lambda i: final_pc[i], reverse=True)
print(f"\n  Head utility ranking (phase_coh):")
print(f"  {'rank':>6} {'head':>6} {'phase_coh':>10} {'utility':>10}")
for r, hh in enumerate(final_rank):
    print(f"  {r:>6} {hh:>6} {final_pc[hh]:>10.4f} {final_pc[hh]:>10.4f}")

print()
print("=" * 80)
print("DISCOVERY 5: C CROSS-TASK TRANSFER")
print("=" * 80)

# Task B: same structure, different feature space (translation of points)
# Same 4 classes but with shifted origin — tests C transfer on similar task
X_tr_alt, Y_tr_alt = gen_geometry(400)
X_te_alt, Y_te_alt = gen_geometry(200)

# Shift the alt task features: same geometric structure, different coordinate frame
shift = torch.randn(6) * 0.5
X_tr_alt = X_tr_alt + shift
X_te_alt = X_te_alt + shift

model_a = BornWithInstruments(mid_dim=8, total_heads=8)
opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-2)
for e in range(80):
    loss = F.cross_entropy(model_a(X_tr), Y_tr)
    opt_a.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
    opt_a.step()

frozen_align = model_a.align.data.clone()
acc_a, ab_a, delta_a = phase_ablate(model_a, X_te, Y_te)

# Test 1: train new model from scratch on alt task
model_b_scratch = BornWithInstruments(mid_dim=8, total_heads=8)
opt_b = torch.optim.AdamW(model_b_scratch.parameters(), lr=1e-2)
for e in range(80):
    loss = F.cross_entropy(model_b_scratch(X_tr_alt), Y_tr_alt)
    opt_b.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model_b_scratch.parameters(), 1.0)
    opt_b.step()
acc_scratch, ab_scratch, delta_scratch = phase_ablate(model_b_scratch, X_te_alt, Y_te_alt)

# Test 2: train new model on alt task WITH frozen C from task A
model_b_transfer = BornWithInstruments(mid_dim=8, total_heads=8)
model_b_transfer.align.data = frozen_align.clone()
model_b_transfer.align.requires_grad = False  # freeze C
opt_bt = torch.optim.AdamW([p for n, p in model_b_transfer.named_parameters() if 'align' not in n], lr=1e-2)
for e in range(80):
    loss = F.cross_entropy(model_b_transfer(X_tr_alt), Y_tr_alt)
    opt_bt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model_b_transfer.parameters(), 1.0)
    opt_bt.step()
acc_transfer, ab_transfer, delta_transfer = phase_ablate(model_b_transfer, X_te_alt, Y_te_alt)

# Test 3: train new model WHERE C is free to adapt (baseline for transfer comparison)
model_b_fine = BornWithInstruments(mid_dim=8, total_heads=8)
model_b_fine.align.data = frozen_align.clone()  # warm start C
opt_bf = torch.optim.AdamW(model_b_fine.parameters(), lr=1e-2)
for e in range(80):
    loss = F.cross_entropy(model_b_fine(X_tr_alt), Y_tr_alt)
    opt_bf.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model_b_fine.parameters(), 1.0)
    opt_bf.step()
acc_fine, ab_fine, delta_fine = phase_ablate(model_b_fine, X_te_alt, Y_te_alt)

print(f"\n  Task A (geometry) C -> Task B (magnitude) transfer:")
print(f"  {'config':>22} {'acc':>7} {'ab_acc':>7} {'delta':>7}")
print("  " + "-" * 45)
print(f"  {'scratch (no transfer)':>22} {acc_scratch:>7.1%} {ab_scratch:>7.1%} {delta_scratch:>+6.1%}")
print(f"  {'frozen C from task A':>22} {acc_transfer:>7.1%} {ab_transfer:>7.1%} {delta_transfer:>+6.1%}")
print(f"  {'warm-start C from A':>22} {acc_fine:>7.1%} {ab_fine:>7.1%} {delta_fine:>+6.1%}")

C_drift_frozen = (model_b_transfer.align.data - frozen_align).norm().item()
C_drift_warm = (model_b_fine.align.data - frozen_align).norm().item()
print(f"\n  C drift from frozen: frozen={C_drift_frozen:.6f}, warm-start={C_drift_warm:.4f}")

if delta_transfer > delta_scratch * 0.8:
    print(f"  C TRANSFERS — pointer state holds {delta_transfer/delta_scratch:.0%} of scratch delta")
else:
    print(f"  C is task-specific — frozen loses {(delta_scratch-delta_transfer)/max(delta_scratch,1e-6):.0%}")

print()
print("=" * 80)
print("DISCOVERY 6: RE-INIT vs PRUNE DEAD HEADS")
print("=" * 80)

# Train 16-head model. At epoch 60, identify dead/laggard heads.
# Variant A: Prune them (remove weights)
# Variant B: Re-init them (random reset, more epochs)
# Variant C: Do nothing (baseline)
h_full = 24
mid_epoch = 80
extra_epochs = 60
np_runs = 5
results_6 = {'baseline': [], 'prune': [], 'reinit': []}

for run in range(np_runs):
    m = BornWithInstruments(mid_dim=8, total_heads=h_full)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for e in range(mid_epoch):
        loss = F.cross_entropy(m(X_tr), Y_tr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        _, pc = m(X_te, instrument=True)
    dead = [hh for hh in range(h_full) if pc[hh] < 0.2]
    leaders = [hh for hh in range(h_full) if pc[hh] > 0.7]

    # Baseline
    m_base = copy.deepcopy(m)
    opt_base = torch.optim.AdamW(m_base.parameters(), lr=1e-2)
    for e in range(extra_epochs):
        loss = F.cross_entropy(m_base(X_tr), Y_tr)
        opt_base.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m_base.parameters(), 1.0)
        opt_base.step()
    acc_b, ab_b, delta_b = phase_ablate(m_base, X_te, Y_te)
    pcb = [pc[i] for i in dead] if dead else [0]
    results_6['baseline'].append((delta_b, len(dead), sum(pcb)/len(pcb)))

    # Prune dead
    if len(dead) >= 1 and len(dead) < h_full:
        kept = [i for i in range(h_full) if i not in dead]
        m_prune = BornWithInstruments(mid_dim=8, total_heads=len(kept))
        for dst, src in enumerate(kept):
            m_prune.qr[dst].weight.data = m.qr[src].weight.data.clone()
            m_prune.qi[dst].weight.data = m.qi[src].weight.data.clone()
            m_prune.kr[dst].weight.data = m.kr[src].weight.data.clone()
            m_prune.ki[dst].weight.data = m.ki[src].weight.data.clone()
            m_prune.vr[dst].weight.data = m.vr[src].weight.data.clone()
            m_prune.vi[dst].weight.data = m.vi[src].weight.data.clone()
            m_prune.phases[dst].data = m.phases[src].data.clone()
        m_prune.align.data = m.align.data.clone()
        opt_p = torch.optim.AdamW(m_prune.parameters(), lr=1e-2)
        for e in range(extra_epochs):
            loss = F.cross_entropy(m_prune(X_tr), Y_tr)
            opt_p.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m_prune.parameters(), 1.0)
            opt_p.step()
        acc_p, ab_p, delta_p = phase_ablate(m_prune, X_te, Y_te)
        results_6['prune'].append((delta_p, len(dead)))

    # Re-init dead
    if len(dead) >= 1:
        m_reinit = copy.deepcopy(m)
        for hh in dead:
            nn.init.normal_(m_reinit.qr[hh].weight, std=0.1)
            nn.init.normal_(m_reinit.qi[hh].weight, std=0.1)
            nn.init.normal_(m_reinit.kr[hh].weight, std=0.1)
            nn.init.normal_(m_reinit.ki[hh].weight, std=0.1)
            nn.init.normal_(m_reinit.vr[hh].weight, std=0.1)
            nn.init.normal_(m_reinit.vi[hh].weight, std=0.1)
            m_reinit.phases[hh].data = torch.tensor(random.random() * 2 * pi)
        opt_r = torch.optim.AdamW(m_reinit.parameters(), lr=1e-2)
        for e in range(extra_epochs):
            loss = F.cross_entropy(m_reinit(X_tr), Y_tr)
            opt_r.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m_reinit.parameters(), 1.0)
            opt_r.step()
        acc_r, ab_r, delta_r = phase_ablate(m_reinit, X_te, Y_te)
        results_6['reinit'].append((delta_r, len(dead)))

print(f"  Mid-epoch analysis (epoch {mid_epoch}) -> {extra_epochs} more epochs:")
print(f"  {'method':>12} {'delta':>8} {'n_dead':>8}")
print("  " + "-" * 32)
for method in ['baseline', 'prune', 'reinit']:
    if results_6[method]:
        avg_d = sum(r[0] for r in results_6[method]) / len(results_6[method])
        avg_n = sum(r[1] for r in results_6[method]) / len(results_6[method])
        print(f"  {method:>12} {avg_d:>+7.1%} {avg_n:>8.1f}")

# Conclusion
base_d = sum(r[0] for r in results_6['baseline']) / len(results_6['baseline'])
if results_6['reinit']:
    reinit_d = sum(r[0] for r in results_6['reinit']) / len(results_6['reinit'])
    if reinit_d > base_d:
        print(f"\n  RE-INIT BEATS BASELINE — dead heads can be awakened")
    else:
        print(f"\n  Re-init doesn't help — dead heads stay dead")
if results_6['prune']:
    prune_d = sum(r[0] for r in results_6['prune']) / len(results_6['prune'])
    if prune_d > base_d:
        print(f"  PRUNE BEATS BASELINE — removing noise improves signal")
    elif abs(prune_d - base_d) < 0.02:
        print(f"  Prune == baseline — dead heads are truly neutral")
