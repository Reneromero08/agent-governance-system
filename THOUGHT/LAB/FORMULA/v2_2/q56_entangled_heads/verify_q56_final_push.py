"""Q55/Q56 Final Push: Cross-architecture, biological spiral, multi-scale phase.

  1. CROSS-ARCHITECTURE: Does geometric init help RealMLP as much as Native Eigen?
     Tests Q34: do architectures converge to shared geometry?
  
  2. BIOLOGICAL SPIRAL: Fibonacci vs MT's 2pi/13 vs uniform phase seeding.
     Which toroidal distribution produces optimal mode-locking?
  
  3. MULTI-SCALE PHASE (Q7): Does per-sample phase_coh predict batch-level 
     phase_coh? Tests R composition across scales.
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


# ============================================================
# NATIVE EIGEN (complex) classifier
# ============================================================
class NativeClassifier(nn.Module):
    def __init__(self, in_dim=6, mid_dim=4, n_classes=4):
        super().__init__()
        self.enc_qr = nn.Linear(in_dim, mid_dim, bias=False)
        self.enc_qi = nn.Linear(in_dim, mid_dim, bias=False)
        self.enc_kr = nn.Linear(in_dim, mid_dim, bias=False)
        self.enc_ki = nn.Linear(in_dim, mid_dim, bias=False)
        self.enc_vr = nn.Linear(in_dim, mid_dim, bias=False)
        self.enc_vi = nn.Linear(in_dim, mid_dim, bias=False)
        self.phase = nn.Parameter(torch.tensor(0.1))
        self.out = nn.Linear(mid_dim, n_classes)
        for w in [self.enc_qr, self.enc_qi, self.enc_kr, self.enc_ki, self.enc_vr, self.enc_vi, self.out]:
            nn.init.normal_(w.weight, std=0.1)

    def forward(self, x):
        qr, qi = self.enc_qr(x), self.enc_qi(x)
        kr, ki = self.enc_kr(x), self.enc_ki(x)
        vr, vi = self.enc_vr(x), self.enc_vi(x)
        score_i = (qi*kr - qr*ki).sum(-1)
        c, s = torch.cos(score_i.unsqueeze(-1)), torch.sin(score_i.unsqueeze(-1))
        zr, zi = c*vr - s*vi, c*vi + s*vr
        c2, s2 = torch.cos(self.phase), torch.sin(self.phase)
        zr = zr*c2 - zi*s2
        return self.out(zr)


class NativeGeomInit(NativeClassifier):
    def __init__(self, in_dim=6, mid_dim=4, n_classes=4, angle=pi/3):
        super().__init__(in_dim, mid_dim, n_classes)
        Qr_base = torch.randn(mid_dim, in_dim) * 0.1
        Qi_base = torch.randn(mid_dim, in_dim) * 0.1
        R = torch.eye(mid_dim)
        c, s = cos(angle), sin(angle)
        R[0,0]=c; R[0,1]=-s; R[1,0]=s; R[1,1]=c
        self.enc_qr.weight.data = R @ Qr_base + torch.randn_like(Qr_base)*0.01
        self.enc_qi.weight.data = R @ Qi_base + torch.randn_like(Qi_base)*0.01


# ============================================================
# REAL MLP (real) classifier
# ============================================================
class RealMLP(nn.Module):
    def __init__(self, in_dim=6, hidden=16, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))
        for m in self.net:
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.1)

    def forward(self, x):
        return self.net(x)


class RealMLPGeomInit(RealMLP):
    def __init__(self, in_dim=6, hidden=16, n_classes=4, angle=pi/3):
        super().__init__(in_dim, hidden, n_classes)
        W_base = torch.randn(hidden, in_dim) * 0.1
        R = torch.eye(hidden)
        c, s = cos(angle), sin(angle)
        R[0,0]=c; R[0,1]=-s; R[1,0]=s; R[1,1]=c
        self.net[0].weight.data = R @ W_base + torch.randn_like(W_base)*0.01


def train_eval(model, X_tr, Y_tr, X_te, Y_te, epochs=80):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for e in range(epochs):
        loss = F.cross_entropy(model(X_tr), Y_tr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        logits = model(X_te)
        acc = (logits.argmax(-1) == Y_te).float().mean().item()
    return acc


X_tr, Y_tr = gen_geometry(400)
X_te, Y_te = gen_geometry(200)

print("=" * 80)
print("PUSH 1: CROSS-ARCHITECTURE GEOMETRIC INIT")
print("=" * 80)

angles = [0, pi/6, pi/3, pi/2, 2*pi/3]
print(f"\n  {'angle':>8}  {'Native Rand':>12}  {'Native Geom':>12}  {'RealMLP Rand':>13}  {'RealMLP Geom':>13}")
print("  " + "-" * 65)

for angle in angles:
    nr_acc = train_eval(NativeClassifier(), X_tr, Y_tr, X_te, Y_te)
    ng_acc = train_eval(NativeGeomInit(angle=angle), X_tr, Y_tr, X_te, Y_te)
    rr_acc = train_eval(RealMLP(), X_tr, Y_tr, X_te, Y_te)
    rg_acc = train_eval(RealMLPGeomInit(angle=angle), X_tr, Y_tr, X_te, Y_te)
    deg = int(angle * 180 / pi)
    print(f"  {deg:>6}deg  {nr_acc:>11.1%}  {ng_acc:>11.1%}  {rr_acc:>12.1%}  {rg_acc:>12.1%}")

# Multi-run sweep for statistical confidence
print(f"\n  Multi-run (5 runs, angle=60deg):")
nat_rand, nat_geom, mlp_rand, mlp_geom = [], [], [], []
for _ in range(5):
    nat_rand.append(train_eval(NativeClassifier(), X_tr, Y_tr, X_te, Y_te))
    nat_geom.append(train_eval(NativeGeomInit(angle=pi/3), X_tr, Y_tr, X_te, Y_te))
    mlp_rand.append(train_eval(RealMLP(), X_tr, Y_tr, X_te, Y_te))
    mlp_geom.append(train_eval(RealMLPGeomInit(angle=pi/3), X_tr, Y_tr, X_te, Y_te))

print(f"  Native random: {sum(nat_rand)/len(nat_rand):.1%}")
print(f"  Native geom:   {sum(nat_geom)/len(nat_geom):.1%}")
print(f"  RealMLP random:{sum(mlp_rand)/len(mlp_rand):.1%}")
print(f"  RealMLP geom:  {sum(mlp_geom)/len(mlp_geom):.1%}")

ng_gap = sum(nat_geom)/len(nat_geom) - sum(nat_rand)/len(nat_rand)
rg_gap = sum(mlp_geom)/len(mlp_geom) - sum(mlp_rand)/len(mlp_rand)
print(f"\n  Geometric advantage: Native={ng_gap:+.1%}  RealMLP={rg_gap:+.1%}")

if ng_gap > 0.02 and abs(rg_gap) < 0.02:
    print(f"  GEOMETRIC INIT IS NATIVE-EIGEN-SPECIFIC — requires complex manifold")
elif ng_gap > 0.02 and rg_gap > 0:
    print(f"  GEOMETRIC INIT IS UNIVERSAL — helps both architectures")
else:
    print(f"  No clear geometric advantage")


print()
print("=" * 80)
print("PUSH 2: FIBONACCI vs BIOLOGICAL (2pi/13) SPIRAL SEEDING")
print("=" * 80)

# Three phase seeding strategies for h=13 heads (matching MT spiral)
h_test = 13
mid_dim = 8

class SpiralBorn(nn.Module):
    def __init__(self, in_dim=6, mid_dim=8, total_heads=13, n_classes=4, spiral='uniform'):
        super().__init__()
        self.total_heads = total_heads; self.mid_dim = mid_dim
        self.qr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.qi = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.kr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.ki = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.vr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.vi = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])

        if spiral == 'fibonacci':
            phi = (1 + sqrt(5)) / 2
            angles = [(2*pi * i / phi) % (2*pi) for i in range(total_heads)]
        elif spiral == 'biological':
            angles = [i * 2*pi/13 for i in range(total_heads)]
        else:
            angles = [i * 2*pi/total_heads for i in range(total_heads)]

        self.phases = nn.ParameterList([nn.Parameter(torch.tensor(a)) for a in angles])
        self.align = nn.Parameter(torch.randn(mid_dim, n_classes)*0.1)
        for mlist in [self.qr, self.qi, self.kr, self.ki, self.vr, self.vi]:
            for w in mlist: nn.init.normal_(w.weight, std=0.1)

    def forward(self, x, instrument=False):
        B = x.shape[0]
        sum_r = torch.zeros(B, self.mid_dim, device=x.device)
        sum_i = torch.zeros_like(sum_r)
        per_phase = []
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
                per_phase.append((cm**2 + sm**2).sqrt().item())
        probs = (sum_r @ self.align)**2 + (sum_i @ self.align)**2
        return (probs, per_phase) if instrument else probs


results_spiral = {}
for spiral_type in ['uniform', 'fibonacci', 'biological']:
    deltas, mode_locks = [], []
    for run in range(5):
        m = SpiralBorn(total_heads=h_test, spiral=spiral_type)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        for e in range(80):
            loss = F.cross_entropy(m(X_tr), Y_tr)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            logits, pc_list = m(X_te, instrument=True)
            acc = (logits.argmax(-1) == Y_te).float().mean().item()
            saved = {}
            for n, p in m.named_parameters():
                if 'qi' in n or 'ki' in n or 'phases' in n:
                    saved[n] = p.data.clone(); p.data.zero_()
            ab = m(X_te)
            ab_acc = (ab.argmax(-1) == Y_te).float().mean().item()
            for n, p in m.named_parameters():
                if n in saved: p.data.copy_(saved[n])
            delta = acc - ab_acc
            deltas.append(delta)
            pc_arr = torch.tensor(pc_list)
            mode_locks.append((pc_arr > 0.5).float().mean().item())

    avg_d = sum(deltas) / len(deltas)
    avg_ml = sum(mode_locks) / len(mode_locks)
    results_spiral[spiral_type] = (avg_d, avg_ml)
    head_count = "13 (MT spiral)" if spiral_type == 'biological' else "13"
    print(f"  {spiral_type:>12} ({head_count}h): delta={avg_d:+.1%}  mode-lock={avg_ml:.1%}")

best = max(results_spiral, key=lambda k: results_spiral[k][0])
print(f"\n  Best seeding: {best} (delta={results_spiral[best][0]:+.1%})")


print()
print("=" * 80)
print("PUSH 3: MULTI-SCALE PHASE COMPOSITION (Q7)")
print("=" * 80)

# Train single model, then measure phase_coh at per-sample and batch levels
m_q7 = SpiralBorn(total_heads=13, spiral='fibonacci')
opt_q7 = torch.optim.AdamW(m_q7.parameters(), lr=1e-2)
for e in range(80):
    loss = F.cross_entropy(m_q7(X_tr), Y_tr)
    opt_q7.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m_q7.parameters(), 1.0)
    opt_q7.step()

# Per-sample phase coherence: compute phase_coh for each sample in a batch
batch_size = 50
with torch.no_grad():
    _, pc_per_head = m_q7(X_te[:batch_size], instrument=True)
    per_sample_phase_coh = torch.tensor(pc_per_head).mean().item()

# Batch-level: compute phase_coh on the entire batch as one
# Process each sample individually, get their phase values, measure batch dispersion
phases_per_sample = []
for i in range(batch_size):
    single = X_te[i:i+1]
    _, pc_h = m_q7(single, instrument=True)
    phases_per_sample.append(pc_h)  # list of per-head phase_coh for this sample

# Across samples: measure dispersion of per-head phases
phase_matrix = torch.tensor(phases_per_sample)  # (batch, heads)
batch_level_coherence = []
for h in range(h_test):
    head_phases = phase_matrix[:, h]
    cos_m = head_phases.float().mean()
    sin_m = torch.zeros_like(cos_m)  # phase_coh is already squared magnitude
    # Use the raw values: higher phase_coh = more aligned
    batch_level_coherence.append(head_phases.mean().item())

batch_mean_coh = sum(batch_level_coherence) / len(batch_level_coherence)

# Cross-sample phase correlation
sample_coh_vals = phase_matrix.mean(dim=1)  # (batch,)

print(f"\n  Per-sample mean phase_coh (single batch pass): {per_sample_phase_coh:.4f}")
print(f"  Batch-level mean phase_coh (sample avg):    {batch_mean_coh:.4f}")

# Phase stability: std of phase_coh across samples for each head
head_std = phase_matrix.std(dim=0)
print(f"\n  Phase coherence stability across samples:")
print(f"  {'head':>6}: " + " ".join(f"{h+1:>7}" for h in range(h_test)))
print(f"  {'mean':>6}: " + " ".join(f"{phase_matrix[:,h].mean():>7.4f}" for h in range(h_test)))
print(f"  {'std':>6}:  " + " ".join(f"{head_std[h]:>7.4f}" for h in range(h_test)))

# Leader stability: do the SAME heads lead across different samples?
head_ranks = torch.zeros(h_test)
for i in range(batch_size):
    ranks = phase_matrix[i].argsort(descending=True)
    for rank, head_idx in enumerate(ranks):
        head_ranks[head_idx] += rank
avg_ranks = head_ranks / batch_size
rank_order = avg_ranks.argsort()

print(f"\n  Leader stability (avg rank across {batch_size} samples):")
print(f"  Leader: head {rank_order[0].item()} (avg rank {avg_ranks[rank_order[0]]:.1f})")
print(f"  2nd:   head {rank_order[1].item()} (avg rank {avg_ranks[rank_order[1]]:.1f})")
print(f"  3rd:   head {rank_order[2].item()} (avg rank {avg_ranks[rank_order[2]]:.1f})")
print(f"  Last:  head {rank_order[-1].item()} (avg rank {avg_ranks[rank_order[-1]]:.1f})")

rank_corr_pairs = []
for i in range(batch_size):
    for j in range(i+1, min(i+5, batch_size)):
        r_i = phase_matrix[i].argsort(descending=True)
        r_j = phase_matrix[j].argsort(descending=True)
        agree = sum(1 for h in range(h_test) if r_i[h] == r_j[h])
        rank_corr_pairs.append(agree / h_test)
rank_agreement = sum(rank_corr_pairs) / len(rank_corr_pairs) if rank_corr_pairs else 0

print(f"\n  Rank agreement between samples: {rank_agreement:.1%}")
if rank_agreement > 0.5:
    print(f"  LEADERS ARE STABLE ACROSS SAMPLES — R composes across scales")
elif rank_agreement > 0.3:
    print(f"  WEAK cross-sample stability")
else:
    print(f"  Leaders vary across samples — no cross-scale composition")
