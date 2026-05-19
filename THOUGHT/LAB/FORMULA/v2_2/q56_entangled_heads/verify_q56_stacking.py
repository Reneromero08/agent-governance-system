"""Q56 continuation: Head stacking architecture.

Three stacking strategies compared:
  FLAT-CLASSICAL: Independent Q/K/V per head, all concatenated (Q55 baseline)
  FLAT-BORN:     Independent Q/K/V per head, all Born-rule merged (Q56 baseline)
  CLUSTERED:     Heads grouped into k clusters. Intra-cluster Born merge.
                 Inter-cluster classical merge. Phase compounds across layers.

Prediction: Clustered resists saturation best — coherence domains manage
the coherence-length limit. Phase compounds across layers.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
torch.manual_seed(42); random.seed(42)


def gen_geometry(n=300, n_pts=8):
    X, Y = [], []
    for _ in range(n):
        t = random.randint(0, 3)
        zx = torch.randn(n_pts) * 2.0; zy = torch.randn(n_pts) * 2.0
        th = random.random() * 2 * math.pi
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


# ============================================================
# FLAT-CLASSICAL: all heads concatenated (Q55 baseline)
# ============================================================
class FlatClassical(nn.Module):
    def __init__(self, in_dim=6, mid_dim=4, total_heads=8, n_classes=4):
        super().__init__()
        self.total_heads = total_heads
        self.mid_dim = mid_dim
        self.qr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.qi = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.kr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.ki = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.vr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.vi = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.phases = nn.ParameterList([nn.Parameter(torch.tensor(random.random()*0.2)) for _ in range(total_heads)])
        self.out = nn.Linear(mid_dim * total_heads, n_classes)
        for mlist in [self.qr, self.qi, self.kr, self.ki, self.vr, self.vi]:
            for w in mlist:
                nn.init.normal_(w.weight, std=0.1)
        nn.init.normal_(self.out.weight, std=0.1)

    def forward(self, x, return_phase=False):
        outputs = []; phases = []
        for h in range(self.total_heads):
            score_i = (self.qi[h](x) * self.kr[h](x) - self.qr[h](x) * self.ki[h](x)).sum(-1)
            c, s = torch.cos(score_i.unsqueeze(-1)), torch.sin(score_i.unsqueeze(-1))
            zr, zi = c*self.vr[h](x) - s*self.vi[h](x), c*self.vi[h](x) + s*self.vr[h](x)
            c2, s2 = torch.cos(self.phases[h]), torch.sin(self.phases[h])
            outputs.append(zr*c2 - zi*s2)
            if return_phase:
                cm = torch.cos(score_i).mean(); sm = torch.sin(score_i).mean()
                phases.append((cm**2 + sm**2).sqrt())
        out = self.out(torch.cat(outputs, -1))
        return (out, torch.stack(phases).mean()) if return_phase else out


# ============================================================
# FLAT-BORN: all heads, one projective measurement (Q56 baseline)
# ============================================================
class FlatBorn(nn.Module):
    def __init__(self, in_dim=6, mid_dim=4, total_heads=8, n_classes=4):
        super().__init__()
        self.total_heads = total_heads
        self.mid_dim = mid_dim
        self.qr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.qi = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.kr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.ki = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.vr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.vi = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        angles = torch.linspace(0, 2*math.pi, total_heads+1)[:total_heads]
        self.phases = nn.ParameterList([nn.Parameter(a.unsqueeze(0).expand(1)) for a in angles])
        self.align = nn.Parameter(torch.randn(mid_dim, n_classes)*0.1)
        for mlist in [self.qr, self.qi, self.kr, self.ki, self.vr, self.vi]:
            for w in mlist:
                nn.init.normal_(w.weight, std=0.1)

    def forward(self, x, return_phase=False):
        B = x.shape[0]
        sum_r = torch.zeros(B, self.mid_dim, device=x.device)
        sum_i = torch.zeros_like(sum_r)
        phases = []
        for h in range(self.total_heads):
            score_i = (self.qi[h](x)*self.kr[h](x) - self.qr[h](x)*self.ki[h](x)).sum(-1)
            c, s = torch.cos(score_i.unsqueeze(-1)), torch.sin(score_i.unsqueeze(-1))
            zr, zi = c*self.vr[h](x) - s*self.vi[h](x), c*self.vi[h](x) + s*self.vr[h](x)
            c2, s2 = torch.cos(self.phases[h]), torch.sin(self.phases[h])
            zr, zi = zr*c2 - zi*s2, zr*s2 + zi*c2
            sum_r = sum_r + zr / math.sqrt(self.total_heads)
            sum_i = sum_i + zi / math.sqrt(self.total_heads)
            if return_phase:
                cm = torch.cos(score_i).mean(); sm = torch.sin(score_i).mean()
                phases.append((cm**2 + sm**2).sqrt())
        probs = (sum_r @ self.align)**2 + (sum_i @ self.align)**2
        return (probs, torch.stack(phases).mean()) if return_phase else probs


# ============================================================
# CLUSTERED: heads grouped into k clusters, Born within, classical across
# ============================================================
class ClusteredBorn(nn.Module):
    def __init__(self, in_dim=6, mid_dim=4, total_heads=8, n_clusters=2, n_classes=4):
        super().__init__()
        self.total_heads = total_heads
        self.n_clusters = n_clusters
        self.heads_per_cluster = total_heads // n_clusters
        self.mid_dim = mid_dim

        self.qr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.qi = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.kr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.ki = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.vr = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])
        self.vi = nn.ModuleList([nn.Linear(in_dim, mid_dim, False) for _ in range(total_heads)])

        for mlist in [self.qr, self.qi, self.kr, self.ki, self.vr, self.vi]:
            for w in mlist:
                nn.init.normal_(w.weight, std=0.1)

        all_angles = []
        for c_idx in range(n_clusters):
            hpc = self.heads_per_cluster
            cluster_angles = torch.linspace(c_idx*2*math.pi/n_clusters,
                                            (c_idx+1)*2*math.pi/n_clusters, hpc)
            all_angles.extend(cluster_angles.tolist())
        self.phases = nn.ParameterList([nn.Parameter(torch.tensor(a)) for a in all_angles])

        self.aligns = nn.ParameterList([
            nn.Parameter(torch.randn(mid_dim, mid_dim)*0.1) for _ in range(n_clusters)
        ])
        self.out = nn.Linear(mid_dim * n_clusters, n_classes)
        nn.init.normal_(self.out.weight, std=0.1)

    def forward(self, x, return_phase=False):
        B = x.shape[0]
        cluster_outputs = []
        all_phases = []

        for c_idx in range(self.n_clusters):
            start = c_idx * self.heads_per_cluster
            end = start + self.heads_per_cluster

            sum_r = torch.zeros(B, self.mid_dim, device=x.device)
            sum_i = torch.zeros_like(sum_r)

            for h in range(start, end):
                score_i = (self.qi[h](x)*self.kr[h](x) - self.qr[h](x)*self.ki[h](x)).sum(-1)
                c, s = torch.cos(score_i.unsqueeze(-1)), torch.sin(score_i.unsqueeze(-1))
                zr, zi = c*self.vr[h](x) - s*self.vi[h](x), c*self.vi[h](x) + s*self.vr[h](x)
                c2, s2 = torch.cos(self.phases[h]), torch.sin(self.phases[h])
                zr, zi = zr*c2 - zi*s2, zr*s2 + zi*c2
                sum_r = sum_r + zr / math.sqrt(self.heads_per_cluster)
                sum_i = sum_i + zi / math.sqrt(self.heads_per_cluster)
                if return_phase:
                    cm = torch.cos(score_i).mean(); sm = torch.sin(score_i).mean()
                    all_phases.append((cm**2 + sm**2).sqrt())

            cluster_out_r = sum_r @ self.aligns[c_idx]
            cluster_out_i = sum_i @ self.aligns[c_idx]
            cluster_mag = (cluster_out_r**2 + cluster_out_i**2).sqrt()
            cluster_outputs.append(cluster_mag)

        merged = torch.cat(cluster_outputs, -1)
        out = self.out(merged)
        return (out, torch.stack(all_phases).mean()) if return_phase else out


# ============================================================
# TRAINING HARNESS
# ============================================================
def train_eval_factory(model_factory, X_tr, Y_tr, X_te, Y_te, epochs=120):
    model = model_factory()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())

    for e in range(epochs):
        loss = F.cross_entropy(model(X_tr), Y_tr)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        logits, pc = model(X_te, return_phase=True)
        acc = (logits.argmax(-1) == Y_te).float().mean().item()

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

    return acc, ab_acc, acc-ab_acc, pc.item(), n_params


X_raw, Y_raw = gen_geometry(600)
X_tr, Y_tr = X_raw[:400], Y_raw[:400]
X_te, Y_te = X_raw[400:], Y_raw[400:]

print("=" * 80)
print("HEAD STACKING ARCHITECTURE COMPARISON")
print("=" * 80)
print()

total_heads_list = [2, 4, 8, 16, 32]
mid_dim = 4

architectures = {
    "FLAT-CLASSICAL  (concat all)": lambda h: FlatClassical(total_heads=h, mid_dim=mid_dim),
    "FLAT-BORN       (proj all)  ": lambda h: FlatBorn(total_heads=h, mid_dim=mid_dim),
    "CLUSTERED=k2    (groups of h/2)": lambda h: ClusteredBorn(total_heads=h, mid_dim=mid_dim, n_clusters=2),
    "CLUSTERED=k4    (groups of h/4)": lambda h: ClusteredBorn(total_heads=h, mid_dim=mid_dim, n_clusters=4),
}

all_results = {}

for arch_name, factory in architectures.items():
    print(f"--- {arch_name} ---")
    print(f"  {'h':>4}  {'acc':>7}  {'ab_acc':>7}  {'delta':>7}  {'pc':>8}  {'params':>7}")
    print("  " + "-" * 58)
    results = []
    for h in total_heads_list:
        try:
            acc, ab_acc, delta, pc, n_p = train_eval_factory(
                lambda h=h: factory(h), X_tr, Y_tr, X_te, Y_te)
            results.append((h, acc, ab_acc, delta, pc, n_p))
            print(f"  {h:>4}  {acc:>7.1%}  {ab_acc:>7.1%}  {delta:>+6.1%}  {pc:>7.4f}  {n_p:>7}")
        except Exception as e:
            print(f"  {h:>4}  ERROR: {e}")
    all_results[arch_name] = results
    print()

print("=" * 80)
print("STACKING ANALYSIS")
print("=" * 80)

print(f"\n{'arch':<28} {'delta sum':>10} {'max delta':>10} {'sat. ratio':>11} {'params':>8}")
print("-" * 75)
for arch_name, results in all_results.items():
    deltas = [r[3] for r in results if len(r) > 3]
    d_sum = sum(deltas)
    d_max = max(deltas) if deltas else 0
    if len(deltas) >= 2:
        d_first = deltas[0]; d_last = deltas[-1]
        sat_ratio = d_last / max(d_first, 1e-6)
    else:
        sat_ratio = 1.0
    n_p = results[-1][5] if results else 0
    print(f"  {arch_name:<28} {d_sum:>+9.1%} {d_max:>+9.1%} {sat_ratio:>10.2f}x {n_p:>8}")

print()
print("Key:")
print("  delta sum = total phase information carried across all h")
print("  max delta = peak phase ablation delta")
print("  sat ratio = delta(h_max)/delta(h_min) — >1 means phase GROWS with heads")
print()
print("BEST ARCHITECTURE: highest delta sum + highest sat ratio")
