"""Q56 v2: True entangled architecture — independent Q/K/V per head (D_f=h),
projective measurement merge (Born rule) vs classical concatenation merge.

Isolates merge mechanism: both architectures use independent encoders.
Difference is only in how head outputs combine.
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


class EntangledHeads(nn.Module):
    """Independent Q/K/V per head, projective measurement merge (interference)."""
    def __init__(self, in_dim=6, mid_dim=4, n_heads=2, n_classes=4):
        super().__init__()
        self.n_heads = n_heads
        self.enc_qr = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
        self.enc_qi = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
        self.enc_kr = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
        self.enc_ki = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
        self.enc_vr = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
        self.enc_vi = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
        angles = torch.linspace(0, 2*math.pi, n_heads + 1)[:n_heads]
        self.phases = nn.ParameterList([nn.Parameter(angle.unsqueeze(0).expand(1)) for angle in angles])
        self.align = nn.Parameter(torch.randn(mid_dim, n_classes) * 0.1)
        for mlist in [self.enc_qr, self.enc_qi, self.enc_kr, self.enc_ki, self.enc_vr, self.enc_vi]:
            for w in mlist:
                nn.init.normal_(w.weight, std=0.1)

    def forward(self, x, return_phase=False):
        B = x.shape[0]
        head_sum_r = torch.zeros(B, self.enc_vr[0].weight.shape[0], device=x.device)
        head_sum_i = torch.zeros_like(head_sum_r)
        all_phases = []

        for h in range(self.n_heads):
            qr = self.enc_qr[h](x); qi = self.enc_qi[h](x)
            kr = self.enc_kr[h](x); ki = self.enc_ki[h](x)
            vr = self.enc_vr[h](x); vi = self.enc_vi[h](x)
            score_i = (qi * kr - qr * ki).sum(dim=-1)
            c, s = torch.cos(score_i.unsqueeze(-1)), torch.sin(score_i.unsqueeze(-1))
            zr, zi = c * vr - s * vi, c * vi + s * vr
            c2, s2 = torch.cos(self.phases[h]), torch.sin(self.phases[h])
            zr = zr * c2 - zi * s2; zi = zr * s2 + zi * c2
            head_sum_r = head_sum_r + zr / math.sqrt(self.n_heads)
            head_sum_i = head_sum_i + zi / math.sqrt(self.n_heads)
            if return_phase:
                cos_mean = torch.cos(score_i).mean()
                sin_mean = torch.sin(score_i).mean()
                all_phases.append((cos_mean**2 + sin_mean**2).sqrt())

        probs = (head_sum_r @ self.align)**2 + (head_sum_i @ self.align)**2
        if return_phase:
            return probs, torch.stack(all_phases).mean() if all_phases else torch.tensor(0.0)
        return probs


class ClassicalHeads(nn.Module):
    """Independent Q/K/V per head, concatenation merge (baseline)."""
    def __init__(self, in_dim=6, mid_dim=4, n_heads=2, n_classes=4):
        super().__init__()
        self.n_heads = n_heads
        self.enc_qr = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
        self.enc_qi = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
        self.enc_kr = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
        self.enc_ki = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
        self.enc_vr = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
        self.enc_vi = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
        self.phases = nn.ParameterList([nn.Parameter(torch.tensor(random.random() * 0.2)) for _ in range(n_heads)])
        self.out = nn.Linear(mid_dim * n_heads, n_classes)
        for mlist in [self.enc_qr, self.enc_qi, self.enc_kr, self.enc_ki, self.enc_vr, self.enc_vi]:
            for w in mlist:
                nn.init.normal_(w.weight, std=0.1)
        nn.init.normal_(self.out.weight, std=0.1)

    def forward(self, x, return_phase=False):
        B = x.shape[0]
        head_outputs = []; all_phases = []
        for h in range(self.n_heads):
            qr = self.enc_qr[h](x); qi = self.enc_qi[h](x)
            kr = self.enc_kr[h](x); ki = self.enc_ki[h](x)
            vr = self.enc_vr[h](x); vi = self.enc_vi[h](x)
            score_i = (qi * kr - qr * ki).sum(dim=-1)
            c, s = torch.cos(score_i.unsqueeze(-1)), torch.sin(score_i.unsqueeze(-1))
            zr, zi = c * vr - s * vi, c * vi + s * vr
            c2, s2 = torch.cos(self.phases[h]), torch.sin(self.phases[h])
            zr = zr * c2 - zi * s2
            head_outputs.append(zr)
            if return_phase:
                cos_mean = torch.cos(score_i).mean()
                sin_mean = torch.sin(score_i).mean()
                all_phases.append((cos_mean**2 + sin_mean**2).sqrt())
        merged = torch.cat(head_outputs, dim=-1)
        if return_phase:
            return self.out(merged), torch.stack(all_phases).mean()
        return self.out(merged)


def train_eval(model_class, h, X_tr, Y_tr, X_te, Y_te, epochs=100):
    model = model_class(n_heads=h)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    best_acc = 0

    for e in range(epochs):
        loss = F.cross_entropy(model(X_tr), Y_tr)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            acc = (model(X_te).argmax(-1) == Y_te).float().mean().item()
            if acc > best_acc:
                best_acc = acc

    with torch.no_grad():
        logits, pc = model(X_te, return_phase=True)
        acc = (logits.argmax(-1) == Y_te).float().mean().item()

        saved = {}
        for n, p in model.named_parameters():
            if 'enc_qi' in n or 'enc_ki' in n or 'phases' in n:
                saved[n] = p.data.clone()
                p.data.zero_()

        ab_logits = model(X_te)
        ab_acc = (ab_logits.argmax(-1) == Y_te).float().mean().item()

        for n, p in model.named_parameters():
            if n in saved:
                p.data.copy_(saved[n])

    return acc, ab_acc, acc - ab_acc, pc.item(), n_params


X_raw, Y_raw = gen_geometry(600)
X_train, Y_train = X_raw[:400], Y_raw[:400]
X_test, Y_test = X_raw[400:], Y_raw[400:]

print("=" * 80)
print("Q56 v2: ENTANGLED MERGE vs CLASSICAL MERGE")
print("Both use independent Q/K/V per head (D_f = h)")
print("Difference: projective measurement (Born) vs concatenation (linear)")
print("=" * 80)
print()

ent_results = {}
cls_results = {}

for label, model_class in [("ENTANGLED (projective)", EntangledHeads),
                            ("CLASSICAL   (concat)", ClassicalHeads)]:
    print(f"--- {label} ---")
    print(f"  {'h':>4}  {'acc':>7}  {'ab_acc':>7}  {'delta':>7}  {'phase_coh':>9}  {'params':>7}")
    print("  " + "-" * 55)
    store = ent_results if "ENT" in label else cls_results
    for h in [1, 2, 4, 8, 16]:
        acc, ab_acc, delta, pc, n_p = train_eval(model_class, h, X_train, Y_train, X_test, Y_test)
        store[h] = (acc, delta, pc, n_p)
        print(f"  {h:>4}  {acc:>7.1%}  {ab_acc:>7.1%}  {delta:>+6.1%}  {pc:>9.4f}  {n_p:>7}")

print()
print("=" * 80)
print("SCALING ANALYSIS")
print("=" * 80)

print(f"\n{'h':>4}  {'Ent delta':>9}  {'Cls delta':>9}  {'gap':>9}  {'E/h':>8}  {'C/h':>8}")
print("-" * 60)
for h in [1, 2, 4, 8, 16]:
    ent_d = ent_results[h][1]
    cls_d = cls_results[h][1]
    gap = ent_d - cls_d
    e_ph = ent_d / h
    c_ph = cls_d / h
    print(f"  {h:>4}  {ent_d:>+8.1%}  {cls_d:>+8.1%}  {gap:>+8.1%}  {e_ph:>7.3f}  {c_ph:>7.3f}")

print()
print("MERGE EFFICIENCY (deltas):")
ent_deltas = [ent_results[h][1] for h in [1, 2, 4, 8, 16]]
cls_deltas = [cls_results[h][1] for h in [1, 2, 4, 8, 16]]
ent_sum = sum(ent_deltas)
cls_sum = sum(cls_deltas)
print(f"  Total phase delta: Entangled={ent_sum:+.1%}  Classical={cls_sum:+.1%}")
if ent_sum > cls_sum * 1.03:
    print("  PROJECTIVE MERGE MORE PHASE-EFFICIENT — interference amplifies phase signal")
elif abs(ent_sum - cls_sum) / max(abs(cls_sum), 1e-6) < 0.03:
    print("  TIED — merge mechanism doesn't dominate at this scale")
else:
    print("  Classical merge retains more phase information")

print()
print("ANTI-SATURATION (per-head decay from h=1 to h=16):")
ent_ph = [ent_results[h][1] / h for h in [1, 2, 4, 8, 16]]
cls_ph = [cls_results[h][1] / h for h in [1, 2, 4, 8, 16]]
ent_decay = ent_ph[0] - ent_ph[-1]
cls_decay = cls_ph[0] - cls_ph[-1]
print(f"  Per-head delta: Entangled {[f'{d:.3f}' for d in ent_ph]}")
print(f"  Per-head delta: Classical {[f'{d:.3f}' for d in cls_ph]}")
print(f"  Decay: Entangled={ent_decay:.3f}  Classical={cls_decay:.3f}")
if ent_decay < cls_decay * 0.5:
    print("  ENTANGLED RESISTS SATURATION — Born rule preserves per-head phase contribution")
elif ent_decay < cls_decay:
    print("  WEAK anti-saturation advantage")
else:
    print("  Both saturate similarly")

print()
print("PARAMETER EFFICIENCY:")
for h in [8, 16]:
    ent_p = ent_results[h][3]
    cls_p = cls_results[h][3]
    ent_a = ent_results[h][0]
    cls_a = cls_results[h][0]
    print(f"  h={h:>2}: Entangled {ent_p:>5} params @ {ent_a:.1%}  |  Classical {cls_p:>5} params @ {cls_a:.1%}  |  param ratio: {cls_p/ent_p:.1f}x")
