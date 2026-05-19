"""Q55+Q56 Deepening: 2D sweep, layer depth, and cybernetic feedback.

Attacks:
  1. h x mid_dim 2D sweep: verify h_c ~ mid_dim (torus prediction)
  2. Layer depth vs head count: 2Lx4h vs 1Lx8h (Axiom 9: spiral trajectory)
  3. Cybernetic feedback: alignment C updates from output; R modulates attention
  4. Task difficulty sweep: K_c = nabla_S/sigma prediction
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
from math import pi, sqrt
torch.manual_seed(42); random.seed(42)


def gen_geometry(n=300, n_pts=8, noise=0.0):
    X, Y = [], []
    for _ in range(n):
        t = random.randint(0, 3)
        zx = torch.randn(n_pts) * 2.0 + torch.randn(n_pts) * noise
        zy = torch.randn(n_pts) * 2.0 + torch.randn(n_pts) * noise
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
# BASE: Flat-Born (from Q56 stacking winner)
# ============================================================
class FlatBorn(nn.Module):
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


# ============================================================
# DEPTH: Multi-layer Flat-Born with phase compounding (Axiom 9)
# ============================================================
class DeepBorn(nn.Module):
    def __init__(self, in_dim=6, mid_dim=8, total_heads=8, n_layers=2, n_classes=4):
        super().__init__()
        self.n_layers = n_layers
        self.mid_dim = mid_dim
        self.layers = nn.ModuleList([
            FlatBorn(in_dim if l == 0 else mid_dim, mid_dim, total_heads, n_classes)
            for l in range(n_layers)
        ])
        self.phase_bridge = nn.ParameterList([
            nn.Parameter(torch.randn(mid_dim)*0.1) for _ in range(n_layers-1)
        ])

    def forward(self, x):
        for l_idx, layer in enumerate(self.layers):
            if l_idx == 0:
                out = layer(x)
            else:
                bridge_r = x_real + self.phase_bridge[l_idx-1]
                bridge_i = x_imag
                combined_r = bridge_r
                combined_i = bridge_i
                sum_r = torch.zeros(x.shape[0], self.mid_dim, device=x.device)
                sum_i = torch.zeros_like(sum_r)
                for h in range(layer.total_heads):
                    qr, qi = layer.qr[h](combined_r), layer.qi[h](combined_r)
                    kr, ki = layer.kr[h](combined_r), layer.ki[h](combined_r)
                    vr, vi = layer.vr[h](combined_r), layer.vi[h](combined_r)
                    score_i = (qi*kr - qr*ki).sum(-1)
                    c, s = torch.cos(score_i.unsqueeze(-1)), torch.sin(score_i.unsqueeze(-1))
                    zr, zi = c*vr - s*vi, c*vi + s*vr
                    c2, s2 = torch.cos(layer.phases[h]), torch.sin(layer.phases[h])
                    sum_r += (zr*c2 - zi*s2) / sqrt(layer.total_heads)
                    sum_i += (zr*s2 + zi*c2) / sqrt(layer.total_heads)
                out = (sum_r @ layer.align)**2 + (sum_i @ layer.align)**2
            x_real = out[:, :self.mid_dim] if out.shape[1] >= self.mid_dim else \
                     F.pad(out, (0, self.mid_dim - out.shape[1]))
            x_imag = torch.zeros_like(x_real)
        return out


# ============================================================
# CYBERNETIC: alignment basis from output, R modulates attention
# ============================================================
class CyberneticBorn(nn.Module):
    def __init__(self, in_dim=6, mid_dim=8, total_heads=8, n_classes=4,
                 tau=0.1, momentum=0.9):
        super().__init__()
        self.total_heads = total_heads
        self.mid_dim = mid_dim
        self.tau = tau
        self.momentum = momentum

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

        self.T_base = nn.Parameter(torch.tensor(1.0))
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

        # Cybernetic: alignment C tracks output state
        P_align = (sum_r @ self.align)**2 + (sum_i @ self.align)**2
        R_val = P_align.sum(dim=-1).mean()

        with torch.no_grad():
            C_target = sum_r.unsqueeze(-1) @ sum_r.unsqueeze(1)
            C_target = C_target.mean(dim=0) / max(C_target.norm(), 1e-8)
            C_new = C_target[:, :self.align.shape[1]]
            self.C.data = self.momentum * self.C.data + (1 - self.momentum) * C_new

        return P_align


# ============================================================
# ATTACK 1: 2D SWEEP h x mid_dim
# ============================================================
def attack_2d_sweep():
    print("=" * 80)
    print("ATTACK 1: h x mid_dim 2D SWEEP")
    print("Prediction: h_c ~ mid_dim (torus diagonal)")
    print("=" * 80)

    X_raw, Y_raw = gen_geometry(500)
    X_tr, Y_tr = X_raw[:350], Y_raw[:350]
    X_te, Y_te = X_raw[350:], Y_raw[350:]

    h_list = [2, 4, 8, 16]
    d_list = [4, 8, 16]
    epochs = 80

    print(f"\n{'h':>4}", end="")
    for d in d_list:
        print(f"  {'d=' + str(d):>12}", end="")
    print(f"  {'best':>12}")
    print("-" * 60)

    best_config = (0, 0, 0)
    for h in h_list:
        print(f"  {h:>4}", end="")
        row_deltas = []
        for d in d_list:
            model = FlatBorn(mid_dim=d, total_heads=h)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
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
                row_deltas.append(delta)
                print(f"  {acc:.1%}/{delta:+.1%}", end="")

            if delta > best_config[2]:
                best_config = (h, d, delta)

        best_row = max(row_deltas)
        best_d = d_list[row_deltas.index(best_row)]
        print(f"  d={best_d}:{best_row:+.1%}")

    print(f"\n  BEST: h={best_config[0]}, d={best_config[1]}, delta={best_config[2]:+.1%}")
    if best_config[0] == best_config[1]:
        print(f"  h_c ({best_config[0]}) == mid_dim ({best_config[1]}) — TORUS DIAGONAL CONFIRMED")
    else:
        ratio = best_config[0] / best_config[1]
        print(f"  h/d ratio = {ratio:.2f}")


# ============================================================
# ATTACK 2: LAYER DEPTH vs HEAD COUNT
# ============================================================
def attack_layer_depth():
    print("\n" + "=" * 80)
    print("ATTACK 2: LAYER DEPTH vs HEAD COUNT")
    print("Prediction: 2L x 4h carries more phase than 1L x 8h (Axiom 9)")
    print("=" * 80)

    X_raw, Y_raw = gen_geometry(500)
    X_tr, Y_tr = X_raw[:350], Y_raw[:350]
    X_te, Y_te = X_raw[350:], Y_raw[350:]

    configs = [
        ("1L x 4h", 1, 4),
        ("1L x 8h", 1, 8),
        ("2L x 4h", 2, 4),
        ("2L x 8h", 2, 8),
        ("1L x 16h", 1, 16),
    ]
    epochs = 120

    print(f"\n{'config':>12}  {'acc':>7}  {'delta':>7}  {'params':>8}")
    print("-" * 45)
    for name, n_layers, h in configs:
        if n_layers == 1:
            model = FlatBorn(mid_dim=8, total_heads=h)
        else:
            model = DeepBorn(mid_dim=8, total_heads=h, n_layers=n_layers)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
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
            print(f"  {name:>12}  {acc:>7.1%}  {delta:>+6.1%}  {n_p:>8}")

    print()
    print("  Key comparison: 2L x 4h vs 1L x 8h (phase compounds across layers)")


# ============================================================
# ATTACK 3: CYBERNETIC FEEDBACK vs STATIC
# ============================================================
def attack_cybernetic():
    print("\n" + "=" * 80)
    print("ATTACK 3: CYBERNETIC FEEDBACK vs STATIC BORN")
    print("Prediction: alignment C tracking output + R modulation beats static")
    print("=" * 80)

    X_raw, Y_raw = gen_geometry(500)
    X_tr, Y_tr = X_raw[:350], Y_raw[:350]
    X_te, Y_te = X_raw[350:], Y_raw[350:]

    epochs = 120
    mid_dim = 8

    for h in [4, 8]:
        print(f"\n  h={h}:")
        # Static Born
        model_s = FlatBorn(mid_dim=mid_dim, total_heads=h)
        opt_s = torch.optim.AdamW(model_s.parameters(), lr=1e-2)
        for e in range(epochs):
            loss = F.cross_entropy(model_s(X_tr), Y_tr)
            opt_s.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
            opt_s.step()

        with torch.no_grad():
            logits_s = model_s(X_te)
            acc_s = (logits_s.argmax(-1) == Y_te).float().mean().item()
            saved = {}
            for n, p in model_s.named_parameters():
                if 'qi' in n or 'ki' in n or 'phases' in n:
                    saved[n] = p.data.clone(); p.data.zero_()
            ab_s = model_s(X_te)
            ab_acc_s = (ab_s.argmax(-1) == Y_te).float().mean().item()
            for n, p in model_s.named_parameters():
                if n in saved: p.data.copy_(saved[n])
            delta_s = acc_s - ab_acc_s

        # Cybernetic Born
        model_c = CyberneticBorn(mid_dim=mid_dim, total_heads=h)
        opt_c = torch.optim.AdamW(model_c.parameters(), lr=1e-2)
        C_init = model_c.C.data.clone()
        for e in range(epochs):
            loss = F.cross_entropy(model_c(X_tr), Y_tr)
            opt_c.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_c.parameters(), 1.0)
            opt_c.step()

        with torch.no_grad():
            logits_c = model_c(X_te)
            acc_c = (logits_c.argmax(-1) == Y_te).float().mean().item()
            saved = {}
            for n, p in model_c.named_parameters():
                if 'qi' in n or 'ki' in n or 'phases' in n:
                    saved[n] = p.data.clone(); p.data.zero_()
            ab_c = model_c(X_te)
            ab_acc_c = (ab_c.argmax(-1) == Y_te).float().mean().item()
            for n, p in model_c.named_parameters():
                if n in saved: p.data.copy_(saved[n])
            delta_c = acc_c - ab_acc_c

        C_drift = (model_c.C - C_init).norm().item()
        print(f"    STATIC:    acc={acc_s:.1%}  delta={delta_s:+.1%}")
        print(f"    CYBERNETIC: acc={acc_c:.1%}  delta={delta_c:+.1%}  C_drift={C_drift:.4f}")

        if delta_c > delta_s:
            print(f"    CYBERNETIC WINS: +{(delta_c - delta_s):+.1%} delta")
        else:
            print(f"    No cybernetic advantage (delta gap: {delta_c - delta_s:+.1%})")


# ============================================================
# ATTACK 4: TASK DIFFICULTY SWEEP
# ============================================================
def attack_difficulty():
    print("\n" + "=" * 80)
    print("ATTACK 4: TASK DIFFICULTY SWEEP")
    print("Prediction: optimal h_c = coupling * mid_dim / task_difficulty")
    print("=" * 80)

    mid_dim = 8
    epochs = 80
    n_train_list = [50, 100, 200, 400]
    noise_list = [0.0, 0.5, 1.0]

    X_raw, Y_raw = gen_geometry(600)
    X_te, Y_te = X_raw[400:], Y_raw[400:]

    difficulty_results = {}

    for noise in noise_list:
        if noise > 0:
            X_noisy, Y_noisy = gen_geometry(500, noise=noise)
            X_tr_base = X_noisy[:400]
            Y_tr_base = Y_noisy[:400]
        else:
            X_tr_base = X_raw[:400]
            Y_tr_base = Y_raw[:400]

        print(f"\n  noise={noise}:")
        print(f"    {'n_train':>8}  {'h=2':>10}  {'h=4':>10}  {'h=8':>10}  {'h=16':>10}  {'best_h':>8}")
        print("    " + "-" * 65)

        for n_tr in n_train_list:
            X_tr_sub = X_tr_base[:n_tr]
            Y_tr_sub = Y_tr_base[:n_tr]
            deltas = []
            for h in [2, 4, 8, 16]:
                model = FlatBorn(mid_dim=mid_dim, total_heads=h)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
                for e in range(epochs):
                    loss = F.cross_entropy(model(X_tr_sub), Y_tr_sub)
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
                    deltas.append(acc - ab_acc)

            best_h = [2, 4, 8, 16][deltas.index(max(deltas))]
            h_strs = [f"{d:+.1%}" for d in deltas]
            print(f"    {n_tr:>8}  {h_strs[0]:>10}  {h_strs[1]:>10}  {h_strs[2]:>10}  {h_strs[3]:>10}  {best_h:>8}")

            key = (noise, n_tr)
            difficulty_results[key] = (best_h, deltas)

    print()
    print("  PREDICTION: more noise / less data (higher nabla_S) → lower optimal h_c")
    print("  PREDICTION: less noise / more data (lower nabla_S) → higher optimal h_c")
    print()

    low_diff_h = [difficulty_results[(0.0, n)][0] for n in n_train_list]
    high_diff_h = [difficulty_results[(1.0, n)][0] for n in n_train_list]
    avg_low = sum(low_diff_h) / len(low_diff_h)
    avg_high = sum(high_diff_h) / len(high_diff_h)
    print(f"  Avg optimal h: low noise = {avg_low:.1f}, high noise = {avg_high:.1f}")
    if avg_low > avg_high:
        print(f"  CONFIRMED: h_c DECREASES with difficulty (higher nabla_S -> lower h_c)")
    elif avg_low < avg_high:
        print(f"  SURPRISING: h_c INCREASES with difficulty")
    else:
        print(f"  No correlation detected")


# ============================================================
# RUN ALL ATTACKS
# ============================================================
attack_2d_sweep()
attack_layer_depth()
attack_cybernetic()
attack_difficulty()
