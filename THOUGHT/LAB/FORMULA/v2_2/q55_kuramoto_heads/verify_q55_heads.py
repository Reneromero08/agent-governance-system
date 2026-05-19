"""Q55: Kuramoto head independence threshold on Native Eigen geometry classification.

Predicts: K_c = nabla_S/sigma predicts minimum h for phase coherence.
Independent heads = genuine D_f. Shared heads = pseudo-redundancy (D_f=1).
Expectation: accuracy shows Kuramoto-style phase transition at critical h_c.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
torch.manual_seed(42); random.seed(42)

# ---- Data: geometry classification (4-class) ----
def gen_geometry(n=300, n_pts=8):
    X, Y = [], []
    for _ in range(n):
        t = random.randint(0, 3)
        zx = torch.randn(n_pts) * 2.0; zy = torch.randn(n_pts) * 2.0
        th = random.random() * 2 * math.pi
        if t == 0:
            c, s = math.cos(th), math.sin(th)
            ox = zx*c - zy*s; oy = zx*s + zy*c
        elif t == 1:
            c, s = math.cos(th), math.sin(th)
            ox = zx*c + zy*s; oy = zx*s - zy*c
        elif t == 2:
            sc = 0.2 + random.random() * 3.0
            ox = zx*sc; oy = zy*sc
        else:
            k = random.random() * 2 - 1
            ox = zx + k*zy; oy = zy
        z = torch.complex(zx, zy); zp = torch.complex(ox, oy)
        ratio = zp / (z + 1e-8)
        feats = torch.stack([
            torch.cos(torch.angle(ratio)).mean(), torch.sin(torch.angle(ratio)).mean(),
            torch.cos(torch.angle(ratio)).std(),  torch.sin(torch.angle(ratio)).std(),
            torch.abs(ratio).mean(), torch.abs(ratio).std(),
        ])
        X.append(feats); Y.append(t)
    return torch.stack(X), torch.tensor(Y)


class MultiHeadNative(nn.Module):
    def __init__(self, in_dim=6, mid_dim=4, n_heads=2, n_classes=4, shared=False):
        super().__init__()
        self.n_heads = n_heads
        self.shared = shared
        self.mid_dim = mid_dim

        if shared:
            self.enc_qr = nn.Linear(in_dim, mid_dim, bias=False)
            self.enc_qi = nn.Linear(in_dim, mid_dim, bias=False)
            self.enc_kr = nn.Linear(in_dim, mid_dim, bias=False)
            self.enc_ki = nn.Linear(in_dim, mid_dim, bias=False)
            self.enc_vr = nn.Linear(in_dim, mid_dim, bias=False)
            self.enc_vi = nn.Linear(in_dim, mid_dim, bias=False)
            self.phases = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(n_heads)])
        else:
            self.enc_qr = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
            self.enc_qi = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
            self.enc_kr = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
            self.enc_ki = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
            self.enc_vr = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
            self.enc_vi = nn.ModuleList([nn.Linear(in_dim, mid_dim, bias=False) for _ in range(n_heads)])
            self.phases = nn.ParameterList([nn.Parameter(torch.tensor(random.random() * 0.2)) for _ in range(n_heads)])

        self.out = nn.Linear(mid_dim * n_heads, n_classes)

        for w in self._all_linears():
            nn.init.normal_(w.weight, std=0.1)
        nn.init.normal_(self.out.weight, std=0.1)

    def _all_linears(self):
        if self.shared:
            return [self.enc_qr, self.enc_qi, self.enc_kr, self.enc_ki, self.enc_vr, self.enc_vi]
        else:
            result = []
            for h in range(self.n_heads):
                result.extend([self.enc_qr[h], self.enc_qi[h], self.enc_kr[h],
                               self.enc_ki[h], self.enc_vr[h], self.enc_vi[h]])
            return result

    def forward(self, x, return_phase=False):
        B = x.shape[0]
        head_outputs = []
        all_phases = []

        for h in range(self.n_heads):
            if self.shared:
                qr, qi = self.enc_qr(x), self.enc_qi(x)
                kr, ki = self.enc_kr(x), self.enc_ki(x)
                vr, vi = self.enc_vr(x), self.enc_vi(x)
            else:
                qr, qi = self.enc_qr[h](x), self.enc_qi[h](x)
                kr, ki = self.enc_kr[h](x), self.enc_ki[h](x)
                vr, vi = self.enc_vr[h](x), self.enc_vi[h](x)

            score_i = (qi * kr - qr * ki).sum(dim=-1)
            c, s = torch.cos(score_i.unsqueeze(-1)), torch.sin(score_i.unsqueeze(-1))
            zr, zi = c * vr - s * vi, c * vi + s * vr

            c2, s2 = torch.cos(self.phases[h]), torch.sin(self.phases[h])
            zr = zr * c2 - zi * s2

            if return_phase:
                cos_mean = torch.cos(score_i).mean()
                sin_mean = torch.sin(score_i).mean()
                phase_coh = (cos_mean**2 + sin_mean**2).sqrt()
                all_phases.append(phase_coh)

            head_outputs.append(zr)

        merged = torch.cat(head_outputs, dim=-1)
        if return_phase:
            return self.out(merged), torch.stack(all_phases).mean()
        return self.out(merged)


def test_config(h, shared, X_train, Y_train, X_test, Y_test, epochs=100):
    model = MultiHeadNative(n_heads=h, shared=shared)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    n_params = sum(p.numel() for p in model.parameters())

    for e in range(epochs):
        loss = F.cross_entropy(model(X_train), Y_train)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        logits, phase_coh = model(X_test, return_phase=True)
        acc = (logits.argmax(-1) == Y_test).float().mean().item()

        saved = {k: p.data.clone() for k, p in model.named_parameters() if 'enc_qi' in k or 'enc_ki' in k or 'phases' in k}
        for n, p in model.named_parameters():
            if 'enc_qi' in n or 'enc_ki' in n:
                p.data.zero_()
            if 'phases' in n:
                p.data.zero_()

        ab_logits = model(X_test)
        ab_acc = (ab_logits.argmax(-1) == Y_test).float().mean().item()

        for k, p in model.named_parameters():
            if k in saved:
                p.data.copy_(saved[k])

    phase_delta = acc - ab_acc
    return acc, ab_acc, phase_delta, phase_coh.item(), n_params


X_raw, Y_raw = gen_geometry(600)
X_train, Y_train = X_raw[:400], Y_raw[:400]
X_test, Y_test = X_raw[400:], Y_raw[400:]

print("=" * 80)
print("Q55: Kuramoto Head Independence Threshold")
print("=" * 80)
print()

results = []
for shared in [False, True]:
    label = "INDEPENDENT" if not shared else "SHARED"
    print(f"--- {label} HEADS ---")
    print(f"{'h':>4}  {'acc':>7}  {'ab_acc':>7}  {'delta':>7}  {'phase_coh':>9}  {'params':>7}")
    print("-" * 55)
    for h in [1, 2, 4, 8, 16]:
        acc, ab_acc, delta, pc, n_p = test_config(h, shared, X_train, Y_train, X_test, Y_test)
        results.append((h, shared, acc, ab_acc, delta, pc, n_p))
        print(f"{h:>4}  {acc:>7.1%}  {ab_acc:>7.1%}  {delta:>+6.1%}  {pc:>9.4f}  {n_p:>7}")

print()
print("=" * 80)
print("KUIRAMOTO ANALYSIS")
print("=" * 80)

ind_results = [(h, acc, delta, pc, params) for h, s, acc, ab_acc, delta, pc, params in results if not s]
shr_results = [(h, acc, delta, pc, params) for h, s, acc, ab_acc, delta, pc, params in results if s]

print()
print("Independent heads:")
print(f"  {'h':>4}  {'acc':>7}  {'delta':>7}  {'phase_coh':>9}  {'params':>7}")
for h, acc, delta, pc, params in ind_results:
    print(f"  {h:>4}  {acc:>7.1%}  {delta:>+6.1%}  {pc:>9.4f}  {params:>7}")

print()
print("Shared heads:")
print(f"  {'h':>4}  {'acc':>7}  {'delta':>7}  {'phase_coh':>9}  {'params':>7}")
for h, acc, delta, pc, params in shr_results:
    print(f"  {h:>4}  {acc:>7.1%}  {delta:>+6.1%}  {pc:>9.4f}  {params:>7}")

print()
print("Phase transition detection:")
ind_accs = [a for _, a, _, _, _ in ind_results]
if len(ind_accs) >= 4:
    diffs = [ind_accs[i+1] - ind_accs[i] for i in range(len(ind_accs)-1)]
    max_diff = max(diffs)
    max_idx = diffs.index(max_diff) + 1
    max_h = ind_results[max_idx][0]
    # Normalize differences to detect jump vs trend
    mean_diff = sum(diffs) / len(diffs)
    print(f"  Accuracy jumps: {[f'{d:+.1%}' for d in diffs]}")
    print(f"  Max jump: {max_diff:+.1%} at h={max_h}")
    if max_diff > 2 * mean_diff and max_diff > 0.02:
        print(f"  KURAMOTO PHASE TRANSITION DETECTED at h_c={max_h}")
    else:
        print("  No clear phase transition detected (gradual or flat)")

print()
print("Independence test (independent vs shared at h=8):")
ind8 = [r for r in results if r[0] == 8 and not r[1]][0]
shr8 = [r for r in results if r[0] == 8 and r[1]][0]
ind_gap = ind8[2] - shr8[2]
print(f"  Independent h=8: {ind8[2]:.1%}")
print(f"  Shared h=8:      {shr8[2]:.1%}")
print(f"  Gap:             {ind_gap:+.1%}")
if ind_gap > 0.05:
    print("  INDEPENDENCE IS CAUSAL — D_f > 1 requires separate projections")
elif ind_gap > 0.02:
    print("  WEAK independence signal")
else:
    print("  Independence not causal at h=8")
