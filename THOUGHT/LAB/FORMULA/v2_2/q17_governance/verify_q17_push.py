"""Q17 Push: Phase coherence-gated correction, multi-metric, adaptive threshold."""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
import numpy as np
from scipy import stats
torch.manual_seed(42); random.seed(42)

def gen_transforms(n=200, n_pts=8):
    X, Y = [], []
    for _ in range(n):
        t = random.randint(0, 3)
        zx = torch.randn(n_pts)*2.0; zy = torch.randn(n_pts)*2.0
        th = random.random()*2*math.pi
        if t==0: c,s=math.cos(th),math.sin(th); ox=zx*c-zy*s; oy=zx*s+zy*c
        elif t==1: c,s=math.cos(th),math.sin(th); ox=zx*c+zy*s; oy=zx*s-zy*c
        elif t==2: sc=0.2+random.random()*3.0; ox=zx*sc; oy=zy*sc
        else: k=random.random()*2-1; ox=zx+k*zy; oy=zy
        z=torch.complex(zx,zy); zp=torch.complex(ox,oy)
        ratio=zp/(z+1e-8)
        feats=torch.stack([torch.cos(torch.angle(ratio)).mean(),torch.sin(torch.angle(ratio)).mean(),torch.cos(torch.angle(ratio)).std(),torch.sin(torch.angle(ratio)).std(),torch.abs(ratio).mean(),torch.abs(ratio).std()])
        X.append(feats); Y.append(t)
    return torch.stack(X), torch.tensor(Y)

class RealMLP(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(6,16),nn.ReLU(),nn.Linear(16,4))
    def forward(self,x): return self.net(x)

def metrics(model, X, Y):
    with torch.no_grad():
        logits = model(X); probs = F.softmax(logits, dim=-1)
        correct = (logits.argmax(-1)==Y).float()
        correct_probs = probs[torch.arange(len(Y)), Y]
        E = correct_probs.mean().item()
        entropies = -(probs*torch.log(probs+1e-8)).sum(-1)
        nabla_S = entropies.std().item() + 1e-8
        phase_coh = 1.0 - entropies.mean().item()/math.log(4)
        R = E / max(nabla_S, 1e-8)
        acc = correct.mean().item()
    return {"R":R,"nabla_S":nabla_S,"phase_coh":phase_coh,"E":E,"acc":acc}

def train_gated(gate_type, seed, epochs=100):
    torch.manual_seed(seed); random.seed(seed)
    X_train, Y_train = gen_transforms(200)
    random.seed(seed+1000); torch.manual_seed(seed+1000)
    X_test, Y_test = gen_transforms(200)
    m = RealMLP(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    hist_phase = []
    for e in range(epochs):
        logits = m(X_train); loss = F.cross_entropy(logits, Y_train)
        met = metrics(m, X_train, Y_train)
        hist_phase.append(met["phase_coh"])
        should_correct = False

        if e > 3:
            if gate_type == "phase_drop" and len(hist_phase) > 1:
                # Correct when phase_coh drops from its recent peak
                recent_peak = max(hist_phase[-5:]) if len(hist_phase) >= 5 else hist_phase[0]
                should_correct = met["phase_coh"] < recent_peak * 0.98
            elif gate_type == "phase_low":
                should_correct = met["phase_coh"] < 0.85
            elif gate_type == "phase_ema":
                if len(hist_phase) >= 3:
                    ema = 0.9*hist_phase[-2] + 0.1*met["phase_coh"]
                    should_correct = met["phase_coh"] < ema - 0.01
            elif gate_type == "multi":
                # Both phase dropping AND R dropping
                if len(hist_phase) > 1:
                    phase_dropping = met["phase_coh"] < hist_phase[-2]
                    R_vals = [0]  # simplified - need R history too
                    should_correct = phase_dropping and met["phase_coh"] < 0.88

        if should_correct:
            preds = logits.argmax(-1); wrong = preds != Y_train
            if wrong.sum() > 0:
                widx = wrong.nonzero(as_tuple=True)[0]
                # Correction weight proportional to phase_coh drop
                weight = max(0.1, (1.0 - met["phase_coh"]) * 2.0)
                loss = loss + weight * F.cross_entropy(m(X_train[widx]), Y_train[widx])

        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        acc = (m(X_test).argmax(-1)==Y_test).float().mean().item()
    return acc

print("Q17 PUSH — Phase coherence as governance gate")
print("=" * 60)

# Baseline once
ctrl_accs = []; cass_accs = []
for seed in range(10):
    torch.manual_seed(seed); random.seed(seed)
    X_tr, Y_tr = gen_transforms(200)
    random.seed(seed+1000); torch.manual_seed(seed+1000)
    X_te, Y_te = gen_transforms(200)

    # CONTROL
    mc = RealMLP(); opt = torch.optim.AdamW(mc.parameters(), lr=1e-2)
    for e in range(100):
        loss = F.cross_entropy(mc(X_tr), Y_tr); opt.zero_grad(); loss.backward(); opt.step()
    ctrl_accs.append((mc(X_te).argmax(-1)==Y_te).float().mean().item())

    # CASSETTE
    ms = RealMLP(); opt = torch.optim.AdamW(ms.parameters(), lr=1e-2)
    for e in range(100):
        logits = ms(X_tr); preds = logits.argmax(-1)
        loss = F.cross_entropy(logits, Y_tr)
        wrong = preds != Y_tr
        if wrong.sum()>0 and e>3:
            widx = wrong.nonzero(as_tuple=True)[0]
            loss = loss + 0.5*F.cross_entropy(ms(X_tr[widx]), Y_tr[widx])
        opt.zero_grad(); loss.backward(); opt.step()
    cass_accs.append((ms(X_te).argmax(-1)==Y_te).float().mean().item())

ctrl_arr = np.array(ctrl_accs); cass_arr = np.array(cass_accs)
print(f"\nCONTROL:  {ctrl_arr.mean():.1%}+/-{ctrl_arr.std():.1%}")
print(f"CASSETTE: {cass_arr.mean():.1%}+/-{cass_arr.std():.1%}")

# Test different gating strategies
for gate_type, label in [
    ("phase_drop", "Phase drop from peak"),
    ("phase_low", "Phase below 0.85"),
    ("phase_ema", "Phase EMA deviation"),
    ("multi", "Multi-metric"),
]:
    accs = []
    for seed in range(10):
        accs.append(train_gated(gate_type, seed))
    acc_arr = np.array(accs)
    delta = acc_arr.mean() - ctrl_arr.mean()
    t, p = stats.ttest_ind(acc_arr, ctrl_arr)
    sig = "***" if p<0.01 else "**" if p<0.05 else ""
    print(f"\n{label}: {acc_arr.mean():.1%}+/-{acc_arr.std():.1%} ({delta:+.1%}) p={p:.4f} {sig}")

    # Does it beat CASSETTE?
    t2, p2 = stats.ttest_ind(acc_arr, cass_arr)
    if p2 < 0.05 and acc_arr.mean() > cass_arr.mean():
        print(f"  BEATS CASSETTE (p={p2:.4f})!")
