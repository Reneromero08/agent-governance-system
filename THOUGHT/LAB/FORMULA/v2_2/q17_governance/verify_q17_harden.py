"""Q17 Hardening: R-threshold sweep, seed stability, phase_coh as gate."""
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
        phase_coh = 1.0-entropies.mean().item()/math.log(4)
        R = E / max(nabla_S, 1e-8)
        acc = correct.mean().item()
    return {"R":R,"nabla_S":nabla_S,"phase_coh":phase_coh,"E":E,"acc":acc}

def train_r_gated(threshold_pct, seed, epochs=100):
    torch.manual_seed(seed); random.seed(seed)
    X_train, Y_train = gen_transforms(200)
    random.seed(seed+1000); torch.manual_seed(seed+1000)
    X_test, Y_test = gen_transforms(200)
    m = RealMLP(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    hist = []
    for e in range(epochs):
        logits = m(X_train); loss = F.cross_entropy(logits, Y_train)
        met = metrics(m, X_train, Y_train)
        if e > 3 and len(hist) > 0 and met["R"] < hist[-1]["R"] * (1 - threshold_pct/100):
            preds = logits.argmax(-1); wrong = preds != Y_train
            if wrong.sum()>0:
                widx = wrong.nonzero(as_tuple=True)[0]
                loss = loss + 0.5*F.cross_entropy(m(X_train[widx]), Y_train[widx])
        opt.zero_grad(); loss.backward(); opt.step()
        if e%10==0: met["epoch"]=e; hist.append(met)
    with torch.no_grad():
        acc = (m(X_test).argmax(-1)==Y_test).float().mean().item()
    return acc, hist

print("Q17 HARDENING")
print("=" * 60)

# ANGLE 1: R-drop threshold sweep
print("\n=== ANGLE 1: R-drop threshold sweep ===")
for pct in [1, 2, 5, 10, 20, 50]:
    acc, _ = train_r_gated(pct, seed=42)
    print(f"  threshold={pct:3d}% drop: acc={acc:.1%}")

# ANGLE 2: Seed stability (best threshold from sweep)
print("\n=== ANGLE 2: Seed stability (5% threshold) ===")
accs = []
for seed in range(10):
    acc, hist = train_r_gated(5, seed=seed)
    accs.append(acc)
accs = np.array(accs)
print(f"  R-gated acc: {accs.mean():.1%}+/-{accs.std():.1%}")

# ANGLE 3: Baseline comparisons
print("\n=== ANGLE 3: Baseline comparison ===")
# CONTROL
ctrl_accs = []
for seed in range(10):
    torch.manual_seed(seed); random.seed(seed)
    X_tr, Y_tr = gen_transforms(200); random.seed(seed+1000); torch.manual_seed(seed+1000)
    X_te, Y_te = gen_transforms(200)
    m = RealMLP(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for e in range(100):
        loss = F.cross_entropy(m(X_tr), Y_tr); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): ctrl_accs.append((m(X_te).argmax(-1)==Y_te).float().mean().item())
ctrl_accs = np.array(ctrl_accs)

# CASSETTE
cass_accs = []
for seed in range(10):
    torch.manual_seed(seed); random.seed(seed)
    X_tr, Y_tr = gen_transforms(200); random.seed(seed+1000); torch.manual_seed(seed+1000)
    X_te, Y_te = gen_transforms(200)
    m = RealMLP(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for e in range(100):
        logits = m(X_tr); preds = logits.argmax(-1)
        loss = F.cross_entropy(logits, Y_tr)
        wrong = preds != Y_tr
        if wrong.sum()>0 and e>3:
            widx = wrong.nonzero(as_tuple=True)[0]
            loss = loss + 0.5*F.cross_entropy(m(X_tr[widx]), Y_tr[widx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): cass_accs.append((m(X_te).argmax(-1)==Y_te).float().mean().item())
cass_accs = np.array(cass_accs)

print(f"  CONTROL:       {ctrl_accs.mean():.1%}+/-{ctrl_accs.std():.1%}")
print(f"  CASSETTE:      {cass_accs.mean():.1%}+/-{cass_accs.std():.1%}")
print(f"  R-GATED:       {accs.mean():.1%}+/-{accs.std():.1%}")

# Does R-gated beat control? 
t, p = stats.ttest_ind(accs, ctrl_accs)
print(f"  R-GATED vs CONTROL t-test: t={t:.1f} p={p:.4f}")
print(f"  {'R-GATED BEATS CONTROL' if p<0.05 and accs.mean()>ctrl_accs.mean() else 'NO SIGNIFICANT DIFFERENCE'}")

# ANGLE 4: Phase coherence as gate — does it predict wrong predictions?
print("\n=== ANGLE 4: Phase_coh predicts wrong predictions ===")
torch.manual_seed(42); random.seed(42)
X_tr, Y_tr = gen_transforms(200); random.seed(1042); torch.manual_seed(1042)
X_te, Y_te = gen_transforms(200)
m = RealMLP(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
all_phase_coh = []; all_wrong = []
for e in range(100):
    logits = m(X_tr); loss = F.cross_entropy(logits, Y_tr)
    met = metrics(m, X_tr, Y_tr)
    preds = logits.argmax(-1); wrong = preds != Y_tr
    all_phase_coh.append(met["phase_coh"]); all_wrong.append(wrong.float().mean().item())
    opt.zero_grad(); loss.backward(); opt.step()

pc = np.array(all_phase_coh); wr = np.array(all_wrong)
r, p_val = stats.pearsonr(pc, wr)
print(f"  Corr(phase_coh, wrong_rate): r={r:.3f} p={p_val:.4f}")
print(f"  {'PHASE PREDICTS ERRORS' if p_val<0.01 and r<-0.5 else 'WEAK SIGNAL'}")
