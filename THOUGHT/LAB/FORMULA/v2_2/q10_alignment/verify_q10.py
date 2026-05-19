"""Q10: Phase coherence as leading indicator of misalignment."""
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
        X.append(torch.stack([torch.cos(torch.angle(ratio)).mean(),torch.sin(torch.angle(ratio)).mean(),torch.cos(torch.angle(ratio)).std(),torch.sin(torch.angle(ratio)).std(),torch.abs(ratio).mean(),torch.abs(ratio).std()]))
        Y.append(t)
    return torch.stack(X), torch.tensor(Y)

class RealMLP(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(6,16),nn.ReLU(),nn.Linear(16,4))
    def forward(self,x): return self.net(x)

def pc(model, X, Y):
    with torch.no_grad():
        logits = model(X); probs = F.softmax(logits, dim=-1)
        h = -(probs*torch.log(probs+1e-8)).sum(-1)
        return 1.0 - h.mean().item()/math.log(4)

def epoch_metrics(model, X, Y):
    with torch.no_grad():
        logits = model(X)
        acc = (logits.argmax(-1)==Y).float().mean().item()
        probs = F.softmax(logits, dim=-1)
        correct_probs = probs[torch.arange(len(Y)), Y]
        E = correct_probs.mean().item()
        h = -(probs*torch.log(probs+1e-8)).sum(-1)
        nabla = h.std().item() + 1e-8
        coh = 1.0 - h.mean().item()/math.log(4)
        R = E / max(nabla, 1e-8)
    return {"acc": acc, "phase_coh": coh, "R": R, "E": E, "nabla_S": nabla}

print("Q10: Phase coherence as leading alignment indicator")
print("=" * 60)

# Train 50 seeds, track epoch-by-epoch metrics
all_histories = []
for seed in range(50):
    torch.manual_seed(seed); random.seed(seed)
    X_tr, Y_tr = gen_transforms(200)
    random.seed(seed+1000); torch.manual_seed(seed+1000)
    X_te, Y_te = gen_transforms(200)
    m = RealMLP(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    history = []
    for e in range(100):
        loss = F.cross_entropy(m(X_tr), Y_tr); opt.zero_grad(); loss.backward(); opt.step()
        met = epoch_metrics(m, X_tr, Y_tr); met["epoch"] = e
        history.append(met)
    all_histories.append(history)

# For each training run, find: epoch where phase_coh crosses 0.85 (drops below)
# and epoch where accuracy crosses 0.85 (rises above).
# If phase crosses BEFORE accuracy, it's a leading indicator.
phase_cross_epochs = []; acc_cross_epochs = []
for hist in all_histories:
    pc_vals = [h["phase_coh"] for h in hist]
    acc_vals = [h["acc"] for h in hist]
    # Find first epoch where phase_coh drops below 0.85
    pc_cross = next((i for i, v in enumerate(pc_vals) if v < 0.85), 100)
    # Find first epoch where accuracy rises above 0.85
    acc_cross = next((i for i, v in enumerate(acc_vals) if v > 0.85), 100)
    if pc_cross < 100 and acc_cross < 100:
        phase_cross_epochs.append(pc_cross)
        acc_cross_epochs.append(acc_cross)

pc_arr = np.array(phase_cross_epochs); ac_arr = np.array(acc_cross_epochs)
delta = ac_arr - pc_arr  # positive = acc crosses AFTER phase drops

print(f"\n  Training runs with both crossings: {len(pc_arr)}/50")
print(f"  Phase drops below 0.85 at epoch: {pc_arr.mean():.1f}+/-{pc_arr.std():.1f}")
print(f"  Accuracy rises above 0.85 at epoch: {ac_arr.mean():.1f}+/-{ac_arr.std():.1f}")
print(f"  Delta (acc - phase): {delta.mean():+.1f} epochs")
print(f"  Phase drops BEFORE accuracy in: {(delta > 0).mean()*100:.0f}% of runs")
t, p = stats.ttest_1samp(delta, 0)
print(f"  t-test (delta > 0): t={t:.1f} p={p:.6f}")

if p < 0.01 and delta.mean() > 0:
    print(f"\n  PHASE COHERENCE IS A LEADING INDICATOR")
    print(f"  It drops {delta.mean():.0f} epochs BEFORE accuracy degrades.")
else:
    print(f"\n  No temporal precedence detected.")

# Also: correlation between phase_coh at epoch N and accuracy at epoch N+1
lead_corrs = []
for lag in [1, 2, 3, 5]:
    corrs = []
    for hist in all_histories:
        pc_vec = np.array([h["phase_coh"] for h in hist])
        acc_vec = np.array([h["acc"] for h in hist])
        if len(pc_vec) > lag:
            r = np.corrcoef(pc_vec[:-lag], acc_vec[lag:])[0,1]
            if not np.isnan(r): corrs.append(r)
    lead_corrs.append(np.mean(corrs))
    print(f"  Corr(phase_coh[t], acc[t+{lag}]): {np.mean(corrs):.4f}")
