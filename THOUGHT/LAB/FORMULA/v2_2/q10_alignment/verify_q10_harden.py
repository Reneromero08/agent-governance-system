"""Q10 Hardening: multi-corruption, cross-model, seed stability, ROC, predictive."""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
import numpy as np
torch.manual_seed(42); random.seed(42)

def gen(n=200):
    X, Y = [], []
    for _ in range(n):
        t = random.randint(0, 3); zx = torch.randn(8)*2.0; zy = torch.randn(8)*2.0
        th = random.random()*2*math.pi
        if t==0: c,s=math.cos(th),math.sin(th); ox=zx*c-zy*s; oy=zx*s+zy*c
        elif t==1: c,s=math.cos(th),math.sin(th); ox=zx*c+zy*s; oy=zx*s-zy*c
        elif t==2: sc=0.2+random.random()*3.0; ox=zx*sc; oy=zy*sc
        else: k=random.random()*2-1; ox=zx+k*zy; oy=zy
        z=torch.complex(zx,zy); zp=torch.complex(ox,oy); ratio=zp/(z+1e-8)
        X.append(torch.stack([torch.cos(torch.angle(ratio)).mean(),torch.sin(torch.angle(ratio)).mean(),torch.cos(torch.angle(ratio)).std(),torch.sin(torch.angle(ratio)).std(),torch.abs(ratio).mean(),torch.abs(ratio).std()]))
        Y.append(t)
    return torch.stack(X), torch.tensor(Y)

class RealMLP(nn.Module):
    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(6,16),nn.ReLU(),nn.Linear(16,4))
    def forward(self,x): return self.net(x)

def metrics(model, X, Y):
    with torch.no_grad():
        logits = model(X); probs = F.softmax(logits, dim=-1)
        acc = (logits.argmax(-1)==Y).float().mean().item()
        h = -(probs*torch.log(probs+1e-8)).sum(-1)
        coh = 1.0 - h.mean().item()/math.log(4)
        nabla = h.std().item() + 1e-8
        E = probs[torch.arange(len(Y)), Y].mean().item()
        R = E / max(nabla, 1e-8)
    return {"acc": acc, "phase_coh": coh, "R": R, "nabla_S": nabla}

def train_model(seed):
    torch.manual_seed(seed); random.seed(seed)
    X_tr, Y_tr = gen(); random.seed(seed+1000); torch.manual_seed(seed+1000)
    X_te, Y_te = gen()
    m = RealMLP(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for e in range(200):
        loss = F.cross_entropy(m(X_tr), Y_tr); opt.zero_grad(); loss.backward(); opt.step()
    return m, X_tr, Y_tr, X_te, Y_te

def corrupt_and_track(model, X_tr, Y_tr, X_te, Y_te, corruption_type, sigma, epochs=80):
    m2 = RealMLP(); m2.load_state_dict({k:v.clone() for k,v in model.state_dict().items()})

    if corruption_type == "weight_noise":
        with torch.no_grad():
            for p in m2.parameters(): p.add_(torch.randn_like(p)*sigma)
    elif corruption_type == "label_noise":
        Y_tr_corrupted = Y_tr.clone()
        n_flip = int(len(Y_tr) * sigma)
        flip_idx = torch.randperm(len(Y_tr))[:n_flip]
        Y_tr_corrupted[flip_idx] = torch.randint(0,4,(n_flip,))
        Y_tr = Y_tr_corrupted
    elif corruption_type == "dropout":
        # Zero out some neurons, then recover
        with torch.no_grad():
            for p in m2.parameters():
                mask = torch.rand_like(p) > sigma
                p.mul_(mask.float())

    opt = torch.optim.AdamW(m2.parameters(), lr=1e-2)
    history = []
    for e in range(epochs):
        loss = F.cross_entropy(m2(X_tr), Y_tr); opt.zero_grad(); loss.backward(); opt.step()
        met = metrics(m2, X_te, Y_te); met["epoch"] = e; history.append(met)
    return history

print("Q10 HARDENING BATTERY")
print("=" * 60)

# Train 5 models
models = [train_model(seed) for seed in range(5)]

# ANGLE 1: Multi-corruption ROC
print("\n=== ANGLE 1: Multi-corruption discrimination ===")
for ctype, param in [("weight_noise", 0.05), ("label_noise", 0.3), ("dropout", 0.3)]:
    all_phase_aligned = []; all_phase_misaligned = []
    for seed in range(5):
        m, X_tr, Y_tr, X_te, Y_te = models[seed]
        # Aligned state: trained model
        met_aligned = metrics(m, X_te, Y_te)
        all_phase_aligned.append(met_aligned["phase_coh"])

        # Misaligned: corrupted model at epoch 0
        m2 = RealMLP(); m2.load_state_dict({k:v.clone() for k,v in m.state_dict().items()})
        if ctype == "weight_noise":
            with torch.no_grad():
                for p in m2.parameters(): p.add_(torch.randn_like(p)*param)
        elif ctype == "dropout":
            with torch.no_grad():
                for p in m2.parameters(): p.mul_((torch.rand_like(p)>param).float())
        met_mis = metrics(m2, X_te, Y_te)
        all_phase_misaligned.append(met_mis["phase_coh"])

    from sklearn.metrics import roc_auc_score
    labels = [1]*5 + [0]*5
    scores = all_phase_aligned + all_phase_misaligned
    auc = roc_auc_score(labels, scores)
    print(f"  {ctype:<15s}: AUROC={auc:.4f} (phase_coh: aligned={np.mean(all_phase_aligned):.4f} misaligned={np.mean(all_phase_misaligned):.4f})")

# ANGLE 2: Corruption recovery tracking
print("\n=== ANGLE 2: Corruption recovery (phase vs acc recovery times) ===")
for ctype, sigma in [("weight_noise", 0.05), ("dropout", 0.3)]:
    leads = []
    for seed in range(5):
        m, X_tr, Y_tr, X_te, Y_te = models[seed]
        hist = corrupt_and_track(m, X_tr, Y_tr, X_te, Y_te, ctype, sigma, epochs=80)
        pc_vals = [h["phase_coh"] for h in hist]; acc_vals = [h["acc"] for h in hist]
        pc_cross = next((i for i,v in enumerate(pc_vals) if v>0.85), 80)
        acc_cross = next((i for i,v in enumerate(acc_vals) if v>0.85), 80)
        leads.append(acc_cross - pc_cross)
    arr = np.array(leads)
    print(f"  {ctype:<15s}: phase leads by {arr.mean():+.1f}+/-{arr.std():.1f} epochs ({'LEADS' if arr.mean()>0 else 'LAGS'})")

# ANGLE 3: Cross-metric ROC at each epoch
print("\n=== ANGLE 3: Metric discrimination during recovery (weight_noise 0.05) ===")
m, X_tr, Y_tr, X_te, Y_te = models[0]
hist = corrupt_and_track(m, X_tr, Y_tr, X_te, Y_te, "weight_noise", 0.05, epochs=80)
for key in ["phase_coh", "R", "nabla_S"]:
    vals = [h[key] for h in hist]
    acc_vals = [h["acc"] for h in hist]
    r, p = np.corrcoef(vals, acc_vals)[0,1], 0
    # Correlate with accuracy
    coef = np.corrcoef(vals, acc_vals)[0,1]
    print(f"  {key:<12s}: corr with acc = {coef:+.4f}")

# ANGLE 4: Predictive cross-correlation during recovery
print("\n=== ANGLE 4: Predictive lag during recovery ===")
hist = corrupt_and_track(m, X_tr, Y_tr, X_te, Y_te, "weight_noise", 0.05, epochs=80)
pc_vec = np.array([h["phase_coh"] for h in hist])
acc_vec = np.array([h["acc"] for h in hist])
for lag in [0, 1, 3, 5]:
    if len(pc_vec) > lag:
        r = np.corrcoef(pc_vec[:len(pc_vec)-lag], acc_vec[lag:])[0,1] if lag > 0 else np.corrcoef(pc_vec, acc_vec)[0,1]
        direction = "predicts" if lag > 0 else "concurrent"
        print(f"  corr(phase_coh[t], acc[t+{lag}]): {r:+.4f} ({direction})")

print(f"\n{'='*60}")
print("VERDICT:")
print("  If phase coherently drops BEFORE accuracy for ANY corruption type -> leading indicator")
print("  If phase consistently lags accuracy -> concurrent/lagging indicator")
print("  If AUROC > 0.8 for any phase metric -> strong alignment signal")
