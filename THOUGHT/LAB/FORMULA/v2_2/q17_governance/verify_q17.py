"""Q17: R-gating improves governance — instrumented cybernetic loop."""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, json
import numpy as np
from scipy import stats
torch.manual_seed(42); random.seed(42)

# ---- Data (same as cybernetic_loop.py) ----
def gen_transforms(n=200, n_pts=8):
    X, Y = [], []
    for _ in range(n):
        t = random.randint(0, 3)
        zx = torch.randn(n_pts) * 2.0; zy = torch.randn(n_pts) * 2.0
        th = random.random() * 2 * math.pi
        if t == 0:
            c, s = math.cos(th), math.sin(th); ox = zx*c - zy*s; oy = zx*s + zy*c
        elif t == 1:
            c, s = math.cos(th), math.sin(th); ox = zx*c + zy*s; oy = zx*s - zy*c
        elif t == 2:
            sc = 0.2 + random.random() * 3.0; ox = zx*sc; oy = zy*sc
        else:
            k = random.random() * 2 - 1; ox = zx + k*zy; oy = zy
        z = torch.complex(zx, zy); zp = torch.complex(ox, oy)
        ratio = zp / (z + 1e-8)
        feats = torch.stack([
            torch.cos(torch.angle(ratio)).mean(), torch.sin(torch.angle(ratio)).mean(),
            torch.cos(torch.angle(ratio)).std(), torch.sin(torch.angle(ratio)).std(),
            torch.abs(ratio).mean(), torch.abs(ratio).std(),
        ])
        X.append(feats); Y.append(t)
    return torch.stack(X), torch.tensor(Y)

class RealMLP(nn.Module):
    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(6,16),nn.ReLU(),nn.Linear(16,4))
    def forward(self,x): return self.net(x)
    def clone(self):
        m=RealMLP();m.load_state_dict({k:v.clone() for k,v in self.state_dict().items()});return m

# ---- R Metrics for Q17 ----
def compute_R_metrics(X_train, Y_train, model):
    """Compute R = E/nabla_S from model predictions on training data."""
    with torch.no_grad():
        logits = model(X_train)
        probs = F.softmax(logits, dim=-1)
        correct = (logits.argmax(-1) == Y_train).float()

        # E = mean probability of correct class
        correct_probs = probs[torch.arange(len(Y_train)), Y_train]
        E = correct_probs.mean().item()

        # nabla_S = std of prediction entropy across samples
        entropies = -(probs * torch.log(probs + 1e-8)).sum(-1)
        nabla_S = entropies.std().item() + 1e-8

        # sigma = 1/Df of probability distribution
        probs_np = probs.numpy()
        cov = np.cov(probs_np.T)
        ev = np.linalg.eigvalsh(cov); ev = np.maximum(ev, 1e-15); ev = ev / ev.sum()
        sigma = 1.0 / max(ev.sum()**2 / (ev**2).sum(), 1e-10)

        # Phase coherence = 1 - entropy/ln(4)
        phase_coh = 1.0 - entropies.mean().item() / math.log(4)

        R = E / max(nabla_S, 1e-8)
        accuracy = correct.mean().item()

    return {"R": R, "sigma": sigma, "nabla_S": nabla_S,
            "phase_coh": phase_coh, "E": E, "accuracy": accuracy}

# ---- Q17 Test: does R predict when correction helps? ----
def train_control(model_cls, X_train, Y_train, X_test, Y_test, epochs=100):
    m = model_cls(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    history = []
    for e in range(epochs):
        logits = m(X_train); loss = F.cross_entropy(logits, Y_train)
        opt.zero_grad(); loss.backward(); opt.step()
        if e % 10 == 0:
            met = compute_R_metrics(X_train, Y_train, m)
            met["epoch"] = e; history.append(met)
    with torch.no_grad():
        acc = (m(X_test).argmax(-1) == Y_test).float().mean().item()
    return acc, history

def train_cassette(model_cls, X_train, Y_train, X_test, Y_test, epochs=100):
    m = model_cls(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    history = []
    for e in range(epochs):
        logits = m(X_train); preds = logits.argmax(-1)
        loss = F.cross_entropy(logits, Y_train)
        wrong = preds != Y_train
        if wrong.sum() > 0 and e > 3:
            widx = wrong.nonzero(as_tuple=True)[0]
            loss = loss + 0.5 * F.cross_entropy(m(X_train[widx]), Y_train[widx])
        opt.zero_grad(); loss.backward(); opt.step()
        if e % 10 == 0:
            met = compute_R_metrics(X_train, Y_train, m)
            met["epoch"] = e; history.append(met)
    with torch.no_grad():
        acc = (m(X_test).argmax(-1) == Y_test).float().mean().item()
    return acc, history

def train_r_gated(model_cls, X_train, Y_train, X_test, Y_test, epochs=100):
    """R-gated: correct when R drops below threshold."""
    m = model_cls(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    history = []
    for e in range(epochs):
        logits = m(X_train); loss = F.cross_entropy(logits, Y_train)
        met = compute_R_metrics(X_train, Y_train, m)
        # R-gated correction: if R drops, the model is uncertain -> extra training on wrong examples
        if e > 3 and met["R"] < history[-1]["R"] * 0.95:  # R dropped 5%
            preds = logits.argmax(-1)
            wrong = preds != Y_train
            if wrong.sum() > 0:
                widx = wrong.nonzero(as_tuple=True)[0]
                loss = loss + 0.5 * F.cross_entropy(m(X_train[widx]), Y_train[widx])
        opt.zero_grad(); loss.backward(); opt.step()
        if e % 10 == 0:
            met["epoch"] = e; history.append(met)
    with torch.no_grad():
        acc = (m(X_test).argmax(-1) == Y_test).float().mean().item()
    return acc, history


X_train, Y_train = gen_transforms(200)
X_test, Y_test = gen_transforms(200)

print("Q17: R-gating improves governance")
print("=" * 60)

for model_cls, name in [(RealMLP, "Real MLP")]:
    print(f"\n{name}:")
    ctrl_acc, ctrl_hist = train_control(model_cls, X_train, Y_train, X_test, Y_test)
    cass_acc, cass_hist = train_cassette(model_cls, X_train, Y_train, X_test, Y_test)
    r_acc, r_hist = train_r_gated(model_cls, X_train, Y_train, X_test, Y_test)

    print(f"  CONTROL:       {ctrl_acc:.1%}")
    print(f"  CASSETTE:      {cass_acc:.1%}  ({cass_acc-ctrl_acc:+.1%})")
    print(f"  R-GATED:       {r_acc:.1%}  ({r_acc-ctrl_acc:+.1%})")

    # Does R-track accuracy?
    R_vals = [h["R"] for h in ctrl_hist]
    acc_vals = [h["accuracy"] for h in ctrl_hist]

    # Does R predict wrong predictions?
    for label, hist in [("CONTROL", ctrl_hist), ("CASSETTE", cass_hist)]:
        phase_coh = [h["phase_coh"] for h in hist]
        acc = [h["accuracy"] for h in hist]
        r, p = stats.pearsonr(phase_coh, acc)
        print(f"  {label} corr(phase_coh, accuracy): r={r:.3f} p={p:.4f}")

    # Key question: does the R-gated approach match or beat the cassette?
    winner = max([("CONTROL", ctrl_acc), ("CASSETTE", cass_acc), ("R-GATED", r_acc)], key=lambda x: x[1])
    print(f"\n  Winner: {winner[0]} at {winner[1]:.1%}")
    if r_acc >= cass_acc:
        print(f"  R-GATED BEATS OR MATCHES CASSETTE — R suffices as governance signal")
    else:
        print(f"  R-GATED trails CASSETTE by {cass_acc-r_acc:.1%}")

print(f"\n{'='*60}")
print("VERDICT:")
print("  If R-gated accuracy >= cassette accuracy, R-metrics suffice for governance.")
print("  The cybernetic loop + R = self-correcting governance without explicit labels.")
