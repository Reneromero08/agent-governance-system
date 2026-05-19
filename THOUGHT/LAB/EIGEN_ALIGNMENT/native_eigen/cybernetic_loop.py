"""Cybernetic Loop v0 — Self-correcting Native Eigen on geometry classification.

Conditions:
  CONTROL: Model trains on 800 examples, tested on 200.
  CASSETTE: Model trains, wrong outputs are corrected via cassette, retrained.
  
The question: does self-correction via cassette improve accuracy
beyond the baseline? If yes, the cybernetic loop works.

Control parameters:
  - Same model architecture across conditions
  - Same training data, same number of epochs
  - Only difference: cassette correction loop
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
torch.manual_seed(42); random.seed(42)

# ---- Data + Model (geometry classification from proven architecture) ----
def gen_transforms(n=1000, n_pts=8):
    X, Y = [], []
    for _ in range(n):
        t = random.randint(0, 3)
        zx = torch.randn(n_pts) * 2.0; zy = torch.randn(n_pts) * 2.0
        th = random.random() * 2 * math.pi
        if t == 0:  # Rotation
            c, s = math.cos(th), math.sin(th)
            ox = zx*c - zy*s; oy = zx*s + zy*c
        elif t == 1:  # Reflection
            c, s = math.cos(th), math.sin(th)
            ox = zx*c + zy*s; oy = zx*s - zy*c  # conjugate + rotate
        elif t == 2:  # Scaling
            sc = 0.2 + random.random() * 3.0
            ox = zx*sc; oy = zy*sc
        else:  # Shear
            k = random.random() * 2 - 1
            ox = zx + k*zy; oy = zy
        # Complex ratio features
        z = torch.complex(zx, zy); zp = torch.complex(ox, oy)
        ratio = zp / (z + 1e-8)
        feats = torch.stack([
            torch.cos(torch.angle(ratio)).mean(), torch.sin(torch.angle(ratio)).mean(),
            torch.cos(torch.angle(ratio)).std(), torch.sin(torch.angle(ratio)).std(),
            torch.abs(ratio).mean(), torch.abs(ratio).std(),
        ])
        X.append(feats); Y.append(t)
    return torch.stack(X), torch.tensor(Y)

class GeoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 4))
    def forward(self, x): return self.net(x)
    def clone(self):
        m = GeoNet(); m.load_state_dict({k:v.clone() for k,v in self.state_dict().items()}); return m

# ---- Cassette (perfect knowledge of correct answers) ----
class Cassette:
    def __init__(self, X, Y): self.X = X; self.Y = Y
    def correct(self, idx): return self.X[idx], self.Y[idx]

# ---- CONDITIONS ----
X_train, Y_train = gen_transforms(200)  # Harder: only 200 examples
X_test, Y_test = gen_transforms(200)

# CONDITION A: CONTROL — standard training
print("=" * 55)
print("CONDITION A: CONTROL (standard training)")
ctrl_model = GeoNet()
opt = torch.optim.AdamW(ctrl_model.parameters(), lr=1e-2)
for e in range(100):
    loss = F.cross_entropy(ctrl_model(X_train), Y_train)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    ctrl_acc = (ctrl_model(X_test).argmax(-1) == Y_test).float().mean()
print("  Accuracy: {:.1%}".format(ctrl_acc))

# CONDITION B: CASSETTE — self-correction loop
print("\nCONDITION B: CASSETTE (self-correcting)")
cass_model = GeoNet()
cassette = Cassette(X_train, Y_train)
opt = torch.optim.AdamW(cass_model.parameters(), lr=1e-2)
corrections = 0
for e in range(100):
    # Forward
    logits = cass_model(X_train)
    preds = logits.argmax(-1)
    # Standard loss
    loss = F.cross_entropy(logits, Y_train)
    
    # +++ CYBERNETIC LOOP +++
    # Find wrong predictions
    wrong_mask = preds != Y_train
    n_wrong = wrong_mask.sum().item()
    if n_wrong > 0 and e > 3:  # Start correcting early with small dataset
        # Cassette provides correct answers
        wrong_idx = wrong_mask.nonzero(as_tuple=True)[0]
        correct_logits = cass_model(cassette.X[wrong_idx])
        correct_targets = cassette.Y[wrong_idx]
        # Correction loss: push wrong outputs toward correct
        correction_loss = F.cross_entropy(correct_logits, correct_targets)
        # Combined loss with higher weight on corrections
        loss = loss + 0.5 * correction_loss
        corrections += n_wrong
    
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    cass_acc = (cass_model(X_test).argmax(-1) == Y_test).float().mean()
print("  Accuracy: {:.1%}".format(cass_acc))

# CONDITION C: MORE EPOCHS (ablation — is it just more compute?)
print("\nCONDITION C: MORE EPOCHS (ablation)")
more_model = GeoNet()
opt = torch.optim.AdamW(more_model.parameters(), lr=1e-2)
for e in range(150):  # 50% more epochs than control
    loss = F.cross_entropy(more_model(X_train), Y_train)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    more_acc = (more_model(X_test).argmax(-1) == Y_test).float().mean()
print("  Accuracy: {:.1%}".format(more_acc))

# ---- RESULTS ----
print("\n" + "=" * 55)
print("RESULTS")
print("=" * 55)
print("CONTROL:         {:.1%}".format(ctrl_acc))
print("CASSETTE:        {:.1%}".format(cass_acc))
print("MORE EPOCHS:     {:.1%}".format(more_acc))
print()
if cass_acc > ctrl_acc and cass_acc > more_acc:
    print("CYBERNETIC LOOP WORKS — correction signal beats pure compute")
elif cass_acc > ctrl_acc:
    print("WEAK — cassette helps but not more than extra epochs")
else:
    print("LOOP NOT PROVEN — cassette doesn't improve over baseline")
