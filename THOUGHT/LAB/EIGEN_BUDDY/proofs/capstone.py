"""Phase Coherence Gate on Native Eigen — Q17 closure.

Q17 proved: phase_coh gate matches CASSETTE on RealMLP (94.8%).
This test: does the gate work on Native Eigen's complex attention?

Task: Geometry classification (4 classes). Hard (200 examples).
Conditions: CONTROL | CASSETTE (label-guided) | PHASE-GATED (autonomous)
If PHASE-GATED ≈ CASSETTE, the gate works on phase-native architecture.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
torch.manual_seed(42); random.seed(42)

# ---- Data (geometry: 4-class) ----
def gen_geometry(n=200, n_pts=8):
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
            torch.cos(torch.angle(ratio)).std(), torch.sin(torch.angle(ratio)).std(),
            torch.abs(ratio).mean(), torch.abs(ratio).std(),
        ])
        X.append(feats); Y.append(t)
    return torch.stack(X), torch.tensor(Y)

# ---- Native Eigen classifier with phase coherence output ----
class NativeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Separate encodings: Q-path and K-path see different aspects
        self.enc_qr = nn.Linear(6, 2, bias=False); self.enc_qi = nn.Linear(6, 2, bias=False)
        self.enc_kr = nn.Linear(6, 2, bias=False); self.enc_ki = nn.Linear(6, 2, bias=False)
        self.enc_vr = nn.Linear(6, 2, bias=False); self.enc_vi = nn.Linear(6, 2, bias=False)
        self.out = nn.Linear(2, 4); self.phase = nn.Parameter(torch.tensor(0.1))
        for w in [self.enc_qr, self.enc_qi, self.enc_kr, self.enc_ki, self.enc_vr, self.enc_vi, self.out]:
            nn.init.normal_(w.weight, std=0.1)

    def forward(self, x, return_phase=False):
        # Q from input (what to look for)
        qr = self.enc_qr(x); qi = self.enc_qi(x)
        # K from input (what is present)
        kr = self.enc_kr(x); ki = self.enc_ki(x)
        # V from input (what to output)
        vr = self.enc_vr(x); vi = self.enc_vi(x)
        # Phase difference: Q vs K — varies per sample since they encode differently
        score_i = (qi * kr - qr * ki).sum(dim=-1)
        c, s = torch.cos(score_i.unsqueeze(-1)), torch.sin(score_i.unsqueeze(-1))
        zr, zi = c * vr - s * vi, c * vi + s * vr
        c2, s2 = torch.cos(self.phase), torch.sin(self.phase)
        zr = zr * c2 - zi * s2
        if return_phase:
            phases = score_i
            cos_mean = torch.cos(phases).mean()
            sin_mean = torch.sin(phases).mean()
            phase_coh = (cos_mean**2 + sin_mean**2).sqrt()
            return self.out(zr), phase_coh
        return self.out(zr)

X, Y = gen_geometry(300)
X_train, Y_train = X[:200], Y[:200]
X_test, Y_test = X[200:], Y[200:]

# ---- CONTROL ----
print("NATIVE EIGEN: Phase Coherence Gate (Q17)")
print("=" * 55)
ctrl = NativeClassifier(); opt = torch.optim.AdamW(ctrl.parameters(), lr=1e-2)
for e in range(100):
    loss = F.cross_entropy(ctrl(X_train), Y_train)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    ctrl_acc = (ctrl(X_test).argmax(-1) == Y_test).float().mean()

# ---- CASSETTE (label-guided) ----
cass = NativeClassifier(); opt = torch.optim.AdamW(cass.parameters(), lr=1e-2)
for e in range(100):
    pred = cass(X_train); loss = F.cross_entropy(pred, Y_train)
    wrong = pred.argmax(-1) != Y_train
    if wrong.sum() > 0 and e > 3:
        widx = wrong.nonzero(as_tuple=True)[0]
        loss = loss + 0.5 * F.cross_entropy(cass(X_train[widx]), Y_train[widx])
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    cass_acc = (cass(X_test).argmax(-1) == Y_test).float().mean()

# ---- PHASE-GATED (autonomous, Q17 gate) ----
gate = NativeClassifier(); opt = torch.optim.AdamW(gate.parameters(), lr=1e-2)
gates_fired = 0; corrections = 0
for e in range(100):
    pred, phase_coh = gate(X_train, return_phase=True)
    loss = F.cross_entropy(pred, Y_train)
    # Q17 gate: phase_coh < 0.85 triggers correction
    if phase_coh < 0.85 and e > 3:
        gates_fired += 1
        wrong = pred.argmax(-1) != Y_train
        if wrong.sum() > 0:
            widx = wrong.nonzero(as_tuple=True)[0]
            loss = loss + 0.5 * F.cross_entropy(gate(X_train[widx]), Y_train[widx])
            corrections += wrong.sum().item()
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    gate_acc = (gate(X_test).argmax(-1) == Y_test).float().mean()

# ---- MORE EPOCHS ----
more = NativeClassifier(); opt = torch.optim.AdamW(more.parameters(), lr=1e-2)
for e in range(150):
    loss = F.cross_entropy(more(X_train), Y_train)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    more_acc = (more(X_test).argmax(-1) == Y_test).float().mean()

print("\nRESULTS")
print("=" * 55)
print("CONTROL:      {:.1%}".format(ctrl_acc))
print("CASSETTE:     {:.1%}".format(cass_acc))
print("PHASE-GATED:  {:.1%}  gates={} corrections={}".format(gate_acc, gates_fired, corrections))
print("MORE EPOCHS:  {:.1%}".format(more_acc))

best = max(ctrl_acc, more_acc)
print("\nPhase-gated vs best non-gated: {:+.1%}".format(gate_acc - best))
if gate_acc >= cass_acc * 0.98:
    print("PHASE COHERENCE GATE WORKS — matches label-guided on Native Eigen")
elif gate_acc > best:
    print("WEAK — gate helps but doesn't match CASSETTE")
else:
    print("GATE FAILED — phase_coh doesn't detect errors on Native Eigen")
