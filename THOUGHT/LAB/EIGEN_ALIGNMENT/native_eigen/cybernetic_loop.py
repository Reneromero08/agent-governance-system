"""Cybernetic Loop — Native Eigen vs Real MLP on geometry classification.

CONTROL: Standard training. CASSETTE: Self-correction. MORE_EPOCHS: Ablation.
The question: does phase-structured attention learn faster than real MLP?
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
torch.manual_seed(42); random.seed(42)

# ---- Data ----
def gen_transforms(n=200, n_pts=8):
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

# ---- Real MLP (baseline) ----
class RealMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 4))
    def forward(self, x): return self.net(x)
    def clone(self):
        m = RealMLP()
        m.load_state_dict({k: v.clone() for k, v in self.state_dict().items()})
        return m
    @property
    def params(self): return sum(p.numel() for p in self.parameters())

class NativeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.qr = nn.Parameter(torch.randn(2) * 0.1)
        self.qi = nn.Parameter(torch.randn(2) * 0.1)
        self.kr = nn.Parameter(torch.randn(2) * 0.1)
        self.ki = nn.Parameter(torch.randn(2) * 0.1)
        self.vr = nn.Linear(6, 2, bias=False)
        self.vi = nn.Linear(6, 2, bias=False)
        nn.init.normal_(self.vr.weight, std=0.1)
        nn.init.normal_(self.vi.weight, std=0.1)
        self.out = nn.Linear(2, 4)
        nn.init.normal_(self.out.weight, std=0.1)
        self.phase = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B = x.shape[0]
        vr = self.vr(x); vi = self.vi(x)
        qr = self.qr.unsqueeze(0); qi = self.qi.unsqueeze(0)
        kr = self.kr.unsqueeze(0); ki = self.ki.unsqueeze(0)
        score_r = (qr * kr + qi * ki).sum(dim=-1, keepdim=True)
        score_i = (qi * kr - qr * ki).sum(dim=-1, keepdim=True)
        c, s = torch.cos(score_i), torch.sin(score_i)
        out_r = c * vr - s * vi
        out_i = c * vi + s * vr
        c2, s2 = torch.cos(self.phase), torch.sin(self.phase)
        out_r = out_r * c2 - out_i * s2
        return self.out(out_r)

    def clone(self):
        m = NativeClassifier()
        m.load_state_dict({k: v.clone() for k, v in self.state_dict().items()})
        return m
    @property
    def params(self): return sum(p.numel() for p in self.parameters())

# ---- Test both architectures ----
def test(model_class, name):
    m = model_class()
    print("\n{}: {:,} params".format(name, m.params))
    
    # CONTROL
    ctrl = model_class(); opt = torch.optim.AdamW(ctrl.parameters(), lr=1e-2)
    for e in range(100):
        loss = F.cross_entropy(ctrl(X_train), Y_train)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ctrl_acc = (ctrl(X_test).argmax(-1) == Y_test).float().mean()
    
    # CASSETTE
    cass = model_class(); opt = torch.optim.AdamW(cass.parameters(), lr=1e-2)
    for e in range(100):
        logits = cass(X_train); preds = logits.argmax(-1)
        loss = F.cross_entropy(logits, Y_train)
        wrong = preds != Y_train
        if wrong.sum() > 0 and e > 3:
            widx = wrong.nonzero(as_tuple=True)[0]
            loss = loss + 0.5 * F.cross_entropy(cass(X_train[widx]), Y_train[widx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        cass_acc = (cass(X_test).argmax(-1) == Y_test).float().mean()
    
    # MORE EPOCHS
    more = model_class(); opt = torch.optim.AdamW(more.parameters(), lr=1e-2)
    for e in range(150):
        loss = F.cross_entropy(more(X_train), Y_train)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        more_acc = (more(X_test).argmax(-1) == Y_test).float().mean()
    
    return ctrl_acc, cass_acc, more_acc

X_train, Y_train = gen_transforms(200)
X_test, Y_test = gen_transforms(200)

print("=" * 60)
print("NATIVE EIGEN vs REAL MLP — Cybernetic Loop")
print("=" * 60)

mlp = test(RealMLP, "Real MLP")
ne = test(NativeClassifier, "Native Eigen")

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print("              REAL MLP    NATIVE EIGEN")
print("CONTROL:      {:.1%}       {:.1%}".format(mlp[0], ne[0]))
print("CASSETTE:     {:.1%}       {:.1%}".format(mlp[1], ne[1]))
print("MORE EPOCHS:  {:.1%}       {:.1%}".format(mlp[2], ne[2]))
mlp_delta = mlp[1] - mlp[0]; ne_delta = ne[1] - ne[0]
print("\nCassette delta: MLP={:+.1%}  Native={:+.1%}".format(mlp_delta, ne_delta))

if ne[0] >= mlp[0] and NativeClassifier().params < RealMLP().params:
    print("FEWER PARAMS, SAME ACCURACY — phase structure is efficient")
elif ne_delta > mlp_delta:
    print("LARGER CASSETTE DELTA — phase learns better from corrections")
else:
    print("Real MLP wins — phase not load-bearing at this scale")
