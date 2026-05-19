"""Curvature: d(theta)/ds — rate of phase change along a path.

Curved paths have non-constant phase velocity. The model must detect
whether a sequence of complex numbers follows a straight line (constant dθ/ds),
a curve (varying dθ/ds), or oscillates (periodic dθ/ds).

This is the same operation Q·K^† performs between tokens in Native Eigen.
Curvature = second derivative of phase = d²θ/ds².
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
torch.manual_seed(42); random.seed(42)

def gen_paths(n=1000, length=8):
    X, Y = [], []
    for _ in range(n):
        path_type = random.randint(0, 2)
        # Start at origin with random initial phase
        phase = random.random() * 2 * math.pi
        r = 1.0
        pts = []
        
        for i in range(length):
            z_real = r * math.cos(phase)
            z_imag = r * math.sin(phase)
            pts.extend([z_real, z_imag])
            
            if path_type == 0:       # STRAIGHT: constant dθ/ds
                dtheta = 0.3 + random.random() * 0.1
            elif path_type == 1:     # CURVED: accelerating dθ/ds
                dtheta = 0.1 + i * 0.1
            else:                     # OSCILLATING: sinusoidal dθ/ds
                dtheta = 0.3 * math.sin(i * 0.8)
            
            phase = (phase + dtheta) % (2 * math.pi)
        
        X.append(torch.tensor(pts, dtype=torch.float32))
        Y.append(path_type)
    
    return torch.stack(X), torch.tensor(Y)

class CurvatureDetector(nn.Module):
    """Detect path curvature from phase velocity and acceleration."""
    def __init__(self, length=8):
        super().__init__()
        self.L = length
        # Extract per-token complex numbers -> compute phase differences -> classify
        self.phase_net = nn.Sequential(
            nn.Linear(length * 2, 32),  # 2 per token (real, imag)
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )
        # Also explicitly compute phase derivatives
        n_deriv = (length-1) * 2 + (length-2) * 2  # cos+sin for dtheta and d2theta
        self.deriv_net = nn.Sequential(
            nn.Linear(n_deriv, 16),  # dθ/ds + d²θ/ds² per step
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.out = nn.Linear(8 + 3, 3)  # deriv_out(8) + raw_out(3) -> 3 classes

    def forward(self, x):
        B = x.shape[0]
        pts = x.view(B, self.L, 2)
        # Path 1: raw complex points
        raw_out = self.phase_net(x)
        # Path 2: explicit phase derivatives
        z = torch.complex(pts[:, :, 0], pts[:, :, 1])
        # dθ = phase(z_{i+1} / z_i) — first derivative
        dtheta = torch.angle(z[:, 1:] / (z[:, :-1] + 1e-8))  # (B, L-1)
        # d²θ = dtheta_{i+1} - dtheta_i — second derivative = CURVATURE
        d2theta = dtheta[:, 1:] - dtheta[:, :-1]  # (B, L-2)
        deriv_feats = torch.cat([
            torch.cos(dtheta), torch.sin(dtheta),
            torch.cos(d2theta), torch.sin(d2theta),
        ], dim=-1).reshape(B, -1)
        deriv_out = self.deriv_net(deriv_feats)
        return self.out(torch.cat([deriv_out, raw_out], dim=-1))

model = CurvatureDetector()
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
X, Y = gen_paths(1000); vX, vY = gen_paths(200)
names = ["STRAIGHT", "CURVED", "OSCILLATING"]

print("Curvature: detect d^2(theta)/ds^2 from complex path")
print("=" * 55)
for e in range(150):
    logits = model(X); loss = F.cross_entropy(logits, Y)
    opt.zero_grad(); loss.backward(); opt.step()
    if e % 30 == 0:
        with torch.no_grad():
            acc = (model(vX).argmax(-1) == vY).float().mean()
        print("  {:3d}: loss={:.4f}  acc={:.1%}".format(e, loss.item(), acc))

with torch.no_grad():
    pred = model(vX); acc = (pred.argmax(-1) == vY).float().mean()
    print("\nAccuracy: {:.1%}".format(acc))
    for t in range(3):
        mask = vY == t
        if mask.any(): print("  {}: {:.1%}".format(names[t], (pred[mask].argmax(-1)==t).float().mean()))
    print("Verdict: {}".format("CURVATURE DETECTED VIA PHASE" if acc > 0.85 else "not learned"))
