"""Continuum Math as Phase — trainable phase rotation learns the dispersion relation.

Level 4 (Schrodinger): i*hbar * dpsi/dt = H*psi
The evolution is e^(-i*H*dt/hbar) — a phase rotation.
The model learns omega = k^2/2m from data.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math
torch.manual_seed(42)

# Free particle wave packet
k = 5.0; hbar = 1.0; m = 1.0
omega_true = hbar * k*k / (2*m)  # dispersion: omega = k^2/2m

x = torch.linspace(-5, 5, 200).unsqueeze(0)
psi_r = torch.cos(k * x) * torch.exp(-x**2 / 4)
psi_i = torch.sin(k * x) * torch.exp(-x**2 / 4)

# Target: packet after time dt
dt_target = 0.05
c_t = math.cos(omega_true * dt_target)
s_t = math.sin(omega_true * dt_target)
target_r = psi_r * c_t - psi_i * s_t
target_i = psi_r * s_t + psi_i * c_t

# Model: learns omega (the dispersion relation)
class PhaseLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor(2.0))  # wrong initial guess

    def forward(self, pr, pi, dt):
        c = torch.cos(self.omega * dt)
        s = torch.sin(self.omega * dt)
        return pr * c - pi * s, pr * s + pi * c

model = PhaseLearner()
opt = torch.optim.AdamW([model.omega], lr=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)

print("Learning dispersion relation via phase rotation")
print("=" * 50)
print("True omega = k^2/2m = {:.3f}".format(omega_true))
print("Initial guess: {:.3f}".format(float(model.omega)))

for epoch in range(300):
    pr_out, pi_out = model(psi_r, psi_i, dt_target)
    loss = ((pr_out - target_r)**2 + (pi_out - target_i)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
    if epoch % 50 == 0:
        print("  {:3d}: omega={:.3f} loss={:.6f}".format(
            epoch, float(model.omega), loss.item()))

print("\nLearned omega: {:.4f}".format(float(model.omega)))
print("True omega:    {:.4f}".format(omega_true))
error = abs(float(model.omega) - omega_true) / omega_true * 100
print("Error: {:.1f}%".format(error))
print("Verdict: {}".format("PHASE LEARNS PHYSICS" if error < 5 else "not converged"))
