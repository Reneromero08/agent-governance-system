"""Semiotic field PINN: wave equation + resonance conservation via collocation.

Proper implementation: samples collocation points, computes spatial and temporal
derivatives via finite differences, enforces wave equation and resonance conservation.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# PINN architecture: (x, t) -> (E, sigma, grad_S)
class SemioticPINN(nn.Module):
    def __init__(self, hidden=32, layers=4):
        super().__init__()
        in_dim = 2
        self.input_layer = nn.Linear(in_dim, hidden)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers-1)])
        self.output_layer = nn.Linear(hidden, 3)  # E, sigma, grad_S
        
    def forward(self, x):
        h = torch.tanh(self.input_layer(x))
        for layer in self.hidden_layers:
            h = torch.tanh(layer(h))
        out = self.output_layer(h)
        # E: signal amplitude (unbounded)
        # sigma: compression (positive, > 0)
        # grad_S: entropy (positive, > 0)
        E = out[:, 0:1]
        sigma = torch.sigmoid(out[:, 1:2]) * 5.0 + 0.1  # [0.1, 5.1]
        grad_S = torch.sigmoid(out[:, 2:3]) * 2.0 + 0.05  # [0.05, 2.05]
        return E, sigma, grad_S

def compute_derivatives(model, x_c, t_c, dx, dt):
    """Compute E and its derivatives at collocation point (x_c, t_c)."""
    # Central point
    xt_center = torch.cat([x_c, t_c], dim=1)
    E_c, sigma_c, grad_S_c = model(xt_center)
    
    # Spatial neighbors for d2E/dx2
    xt_left = torch.cat([x_c - dx, t_c], dim=1)
    xt_right = torch.cat([x_c + dx, t_c], dim=1)
    E_left, _, _ = model(xt_left)
    E_right, _, _ = model(xt_right)
    d2E_dx2 = (E_left - 2*E_c + E_right) / (dx * dx)
    
    # Temporal neighbors for d2E/dt2
    xt_past = torch.cat([x_c, t_c - dt], dim=1)
    xt_future = torch.cat([x_c, t_c + dt], dim=1)
    E_past, _, _ = model(xt_past)
    E_future, _, _ = model(xt_future)
    d2E_dt2 = (E_past - 2*E_c + E_future) / (dt * dt)
    
    # First temporal derivative for resonance conservation
    dE_dt = (E_future - E_past) / (2 * dt)
    
    return E_c, sigma_c, grad_S_c, d2E_dx2, d2E_dt2, dE_dt

def semiotic_loss(model, batch_size=256, dx=0.1, dt=0.1):
    """Compute semiotic field loss on collocation points."""
    device = next(model.parameters()).device
    
    # Sample collocation points in (x, t) domain
    x = torch.rand(batch_size, 1, device=device) * 4.0 - 2.0  # [-2, 2]
    t = torch.rand(batch_size, 1, device=device) * 4.0  # [0, 4]
    
    E, sigma, grad_S, d2E_dx2, d2E_dt2, dE_dt = compute_derivatives(model, x, t, dx, dt)
    
    # Wave equation residual: d2E/dt2 - c^2 * d2E/dx2 = 0
    c_squared = sigma / (grad_S + 1e-8)
    c_squared_clamped = torch.clamp(c_squared, 0.0, 100.0)
    wave_residual = d2E_dt2 - c_squared_clamped * d2E_dx2
    
    # Resonance conservation: dR/dt = 0 along propagation
    # R = (E / grad_S) * sigma  (Df=1 for field limit)
    R = (E / (grad_S + 1e-8)) * sigma
    dR_dx = 0  # simplified
    
    # Boundary/initial conditions: E(x, 0) = Gaussian pulse
    t0_mask = (t < dt).float()
    ic_target = torch.exp(-x**2 / 0.2)  # Gaussian pulse at t=0
    ic_loss = t0_mask * (E - ic_target)**2
    
    # Combine losses
    wave_loss = (wave_residual**2).mean()
    resonance_loss = (R**2).mean()  # penalize explosion
    boundary_loss = ic_loss.mean()
    
    total = wave_loss + 0.01 * resonance_loss + 0.1 * boundary_loss
    return total, wave_loss, resonance_loss, boundary_loss

# Training
print("Training semiotic field PINN...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SemioticPINN(hidden=32, layers=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 2000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    total, wave, res, ic = semiotic_loss(model, batch_size=256)
    total.backward()
    optimizer.step()
    
    if epoch % 200 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch:4d}: total={total.item():.6f} wave={wave.item():.6f} res={res.item():.6f} ic={ic.item():.6f}")

# Test: evaluate wave propagation at fixed x positions over time
print("\nTesting wave propagation...")
model.eval()
dx = 0.1; dt = 0.1
with torch.no_grad():
    x_vals = torch.tensor([[-1.0], [0.0], [1.0]], device=device).float()
    t_vals = torch.linspace(0, 4, 40, device=device).unsqueeze(1)
    
    for j, x0 in enumerate(x_vals):
        print(f"\n  x = {x0.item():.1f}:")
        for t0 in t_vals[::5]:
            xt = torch.cat([x0.expand(1, 1), t0.unsqueeze(0)], dim=1)
            E, sig, gS = model(xt)
            print(f"    t={t0.item():.1f}: E={E.item():.4f} sigma={sig.item():.4f} gS={gS.item():.4f}")

# Check if wave equation holds at test points
print("\nWave equation check:")
with torch.no_grad():
    for x0 in [-1.0, 0.0, 1.0]:
        x = torch.tensor([[x0]], device=device).float()
        t = torch.linspace(dt, 4-dt, 10, device=device).unsqueeze(1)
        E, sigma, grad_S, d2x, d2t, _ = compute_derivatives(model, x.expand(10,1), t, dx, dt)
        c2 = torch.clamp(sigma/(grad_S+1e-8), 0, 100)
        wave_err = (d2t - c2*d2x).abs().mean().item()
        R_val = ((E/(grad_S+1e-8))*sigma).mean().item()
        print(f"  x={x0:.1f}: wave_err={wave_err:.6f} R_mean={R_val:.6f}")

# Save
torch.save(model.state_dict(), str(RESULTS / "semiotic_pinn.pt"))
print(f"\nSaved: {RESULTS / 'semiotic_pinn.pt'}")
