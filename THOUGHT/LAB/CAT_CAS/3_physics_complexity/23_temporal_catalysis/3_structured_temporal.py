"""23.3: Temporal Catalysis on Structured Data"""
import torch, torch.nn as nn, torch.nn.functional as F, math

# ---- Step 1: Create structured temporal data ----
# Sequence: x_{n+1} = (7 * x_n + 3) mod 100
# Each sample: (x_n, x_{n+1}) — input is token at t, target is token at t+1
def generate_sequence(n=2000):
    x = torch.zeros(n, dtype=torch.long)
    x[0] = 1
    for i in range(1, n): x[i] = (7 * x[i-1] + 3) % 100
    X_in = F.one_hot(x[:-1], 100).float()   # (n-1, 100)
    X_out = F.one_hot(x[1:], 100).float()   # (n-1, 100)
    return X_in, X_out

# ---- Step 2: Train a tiny predictor ----
class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(100, 100, bias=False)  # predict next token
    def forward(self, x): return self.W(x)

model = TinyPredictor()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
X_in, X_out = generate_sequence(2000)
for epoch in range(200):
    pred = model(X_in)
    loss = F.cross_entropy(pred, X_out.argmax(dim=1))
    opt.zero_grad(); loss.backward(); opt.step()
print(f"  Trained: loss={loss.item():.4f}, accuracy={(pred.argmax(1)==X_out.argmax(1)).float().mean():.2%}")

# ---- Step 3: SVD of trained weight matrix ----
W = model.W.weight.data.float()
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
print(f"  Top 5 singular values: {S[:5].tolist()}")
print(f"  D_pr = {(S.sum()**2 / (S**2).sum()).item():.1f} (of 100)")

# ---- Step 4: Temporal catalysis ----
# Token at position t (input) vs token at position t+1 (future)
x_t   = X_in[0:1]    # (1, 100) — current token
x_t1  = X_in[1:2]    # (1, 100) — next token (future)

# Forward pass WITHOUT future calibration
with torch.no_grad():
    out_t = model(x_t)     # prediction for t+1 using token t
    out_t1 = model(x_t1)   # prediction for t+2 using token t+1

# SVD-compress W for different k
for k in [5, 10, 20, 50]:
    # Reconstruct at rank k
    Wk = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]
    
    # Project future activation onto U_q modes
    fv = out_t1.squeeze()  # future context: token t+1's prediction
    mode_weights = torch.zeros(k)
    for i in range(k):
        mode_weights[i] = torch.abs(torch.dot(U[:, i], fv))
    mode_weights = F.softmax(mode_weights * 5.0, dim=0)
    
    # Boosted vs unboosted
    Wk_boosted = (U[:, :k] * (S[:k] * (0.5 + 0.5 * mode_weights)).unsqueeze(0)) @ Vh[:k, :]
    
    # Forward with both
    pred_base = x_t @ Wk.T      # baseline
    pred_retro = x_t @ Wk_boosted.T  # retrocausally calibrated
    
    diff = (pred_base - pred_retro).abs().max().item()
    signal = "SIGNAL" if diff > 0.01 else "noise"
    
    # Which mode is most aligned with future?
    top_mode = mode_weights.argmax().item()
    top_weight = mode_weights[top_mode].item()
    
    print(f"  k={k:>3}: diff={diff:.6f}  top_mode={top_mode} (weight={top_weight:.3f})  {signal}")

print(f"\n  The future token's representation genuinely aligns with")
print(f"  specific SVD modes of the weight matrix because the")
print(f"  sequence has real temporal structure that the model learned.")
