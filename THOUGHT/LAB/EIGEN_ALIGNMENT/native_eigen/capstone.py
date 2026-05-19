"""Capstone: Native Eigen + Facts Cassette + Cybernetic Loop.

The full architecture in one script. Native Eigen (phase attention) reasons.
Facts cassette provides ground truth. Cybernetic loop self-corrects.

Task: Arithmetic. The model predicts sums. The cassette knows the right answer.
When wrong, cassette corrects. Model fine-tunes. The loop closes.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random
torch.manual_seed(42); random.seed(42)

# ============================================================================
# 1. Facts Cassette — stores correct answers
# ============================================================================
class Cassette:
    def __init__(self, n_facts=100):
        # Generate random arithmetic facts the model must learn
        self.a = torch.randint(10, 100, (n_facts,)).float()
        self.b = torch.randint(10, 100, (n_facts,)).float()
        self.answer = self.a + self.b
        # Features: [a/100, b/100, a%10/10, b%10/10] — normalized
        self.X = torch.stack([
            self.a/100, self.b/100,
            (self.a%10)/10, (self.b%10)/10
        ], dim=-1)
        self.Y = self.answer / 200  # normalize to ~[0,1]

# ============================================================================
# 2. Native Eigen Model — phase-structured reasoning
# ============================================================================
class NativeReasoner(nn.Module):
    def __init__(self):
        super().__init__()
        # Encode to complex: 4D -> 2D
        self.enc_r = nn.Linear(4, 2, bias=False)
        self.enc_i = nn.Linear(4, 2, bias=False)
        # Q, K from input (not fixed constants)
        self.Wq_r = nn.Linear(2, 2, bias=False); self.Wq_i = nn.Linear(2, 2, bias=False)
        self.Wk_r = nn.Linear(2, 2, bias=False); self.Wk_i = nn.Linear(2, 2, bias=False)
        self.out = nn.Linear(2, 1)
        self.phase = nn.Parameter(torch.tensor(0.1))
        for w in [self.enc_r, self.enc_i, self.Wq_r, self.Wq_i, self.Wk_r, self.Wk_i, self.out]:
            nn.init.normal_(w.weight, std=0.1)

    def forward(self, x):
        # Encode to complex
        zr = self.enc_r(x); zi = self.enc_i(x)  # (B, 2)
        # Q, K from encoded input
        qr = self.Wq_r(zr) - self.Wq_i(zi)
        qi = self.Wq_r(zi) + self.Wq_i(zr)
        kr = self.Wk_r(zr) - self.Wk_i(zi)
        ki = self.Wk_r(zi) + self.Wk_i(zr)
        # Score = <Q|K> — per-example phase difference
        score_r = (qr * kr + qi * ki).sum(dim=-1, keepdim=True)
        score_i = (qi * kr - qr * ki).sum(dim=-1, keepdim=True)
        # Rotate Z by attention phase
        c, s = torch.cos(score_i), torch.sin(score_i)
        zr = c * zr - s * zi
        # Phase accumulation
        c2, s2 = torch.cos(self.phase), torch.sin(self.phase)
        zr = zr * c2 - (c * zi + s * zr) * s2
        return self.out(zr).squeeze(-1)

# ============================================================================
# 3. Cybernetic Loop — self-correction via cassette
# ============================================================================
cassette = Cassette(n_facts=200)
X, Y = cassette.X, cassette.Y

# Split
n_train = 150
X_train, Y_train = X[:n_train], Y[:n_train]
X_test, Y_test = X[n_train:], Y[n_train:]

def mae(pred, true):
    return (pred * 200 - true * 200).abs().mean().item()  # denormalized

# ---- CONTROL ----
print("CAPSTONE: Native Eigen + Cassette + Cybernetic Loop")
print("=" * 55)
print("Task: Arithmetic (a + b), {} train / {} test".format(n_train, len(Y)-n_train))

ctrl = NativeReasoner(); opt = torch.optim.AdamW(ctrl.parameters(), lr=1e-2)
for e in range(200):
    loss = F.mse_loss(ctrl(X_train), Y_train)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    ctrl_mae = mae(ctrl(X_test), Y_test)
print("CONTROL:        MAE={:.1f}  loss={:.4f}".format(ctrl_mae, loss.item()))

# ---- CASSETTE LOOP ----
cass_model = NativeReasoner(); opt = torch.optim.AdamW(cass_model.parameters(), lr=1e-2)
corrections = 0
for e in range(200):
    pred = cass_model(X_train)
    loss = F.mse_loss(pred, Y_train)
    # Self-correction: find wrong predictions
    error = (pred - Y_train).abs()
    wrong_mask = error > 0.02  # >4 absolute error
    if wrong_mask.sum() > 0 and e > 10:
        widx = wrong_mask.nonzero(as_tuple=True)[0]
        corr_loss = F.mse_loss(cass_model(X_train[widx]), Y_train[widx])
        loss = loss + 0.5 * corr_loss
        corrections += wrong_mask.sum().item()
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    cass_mae = mae(cass_model(X_test), Y_test)
print("CASSETTE LOOP:  MAE={:.1f}  corrections={}".format(cass_mae, corrections))

# ---- MORE EPOCHS ----
more_model = NativeReasoner(); opt = torch.optim.AdamW(more_model.parameters(), lr=1e-2)
for e in range(300):
    loss = F.mse_loss(more_model(X_train), Y_train)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    more_mae = mae(more_model(X_test), Y_test)
print("MORE EPOCHS:    MAE={:.1f}".format(more_mae))

# ---- Phase Ablation ----
ablated = NativeReasoner()
ablated.load_state_dict({k: v.clone() for k, v in ctrl.state_dict().items()})
ablated.phase.data.zero_()
with torch.no_grad():
    abl_mae = mae(ablated(X_test), Y_test)

print("\n" + "=" * 55)
print("INFERENCE — before and after cybernetic loop")
print("=" * 55)
with torch.no_grad():
    for i in range(5):
        a_val = float(cassette.a[n_train + i])
        b_val = float(cassette.b[n_train + i])
        true = a_val + b_val
        before = float(ctrl(X_test[i:i+1]) * 200)
        after = float(cass_model(X_test[i:i+1]) * 200)
        print("  {:3.0f} + {:3.0f} = {:3.0f}  |  before: {:5.1f}  after: {:5.1f}".format(
            a_val, b_val, true, before, after))

print()
print("CONTROL:         MAE={:.1f}".format(ctrl_mae))
print("CASSETTE:        MAE={:.1f}".format(cass_mae))
print("MORE EPOCHS:     MAE={:.1f}".format(more_mae))
print("PHASE ABLATED:   MAE={:.1f}".format(abl_mae))

best_non_cassette = min(ctrl_mae, more_mae, abl_mae)
delta = best_non_cassette - cass_mae
print("\nPhase ablation delta: {:+.1f}".format(abl_mae - ctrl_mae))
print("Cassette delta:       {:+.1f} over best non-cassette".format(delta))

if delta > 0.1:
    print("CAPSTONE PROVEN — cassette self-correction beats all controls")
elif cass_mae <= best_non_cassette:
    print("WEAK — cassette matches but doesn't clearly beat")
else:
    print("NOT PROVEN — cassette doesn't improve over baseline")
