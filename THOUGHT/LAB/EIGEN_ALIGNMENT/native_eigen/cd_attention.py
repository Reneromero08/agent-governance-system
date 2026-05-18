"""C^d Complex Attention — d-dimensional Hermitian attention matrix.

Q·K^† is a full matrix of phase relationships between components.
Each attention weight is complex: magnitude = similarity, phase = twist.
Output = complex-weighted sum of values WITH rotation by Q-K phase.

Geometric operation on C^d:
    Q: (d, S)  K: (d, S)  V: (d, S)
    Scores[i,j] = <Q_i|K_j> = sum_k Q*[k,i] * K[k,j]  (Hermitian, complex)
    Weights = softmax(|Scores|) with phase preserved
    Output[i] = sum_j Weights[i,j] * e^(i*arg(Scores[i,j])) * V_j

This is the jump from C^1 (scalar rotation) to C^d (matrix phase algebra).
"""
import torch, torch.nn as nn, torch.nn.functional as F, math
torch.manual_seed(42)

class ComplexAttention(nn.Module):
    """Hermitian attention in C^d. Each attention weight is complex."""
    def __init__(self, d_model=8, n_heads=2):
        super().__init__()
        self.d = d_model
        self.H = n_heads
        self.dh = d_model // n_heads  # dims per head
        # Q, K, V projections: map real input to complex
        hd = d_model  # each head gets 2*dh real dims = dh complex dims
        self.Wq_r = nn.Linear(d_model, d_model, bias=False)
        self.Wq_i = nn.Linear(d_model, d_model, bias=False)
        self.Wk_r = nn.Linear(d_model, d_model, bias=False)
        self.Wk_i = nn.Linear(d_model, d_model, bias=False)
        self.Wv_r = nn.Linear(d_model, d_model, bias=False)
        self.Wv_i = nn.Linear(d_model, d_model, bias=False)
        self.Wo_r = nn.Linear(d_model, d_model, bias=False)
        self.Wo_i = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(self.dh)
        for w in [self.Wq_r, self.Wq_i, self.Wk_r, self.Wk_i,
                  self.Wv_r, self.Wv_i, self.Wo_r, self.Wo_i]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, x):
        # x: (B, S, d) real — input sequence
        B, S, d = x.shape
        # Project to complex Q, K, V
        qr, qi = self.Wq_r(x), self.Wq_i(x)  # (B, S, d)
        kr, ki = self.Wk_r(x), self.Wk_i(x)
        vr, vi = self.Wv_r(x), self.Wv_i(x)

        # Reshape for multi-head: (B, H, dh, S)
        qr = qr.view(B, S, self.H, self.dh).permute(0,2,3,1)  # (B, H, dh, S)
        qi = qi.view(B, S, self.H, self.dh).permute(0,2,3,1)
        kr = kr.view(B, S, self.H, self.dh).permute(0,2,3,1)
        ki = ki.view(B, S, self.H, self.dh).permute(0,2,3,1)
        vr = vr.view(B, S, self.H, self.dh).permute(0,2,3,1)
        vi = vi.view(B, S, self.H, self.dh).permute(0,2,3,1)

        # Hermitian attention: Q^† @ K = (dh, S) @ (dh, S)^T -> (S, S)
        # Real part: Qr·Kr + Qi·Ki
        # Imag part: Qi·Kr - Qr·Ki
        score_r = (qr.transpose(-2,-1) @ kr + qi.transpose(-2,-1) @ ki) * self.scale  # (B, H, S, S)
        score_i = (qi.transpose(-2,-1) @ kr - qr.transpose(-2,-1) @ ki) * self.scale

        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        score_r = score_r.masked_fill(mask, float('-inf'))
        score_i = score_i.masked_fill(mask, 0.0)

        # Complex attention weights: magnitude softmax, phase preserved
        attn_mag = F.softmax(score_r, dim=-1)  # (B, H, S, S)
        # Phase of each attention weight
        cos_p = torch.cos(score_i)
        sin_p = torch.sin(score_i)

        # Apply complex attention: rotate V by phase, weight by magnitude
        # Output_r = sum_j A_ij * (cos(p_ij)*V_rj - sin(p_ij)*V_ij)
        # Output_i = sum_j A_ij * (cos(p_ij)*V_ij + sin(p_ij)*V_rj)
        out_r = (attn_mag * cos_p) @ vr.transpose(-2,-1) - (attn_mag * sin_p) @ vi.transpose(-2,-1)
        out_i = (attn_mag * cos_p) @ vi.transpose(-2,-1) + (attn_mag * sin_p) @ vr.transpose(-2,-1)

        # Merge heads: (B, H, dh, S) -> (B, S, d)
        out_r = out_r.permute(0,3,1,2).contiguous().view(B, S, d)
        out_i = out_i.permute(0,3,1,2).contiguous().view(B, S, d)

        # Output projection (complex linear)
        out_r = self.Wo_r(out_r) - self.Wo_i(out_i)
        out_i = self.Wo_r(out_i) + self.Wo_i(out_r)

        return out_r, out_i  # real and imag parts of output


# ---- Test: learn to route information through phase ----
# Task: given 2 tokens in C^4, produce the Hadamard product (element-wise multiply)
def test_cd_attention():
    model = ComplexAttention(d_model=8, n_heads=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    B, S, d = 64, 4, 8

    for epoch in range(100):
        x = torch.randn(B, S, d)
        # Target: learn identity (output should match input at last position)
        target = x[:, -1, :]  # (B, d)

        out_r, out_i = model(x)
        # Use real part to predict target, imag part as auxiliary
        pred = out_r[:, -1, :]  # last token output
        loss = F.mse_loss(pred, target)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 20 == 0:
            # Check if imaginary channel is non-zero (phase is alive)
            imag_norm = out_i.abs().mean().item()
            print("  {:3d}: loss={:.4f}  |imag|={:.4f}".format(
                epoch, loss.item(), imag_norm))

    # Phase ablation test
    model.eval()
    with torch.no_grad():
        x = torch.randn(B, S, d)
        out_r_norm, out_i_norm = model(x)
        loss_norm = F.mse_loss(out_r_norm[:, -1, :], x[:, -1, :]).item()

        # Ablate: zero out Q and K imaginary projections
        saved = {n: p.data.clone() for n, p in model.named_parameters() if 'Wq_i' in n or 'Wk_i' in n}
        for n, p in model.named_parameters():
            if 'Wq_i' in n or 'Wk_i' in n:
                p.data.zero_()

        out_r_ab, out_i_ab = model(x)
        loss_ab = F.mse_loss(out_r_ab[:, -1, :], x[:, -1, :]).item()

        # Restore
        for n, p in model.named_parameters():
            if n in saved:
                p.data.copy_(saved[n])

    print("\nNormal loss:  {:.4f}".format(loss_norm))
    print("Ablated loss: {:.4f}".format(loss_ab))
    delta = (loss_ab - loss_norm) / max(loss_norm, 1e-8) * 100
    print("Delta: {:+.1f}%".format(delta))
    print("Verdict: {}".format(
        "C^d PHASE CARRIES INFORMATION" if delta > 5 else
        "WEAK" if delta > 1 else "phase not used"))

test_cd_attention()
