"""23.2: Temporal Catalysis on Real Weights — Qwen 0.5B Layer 0"""
import time, math, glob, torch, torch.nn.functional as F
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
MODEL_DIR = str(next(p for p in Path(__file__).resolve().parents if p.name == "CAT_CAS") / "16_catalytic_27b_inference" / "gemini_update" / "qwen_0.5b")
from safetensors.torch import load_file

def forward_catalytic(x, Wq_raw, Wk_raw, Wv_raw, Wo_raw, k, future_tape=None):
    def svd_if_square(W):
        if W.shape[0] == W.shape[1]:
            U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
            ka = min(k, U.shape[1])
            Wr = (U[:, :ka] * S[:ka].unsqueeze(0)) @ Vh[:ka, :]
            return Wr, (U, S, Vh, ka)
        return W.to(x.dtype), (None, None, None, W.shape[0])

    Wq, (Uq, Sq, Vhq, kq) = svd_if_square(Wq_raw)
    Wk, _ = svd_if_square(Wk_raw)
    Wv, _ = svd_if_square(Wv_raw)
    Wo, _ = svd_if_square(Wo_raw)

    Wq_u = Wq

    if future_tape is not None and Uq is not None:
        fv = future_tape.mean(dim=(0, 1))
        fv = fv / (fv.norm() + 1e-8)
        mw = torch.zeros(kq)
        for i in range(kq):
            mw[i] = torch.abs(torch.dot(Uq[:, i], fv))
        mw = F.softmax(mw * 10.0, dim=0)
        boosted = torch.where(mw > 1.0 / kq,
                              torch.ones(kq) * 2.0, torch.ones(kq) * 0.1)
        Wq_u = (Uq[:, :kq] * (Sq[:kq] * boosted).unsqueeze(0)) @ Vhq[:kq, :]

    # Project to Q's dimension (the master dim)
    D = Wq_u.shape[0]  # Q's output dim
    q = F.linear(x, Wq_u.to(x.dtype))  # (B, S, D)
    k = F.linear(x, Wk.to(x.dtype))    # (B, S, dk)
    v = F.linear(x, Wv.to(x.dtype))    # (B, S, dv)

    # Pad if needed
    if k.shape[-1] < D:
        k = F.pad(k, (0, D - k.shape[-1]))
    if v.shape[-1] < D:
        v = F.pad(v, (0, D - v.shape[-1]))

    attn = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(D), dim=-1)
    out = attn @ v
    out = F.linear(out, Wo.to(x.dtype))
    return out + x

def main():
    print("=" * 78)
    print("23.2: TEMPORAL CATALYSIS ON REAL WEIGHTS")
    print("=" * 78)
    w = {}
    for f in sorted(glob.glob(f"{MODEL_DIR}/*.safetensors")): w.update(load_file(f))
    L0 = {}
    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        for k in w:
            if '.0.' in k and proj in k and 'weight' in k:
                L0[proj] = w[k].float()

    print(f"  Q: {list(L0['q_proj'].shape)}  K: {list(L0['k_proj'].shape)}  V: {list(L0['v_proj'].shape)}  O: {list(L0['o_proj'].shape)}")

    B, S = 2, 16
    D = L0['q_proj'].shape[0]
    x = torch.randn(B, S, D) * 0.02
    future = torch.randn(B, S, D) * 0.1

    print(f"\n  {'k':>6}  {'baseline':>12}  {'retro':>12}  {'diff':>12}  {'signal?':>10}")
    print(f"  {'-'*60}")
    B, S = 2, 16
    D = L0['q_proj'].shape[0]
    
    # Two different inputs — simulating consecutive tokens in a sequence
    x1 = torch.randn(B, S, D) * 0.02  # token at position t
    x2 = torch.randn(B, S, D) * 0.02  # token at position t+1 (future)
    
    print(f"\n  Token t -> SVD -> output. Token t+1 output calibrates token t's SVD.")
    print(f"\n  {'k':>6}  {'baseline':>12}  {'retro':>12}  {'diff':>12}  {'signal?':>10}")
    print(f"  {'-'*60}")
    
    diffs = []
    for k_val in [16, 32, 64, 96, 128]:
        with torch.no_grad():
            # Baseline: token t, no future calibration
            h0 = forward_catalytic(x1, L0['q_proj'], L0['k_proj'], L0['v_proj'], L0['o_proj'], k_val)
            # Process token t+1 to get future context
            h_future = forward_catalytic(x2, L0['q_proj'], L0['k_proj'], L0['v_proj'], L0['o_proj'], k_val)
            # Retro: token t, calibrated by token t+1's output
            h1 = forward_catalytic(x1, L0['q_proj'], L0['k_proj'], L0['v_proj'], L0['o_proj'], k_val, h_future)
        
        diff = torch.max(torch.abs(h0 - h1)).item()
        signal = "SIGNAL" if diff > 1e-3 else "noise"
        diffs.append(diff)
        print(f"  {k_val:>6}  {h0.norm().item():>12.4f}  {h1.norm().item():>12.4f}  {diff:>12.6f}  {signal:>10}")
    import numpy as np
    print(f"\n  diff across k: mean={np.mean(diffs):.6f}  std={np.std(diffs):.6f}  "
          f"min={np.min(diffs):.6f}  max={np.max(diffs):.6f}")
    print()
    print()
if __name__ == "__main__": main()
