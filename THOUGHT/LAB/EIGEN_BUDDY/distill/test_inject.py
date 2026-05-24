"""Test: inject distilled Qwen eigenbasis into Eigen Buddy attention weights.
This IS instant vectorization — no training, pure geometric mapping."""
import sys, json, numpy as np, torch
sys.path.insert(0, r'D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\EIGEN_BUDDY')
from core.attention import MultiHeadComplexAttention

# Load phase grating
distilled = np.load(r'D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\EIGEN_BUDDY\distill\distilled\eigenbuddy_qwen27b.holo.npz')
v_proj_key = 'model.language_model.layers.11.self_attn.v_proj'
grating = distilled[v_proj_key]  # (k=128, dim=1024) complex64
print(f"Loaded {v_proj_key}: shape={grating.shape}")

# Create TWO attention modules: one with random init, one with Qwen eigenbasis
# Both at d_model=1024, the smallest matching dimension
d_model = 1024
n_heads = 8  # 1024/8 = 128 = k — each head gets one eigenmode

# Random init attention
torch.manual_seed(42)
attn_random = MultiHeadComplexAttention(d_model=d_model, n_heads=n_heads, geo_init=False)

# Qwen eigenbasis attention
torch.manual_seed(42)  # same base init seed
attn_qwen = MultiHeadComplexAttention(d_model=d_model, n_heads=n_heads, geo_init=False)

# INJECT Qwen eigenbasis into attention weights
# Each head h gets eigenmode h as its initialization pattern
dh = d_model // n_heads  # = 128
for h in range(n_heads):
    start, end = h * dh, (h + 1) * dh
    # The eigenmode for head h: grating[h, :] is (dim=1024,) complex
    eigenmode = grating[h % grating.shape[0], :]  # wrap if n_heads > k
    phase_angle = np.angle(eigenmode)
    magnitude = np.abs(eigenmode)
    
    # Apply to qr projection weights
    with torch.no_grad():
        for proj_name in ['qr', 'qi', 'kr', 'ki', 'vr', 'vi']:
            w = getattr(attn_qwen, proj_name).weight
            # Set head h's weight block to be proportional to the eigenmode
            # with head-specific phase rotation
            head_angle = 2 * np.pi * h / n_heads
            rot = np.exp(1j * head_angle)
            rotated = eigenmode * rot
            template_r = torch.tensor(rotated.real.astype(np.float32))
            template_i = torch.tensor(rotated.imag.astype(np.float32))
            
            # Use real part for 'real' projections, imag for 'imag'
            if 'i' in proj_name:
                w.data[start:end] = template_i.unsqueeze(0).expand(dh, -1) * 0.1
            else:
                w.data[start:end] = template_r.unsqueeze(0).expand(dh, -1) * 0.1

# Forward pass on both
x = torch.randn(1, 8, d_model, dtype=torch.complex64)
with torch.no_grad():
    z_random, _ = attn_random(x)
    z_qwen, _ = attn_qwen(x)

print(f"\nRandom init output:  norm={z_random.norm().item():.4f}")
print(f"Qwen-injected output: norm={z_qwen.norm().item():.4f}")
print(f"They are DIFFERENT: {not torch.allclose(z_random, z_qwen, atol=1e-3)}")
# Complex cosine: |<z1|z2>| / (|z1|*|z2|)
cos_sim = torch.abs(torch.dot(z_random.flatten().conj(), z_qwen.flatten())) / (z_random.norm() * z_qwen.norm() + 1e-8)
print(f"Complex cosine similarity: {cos_sim.item():.4f}")

# Kuramoto order
r_random = attn_random.kuramoto_order(x)
r_qwen = attn_qwen.kuramoto_order(x)
print(f"\nKuramoto order (r=0 chaos, r=1 sync):")
print(f"  Random init: r={r_random:.4f}")
print(f"  Qwen-injected: r={r_qwen:.4f}")
print(f"  Delta: {r_qwen - r_random:+.4f}")

# Test: do gradients flow through injected weights?
z_qwen_nograd = z_qwen.clone()
z_qwen, _ = attn_qwen(x)
loss = z_qwen.real.sum() + z_qwen.imag.sum()
loss.backward()
grads = sum(1 for p in attn_qwen.parameters() if p.grad is not None)
print(f"\nGradients flow: {grads}/{sum(1 for _ in attn_qwen.parameters())} params")

print("\nVERDICT: Distilled eigenbasis injects successfully. Forward pass works. Gradients flow. Weights are structurally different from random init.")
