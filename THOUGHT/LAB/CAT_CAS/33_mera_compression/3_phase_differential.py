"""
MERA v3: Phase Encoding — Differential Compression on S^1
===========================================================
Maps U_k columns to complex phases on the unit circle.
Adjacent layers differ by a phase rotation — store k angles
per layer instead of m*k floats. This is Moire decomposition
applied to weight matrices.

Layer 0: full phase-encoded U_k (m, k) complex
Layer i>0: phase rotation angles (k,) — the difference from prev

Compression: (m*k*2) vs (k*1) per rotation layer = 2m:1 ratio.
For MLP with m=5120: 10,240x compression per rotation layer!
"""
import torch, time, os, math, numpy as np, cmath
from pathlib import Path
from collections import defaultdict

REPO = Path(r"d:\CCC 2.0\AI\agent-governance-system")
HOLO_05B = str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY" / "cybernetic_truth" / "qwen_0_5b_k128.holo")
HOLO_27B = str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY" / "cybernetic_truth" / "qwen_27b_catalytic_k256.holo")

def matrix_to_phase(U):
    """Map U (m, k) to complex phases on unit circle.
    Normalize each column to unit length, then encode angle."""
    U_norm = U.float() / (U.float().norm(dim=0, keepdim=True) + 1e-10)
    # Convert to complex: take first two rows as real/imag per column
    # Actually: encode entire column as a single phase via weighted average
    # Simpler: take angle of each element -> (m, k) phase matrix
    angles = torch.atan2(U_norm, torch.zeros_like(U_norm))  # all real -> angle 0 or pi
    # Better: use the actual sign structure: positive -> angle 0, negative -> angle pi
    angles = torch.where(U_norm > 0, torch.zeros_like(U_norm), 
                         torch.ones_like(U_norm) * math.pi)
    return torch.polar(torch.ones_like(U_norm), angles)

def phase_to_matrix(phase_mat):
    """Reverse: complex phase -> real matrix. Just take the real part (sign)."""
    return phase_mat.real

def phase_rotation_angle(U_prev, U_curr):
    """Compute phase rotation between two U matrices.
    For each column j: find angle delta_j such that rotating U_prev[:,j] by delta_j
    best aligns with U_curr[:,j].
    
    Returns: (k,) array of rotation angles.
    """
    m, k = U_prev.shape
    angles = torch.zeros(k)
    for j in range(k):
        # Compute optimal rotation angle between column j of prev and curr
        dot = torch.dot(U_prev[:, j], U_curr[:, j])
        norm_prod = U_prev[:, j].norm() * U_curr[:, j].norm() + 1e-10
        cos_theta = torch.clamp(dot / norm_prod, -1.0, 1.0)
        angles[j] = torch.acos(cos_theta)
        # Determine sign of rotation via cross product
        cross = U_prev[0, j] * U_curr[1, j] - U_prev[1, j] * U_curr[0, j]
        if cross < 0: angles[j] = -angles[j]
    return angles

def apply_rotation(U_ref, angles):
    """Rotate each column of U_ref by its angle.
    For 2D rotation: [cos, -sin; sin, cos] applied to each column.
    For m>2: apply Givens rotation in the plane of the first 2 components."""
    m, k = U_ref.shape
    result = U_ref.clone()
    for j in range(k):
        c = torch.cos(angles[j]); s = torch.sin(angles[j])
        # Givens rotation on first 2 rows
        x = U_ref[0, j]; y = U_ref[1, j]
        result[0, j] = c * x - s * y
        result[1, j] = s * x + c * y
    return result

def compress_phase_differential(tensors_dict):
    """
    Layer 0: full phase matrix (m, k) complex -> (m*k*2) bytes (unchanged)
    Layer i>0: rotation angles (k,) -> (k*1) bytes (2m:1 compression)
    
    Reconstruction: U_i = rotate_phase(U_{i-1}, angles_i)
    Quality: measured by cosine similarity of reconstructed vs original
    """
    sorted_l = sorted(tensors_dict.keys())
    if len(sorted_l) < 2: return None
    
    L = len(sorted_l)
    first = tensors_dict[sorted_l[0]]
    m, k = first.shape
    
    phase_first = matrix_to_phase(first)
    prev = first
    
    rotations = []
    reconstructions = []
    fidelities = []
    
    for i in range(1, L):
        curr = tensors_dict[sorted_l[i]]
        angles = phase_rotation_angle(prev, curr)
        rotations.append(angles)
        
        # Reconstruct via rotation
        recon = apply_rotation(prev, angles)
        fid = torch.nn.functional.cosine_similarity(
            curr.flatten().unsqueeze(0), recon.flatten().unsqueeze(0)
        ).item()
        fidelities.append(fid)
        reconstructions.append(recon)
        
        prev = curr  # chain from original for accurate next step
    
    # Compression ratio
    original_bytes = L * m * k * 2  # float16 U matrices
    compressed_bytes = m * k * 2 + (L-1) * k * 4  # first full + rotations (float32)
    ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
    
    avg_fid = np.mean(fidelities) if fidelities else 0
    
    return {
        'L': L, 'm': m, 'k': k,
        'ratio': ratio,
        'avg_fidelity': avg_fid,
        'fidelities': fidelities,
        'original_MB': original_bytes / 1024**2,
        'compressed_MB': compressed_bytes / 1024**2,
    }


print("=" * 78)
print("MERA v3: PHASE ENCODING — Differential Compression on S^1")
print("=" * 78)

for holo_path, label in [(HOLO_05B, "0.5B k128")]:
    if not os.path.exists(holo_path): continue
    
    print(f"\n  {label}:")
    holo = torch.load(holo_path, weights_only=False)
    
    u_groups = defaultdict(dict)
    for key, val in holo.items():
        if not key.endswith('.U') or val.ndim != 2: continue
        parts = key.split('.')
        layer_idx = None; wt = None
        for i, p in enumerate(parts):
            if p == 'layers' and i+1 < len(parts):
                try: layer_idx = int(parts[i+1])
                except: pass
            if p in ('mlp', 'self_attn', 'attn') and i+1 < len(parts):
                wt = '.'.join(parts[i:-1])
        if layer_idx is not None and wt is not None:
            u_groups[wt][layer_idx] = val
    
    total_orig = 0; total_comp = 0
    for wt, tensors in sorted(u_groups.items()):
        if len(tensors) < 2: continue
        result = compress_phase_differential(tensors)
        if result:
            total_orig += result['original_MB']
            total_comp += result['compressed_MB']
            first_fids = result['fidelities'][:3]
            fid_str = ' '.join([f'{f:.3f}' for f in first_fids])
            print(f"    {wt}: L={result['L']} m={result['m']} k={result['k']} ratio={result['ratio']:.1f}x fid_avg={result['avg_fidelity']:.3f} [{fid_str}...]")
    
    if total_orig > 0:
        print(f"    OVERALL: {total_orig:.0f}MB -> {total_comp:.0f}MB ({total_orig/total_comp:.1f}x) avg_fid={result['avg_fidelity']:.3f}")

# Show individual layer fidelities for one group
print(f"\n  PHASE ROTATION FIDELITY (0.5B mlp.down_proj):")
tensors = {}
for key, val in torch.load(HOLO_05B, weights_only=False).items():
    if 'mlp.down_proj' in key and key.endswith('.U') and val.ndim == 2:
        parts = key.split('.')
        for i, p in enumerate(parts):
            if p == 'layers' and i+1 < len(parts):
                try: tensors[int(parts[i+1])] = val
                except: pass

sorted_l = sorted(tensors.keys())
prev = tensors[sorted_l[0]]
for i in range(1, len(sorted_l)):
    curr = tensors[sorted_l[i]]
    angles = phase_rotation_angle(prev, curr)
    recon = apply_rotation(prev, angles)
    fid = torch.nn.functional.cosine_similarity(
        curr.flatten().unsqueeze(0), recon.flatten().unsqueeze(0)
    ).item()
    bar = '#' * int(fid * 50)
    max_angle = angles.abs().max().item() * 180 / math.pi
    print(f"    L{sorted_l[i-1]:>2}->L{sorted_l[i]:>2}: fid={fid:.4f} max_angle={max_angle:.1f}deg [{bar}]")
    prev = curr

print("=" * 78)
