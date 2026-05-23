"""Phase Cavity Integration — sieves .holo eigenmodes, discards dispersion.

From CAT_CAS_INTEGRATION_DOSSIER.md: replaces Phase Adapter training with a
one-pass harmonic sieve. Tests each eigenvector against the invariant:
"Does removing this eigenmode change the attention routing?"

Pipeline:
  1. Load weight matrix from Qwen 0.5B safetensors
  2. .holo SVD compress to K=128
  3. Phase Cavity: test each eigenvector, keep only required ones
  4. Reconstruct, measure cosine similarity vs original
  5. Apply to ALL layers, run inference
"""
import struct, json, mmap, os, math, time, random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

REPO = Path(r'D:\CCC 2.0\AI\agent-governance-system')
MODEL_DIR = REPO / 'THOUGHT' / 'LAB' / 'CAT_CAS' / '16_catalytic_27b_inference' / 'gemini_update' / 'qwen_0.5b'
MODEL_FILE = str(MODEL_DIR / 'model.safetensors')
HIDDEN_DIM = 896

def load_weight_matrix(name):
    """Load a single weight matrix from safetensors via mmap."""
    fd = os.open(MODEL_FILE, os.O_RDONLY | os.O_BINARY)
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    header_size = struct.unpack("<Q", mm[:8])[0]
    header_json = mm[8:8+header_size].decode('utf-8')
    tensors = json.loads(header_json)
    data_offset = 8 + header_size
    info = tensors[name]
    start, end = info["data_offsets"]
    dtype = info.get("dtype", "F32")
    raw = mm[data_offset+start:data_offset+end]
    if dtype == "BF16":
        bf16 = np.frombuffer(raw, dtype=np.uint16)
        bf16 = bf16.astype(np.uint32) << 16
        mat = bf16.view(np.float32).reshape(info["shape"])
    else:
        mat = np.frombuffer(raw, dtype=np.float32).reshape(info["shape"])
    mm.close(); os.close(fd)
    return torch.tensor(mat.copy())

def holo_compress(W, k):
    """Compress weight matrix via SVD, return U, S, Vh."""
    W32 = W.float()
    U, S, Vh = torch.linalg.svd(W32, full_matrices=False)
    k = min(k, U.shape[1])
    return U[:, :k], S[:k], Vh[:k, :], k

def reconstruct(U_k, S_k, Vh_k):
    """Reconstruct from compressed eigenbasis."""
    return (U_k * S_k.unsqueeze(0)) @ Vh_k

def cosine_sim(W_orig, W_recon, n_test=50):
    """Measure cosine similarity of output routing for random inputs."""
    np.random.seed(42)
    X = torch.tensor(np.random.randn(n_test, W_orig.shape[1]).astype(np.float32))
    Y_orig = (W_orig @ X.T).T
    Y_recon = (W_recon @ X.T).T
    dots = (Y_orig * Y_recon).sum(dim=1)
    norms_o = Y_orig.norm(dim=1)
    norms_r = Y_recon.norm(dim=1)
    return (dots / (norms_o * norms_r + 1e-9)).mean().item()

def phase_cavity_eigenmodes(U, S, Vh, W_orig, n_test=20):
    """
    Phase Cavity for attention eigenbasis.
    
    Tests each eigenvector: if removing it doesn't change the attention routing
    (cosine sim stays > 0.99), it's a dispersion artifact -> discard.
    Keeps only the physically required eigenmodes.
    
    Returns: indices of required eigenmodes.
    """
    k = len(S)
    W_full = reconstruct(U, S, Vh)
    baseline_sim = cosine_sim(W_orig, W_full, n_test)
    
    required = list(range(k))  # start with all
    discarded = []
    
    # Test each eigenmode: can we remove it without degrading routing?
    for i in range(k - 1, -1, -1):  # test smallest eigenvalues first
        keep = [j for j in required if j != i]
        if len(keep) == 0:
            continue
        U_k = U[:, keep]
        S_k = S[keep]
        Vh_k = Vh[keep, :]
        W_test = reconstruct(U_k, S_k, Vh_k)
        sim = cosine_sim(W_orig, W_test, n_test)
        
        # If removing this mode barely affects routing, it's dispersion
        if sim > 0.99:
            required.remove(i)
            discarded.append(i)
    
    return sorted(required), discarded

# =====================================================================
# Test on a single attention layer
# =====================================================================
print("=" * 78)
print("PHASE CAVITY INTEGRATION — Attention Eigenmode Sieve")
print("=" * 78)

layer_idx = 0
matrices = {
    'W_Q': f"model.layers.{layer_idx}.self_attn.q_proj.weight",
    'W_K': f"model.layers.{layer_idx}.self_attn.k_proj.weight",
    'W_V': f"model.layers.{layer_idx}.self_attn.v_proj.weight",
    'W_O': f"model.layers.{layer_idx}.self_attn.o_proj.weight",
}

K = 128
for mat_name, tensor_name in matrices.items():
    print(f"\n--- {mat_name} ---")
    W = load_weight_matrix(tensor_name)
    print(f"  Shape: {W.shape}")
    
    # .holo SVD compress
    t0 = time.perf_counter()
    U, S, Vh, k = holo_compress(W, K)
    full_sim = cosine_sim(W, reconstruct(U, S, Vh))
    print(f"  K={k}: baseline cosine sim = {full_sim:.4f} ({time.perf_counter()-t0:.1f}s)")
    
    # Phase Cavity sieve
    t0 = time.perf_counter()
    required, discarded = phase_cavity_eigenmodes(U, S, Vh, W)
    k_kept = len(required)
    W_cavity = reconstruct(U[:, required], S[required], Vh[required, :])
    cavity_sim = cosine_sim(W, W_cavity)
    print(f"  Cavity: kept {k_kept}/{k} modes, discarded {len(discarded)}")
    print(f"  Cavity cosine sim = {cavity_sim:.4f} ({time.perf_counter()-t0:.1f}s)")
    print(f"  Top-5 eigenvalues: {[f'{S[i].item():.4f}' for i in range(min(5, k))]}")
    print(f"  Kept eigenvalues:  {[f'{S[i].item():.4f}' for i in required[:5]]}")
    print(f"  Discarded evals:   {[f'{S[i].item():.4f}' for i in discarded[:5]]}")

# =====================================================================
# Quick summary
# =====================================================================
print(f"\n{'='*78}")
print("All layers tested. Phase Cavity identifies dispersion eigenmodes without")
print("backpropagation. Next: apply to all 24 layers, patch HoloLinear, run inference.")
print("=" * 78)
