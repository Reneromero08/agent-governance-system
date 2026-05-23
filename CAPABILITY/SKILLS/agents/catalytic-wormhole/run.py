#!/usr/bin/env python3
"""
Catalytic Wormhole Compressor — Integrated Lab Pipeline
========================================================
run.py: Executes the full 5-stage compression pipeline.

Stages:
  1. Catalytic Eigenbasis Extraction (SVD + cached Vh projection)
  2. Rotation Chain Construction (R = U_i^T @ U_{i+1})
  3. Boundary Stress Decomposition (signal vs noise modes)
  4. Phase Cavity Sieve (optimal rank per weight type)
  5. LoRA Compression (store sieved R as A * B pair)

Input: JSON config with model path or .holo path
Output: Compressed .holo file + stats.json
"""
import json, os, sys, time, math, struct, hashlib
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "THOUGHT" / "LAB" / "HOLO" / "_models"
K = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD_STEPS = [0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01]
FID_TOLERANCE = 0.001

# ---- Safetensors I/O ----
class SafetensorsLoader:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        with open(self.model_dir / "model.safetensors.index.json") as f:
            self.idx = json.load(f)
        self.header_cache = {}
    
    def get_header(self, shard_name):
        if shard_name in self.header_cache:
            return self.header_cache[shard_name]
        path = self.model_dir / shard_name
        with open(path, 'rb') as f:
            hl = struct.unpack('<Q', f.read(8))[0]
            self.header_cache[shard_name] = json.loads(f.read(hl))
        return self.header_cache[shard_name]
    
    def load_int8_weight(self, shard, key):
        h = self.get_header(shard); info = h[key]
        start, end = info['data_offsets']
        shape = info['shape']
        path = self.model_dir / shard
        with open(path, 'rb') as f:
            hl = struct.unpack('<Q', f.read(8))[0]
            f.seek(8 + hl + start)
            data = f.read(end - start)
        return torch.from_numpy(
            np.frombuffer(data, dtype=np.int8).astype(np.float32).copy()
        ).reshape(shape)

# ---- Core Algorithms ----
def cavity_sieve_R(R, threshold):
    """Exp 30: Decompose R into signal (S > threshold) and noise."""
    U, S, Vh = np.linalg.svd(R.numpy() if hasattr(R, 'numpy') else R, full_matrices=False)
    mask = S > threshold
    R_sig = (U[:, mask] * S[mask]) @ Vh[mask, :]
    return R_sig, mask.sum()

def spectral_signature(matrix_np):
    """Exp 31: D_pr, D_sh from covariance of R rows."""
    x = matrix_np.astype(np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    _, sv, _ = np.linalg.svd(centered, full_matrices=False)
    if x.shape[0] > 1:
        ev = (sv ** 2) / (x.shape[0] - 1)
    else:
        ev = sv ** 2
    ev = ev[ev > 1e-12]
    total = float(ev.sum())
    probs = ev / total
    D_pr = (total * total) / float((ev ** 2).sum())
    entropy = -float((probs * np.log(probs + 1e-12)).sum())
    D_sh = float(np.exp(entropy))
    return {'D_pr': D_pr, 'D_sh': D_sh, 'eigenvalues': ev}

def compute_chain_fidelity(Us, Rs_cavity, wt):
    """Propagate sieved rotation chain and measure fidelity vs final layer."""
    layers = sorted(Us[wt].keys())
    anchor = Us[wt][layers[0]].float()
    cum_R = torch.eye(K)
    for R_sig in Rs_cavity:
        cum_R = cum_R @ torch.from_numpy(R_sig).float()
    final = Us[wt][layers[-1]].float()
    recon = anchor @ cum_R
    return torch.nn.functional.cosine_similarity(
        final.flatten(), recon.flatten(), dim=0
    ).item()

# ---- Main Pipeline ----
def main(input_path: Path, output_path: Path, writer=None) -> int:
    payload = json.loads(input_path.read_text())
    
    print(f"Catalytic Wormhole Compressor — Integrated Lab Pipeline")
    print(f"Device: {DEVICE}, K={K}")
    t0 = time.perf_counter()
    
    # Determine input source
    model_dir = payload.get("model_dir")
    rank_k = payload.get("rank_k", K)
    
    if model_dir:
        # Load from safetensors
        loader = SafetensorsLoader(model_dir)
        source = "safetensors"
        print(f"Source: {model_dir}")
        
        # Find expert-0 weight keys
        expert0 = defaultdict(lambda: defaultdict(list))
        for k, shard in loader.idx["weight_map"].items():
            if "ffn.experts.0." in k and k.endswith(".weight"):
                parts = k.split(".")
                layer = None; wt = None
                for i, p in enumerate(parts):
                    if p == "layers":
                        try: layer = int(parts[i+1])
                        except: pass
                    if p == "experts":
                        for j in range(i+2, len(parts)):
                            if parts[j] in ("w1", "w2", "w3"):
                                wt = parts[j]; break
                        break
                if layer is not None and wt is not None:
                    expert0[wt][layer] = (k, shard)
        expert0 = dict(expert0)
        print(f"Expert 0: {sum(len(v) for v in expert0.values())} keys, {len(expert0)} weight types")
        
        # Stage 1: Catalytic SVD
        Us = {}
        svd_cache = {}
        for wt in sorted(expert0.keys()):
            Us[wt] = {}
            for layer in sorted(expert0[wt].keys()):
                key, shard = expert0[wt][layer]
                W = loader.load_int8_weight(shard, key).float().to(DEVICE)
                if wt not in svd_cache:
                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                    svd_cache[wt] = (U[:, :K].cpu(), (S[:K].unsqueeze(1)*Vh[:K,:]).cpu())
                    del U, S, Vh, W; torch.cuda.empty_cache()
                    Us[wt][layer] = svd_cache[wt][0]
                else:
                    _, Vh = svd_cache[wt]; Vh = Vh.to(DEVICE)
                    U_raw = W @ Vh.T; U, _ = torch.linalg.qr(U_raw)
                    Us[wt][layer] = U[:, :K].cpu()
                    del W, Vh, U_raw, U; torch.cuda.empty_cache()
        print(f"Stage 1: {sum(len(v) for v in Us.values())} U tensors computed")
    else:
        # Load from .holo
        holo_path = payload["input_path"]
        holo = torch.load(holo_path, weights_only=False, map_location='cpu')
        source = "holo"
        print(f"Source: {holo_path} ({len(holo)} keys)")
        Us = {}
        for key, val in holo.items():
            if not key.endswith('.U') or val.ndim != 2: continue
            parts = key.split(".")
            layer = None; wt = None
            for i, p in enumerate(parts):
                if p == "layers":
                    try: layer = int(parts[i+1])
                    except: pass
                if p == "experts" and i+1 < len(parts) and int(parts[i+1]) == 0:
                    for j in range(i+2, len(parts)):
                        if parts[j] in ("w1","w2","w3"): wt = parts[j]; break
            if layer is not None and wt is not None:
                Us.setdefault(wt, {})[layer] = val.float()
        del holo
    
    # Stage 2: Rotation chain
    Rs = {}
    for wt in sorted(Us.keys()):
        layers = sorted(Us[wt].keys())
        Rs[wt] = []
        for li in range(len(layers)-1):
            R = Us[wt][layers[li]].float().T @ Us[wt][layers[li+1]].float()
            Rs[wt].append(R)
    
    # Stage 3+4: Cavity Sieve + Optimal Threshold
    optimal = {}
    stats_out = {}
    
    for wt in sorted(Rs.keys()):
        # Find max chain fidelity
        max_fid = 0
        for thresh in [0.5, 0.1]:
            sieved = [cavity_sieve_R(R, thresh)[0] for R in Rs[wt]]
            max_fid = max(max_fid, compute_chain_fidelity(Us, sieved, wt))
        
        best = {'thresh': 0.5, 'r': K, 'fid': 0, 'mb': 0}
        for thresh in THRESHOLD_STEPS:
            modes = [cavity_sieve_R(R, thresh)[1] for R in Rs[wt]]
            avg_r = max(1, int(np.mean(modes)))
            sieved = [cavity_sieve_R(R, thresh)[0] for R in Rs[wt]]
            fid = compute_chain_fidelity(Us, sieved, wt)
            rot_mb = avg_r * K * 4 * len(Rs[wt]) / 1024**2
            anchor_mb = K * Us[wt][sorted(Us[wt].keys())[0]].shape[0] * 2 / 1024**2
            total_mb = anchor_mb + rot_mb
            
            if fid >= max_fid * (1 - FID_TOLERANCE) and avg_r < best['r']:
                best = {'thresh': thresh, 'r': avg_r, 'fid': fid, 'mb': total_mb}
        
        optimal[wt] = best
        
        # Stage 5: Build compressed
        layers = sorted(Us[wt].keys())
        compressed = {}
        compressed[f"{wt}.anchor"] = Us[wt][layers[0]].half()
        compressed[f"{wt}.layers"] = len(layers)
        
        for li in range(len(Rs[wt])):
            R_sig, _ = cavity_sieve_R(Rs[wt][li], best['thresh'])
            U_r, S_r, Vh_r = np.linalg.svd(R_sig, full_matrices=False)
            r = best['r']
            A = torch.from_numpy(U_r[:, :r] * np.sqrt(S_r[:r])).half()
            B = torch.from_numpy(np.sqrt(S_r[:r])[:, None] * Vh_r[:r, :]).half()
            compressed[f"{wt}.L{layers[li+1]}.R_A"] = A
            compressed[f"{wt}.L{layers[li+1]}.R_B"] = B
        
        # Verify
        sieved_final = [np.zeros((K,K)) for _ in range(len(Rs[wt]))]
        for li in range(len(Rs[wt])):
            sieved_final[li], _ = cavity_sieve_R(Rs[wt][li], best['thresh'])
        final_fid = compute_chain_fidelity(Us, sieved_final, wt)
        
        orig_mb = sum(u.numel() * 2 for u in Us[wt].values()) / 1024**2
        stats_out[wt] = {
            'layers': len(layers), 'rank': best['r'], 'fidelity': final_fid,
            'orig_mb': round(orig_mb, 1), 'comp_mb': round(best['mb'], 1),
            'ratio': round(orig_mb / max(best['mb'], 0.01), 1),
            'spectral_D_pr': spectral_signature(Rs[wt][0].numpy())['D_pr'],
        }
        
        if writer:
            output_name = f"THOUGHT/LAB/HOLO/_models/ds_{wt}_wormhole.holo"
            tmp = writer.write_tmp(output_name, "")
            torch.save(compressed, PROJECT_ROOT / output_name)
    
    # Summary
    total_orig = sum(s['orig_mb'] for s in stats_out.values())
    total_comp = sum(s['comp_mb'] for s in stats_out.values())
    
    result = {
        'source': source,
        'rank_k': K,
        'weight_types': len(Us),
        'total_layers': sum(len(v) for v in Us.values()),
        'total_orig_mb': round(total_orig, 1),
        'total_comp_mb': round(total_comp, 1),
        'ratio': round(total_orig / max(total_comp, 0.01), 1),
        'time_s': round(time.perf_counter() - t0, 1),
        'per_weight': stats_out,
    }
    
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nPipeline complete: {result['ratio']}x compression in {result['time_s']}s")
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
