#!/usr/bin/env python3
"""
task3_diagnostics.py -- Three zero-training tests for the 85x KV cache barrier

Test 1: Asymmetric compression -- k_K != k_V since K has Df~8, V has Df~42
Test 2: Per-head PCA -- compress each head's 64-dim signal independently
Test 3: V-dimension Q-gradient diagnostic -- which V dims actually affect attention?

All tests are zero-training, using only PCA projections and linear algebra.
Outputs to diagnostics_results.json for analysis.

Usage:
    python task3_diagnostics.py
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Utility functions (same as flat_llm_adapter.py)
# ============================================================================

def compute_cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    cos = F.cosine_similarity(a_flat, b_flat, dim=-1)
    return cos.mean().item()


def compute_attention_output(q, k, v, num_heads, scale):
    batch, seq, hidden = q.shape
    head_dim = hidden // num_heads
    q_r = q.view(batch, -1, num_heads, head_dim).transpose(1, 2)
    k_r = k.view(batch, -1, num_heads, head_dim).transpose(1, 2)
    v_r = v.view(batch, -1, num_heads, head_dim).transpose(1, 2)
    attn = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale
    causal = torch.triu(torch.ones(seq, seq, device=q.device) * float('-inf'), diagonal=1)
    attn = attn + causal
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v_r)
    return out.transpose(1, 2).contiguous().view(batch, -1, hidden)


@torch.no_grad()
def collect_all_activations(model, tokenizer, texts: List[str], device: str):
    """Collect Q, K, V from every layer. Returns dict[layer_idx] -> {q,k,v}."""
    model.eval()
    n_layers = model.config.n_layer
    hidden_dim = model.config.n_embd
    layer_data = {i: {'q': [], 'k': [], 'v': []} for i in range(n_layers)}

    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        seq_len = inputs['input_ids'].shape[1]

        hidden = model.transformer.wte(inputs['input_ids']) + model.transformer.wpe(
            torch.arange(seq_len, device=device)
        )
        hidden = model.transformer.drop(hidden)

        for idx in range(n_layers):
            block = model.transformer.h[idx]
            normed = block.ln_1(hidden)
            qkv = block.attn.c_attn(normed)
            q, k, v = qkv.chunk(3, dim=-1)
            layer_data[idx]['q'].append(q.reshape(-1, hidden_dim))
            layer_data[idx]['k'].append(k.reshape(-1, hidden_dim))
            layer_data[idx]['v'].append(v.reshape(-1, hidden_dim))
            hidden = block(hidden)[0]

    result = {}
    for idx in range(n_layers):
        q_t = torch.cat(layer_data[idx]['q'], dim=0)
        k_t = torch.cat(layer_data[idx]['k'], dim=0)
        v_t = torch.cat(layer_data[idx]['v'], dim=0)
        result[idx] = {'q': q_t, 'k': k_t, 'v': v_t}
    return result


@torch.no_grad()
def pca_project(data: torch.Tensor, k: int):
    """PCA projection: returns (compressed, decompressed, Vt[:k], mean)."""
    mean = data.mean(dim=0)
    centered = data - mean
    _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
    k_use = min(k, Vt.shape[0])
    basis = Vt[:k_use]
    compressed = (centered @ basis.T)
    decompressed = compressed @ basis + mean
    return compressed, decompressed, basis, mean


# ============================================================================
# Test 1: Asymmetric compression
# ============================================================================

@torch.no_grad()
def test_asymmetric(all_acts, model, tokenizer, test_texts, device, k_pairs):
    """Benchmark asymmetric (k_K, k_V) pairs across all layers."""
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads
    scale = 1.0 / math.sqrt(head_dim)
    n_layers = model.config.n_layer
    hidden_dim = model.config.n_embd

    results = {}
    for (k_k, k_v) in k_pairs:
        total = k_k + k_v
        cr = (2 * hidden_dim) / total
        print(f"\n  --- Asymmetric: k_K={k_k}, k_V={k_v} (total={total}, CR={cr:.1f}x) ---")
        layer_results = []

        for layer_idx in range(n_layers):
            q = all_acts[layer_idx]['q']
            k = all_acts[layer_idx]['k']
            v = all_acts[layer_idx]['v']

            # PCA on K and V
            _, _, basis_k, mean_k = pca_project(k, k_k)
            _, _, basis_v, mean_v = pca_project(v, k_v)

            # Test on per-text basis
            cos_sum = 0.0
            n_samples = 0
            for text in test_texts:
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
                inputs = {k_in: v_in.to(device) for k_in, v_in in inputs.items()}
                seq_len = inputs['input_ids'].shape[1]

                hidden = model.transformer.wte(inputs['input_ids']) + model.transformer.wpe(
                    torch.arange(seq_len, device=device))
                hidden = model.transformer.drop(hidden)
                for i in range(layer_idx):
                    hidden = model.transformer.h[i](hidden)[0]

                block = model.transformer.h[layer_idx]
                normed = block.ln_1(hidden)
                qkv = block.attn.c_attn(normed)
                q_t, k_t, v_t = qkv.chunk(3, dim=-1)
                orig_attn = compute_attention_output(q_t, k_t, v_t, num_heads, scale)

                k_comp = (k_t - mean_k) @ basis_k.T
                k_recon = k_comp @ basis_k + mean_k
                v_comp = (v_t - mean_v) @ basis_v.T
                v_recon = v_comp @ basis_v + mean_v

                recon_attn = compute_attention_output(q_t, k_recon, v_recon, num_heads, scale)
                cos_sum += compute_cosine_sim(orig_attn, recon_attn)
                n_samples += 1

            avg_cos = cos_sum / n_samples
            print(f"    L{layer_idx:2d} | Attn cos: {avg_cos:.4f} | k=({k_k}+{k_v})")
            layer_results.append({'layer': layer_idx, 'attn_cos': avg_cos})

        avg = np.mean([r['attn_cos'] for r in layer_results])
        results[f'K{k_k}_V{k_v}'] = {
            'total_dims': total, 'cr': cr,
            'layers': layer_results, 'avg_cos': float(avg)
        }
        print(f"    AVERAGE: {avg:.4f}")

    return results


# ============================================================================
# Test 2: Per-head PCA
# ============================================================================

@torch.no_grad()
def test_per_head_pca(all_acts, model, tokenizer, test_texts, device, k_per_head_vals):
    """PCA on each head's 64-dim K and V independently."""
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads
    scale = 1.0 / math.sqrt(head_dim)
    n_layers = model.config.n_layer
    hidden_dim = model.config.n_embd

    results = {}
    for k_h in k_per_head_vals:
        total = num_heads * k_h * 2
        cr = (2 * hidden_dim) / total
        print(f"\n  --- Per-head PCA: k={k_h}/head (total={total}, CR={cr:.1f}x) ---")
        layer_results = []

        for layer_idx in range(n_layers):
            q_full = all_acts[layer_idx]['q']
            k_full = all_acts[layer_idx]['k']
            v_full = all_acts[layer_idx]['v']

            # Compute per-head PCA bases
            k_bases = []  # list of (basis, mean) per head
            v_bases = []
            for h in range(num_heads):
                k_h_data = k_full[:, h * head_dim:(h + 1) * head_dim]
                v_h_data = v_full[:, h * head_dim:(h + 1) * head_dim]
                _, _, bk, mk = pca_project(k_h_data, k_h)
                _, _, bv, mv = pca_project(v_h_data, k_h)
                k_bases.append((bk, mk))
                v_bases.append((bv, mv))

            cos_sum = 0.0
            n_samples = 0
            for text in test_texts:
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
                inputs = {k_in: v_in.to(device) for k_in, v_in in inputs.items()}
                seq_len = inputs['input_ids'].shape[1]

                hidden = model.transformer.wte(inputs['input_ids']) + model.transformer.wpe(
                    torch.arange(seq_len, device=device))
                hidden = model.transformer.drop(hidden)
                for i in range(layer_idx):
                    hidden = model.transformer.h[i](hidden)[0]

                block = model.transformer.h[layer_idx]
                normed = block.ln_1(hidden)
                qkv = block.attn.c_attn(normed)
                q_t, k_t, v_t = qkv.chunk(3, dim=-1)
                orig_attn = compute_attention_output(q_t, k_t, v_t, num_heads, scale)

                # Compress/reconstruct per head
                k_recon_parts = []
                v_recon_parts = []
                for h in range(num_heads):
                    k_h_slice = k_t[:, :, h * head_dim:(h + 1) * head_dim]
                    v_h_slice = v_t[:, :, h * head_dim:(h + 1) * head_dim]
                    bk, mk = k_bases[h]
                    bv, mv = v_bases[h]
                    k_h_comp = (k_h_slice - mk) @ bk.T
                    k_h_recon = k_h_comp @ bk + mk
                    v_h_comp = (v_h_slice - mv) @ bv.T
                    v_h_recon = v_h_comp @ bv + mv
                    k_recon_parts.append(k_h_recon)
                    v_recon_parts.append(v_h_recon)
                k_recon = torch.cat(k_recon_parts, dim=-1)
                v_recon = torch.cat(v_recon_parts, dim=-1)

                recon_attn = compute_attention_output(q_t, k_recon, v_recon, num_heads, scale)
                cos_sum += compute_cosine_sim(orig_attn, recon_attn)
                n_samples += 1

            avg_cos = cos_sum / n_samples
            print(f"    L{layer_idx:2d} | Attn cos: {avg_cos:.4f} | k_h={k_h}")
            layer_results.append({'layer': layer_idx, 'attn_cos': avg_cos})

        avg = np.mean([r['attn_cos'] for r in layer_results])
        results[f'perhead_k{k_h}'] = {
            'total_dims': total, 'cr': cr,
            'layers': layer_results, 'avg_cos': float(avg)
        }
        print(f"    AVERAGE: {avg:.4f}")

    return results


# ============================================================================
# Test 3: V-dimension Q-gradient diagnostic
# ============================================================================

@torch.no_grad()
def test_v_dimension_sensitivity(all_acts, model, tokenizer, test_texts, device):
    """Measure which V dimensions actually affect attention output.

    Method: attention output = A @ V where A = softmax(QK^T/sqrt(d)).
    The contribution of V dimension d is ||A @ V[:,d]||^2.
    We compute this per layer and rank dimensions by contribution.
    """
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads
    scale = 1.0 / math.sqrt(head_dim)
    n_layers = model.config.n_layer

    print(f"\n  --- V-dimension Q-gradient diagnostic ---")
    print(f"  Computing per-dimension attention contribution for all layers...")

    results = {}
    for layer_idx in range(n_layers):
        q = all_acts[layer_idx]['q']
        k = all_acts[layer_idx]['k']
        v = all_acts[layer_idx]['v']

        # Compute on a representative single text for speed
        text = test_texts[0]
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        inputs = {k_in: v_in.to(device) for k_in, v_in in inputs.items()}
        seq_len = inputs['input_ids'].shape[1]

        hidden = model.transformer.wte(inputs['input_ids']) + model.transformer.wpe(
            torch.arange(seq_len, device=device))
        hidden = model.transformer.drop(hidden)
        for i in range(layer_idx):
            hidden = model.transformer.h[i](hidden)[0]

        block = model.transformer.h[layer_idx]
        normed = block.ln_1(hidden)
        qkv = block.attn.c_attn(normed)
        q_t, k_t, v_t = qkv.chunk(3, dim=-1)

        # Compute attention weights A
        batch, seq, hidden = q_t.shape
        q_r = q_t.view(batch, -1, num_heads, head_dim).transpose(1, 2)
        k_r = k_t.view(batch, -1, num_heads, head_dim).transpose(1, 2)
        v_r = v_t.view(batch, -1, num_heads, head_dim).transpose(1, 2)
        attn = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale
        causal = torch.triu(torch.ones(seq, seq, device=device) * float('-inf'), diagonal=1)
        attn = attn + causal
        attn = F.softmax(attn, dim=-1)  # (batch, heads, seq, seq)

        # Per-dimension contribution to attention output
        # V shape: (1, seq, 768). Reshape to (1, seq, 12, 64) for per-head
        v_r_reshaped = v_t.view(batch, -1, num_heads, head_dim)  # (1, seq, 12, 64)

        # For each of 768 flattened dims, compute contribution norm
        dim_contributions = torch.zeros(hidden, device=device)
        for d in range(hidden):
            h_idx = d // head_dim
            d_idx = d % head_dim
            # Contribution of this (head, dim) pair
            # For head h_idx, position d_idx: attn @ v[:, h_idx, d_idx]
            contrib = torch.matmul(
                attn[:, h_idx:h_idx+1, :, :],  # (1, 1, seq, seq)
                v_r_reshaped[:, :, h_idx:h_idx+1, d_idx:d_idx+1].transpose(1, 2)
            )  # (1, 1, seq, 1)
            dim_contributions[d] = contrib.norm()

        # Normalize to percentages
        total = dim_contributions.sum()
        pct = dim_contributions / (total + 1e-10)

        # Sort by contribution
        sorted_vals, sorted_idx = torch.sort(dim_contributions, descending=True)
        sorted_pct = sorted_vals / (total + 1e-10)

        # Cumulative
        cum_pct = torch.cumsum(sorted_pct, dim=0)

        # Find how many dims needed for 50%, 80%, 90%, 95% of contribution
        thresholds = [0.50, 0.80, 0.90, 0.95]
        dims_needed = {}
        for t in thresholds:
            n = int((cum_pct >= t).nonzero()[0].item()) + 1
            dims_needed[f'dims_for_{int(t*100)}pct'] = n

        # Per-head breakdown
        head_contribs = []
        for h in range(num_heads):
            h_dims = dim_contributions[h * head_dim:(h + 1) * head_dim]
            h_total = h_dims.sum()
            head_contribs.append({
                'head': h, 'total': float(h_total),
                'pct': float(h_total / (total + 1e-10)),
                'top5_dims_cum_pct': float(
                    torch.topk(h_dims, min(5, head_dim)).values.sum() / (h_total + 1e-10)
                )
            })

        layer_result = {
            'layer': layer_idx,
            'total_contrib': float(total),
            'dims_needed': dims_needed,
            'top10_dim_pct': float(sorted_pct[:10].sum()),
            'head_contribs': head_contribs,
        }

        results[f'L{layer_idx}'] = layer_result
        print(f"    L{layer_idx:2d}: top10 dims={float(sorted_pct[:10].sum())*100:.1f}%  "
              f"dims@80%={dims_needed['dims_for_80pct']}  "
              f"dims@95%={dims_needed['dims_for_95pct']}")

    # Summary across layers
    avg_dims_80 = np.mean([r['dims_needed']['dims_for_80pct'] for r in results.values()])
    avg_dims_95 = np.mean([r['dims_needed']['dims_for_95pct'] for r in results.values()])
    avg_top10 = np.mean([r['top10_dim_pct'] for r in results.values()])

    summary = {
        'avg_dims_needed_80pct': float(avg_dims_80),
        'avg_dims_needed_95pct': float(avg_dims_95),
        'avg_top10_dim_contribution_pct': float(avg_top10),
        'layers': results,
    }
    print(f"\n    CROSS-LAYER AVERAGE:")
    print(f"      Top 10 V dims contribute: {avg_top10*100:.1f}% of attention output")
    print(f"      Dims needed for 80%: {avg_dims_80:.0f}")
    print(f"      Dims needed for 95%: {avg_dims_95:.0f}")
    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("  TASK 3 DIAGNOSTICS: Three zero-training tests for 85x barrier")
    print(f"  Device: {device}")
    print("=" * 72)

    # Load model
    print("\n[1/5] Loading GPT-2...")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    n_layers = model.config.n_layer
    print(f"      Layers: {n_layers}, Hidden: {model.config.n_embd}, Heads: {model.config.n_head}")

    test_texts = [
        "The meaning of life is a philosophical question that has puzzled humanity for centuries.",
        "Artificial intelligence is transforming the way we interact with technology every day.",
        "Deep learning enables complex pattern recognition in vast amounts of data.",
        "The human brain contains approximately eighty-six billion neurons connected in intricate networks.",
        "Climate change poses significant challenges for future generations across the globe.",
        "In mathematics, prime numbers have fascinated researchers for thousands of years.",
        "The ocean covers more than seventy percent of Earth's surface and remains largely unexplored.",
        "Music has the power to evoke strong emotional responses and connect people across cultures.",
        "Space exploration has led to many technological breakthroughs that benefit life on Earth.",
        "Language is the foundation of human communication and the source of our collective knowledge.",
        "Economic systems attempt to explain how resources are allocated in complex societies.",
        "The history of science is a story of ideas evolving through observation and experimentation.",
    ]

    print("\n[2/5] Collecting Q,K,V activations from all layers...")
    t0 = time.time()
    all_acts = collect_all_activations(model, tokenizer, test_texts, device)
    print(f"      Done in {time.time()-t0:.1f}s  |  {len(all_acts)} layers collected")

    all_results = {}

    # ============ Test 1: Asymmetric ============
    print("\n[3/5] TEST 1: Asymmetric compression (k_K != k_V)")
    print("=" * 72)
    symmetric_baseline = [(9, 9), (12, 12)]
    asymmetric_pairs = [(4, 14), (8, 36)]
    # Also test some balanced points for reference
    more_pairs = [(6, 12), (3, 15), (5, 25)]

    k_pairs = symmetric_baseline + asymmetric_pairs + more_pairs
    asym_results = test_asymmetric(all_acts, model, tokenizer, test_texts, device, k_pairs)
    all_results['asymmetric'] = {k: {'avg_cos': v['avg_cos'], 'cr': v['cr'],
                                      'total_dims': v['total_dims']} for k, v in asym_results.items()}

    # ============ Test 2: Per-head PCA ============
    print("\n[4/5] TEST 2: Per-head PCA")
    print("=" * 72)
    k_per_head_vals = [1, 2, 3, 4, 6]
    ph_results = test_per_head_pca(all_acts, model, tokenizer, test_texts, device, k_per_head_vals)
    all_results['per_head'] = {k: {'avg_cos': v['avg_cos'], 'cr': v['cr'],
                                    'total_dims': v['total_dims']} for k, v in ph_results.items()}

    # ============ Test 3: Q-gradient ============
    print("\n[5/5] TEST 3: V-dimension Q-gradient diagnostic")
    print("=" * 72)
    qg_results = test_v_dimension_sensitivity(all_acts, model, tokenizer, test_texts, device)
    all_results['q_gradient'] = qg_results

    # ============ Summary ============
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)

    # Asymmetric table
    print("\n  Asymmetric compression (k_K, k_V):")
    print(f"  {'Config':<16} {'CR':>6} {'Avg Attn Cos':>14} {'vs sym k=9':>12}")
    print(f"  {'-' * 48}")
    sym9_cos = asym_results.get('K9_V9', {}).get('avg_cos', 0.6830)
    for key in sorted(asym_results.keys(), key=lambda k: asym_results[k]['cr'], reverse=True):
        r = asym_results[key]
        delta = (r['avg_cos'] - sym9_cos) / (sym9_cos + 1e-10) * 100
        print(f"  {key:<16} {r['cr']:>6.1f}x {r['avg_cos']:>14.4f} {delta:>+11.1f}%")

    # Per-head table
    print(f"\n  Per-head PCA:")
    print(f"  {'Config':<16} {'CR':>6} {'Avg Attn Cos':>14} {'vs full k=9':>12}")
    print(f"  {'-' * 48}")
    for key in sorted(ph_results.keys(), key=lambda k: ph_results[k]['cr'], reverse=True):
        r = ph_results[key]
        delta = (r['avg_cos'] - sym9_cos) / (sym9_cos + 1e-10) * 100
        print(f"  {key:<16} {r['cr']:>6.1f}x {r['avg_cos']:>14.4f} {delta:>+11.1f}%")

    # Q-gradient summary
    print(f"\n  V-dimension sensitivity:")
    print(f"    Top 10 dimensions: {qg_results['avg_top10_dim_contribution_pct']*100:.1f}% of output")
    print(f"    Dims for 80%: {qg_results['avg_dims_needed_80pct']:.0f} (of 768)")
    print(f"    Dims for 95%: {qg_results['avg_dims_needed_95pct']:.0f} (of 768)")
    print(f"    Implication: {'Q-aware projection could help' if qg_results['avg_dims_needed_80pct'] < 50 else 'V is genuinely high-D -- need nonlinear approach'}")

    # Save
    out_dir = Path(__file__).parent
    out_path = out_dir / "diagnostics_results.json"
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return super().default(obj)

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")
    print("Done.")

    return all_results


if __name__ == '__main__':
    main()
