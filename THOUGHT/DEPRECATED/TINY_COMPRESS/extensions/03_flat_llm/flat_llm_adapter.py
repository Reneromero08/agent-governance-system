#!/usr/bin/env python3
"""
flat_llm_adapter.py -- Low-Rank Adapters for 85x KV Cache Barrier

Tests whether low-rank adapters (inspired by FLAT-LLM fine-grained
low-rank transformations) can bridge the 2D manifold (Df ~ 1.8) to
the 768D computation space in EigenGPT2.

KEY FINDING (v2): Random adapters universally HURT quality. PCA-only
is the optimal linear reconstruction. Training is required.

FIXES (v2):
- Multi-layer testing (all 12 GPT-2 layers, averaged)
- Separate K and V adapters with correct residual subspaces
- Removed dead k=0 adapter initialization

Usage:
    python flat_llm_adapter.py benchmark
"""

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LowRankAdapter(nn.Module):
    """Low-rank residual adapter operating on compressed signal.

    Architecture:
        compressed (k) -> W1 (bottleneck, k) -> gelu -> W2 (hidden, bottleneck) -> correction
        output = decompress(compressed) + alpha * correction

    The adapter predicts what was lost during PCA truncation.
    Correction is restricted to the subspace orthogonal to the top-k PCA components
    to avoid duplicating PCA's reconstruction.

    NOTE: Random weights do NOT help. This adapter must be TRAINED to be effective.
    """

    def __init__(self, k: int = 9, hidden: int = 768, bottleneck: int = 64,
                 seed: int = 42, alpha_init: float = 0.1):
        super().__init__()
        self.k = k
        self.hidden = hidden
        rng = torch.Generator().manual_seed(seed)

        self.W1 = nn.Parameter(
            torch.randn(bottleneck, k, generator=rng) / math.sqrt(k)
        )
        self.W2 = nn.Parameter(
            torch.randn(hidden, bottleneck, generator=rng) / math.sqrt(bottleneck)
        )
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Residual subspace projector: set externally from this adapter's PCA basis
        self.register_buffer('residual_proj', None)

    def set_residual_subspace(self, pca_vectors: torch.Tensor):
        """Build orthonormal basis for the (hidden - k) subspace orthogonal to pca_vectors.

        Args:
            pca_vectors: PCA eigenvectors for THIS signal (K or V), shape (k, hidden)
        """
        k = pca_vectors.shape[0]
        hidden = pca_vectors.shape[1]
        rest = hidden - k
        if rest <= 0:
            self.residual_proj = torch.zeros(1, hidden, device=pca_vectors.device)
            return

        device = pca_vectors.device
        # Random vectors orthogonalized against PCA components
        random_base = torch.randn(rest, hidden, device=device)
        for i in range(k):
            v = pca_vectors[i:i+1]
            proj = (random_base @ v.T) * v
            random_base = random_base - proj
        Q, _ = torch.linalg.qr(random_base.T)
        self.residual_proj = Q.T

    def forward(self, z: torch.Tensor, x_decompressed: torch.Tensor) -> torch.Tensor:
        if self.residual_proj is None or self.residual_proj.shape[0] <= 1:
            return x_decompressed

        h = F.gelu(F.linear(z, self.W1))
        raw_correction = F.linear(h, self.W2)

        # Project correction onto residual subspace
        coeffs = F.linear(raw_correction, self.residual_proj)
        correction = F.linear(coeffs, self.residual_proj.T)

        return x_decompressed + self.alpha * correction

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class EigenProjector(nn.Module):
    """PCA-based projector (same as in eigen_gpt2.py)."""

    def __init__(self, full_dim: int, k: int):
        super().__init__()
        self.full_dim = full_dim
        self.k = k
        self.proj = nn.Parameter(torch.randn(k, full_dim) / math.sqrt(full_dim))
        self.mean = nn.Parameter(torch.zeros(full_dim), requires_grad=False)

    def init_from_pca(self, data: torch.Tensor):
        with torch.no_grad():
            mean = data.mean(dim=0)
            centered = data - mean
            _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
            n_available = Vt.shape[0]
            k_use = min(self.k, n_available)
            self.proj[:k_use].copy_(Vt[:k_use])
            if k_use < self.k:
                noise = torch.randn(self.k - k_use, self.full_dim, device=Vt.device)
                noise = noise / noise.norm(dim=1, keepdim=True) * 0.01
                self.proj[k_use:].copy_(noise)
            self.mean.copy_(mean)

    def get_pca_vectors(self) -> torch.Tensor:
        return self.proj.data.clone()

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x - self.mean, self.proj)

    def decompress(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.proj.T) + self.mean


def compute_cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    cos = F.cosine_similarity(a_flat, b_flat, dim=-1)
    return cos.mean().item()


def compute_relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.norm(a - b).item() / torch.norm(a).item()


@torch.no_grad()
def collect_activations_all_layers(model, tokenizer, texts: List[str],
                                    device: str = "cpu") -> Dict[int, Dict[str, torch.Tensor]]:
    """Collect K, V activations from ALL layers of GPT-2 using hooks."""
    model.eval()
    n_layers = model.config.n_layer
    layer_data = {i: {'k': [], 'v': []} for i in range(n_layers)}

    # Register hooks to capture K,V at each layer
    hooks = []
    def make_hook(layer_idx):
        def hook(module, input, output):
            # GPT2Attention output: (attn_output, present) or just a tensor
            # But we want the K,V BEFORE attention. Use a forward hook on c_attn.
            pass
        return hook

    # Instead, intercept the c_attn output per layer
    def make_qkv_hook(layer_idx):
        def hook(module, input, output):
            # output from c_attn: [batch, seq, 3*hidden]
            split_size = output.shape[-1] // 3
            k = output[..., split_size:2*split_size]
            v = output[..., 2*split_size:]
            layer_data[layer_idx]['k'].append(k.reshape(-1, k.shape[-1]))
            layer_data[layer_idx]['v'].append(v.reshape(-1, v.shape[-1]))
        return hook

    for idx in range(n_layers):
        block = model.transformer.h[idx]
        h = block.attn.c_attn.register_forward_hook(make_qkv_hook(idx))
        hooks.append(h)

    # Run forward pass through the model for each text
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _ = model(**inputs, output_hidden_states=False)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Concatenate and pad
    result = {}
    for idx in range(n_layers):
        k_t = torch.cat(layer_data[idx]['k'], dim=0)
        v_t = torch.cat(layer_data[idx]['v'], dim=0)
        n_samp = k_t.shape[0]
        min_required = 512
        if n_samp < min_required:
            noise_k = torch.randn(min_required - n_samp, 768, device=device) * 0.01
            noise_v = torch.randn(min_required - n_samp, 768, device=device) * 0.01
            k_t = torch.cat([k_t, noise_k], dim=0)
            v_t = torch.cat([v_t, noise_v], dim=0)
        result[idx] = {'k': k_t, 'v': v_t}

    return result


def compute_attention_output_train(q, k, v, num_heads, scale):
    """Same as compute_attention_output but without no_grad for training."""
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
def benchmark_layer(
    layer_idx: int,
    k_dim: int,
    model,
    tokenizer,
    test_texts: List[str],
    acts_per_layer: Dict[int, Dict[str, torch.Tensor]],
    device: str = "cpu",
    seeds: List[int] = None,
) -> Dict:
    """Benchmark PCA vs PCA+adapter on a single layer.

    Uses SEPARATE K and V adapters, each with its own PCA subspace.
    """
    if seeds is None:
        seeds = [1]
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads
    scale = 1.0 / math.sqrt(head_dim)

    # Init projectors from this layer's activations
    proj_k = EigenProjector(768, k_dim).to(device)
    proj_v = EigenProjector(768, k_dim).to(device)
    proj_k.init_from_pca(acts_per_layer[layer_idx]['k'])
    proj_v.init_from_pca(acts_per_layer[layer_idx]['v'])

    pca_vecs_k = proj_k.get_pca_vectors()
    pca_vecs_v = proj_v.get_pca_vectors()

    # Create SEPARATE adapters for K and V, each with correct subspace
    k_adapters = []
    v_adapters = []
    for seed in seeds:
        ka = LowRankAdapter(k=k_dim, hidden=768, bottleneck=64, seed=seed, alpha_init=0.3).to(device)
        ka.set_residual_subspace(pca_vecs_k)
        k_adapters.append(ka)

        va = LowRankAdapter(k=k_dim, hidden=768, bottleneck=64, seed=seed, alpha_init=0.3).to(device)
        va.set_residual_subspace(pca_vecs_v)
        v_adapters.append(va)

    # Ensemble: average multiple seeds
    def ensemble_forward(adapters, z, x_dec):
        outs = [a(z, x_dec) for a in adapters]
        return torch.stack(outs).mean(dim=0)

    # Accumulators
    pca_k_cos_sum = 0.0
    pca_v_cos_sum = 0.0
    pca_attn_cos_sum = 0.0
    pca_k_err_sum = 0.0
    adapter_k_cos_sum = 0.0
    adapter_v_cos_sum = 0.0
    adapter_attn_cos_sum = 0.0
    adapter_k_err_sum = 0.0
    n_samples = 0

    for text in test_texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        seq_len = inputs['input_ids'].shape[1]

        # Forward to this layer
        hidden = model.transformer.wte(inputs['input_ids']) + model.transformer.wpe(
            torch.arange(seq_len, device=device)
        )
        hidden = model.transformer.drop(hidden)
        for i in range(layer_idx):
            hidden = model.transformer.h[i](hidden)[0]

        block = model.transformer.h[layer_idx]
        normed = block.ln_1(hidden)
        qkv = block.attn.c_attn(normed)
        q, k, v = qkv.chunk(3, dim=-1)

        # Original attention output
        orig_attn = compute_attention_output(q, k, v, num_heads, scale)

        # -- PCA only --
        k_comp = proj_k.compress(k)
        k_pca = proj_k.decompress(k_comp)
        v_comp = proj_v.compress(v)
        v_pca = proj_v.decompress(v_comp)
        pca_k_cos_sum += compute_cosine_sim(k, k_pca)
        pca_v_cos_sum += compute_cosine_sim(v, v_pca)
        pca_k_err_sum += compute_relative_error(k, k_pca)
        pca_attn = compute_attention_output(q, k_pca, v_pca, num_heads, scale)
        pca_attn_cos_sum += compute_cosine_sim(orig_attn, pca_attn)

        # -- PCA + adapter (ensemble) --
        k_ad = ensemble_forward(k_adapters, k_comp, k_pca)
        v_ad = ensemble_forward(v_adapters, v_comp, v_pca)
        adapter_k_cos_sum += compute_cosine_sim(k, k_ad)
        adapter_v_cos_sum += compute_cosine_sim(v, v_ad)
        adapter_k_err_sum += compute_relative_error(k, k_ad)
        adapter_attn = compute_attention_output(q, k_ad, v_ad, num_heads, scale)
        adapter_attn_cos_sum += compute_cosine_sim(orig_attn, adapter_attn)

        n_samples += 1

    return {
        'layer': layer_idx,
        'pca': {
            'k_cos': pca_k_cos_sum / n_samples,
            'v_cos': pca_v_cos_sum / n_samples,
            'attn_cos': pca_attn_cos_sum / n_samples,
            'k_rel_error': pca_k_err_sum / n_samples,
        },
        'adapter': {
            'k_cos': adapter_k_cos_sum / n_samples,
            'v_cos': adapter_v_cos_sum / n_samples,
            'attn_cos': adapter_attn_cos_sum / n_samples,
            'k_rel_error': adapter_k_err_sum / n_samples,
        },
    }


def run_benchmark(device: str = "cpu", k_values: List[int] = None, seeds: List[int] = None):
    """Full benchmark across all layers and k values."""
    if k_values is None:
        k_values = [9, 25, 50, 100, 150]
    if seeds is None:
        seeds = [1, 2, 3]

    print("=" * 72)
    print("  Low-Rank Adapter Benchmark for 85x KV Cache Barrier")
    print("  FIXES: multi-layer (all 12), separate K/V subspaces, ensemble")
    print("=" * 72)
    print()

    print("[1/5] Loading pretrained GPT-2...")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()
    n_layers = model.config.n_layer
    print(f"      Hidden dim: {model.config.n_embd}, Heads: {model.config.n_head}, Layers: {n_layers}")

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

    print()
    print("[2/5] Collecting activations from ALL layers...")
    acts = collect_activations_all_layers(model, tokenizer, test_texts, device)
    print(f"      Collected from {len(acts)} layers")

    print()
    print("[3/5] Initializing adapters...")
    adapter = LowRankAdapter(k=k_values[0], hidden=768, bottleneck=64, seed=seeds[0], alpha_init=0.3)
    params_per = adapter.get_param_count()
    total_params = params_per * 2 * 3 + (0 if len(seeds) < 2 else 0)
    print(f"      Per adapter: {params_per:,} params")
    print(f"      Per layer: {params_per * 2:,} params (K + V)")
    print(f"      Seeds per adapter: {len(seeds)} (ensemble)")
    print(f"      Total adapter params per layer: {params_per * 2 * len(seeds):,}")

    print()
    print("[4/5] Running benchmark across ALL layers...")
    print(f"      k values: {k_values}")
    print(f"      Layers: 0-{n_layers - 1}")
    print()

    # Per-layer, per-k results
    all_results = {}

    for k_dim in sorted(k_values):
        print(f"  --- k={k_dim} ({768/k_dim:.1f}x compression) ---")
        layer_results = []

        for layer_idx in range(n_layers):
            t0 = time.time()
            r = benchmark_layer(layer_idx, k_dim, model, tokenizer, test_texts, acts, device, seeds)
            elapsed = time.time() - t0

            pca_cos = r['pca']['attn_cos']
            ada_cos = r['adapter']['attn_cos']
            delta = (ada_cos - pca_cos) / (pca_cos + 1e-10) * 100
            print(f"    L{layer_idx:2d} | PCA attn: {pca_cos:.4f} | Adapter attn: {ada_cos:.4f} ({delta:+.2f}%) | {elapsed:.2f}s")
            layer_results.append(r)

        all_results[k_dim] = layer_results

        # Average across layers
        avg_pca = np.mean([r['pca']['attn_cos'] for r in layer_results])
        avg_ada = np.mean([r['adapter']['attn_cos'] for r in layer_results])
        avg_delta = (avg_ada - avg_pca) / (avg_pca + 1e-10) * 100
        best_ada = max(r['adapter']['attn_cos'] for r in layer_results)
        print(f"    {'---':>4s} | {'AVERAGE':<12s} | PCA: {avg_pca:.4f} | Adapter: {avg_ada:.4f} ({avg_delta:+.2f}%) | best layer: {best_ada:.4f}")
        print()

    # Summary table
    print("[5/5] SUMMARY (averaged across all layers)")
    print()
    header = f"  {'k':<6} {'Compress':<12} {'PCA Attn Cos':<14} {'Adapter Attn Cos':<18} {'Delta':<10} {'Layers Best':<12}"
    print(header)
    print(f"  {'-' * 72}")
    for k_dim in sorted(k_values):
        lr = all_results[k_dim]
        avg_pca = np.mean([r['pca']['attn_cos'] for r in lr])
        avg_ada = np.mean([r['adapter']['attn_cos'] for r in lr])
        delta = (avg_ada - avg_pca) / (avg_pca + 1e-10) * 100
        worst_layer = min(r['adapter']['attn_cos'] - r['pca']['attn_cos'] for r in lr)
        best_layer = max(r['adapter']['attn_cos'] - r['pca']['attn_cos'] for r in lr)
        print(f"  {k_dim:<6} {768/k_dim:<12.1f}x {avg_pca:<14.4f} {avg_ada:<18.4f} {delta:<+10.2f}% worst={worst_layer:+.3f} best={best_layer:+.3f}")

    print()
    print("  VERDICT: Random adapters do NOT help at any k or any layer.")
    print("  Training is required to learn attention-aware corrections.")
    print()
    print("  Done.")

    # Save
    out_dir = Path(__file__).parent
    results_path = out_dir / "benchmark_results.json"
    serializable = {
        str(k): [
            {
                'layer': r['layer'],
                'pca': {k2: float(v) for k2, v in r['pca'].items()},
                'adapter': {k2: float(v) for k2, v in r['adapter'].items()},
            }
            for r in lr
        ]
        for k, lr in all_results.items()
    }
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {results_path}")

    return all_results


def run_gguf_demo(device: str = "cpu"):
    """GGUF backend demo: load LFM2.5, run inference, extract signals."""
    print("=" * 72)
    print("  GGUF Backend Demo (LFM2.5 + CUDA)")
    print("  Phase 3.5: KV Cache Compression signal extraction")
    print("=" * 72)
    print()

    from gguf_backend import GgufBackend

    print("[1/5] Loading LFM2.5 GGUF with full GPU offload...")
    backend = GgufBackend(n_ctx=2048, verbose=False)
    info = backend.info()
    print(f"      Model: {info['model']} ({info['arch']})")
    print(f"      GPU:   {info['gpu']}")
    print(f"      EmbD:  {info['n_embd']}, Vocab: {info['n_vocab']}, Layers: {info['n_layers']}")
    print()

    test_texts = [
        "The capital of France is",
        "The theory of relativity was developed by",
        "Machine learning is a subset of",
        "The largest ocean on Earth is",
        "In computer science, an algorithm is",
    ]

    print("[2/5] Inference benchmark...")
    for text in test_texts:
        out = backend.generate(text, max_tokens=32)
        print(f"      {text}")
        print(f"      -> {out[:80]}...")
        print()

    print("[3/5] Logit extraction (per-token vocabulary distribution)...")
    for text in ["Hello world", "KV cache compression"]:
        logits = backend.get_logits(text)
        # softmax over 65536 vocab
        logits_stable = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits_stable) / np.exp(logits_stable).sum(axis=-1, keepdims=True)
        top5 = np.argsort(-probs[0])[:5]
        top5_str = [backend.detokenize(int(t)).strip() or f"<tok_{t}>" for t in top5]
        top5_p = [probs[0, t] for t in top5]
        print(f"      Prompt: {repr(text)}")
        print(f"      Logits shape: {logits.shape}")
        for i, (t_str, t_p) in enumerate(zip(top5_str, top5_p)):
            print(f"        Top-{i+1}: {t_str!r:20s}  p={t_p:.6f}  (id={top5[i]})")
        print()

    print("[4/5] Embedding extraction (sentence-level hidden states)...")
    for text in test_texts:
        emb = backend.get_embedding(text)
        norm = np.linalg.norm(emb)
        print(f"      {text[:50]:50s}  emb={emb.shape}  norm={norm:.3f}")
    print()

    print("[5/5] Chat completion (LFM2.5 chat template)...")
    reply = backend.chat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain KV cache compression in one sentence."},
    ], max_tokens=64)
    print(f"      {reply}")
    print()

    backend.close()
    print("  Done.")


def run_qwen_demo():
    """Qwen3.6 35B-A3B MoE GGUF demo with partial GPU offload."""
    print("=" * 72)
    print("  Qwen3.6 GGUF Demo (35B-A3B MoE + CUDA)")
    print("  Phase 3.5: Large MoE model inference")
    print("=" * 72)
    print()

    from gguf_backend import GgufBackend, MODELS

    print("[1/3] Loading Qwen3.6 with partial GPU offload (18/42 layers)...")
    b = GgufBackend(gguf_path=MODELS["qwen"], n_gpu_layers=18, n_ctx=512)
    info = b.info()
    print(f"      Model: {info['model']} ({info['arch']})")
    print(f"      GPU:   {info['gpu']}, {info['n_layers']} layers, 18 offloaded")
    print(f"      EmbD:  {info['n_embd']}, Vocab: {info['n_vocab']}")
    print()

    print("[2/3] Text generation (single instance)...")
    prompts = [
        "The capital of France is",
        "The theory of relativity was developed by",
        "Machine learning is a subset of",
    ]
    for text in prompts:
        out = b.generate(text, max_tokens=24)
        print(f"      {text}")
        print(f"      -> {out}")
        print()

    print("[3/3] Chat completion (Qwen3.6 chat template)...")
    reply = b.chat([
        {"role": "user", "content": "Explain what an MoE model is in one sentence."},
    ], max_tokens=64)
    print(f"      {reply}")
    print()

    b.close()
    print("  Done.")


def main():
    parser = argparse.ArgumentParser(description="Low-Rank Adapter Benchmark")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    bench_p = subparsers.add_parser("benchmark", help="Run full benchmark")
    bench_p.add_argument("--k_values", type=int, nargs="+", default=[9, 25, 50, 100, 150],
                         help="K values to test")
    bench_p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                         help="Random seeds for ensemble averaging")

    subparsers.add_parser("gguf-demo", help="Run GGUF backend demo (LFM2.5 + CUDA)")
    subparsers.add_parser("qwen-demo", help="Run Qwen3.6 GGUF demo (35B MoE + CUDA)")

    args = parser.parse_args()

    if args.command == "benchmark":
        run_benchmark(args.device, args.k_values, args.seeds)
    elif args.command == "gguf-demo":
        run_gguf_demo(args.device)
    elif args.command == "qwen-demo":
        run_qwen_demo()


if __name__ == '__main__':
    main()
