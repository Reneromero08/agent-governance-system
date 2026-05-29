"""Phase 3.5: Train Low-Rank Adapters for KV Cache Compression

Trains the LowRankAdapter from flat_llm_adapter.py using attention
output fidelity as the loss function. Extends the existing benchmark
pipeline with a training loop.

Usage:
    python train_adapter.py --k 9 --epochs 20 --device cpu
"""

import argparse, json, math, time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from flat_llm_adapter import (
    LowRankAdapter, EigenProjector,
    collect_activations_all_layers,
    compute_attention_output, compute_cosine_sim, compute_relative_error,
    compute_attention_output_train,
)


def train_adapter_layer(
    layer_idx: int,
    k_dim: int,
    model,
    tokenizer,
    train_texts: List[str],
    test_texts: List[str],
    acts_per_layer: Dict[int, Dict[str, torch.Tensor]],
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict:
    """Train separate K and V adapters for a single GPT-2 layer.

    Loss: MSE between attention output with adapted K,V and original attention.
    The adapter learns to correct PCA reconstruction, preserving what attention
    actually uses rather than what PCA discards.
    """
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads
    scale = 1.0 / math.sqrt(head_dim)

    # Init PCA projectors
    proj_k = EigenProjector(768, k_dim).to(device)
    proj_v = EigenProjector(768, k_dim).to(device)
    proj_k.init_from_pca(acts_per_layer[layer_idx]['k'])
    proj_v.init_from_pca(acts_per_layer[layer_idx]['v'])

    # Create trainable adapters
    adapter_k = LowRankAdapter(k=k_dim, hidden=768, bottleneck=64, seed=42, alpha_init=0.1).to(device)
    adapter_v = LowRankAdapter(k=k_dim, hidden=768, bottleneck=64, seed=43, alpha_init=0.1).to(device)
    adapter_k.set_residual_subspace(proj_k.get_pca_vectors())
    adapter_v.set_residual_subspace(proj_v.get_pca_vectors())

    # Collect training data: Q,K,V from all train texts at this layer using hooks
    train_data = []
    model.eval()
    
    qkv_data = {}
    def make_qkv_hook(layer_idx):
        def hook(module, input, output):
            split_sz = output.shape[-1] // 3
            q = output[..., :split_sz]
            k = output[..., split_sz:2*split_sz]
            v = output[..., 2*split_sz:]
            key = f'L{layer_idx}'
            if key not in qkv_data:
                qkv_data[key] = []
            qkv_data[key].append((q.detach(), k.detach(), v.detach()))
        return hook
    
    # Register hooks on c_attn for the target layer only
    hook = model.transformer.h[layer_idx].attn.c_attn.register_forward_hook(make_qkv_hook(layer_idx))
    
    # Run all train texts through the model
    with torch.no_grad():
        for text in train_texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = model(**inputs)
    
    hook.remove()
    train_data = qkv_data.get(f'L{layer_idx}', [])

    # Optimizer
    optimizer = torch.optim.Adam(
        list(adapter_k.parameters()) + list(adapter_v.parameters()), lr=lr)

    # Evaluate before training
    def evaluate_metrics(test_q, test_k, test_v):
        k_comp = proj_k.compress(test_k)
        k_pca = proj_k.decompress(k_comp)
        v_comp = proj_v.compress(test_v)
        v_pca = proj_v.decompress(v_comp)

        orig_attn = compute_attention_output(test_q, test_k, test_v, num_heads, scale)
        pca_attn = compute_attention_output(test_q, k_pca, v_pca, num_heads, scale)

        k_ad = adapter_k(k_comp, k_pca)
        v_ad = adapter_v(v_comp, v_pca)
        ada_attn = compute_attention_output(test_q, k_ad, v_ad, num_heads, scale)

        return {
            'pca_attn_cos': compute_cosine_sim(orig_attn, pca_attn),
            'ada_attn_cos': compute_cosine_sim(orig_attn, ada_attn),
            'pca_attn_loss': F.mse_loss(pca_attn, orig_attn).item(),
            'ada_attn_loss': F.mse_loss(ada_attn, orig_attn).item(),
        }

    # Pre-train metrics
    test_q, test_k, test_v = train_data[0]
    pre_metrics = evaluate_metrics(test_q, test_k, test_v)

    # Training loop
    loss_history = []
    t0 = time.time()
    for epoch in range(epochs):
        total_loss = 0.0
        for q, k, v in train_data:
            optimizer.zero_grad()

            # Compress and decompress K, V
            k_comp = proj_k.compress(k)
            k_pca = proj_k.decompress(k_comp)
            v_comp = proj_v.compress(v)
            v_pca = proj_v.decompress(v_comp)

            # Apply adapters
            k_ad = adapter_k(k_comp, k_pca)
            v_ad = adapter_v(v_comp, v_pca)

            # Compute attention with adapted K,V (training — needs gradients)
            orig_attn = compute_attention_output_train(q, k, v, num_heads, scale)
            ada_attn = compute_attention_output_train(q, k_ad, v_ad, num_heads, scale)

            # Loss: attention output fidelity
            loss = F.mse_loss(ada_attn, orig_attn)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        loss_history.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.6f}", flush=True)

    train_time = time.time() - t0

    # Post-train metrics
    post_metrics = evaluate_metrics(test_q, test_k, test_v)

    return {
        'layer': layer_idx,
        'k': k_dim,
        'pre_pca_attn_cos': pre_metrics['pca_attn_cos'],
        'pre_ada_attn_cos': pre_metrics['ada_attn_cos'],
        'post_pca_attn_cos': post_metrics['pca_attn_cos'],
        'post_ada_attn_cos': post_metrics['ada_attn_cos'],
        'pre_attn_loss': pre_metrics['ada_attn_loss'],
        'post_attn_loss': post_metrics['ada_attn_loss'],
        'loss_history': loss_history,
        'adapter_k_state': {k: v.cpu().clone() for k, v in adapter_k.state_dict().items()},
        'adapter_v_state': {k: v.cpu().clone() for k, v in adapter_v.state_dict().items()},
        'train_time': train_time,
    }


def run_training(
    k_values: List[int] = None,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
):
    if k_values is None:
        k_values = [9]

    print("=" * 72)
    print("  Low-Rank Adapter TRAINING for KV Cache Compression")
    print(f"  k={k_values}, epochs={epochs}, lr={lr}, device={device}")
    print("=" * 72)
    print()

    # Load GPT-2 from local checkpoint
    print("[1/5] Loading GPT-2 from local model...")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    local_path = str(Path(__file__).parent.parent.parent / "models" / "gpt2")
    model = GPT2LMHeadModel.from_pretrained(local_path).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(local_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    n_layers = model.config.n_layer
    print(f"      Hidden: {model.config.n_embd}, Heads: {model.config.n_head}, Layers: {n_layers}")

    # Calibration texts
    train_texts = [
        "The meaning of life is a philosophical question that has puzzled humanity for centuries.",
        "Artificial intelligence is transforming the way we interact with technology every day.",
        "Deep learning enables complex pattern recognition in vast amounts of data.",
        "The human brain contains approximately eighty-six billion neurons.",
        "Climate change poses significant challenges for future generations.",
        "In mathematics, prime numbers have fascinated researchers for thousands of years.",
        "The ocean covers more than seventy percent of Earth's surface.",
        "Music has the power to evoke strong emotional responses across cultures.",
        "Space exploration has led to many technological breakthroughs.",
        "Language is the foundation of human communication and collective knowledge.",
    ]
    test_texts = [
        "Economic systems attempt to explain how resources are allocated in complex societies.",
        "The history of science is a story of ideas evolving through observation.",
    ]

    print(f"\n[2/5] Collecting activations from all {n_layers} layers...")
    acts = collect_activations_all_layers(model, tokenizer, train_texts + test_texts, device)
    print(f"      Collected from {len(acts)} layers")

    all_results = {}

    for k_dim in sorted(k_values):
        comp_ratio = 768 / k_dim
        print(f"\n[3/5] Training adapters at k={k_dim} ({comp_ratio:.1f}x compression)")
        print(f"      {'=' * 50}")

        layer_results = []
        for layer_idx in range(n_layers):
            print(f"  Layer {layer_idx}:", flush=True)
            r = train_adapter_layer(
                layer_idx, k_dim, model, tokenizer,
                train_texts, test_texts, acts,
                epochs=epochs, lr=lr, device=device,
            )

            delta_pre = r['pre_ada_attn_cos'] - r['pre_pca_attn_cos']
            delta_post = r['post_ada_attn_cos'] - r['post_pca_attn_cos']
            print(f"    pre:  PCA={r['pre_pca_attn_cos']:.4f}  Ada={r['pre_ada_attn_cos']:.4f}  "
                  f"delta={delta_pre:+.4f}  loss={r['pre_attn_loss']:.6f}")
            print(f"    post: PCA={r['post_pca_attn_cos']:.4f}  Ada={r['post_ada_attn_cos']:.4f}  "
                  f"delta={delta_post:+.4f}  loss={r['post_attn_loss']:.6f}  "
                  f"dt={r['train_time']:.1f}s")
            layer_results.append(r)

        all_results[k_dim] = layer_results

        # Summary
        avg_pre = np.mean([r['pre_ada_attn_cos'] for r in layer_results])
        avg_post = np.mean([r['post_ada_attn_cos'] for r in layer_results])
        avg_pca = np.mean([r['post_pca_attn_cos'] for r in layer_results])
        avg_delta = avg_post - avg_pca
        n_improved = sum(1 for r in layer_results if r['post_ada_attn_cos'] > r['pre_ada_attn_cos'])

        print(f"\n  AVERAGE across {n_layers} layers:")
        print(f"    PCA attn cos:     {avg_pca:.4f}")
        print(f"    Pre-train Ada:    {avg_pre:.4f}")
        print(f"    Post-train Ada:   {avg_post:.4f}  (delta vs PCA: {avg_delta:+.4f})")
        print(f"    Layers improved:  {n_improved}/{n_layers}")

    # Save results
    print(f"\n[4/5] Saving results...")
    out_dir = Path(__file__).parent
    results_path = out_dir / "train_results.json"

    serializable = {}
    for k, layers in all_results.items():
        serializable[str(k)] = []
        for r in layers:
            entry = {key: float(val) if isinstance(val, (np.floating, np.integer)) else val
                     for key, val in r.items()
                     if key not in ('adapter_k_state', 'adapter_v_state')}
            serializable[str(k)].append(entry)

    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"    Saved to {results_path}")

    # Save trained adapter weights
    weights_path = out_dir / "trained_adapters.pt"
    weight_data = {}
    for k, layers in all_results.items():
        weight_data[str(k)] = {}
        for r in layers:
            weight_data[str(k)][str(r['layer'])] = {
                'adapter_k': r['adapter_k_state'],
                'adapter_v': r['adapter_v_state'],
            }
    torch.save(weight_data, weights_path)
    print(f"    Weights saved to {weights_path}")

    # Final verdict
    print(f"\n[5/5] VERDICT")
    for k_dim in sorted(k_values):
        lr = all_results[k_dim]
        avg_post = np.mean([r['post_ada_attn_cos'] for r in lr])
        avg_pca = np.mean([r['post_pca_attn_cos'] for r in lr])
        avg_delta = avg_post - avg_pca
        avg_pre = np.mean([r['pre_ada_attn_cos'] for r in lr])
        pre_delta = avg_pre - avg_pca
        improved_layers = sum(1 for r in lr if r['post_ada_attn_cos'] > r['post_pca_attn_cos'])
        print(f"    k={k_dim}: PCA={avg_pca:.4f}  Pre-train Ada={avg_pre:.4f} ({pre_delta:+.4f})  Post-train Ada={avg_post:.4f} ({avg_delta:+.4f})  improved={improved_layers}/{n_layers} layers")
        if avg_delta < 0:
            print(f"      TRAINED ADAPTER STILL HURTS. Delta={avg_delta:+.4f} vs PCA-only.")
        else:
            print(f"      TRAINED ADAPTER HELPS. Delta={avg_delta:+.4f} over PCA-only.")

    print("\nDone.")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Train Low-Rank Adapters for KV Cache Compression")
    parser.add_argument("--k", type=int, nargs="+", default=[9], help="Compression dimensions")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()
    run_training(k_values=args.k, epochs=args.epochs, lr=args.lr, device=args.device)


if __name__ == '__main__':
    main()
