"""Phase 3.5 Sweeps: Fast iteration across 5 tasks on GPT-2.

Task 1: Push past 85x — k=6, 3, 1
Task 2: Asymmetric budget — V gets more dims than K
Task 3: Adapter bottleneck sweep — 32, 64, 128, 256
Task 4: Shared adapter across layers
Task 5: Cross-model transfer (GPT-2 -> DistilGPT-2)

Usage: python sweep.py --task 1
"""

import argparse, json, math, time
from pathlib import Path
from typing import Dict, List
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent dir for flat_llm_adapter imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "extensions" / "03_flat_llm"))
from flat_llm_adapter import (
    LowRankAdapter, EigenProjector,
    collect_activations_all_layers,
    compute_attention_output, compute_cosine_sim,
    compute_attention_output_train,
)

OUT_DIR = Path(__file__).parent


def load_gpt2(device="cpu"):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    local_path = str(Path(__file__).resolve().parent.parent.parent / "models" / "gpt2")
    model = GPT2LMHeadModel.from_pretrained(local_path).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(local_path)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer


def get_train_test_texts():
    train = [
        "The meaning of life is a philosophical question that has puzzled humanity for centuries.",
        "Artificial intelligence is transforming the way we interact with technology.",
        "Deep learning enables complex pattern recognition in vast amounts of data.",
        "The human brain contains approximately eighty-six billion neurons.",
        "Climate change poses significant challenges for future generations.",
        "In mathematics, prime numbers have fascinated researchers for millennia.",
        "The ocean covers more than seventy percent of Earth's surface.",
        "Music has the power to evoke strong emotional responses across cultures.",
    ]
    test = ["Space exploration has led to many technological breakthroughs.",
            "Language is the foundation of human communication."]
    return train, test


def train_one_layer(model, tokenizer, layer_idx, k_K, k_V, bottleneck, epochs, lr, device,
                    train_texts, test_texts, acts):
    """Train K and V adapters. Evaluate on held-out test texts. Returns metrics dict."""
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads
    scale = 1.0 / math.sqrt(head_dim)

    proj_k = EigenProjector(768, k_K).to(device); proj_k.init_from_pca(acts[layer_idx]['k'])
    proj_v = EigenProjector(768, k_V).to(device); proj_v.init_from_pca(acts[layer_idx]['v'])

    adapter_k = LowRankAdapter(k=k_K, hidden=768, bottleneck=bottleneck, seed=42).to(device)
    adapter_v = LowRankAdapter(k=k_V, hidden=768, bottleneck=bottleneck, seed=43).to(device)
    adapter_k.set_residual_subspace(proj_k.get_pca_vectors())
    adapter_v.set_residual_subspace(proj_v.get_pca_vectors())

    # Collect train Q,K,V at this layer
    train_qkv = []
    def hook_fn(module, inp, out):
        sz = out.shape[-1] // 3
        train_qkv.append((out[..., :sz].detach(), out[..., sz:2*sz].detach(), out[..., 2*sz:].detach()))
    hook = model.transformer.h[layer_idx].attn.c_attn.register_forward_hook(hook_fn)
    with torch.no_grad():
        for text in train_texts:
            inp = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            _ = model(**{k: v.to(device) for k, v in inp.items()})
    hook.remove()

    if not train_qkv: return None

    # Train
    optimizer = torch.optim.Adam(list(adapter_k.parameters()) + list(adapter_v.parameters()), lr=lr)
    for epoch in range(epochs):
        for q, k, v in train_qkv:
            optimizer.zero_grad()
            k_comp = proj_k.compress(k); k_pca = proj_k.decompress(k_comp)
            v_comp = proj_v.compress(v); v_pca = proj_v.decompress(v_comp)
            k_ad = adapter_k(k_comp, k_pca); v_ad = adapter_v(v_comp, v_pca)
            orig = compute_attention_output_train(q, k, v, num_heads, scale)
            ada = compute_attention_output_train(q, k_ad, v_ad, num_heads, scale)
            loss = F.mse_loss(ada, orig)
            loss.backward()
            optimizer.step()

    # Evaluate on HELD-OUT test texts
    test_qkv = []
    def test_hook(module, inp, out):
        sz = out.shape[-1] // 3
        test_qkv.append((out[..., :sz].detach(), out[..., sz:2*sz].detach(), out[..., 2*sz:].detach()))
    h_test = model.transformer.h[layer_idx].attn.c_attn.register_forward_hook(test_hook)
    with torch.no_grad():
        for text in test_texts:
            inp = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            _ = model(**{k: v.to(device) for k, v in inp.items()})
    h_test.remove()

    pca_cos_list, ada_cos_list = [], []
    for q, k, v in test_qkv:
        with torch.no_grad():
            k_comp = proj_k.compress(k); k_pca = proj_k.decompress(k_comp)
            v_comp = proj_v.compress(v); v_pca = proj_v.decompress(v_comp)
            orig = compute_attention_output(q, k, v, num_heads, scale)
            pca_attn = compute_attention_output(q, k_pca, v_pca, num_heads, scale)
            k_ad = adapter_k(k_comp, k_pca); v_ad = adapter_v(v_comp, v_pca)
            ada_attn = compute_attention_output(q, k_ad, v_ad, num_heads, scale)
            pca_cos_list.append(compute_cosine_sim(orig, pca_attn))
            ada_cos_list.append(compute_cosine_sim(orig, ada_attn))

    return {
        'pca_cos': float(np.mean(pca_cos_list)),
        'ada_cos': float(np.mean(ada_cos_list)),
    }


def run_task1(device="cpu", epochs=10):
    """Push past 85x: k=9, 6, 3, 1."""
    print("=" * 60)
    print("TASK 1: Push Past 85x (out-of-sample)")
    print("=" * 60)

    model, tokenizer = load_gpt2(device)
    train_texts, test_texts = get_train_test_texts()
    n_layers = model.config.n_layer

    acts = collect_activations_all_layers(model, tokenizer, train_texts + test_texts, device)
    print(f"Collected activations from {len(acts)} layers")

    k_values = [9, 6, 3, 1]
    results = {}

    for k_dim in k_values:
        comp = 768 / k_dim
        print(f"\n--- k={k_dim} ({comp:.1f}x compression) ---")
        layer_results = []
        for l in range(n_layers):
            t0 = time.time()
            r = train_one_layer(model, tokenizer, l, k_dim, k_dim, 64, epochs, 1e-3, device, train_texts, test_texts, acts)
            if r:
                print(f"  L{l:2d}: PCA={r['pca_cos']:.4f}  Ada={r['ada_cos']:.4f}  delta={r['ada_cos']-r['pca_cos']:+.4f}  dt={time.time()-t0:.1f}s")
            layer_results.append(r)

        avg_pca = np.mean([r['pca_cos'] for r in layer_results if r])
        avg_ada = np.mean([r['ada_cos'] for r in layer_results if r])
        results[k_dim] = {'avg_pca': avg_pca, 'avg_ada': avg_ada, 'layers': layer_results}
        print(f"  AVG: PCA={avg_pca:.4f}  Ada={avg_ada:.4f}  delta={avg_ada-avg_pca:+.4f}")

    print(f"\nTask 1 Summary (out-of-sample):")
    for k_dim in k_values:
        r = results[k_dim]
        print(f"  k={k_dim} ({768/k_dim:.1f}x): PCA={r['avg_pca']:.4f}  Ada={r['avg_ada']:.4f}  +{r['avg_ada']-r['avg_pca']:.4f}")

    json.dump({str(k): {'avg_pca': float(v['avg_pca']), 'avg_ada': float(v['avg_ada'])}
               for k, v in results.items()},
              open(OUT_DIR / "sweep_task1.json", 'w'), indent=2)
    return results


def run_task2(device="cpu", epochs=10):
    """Asymmetric budget: give V more dims than K."""
    print("=" * 60)
    print("TASK 2: Asymmetric Budget")
    print("=" * 60)

    model, tokenizer = load_gpt2(device)
    train_texts, test_texts = get_train_test_texts()
    n_layers = model.config.n_layer

    acts = collect_activations_all_layers(model, tokenizer, train_texts + test_texts, device)

    splits = [(3, 15, 18, 85.3), (5, 25, 30, 51.2), (8, 36, 44, 34.9)]
    results = {}

    for k_K, k_V, total, comp_ratio in splits:
        print(f"\n--- K={k_K}, V={k_V} (total={total}, {comp_ratio:.1f}x) ---")
        layer_results = []
        for l in range(n_layers):
            t0 = time.time()
            r = train_one_layer(model, tokenizer, l, k_K, k_V, 64, epochs, 1e-3, device, train_texts, test_texts, acts)
            if r:
                print(f"  L{l:2d}: PCA={r['pca_cos']:.4f}  Ada={r['ada_cos']:.4f}  dt={time.time()-t0:.1f}s")
            layer_results.append(r)

        avg = np.mean([r['ada_cos'] for r in layer_results if r])
        results[f"K{k_K}_V{k_V}"] = avg
        print(f"  AVG Ada: {avg:.4f}")

    # Compare to symmetric at same total budget
    print(f"\nTask 2 Summary:")
    for k_K, k_V, total, cr in splits:
        avg = results[f"K{k_K}_V{k_V}"]
        print(f"  K={k_K} V={k_V} (total={total}, {cr:.1f}x): Ada={avg:.4f}")

    json.dump({k: float(v) for k, v in results.items()},
              open(OUT_DIR / "sweep_task2.json", 'w'), indent=2)
    return results


def run_task3(device="cpu", epochs=10):
    """Adapter bottleneck sweep: 32, 64, 128, 256."""
    print("=" * 60)
    print("TASK 3: Bottleneck Sweep")
    print("=" * 60)

    model, tokenizer = load_gpt2(device)
    train_texts, test_texts = get_train_test_texts()
    n_layers = model.config.n_layer

    acts = collect_activations_all_layers(model, tokenizer, train_texts + test_texts, device)
    k_dim = 9

    bottlenecks = [32, 64, 128, 256]
    results = {}

    for bn in bottlenecks:
        params = (bn * k_dim + 768 * bn + 1) * 2
        print(f"\n--- bottleneck={bn} ({params//1000}K params per layer) ---")
        layer_results = []
        for l in range(n_layers):
            t0 = time.time()
            r = train_one_layer(model, tokenizer, l, k_dim, k_dim, bn, epochs, 1e-3, device, train_texts, test_texts, acts)
            if r:
                print(f"  L{l:2d}: PCA={r['pca_cos']:.4f}  Ada={r['ada_cos']:.4f}  dt={time.time()-t0:.1f}s")
            layer_results.append(r)

        avg = np.mean([r['ada_cos'] for r in layer_results if r])
        results[bn] = avg
        print(f"  AVG: {avg:.4f}")

    print(f"\nTask 3 Summary:")
    for bn in bottlenecks:
        print(f"  bottleneck={bn}: Ada={results[bn]:.4f}")

    json.dump({str(k): float(v) for k, v in results.items()},
              open(OUT_DIR / "sweep_task3.json", 'w'), indent=2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=[1,2,3,4,5], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.task == 1: run_task1(args.device, args.epochs)
    elif args.task == 2: run_task2(args.device, args.epochs)
    elif args.task == 3: run_task3(args.device, args.epochs)
    elif args.task == 4: run_task4(args.device, args.epochs)
    elif args.task == 5: run_task5(args.device, args.epochs)


def run_task4(device="cpu", epochs=15):
    """Shared adapter across ALL layers: train one adapter on pooled data."""
    print("=" * 60)
    print("TASK 4: Shared Adapter Across Layers")
    print("=" * 60)

    model, tokenizer = load_gpt2(device)
    train_texts, _ = get_train_test_texts()
    n_layers = model.config.n_layer
    k_dim = 9
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads
    scale = 1.0 / math.sqrt(head_dim)

    acts = collect_activations_all_layers(model, tokenizer, train_texts, device)

    # Collect Q,K,V from ALL layers pooled together
    all_qkv = []
    hooks = []
    def make_hook():
        def hook_fn(module, inp, out):
            sz = out.shape[-1] // 3
            all_qkv.append((out[..., :sz].detach(), out[..., sz:2*sz].detach(), out[..., 2*sz:].detach()))
        return hook_fn
    for l in range(n_layers):
        hooks.append(model.transformer.h[l].attn.c_attn.register_forward_hook(make_hook()))
    with torch.no_grad():
        for text in train_texts:
            inp = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            _ = model(**{k: v.to(device) for k, v in inp.items()})
    for h in hooks: h.remove()

    print(f"Collected {len(all_qkv)} QKV samples across {n_layers} layers")

    # Train ONE shared adapter on pooled data
    # Use PCA from all layers pooled for init
    all_k = torch.cat([qkv[1].reshape(-1, 768) for qkv in all_qkv], dim=0)
    all_v = torch.cat([qkv[2].reshape(-1, 768) for qkv in all_qkv], dim=0)

    proj_k = EigenProjector(768, k_dim).to(device); proj_k.init_from_pca(all_k)
    proj_v = EigenProjector(768, k_dim).to(device); proj_v.init_from_pca(all_v)

    adapter_k = LowRankAdapter(k=k_dim, hidden=768, bottleneck=64, seed=42).to(device)
    adapter_v = LowRankAdapter(k=k_dim, hidden=768, bottleneck=64, seed=43).to(device)
    adapter_k.set_residual_subspace(proj_k.get_pca_vectors())
    adapter_v.set_residual_subspace(proj_v.get_pca_vectors())

    optimizer = torch.optim.Adam(list(adapter_k.parameters()) + list(adapter_v.parameters()), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0.0
        for q, k, v in all_qkv:
            optimizer.zero_grad()
            k_comp = proj_k.compress(k); k_pca = proj_k.decompress(k_comp)
            v_comp = proj_v.compress(v); v_pca = proj_v.decompress(v_comp)
            k_ad = adapter_k(k_comp, k_pca); v_ad = adapter_v(v_comp, v_pca)
            orig = compute_attention_output_train(q, k, v, num_heads, scale)
            ada = compute_attention_output_train(q, k_ad, v_ad, num_heads, scale)
            loss = F.mse_loss(ada, orig)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}: loss={total_loss/len(all_qkv):.6f}")

    # Evaluate per-layer with shared adapter
    print(f"\n  Per-layer evaluation (shared adapter):")
    per_layer = []
    for l in range(n_layers):
        # Get test sample for this layer
        layer_hook_data = []
        def lhook(module, inp, out):
            sz = out.shape[-1] // 3
            layer_hook_data.append((out[..., :sz].detach(), out[..., sz:2*sz].detach(), out[..., 2*sz:].detach()))
        h = model.transformer.h[l].attn.c_attn.register_forward_hook(lhook)
        with torch.no_grad():
            inp = tokenizer(train_texts[0], return_tensors='pt', truncation=True, max_length=128)
            _ = model(**{k: v.to(device) for k, v in inp.items()})
        h.remove()

        if layer_hook_data:
            q, k, v = layer_hook_data[0]
            with torch.no_grad():
                k_comp = proj_k.compress(k); k_pca = proj_k.decompress(k_comp)
                v_comp = proj_v.compress(v); v_pca = proj_v.decompress(v_comp)
                orig = compute_attention_output(q, k, v, num_heads, scale)
                pca_attn = compute_attention_output(q, k_pca, v_pca, num_heads, scale)
                k_ad = adapter_k(k_comp, k_pca); v_ad = adapter_v(v_comp, v_pca)
                ada_attn = compute_attention_output(q, k_ad, v_ad, num_heads, scale)
            pca_cos = compute_cosine_sim(orig, pca_attn)
            ada_cos = compute_cosine_sim(orig, ada_attn)
            per_layer.append({'pca': pca_cos, 'ada': ada_cos})
            print(f"    L{l:2d}: PCA={pca_cos:.4f}  Shared Ada={ada_cos:.4f}  delta={ada_cos-pca_cos:+.4f}")

    avg_shared = np.mean([r['ada'] for r in per_layer])
    avg_pca = np.mean([r['pca'] for r in per_layer])
    print(f"\n  Shared adapter avg: {avg_shared:.4f}  (PCA avg: {avg_pca:.4f})  delta: {avg_shared-avg_pca:+.4f}")
    print(f"  Compare: per-layer trained avg at k=9 was 0.821")

    json.dump({"shared_ada_cos": float(avg_shared), "pca_cos": float(avg_pca),
               "per_layer": [{"pca": float(r['pca']), "ada": float(r['ada'])} for r in per_layer]},
              open(OUT_DIR / "sweep_task4.json", 'w'), indent=2)
    return per_layer


def run_task5(device="cpu", epochs=15):
    """Cross-model transfer: GPT-2 adapter -> DistilGPT-2."""
    print("=" * 60)
    print("TASK 5: Cross-Model Transfer (GPT-2 -> DistilGPT-2)")
    print("=" * 60)

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # Train on GPT-2
    print("\n[1/3] Training adapter on GPT-2...")
    model_gpt2, tokenizer = load_gpt2(device)
    train_texts, _ = get_train_test_texts()
    n_layers = model_gpt2.config.n_layer
    k_dim = 9

    acts = collect_activations_all_layers(model_gpt2, tokenizer, train_texts, device)
    # Train adapter and return it for transfer
    _ = train_one_layer(model_gpt2, tokenizer, 0, k_dim, k_dim, 64, epochs, 1e-3, device, train_texts, acts)
    
    # Recreate adapter for transfer (same seed, same init)
    proj_k = EigenProjector(768, k_dim).to(device); proj_k.init_from_pca(acts[0]['k'])
    proj_v = EigenProjector(768, k_dim).to(device); proj_v.init_from_pca(acts[0]['v'])
    adapter_k = LowRankAdapter(k=k_dim, hidden=768, bottleneck=64, seed=42).to(device)
    adapter_v = LowRankAdapter(k=k_dim, hidden=768, bottleneck=64, seed=43).to(device)
    adapter_k.set_residual_subspace(proj_k.get_pca_vectors())
    adapter_v.set_residual_subspace(proj_v.get_pca_vectors())
    
    # Train it
    qkv_data = []
    def hook_fn(module, inp, out):
        sz = out.shape[-1] // 3
        qkv_data.append((out[..., :sz].detach(), out[..., sz:2*sz].detach(), out[..., 2*sz:].detach()))
    h = model_gpt2.transformer.h[0].attn.c_attn.register_forward_hook(hook_fn)
    with torch.no_grad():
        for text in train_texts:
            inp = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            _ = model_gpt2(**{k: v.to(device) for k, v in inp.items()})
    h.remove()
    
    num_heads = model_gpt2.config.n_head
    head_dim = model_gpt2.config.n_embd // num_heads
    scale = 1.0 / math.sqrt(head_dim)
    optimizer = torch.optim.Adam(list(adapter_k.parameters()) + list(adapter_v.parameters()), lr=1e-3)
    for epoch in range(epochs):
        for q, k, v in qkv_data:
            optimizer.zero_grad()
            k_comp = proj_k.compress(k); k_pca = proj_k.decompress(k_comp)
            v_comp = proj_v.compress(v); v_pca = proj_v.decompress(v_comp)
            k_ad = adapter_k(k_comp, k_pca); v_ad = adapter_v(v_comp, v_pca)
            orig = compute_attention_output_train(q, k, v, num_heads, scale)
            ada = compute_attention_output_train(q, k_ad, v_ad, num_heads, scale)
            loss = F.mse_loss(ada, orig); loss.backward(); optimizer.step()
    print(f"  GPT-2 adapter trained for transfer")

    # Load DistilGPT-2
    print("\n[2/3] Loading DistilGPT-2...")
    distil_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
    distil_model.eval()
    n_layers_d = distil_model.config.n_layer
    print(f"  DistilGPT-2 layers: {n_layers_d}")

    # Collect DistilGPT-2 activations and train native adapter
    acts_d = collect_activations_all_layers(distil_model, tokenizer, train_texts, device)
    r_native = train_one_layer(distil_model, tokenizer, 0, k_dim, k_dim, 64, epochs, 1e-3, device, train_texts, acts_d)
    if r_native:
        print(f"  DistilGPT-2 Layer 0 (native trained): PCA={r_native['pca_cos']:.4f}  Ada={r_native['ada_cos']:.4f}")

    # Test PCA-only on DistilGPT-2 with GPT-2 PCA basis
    print("\n[3/3] Testing transfer...")
    num_heads = distil_model.config.n_head
    head_dim = distil_model.config.n_embd // num_heads
    scale = 1.0 / math.sqrt(head_dim)

    # Use GPT-2's PCA basis on DistilGPT-2
    proj_k = EigenProjector(768, k_dim).to(device); proj_k.init_from_pca(acts[0]['k'])
    proj_v = EigenProjector(768, k_dim).to(device); proj_v.init_from_pca(acts[0]['v'])

    # Evaluate on DistilGPT-2 with GPT-2's PCA basis
    qkv_data = []
    def hook_fn(module, inp, out):
        sz = out.shape[-1] // 3
        qkv_data.append((out[..., :sz].detach(), out[..., sz:2*sz].detach(), out[..., 2*sz:].detach()))
    h = distil_model.transformer.h[0].attn.c_attn.register_forward_hook(hook_fn)
    with torch.no_grad():
        inp = tokenizer(train_texts[0], return_tensors='pt', truncation=True, max_length=128)
        _ = distil_model(**{k: v.to(device) for k, v in inp.items()})
    h.remove()

    if qkv_data:
        q, k, v = qkv_data[0]
        with torch.no_grad():
            k_comp = proj_k.compress(k); k_pca = proj_k.decompress(k_comp)
            v_comp = proj_v.compress(v); v_pca = proj_v.decompress(v_comp)
            orig = compute_attention_output(q, k, v, num_heads, scale)
            pca_cos = compute_cosine_sim(orig, compute_attention_output(q, k_pca, v_pca, num_heads, scale))
            k_ad = adapter_k(k_comp, k_pca); v_ad = adapter_v(v_comp, v_pca)
            ada_cos = compute_cosine_sim(orig, compute_attention_output(q, k_ad, v_ad, num_heads, scale))

        print(f"\n  DistilGPT-2 with GPT-2 PCA basis: PCA={pca_cos:.4f}")
        print(f"  DistilGPT-2 with GPT-2 adapter (no retrain): Ada={ada_cos:.4f}")
        if r_native:
            print(f"  DistilGPT-2 native trained: Ada={r_native['ada_cos']:.4f}")
            gap = r_native['ada_cos'] - ada_cos
            print(f"  Transfer gap: {gap:+.4f} {'PASS' if abs(gap) < 0.10 else 'FAIL'} (<0.10)")
        print(f"  PCA delta vs adapter: {ada_cos-pca_cos:+.4f}")

    results = {"transfer_ada_cos": float(ada_cos) if qkv_data else None,
               "transfer_pca_cos": float(pca_cos) if qkv_data else None,
               "native_ada_cos": float(r_native['ada_cos']) if r_native else None,
               "native_pca_cos": float(r_native['pca_cos']) if r_native else None}
    json.dump(results, open(OUT_DIR / "sweep_task5.json", 'w'), indent=2)
    return results


if __name__ == '__main__':
    main()
