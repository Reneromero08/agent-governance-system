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
    parser.add_argument("--task", type=int, choices=[1,2,3,4,5,6,7,8], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.task == 1: run_task1(args.device, args.epochs)
    elif args.task == 2: run_task2(args.device, args.epochs)
    elif args.task == 3: run_task3(args.device, args.epochs)
    elif args.task == 4: run_task4(args.device, args.epochs)
    elif args.task == 5: run_task5(args.device, args.epochs)
    elif args.task == 6: run_task6(args.device, args.epochs)
    elif args.task == 7: run_task7(args.device, args.epochs)
    elif args.task == 8: run_task8(args.device, args.epochs)


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


# ---- Task 6: Joint K+V Adapter ----
class JointAdapter(nn.Module):
    """Single adapter: [K_comp, V_comp] -> K_correction, V_correction."""

    def __init__(self, k_K, k_V, hidden=768, bottleneck=64, seed=42):
        super().__init__()
        total_in = k_K + k_V
        rng = torch.Generator().manual_seed(seed)
        self.W1 = nn.Parameter(torch.randn(bottleneck, total_in, generator=rng) / math.sqrt(total_in))
        self.W2 = nn.Parameter(torch.randn(hidden * 2, bottleneck, generator=rng) / math.sqrt(bottleneck))
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, z_k, z_v):
        z = torch.cat([z_k, z_v], dim=-1)
        h = F.gelu(F.linear(z, self.W1))
        raw = F.linear(h, self.W2)
        return raw[..., :768] * self.alpha, raw[..., 768:] * self.alpha


def run_task6(device="cpu", epochs=10):
    """Joint K+V adapter."""
    print("=" * 60)
    print("TASK 6: Joint K+V Adapter (out-of-sample)")
    print("=" * 60)
    model, tokenizer = load_gpt2(device)
    train_texts, test_texts = get_train_test_texts()
    n_layers = model.config.n_layer
    num_heads = model.config.n_head; hd = model.config.n_embd // num_heads; sc = 1.0 / math.sqrt(hd)
    acts = collect_activations_all_layers(model, tokenizer, train_texts + test_texts, device)

    for k_dim in [9, 3]:
        comp = 768 / k_dim
        joint_p = (64*(k_dim*2) + 1536*64 + 1) // 1000
        sep_p = ((64*k_dim + 768*64 + 1) * 2) // 1000
        print(f"\n--- k={k_dim} ({comp:.1f}x) joint={joint_p}K vs sep={sep_p}K params ---")
        layer_results = []
        for l in range(n_layers):
            t0 = time.time()
            proj_k = EigenProjector(768, k_dim).to(device); proj_k.init_from_pca(acts[l]['k'])
            proj_v = EigenProjector(768, k_dim).to(device); proj_v.init_from_pca(acts[l]['v'])
            adapter = JointAdapter(k_dim, k_dim, 768, 64).to(device)
            opt = torch.optim.Adam(adapter.parameters(), lr=1e-3)

            tr_qkv = []
            def hk(m, i, o):
                sz = o.shape[-1]//3; tr_qkv.append((o[...,:sz].detach(), o[...,sz:2*sz].detach(), o[...,2*sz:].detach()))
            hh = model.transformer.h[l].attn.c_attn.register_forward_hook(hk)
            with torch.no_grad():
                for t in train_texts:
                    inp = tokenizer(t, return_tensors='pt', truncation=True, max_length=128)
                    _ = model(**{k: v.to(device) for k,v in inp.items()})
            hh.remove()

            for _ in range(epochs):
                for q, k, v in tr_qkv:
                    opt.zero_grad()
                    kc = proj_k.compress(k); vc = proj_v.compress(v)
                    kp = proj_k.decompress(kc); vp = proj_v.decompress(vc)
                    ck, cv = adapter(kc, vc)
                    o = compute_attention_output_train(q, k, v, num_heads, sc)
                    a = compute_attention_output_train(q, kp+ck, vp+cv, num_heads, sc)
                    F.mse_loss(a, o).backward(); opt.step()

            te_qkv = []
            def hk2(m, i, o):
                sz = o.shape[-1]//3; te_qkv.append((o[...,:sz].detach(), o[...,sz:2*sz].detach(), o[...,2*sz:].detach()))
            hh2 = model.transformer.h[l].attn.c_attn.register_forward_hook(hk2)
            with torch.no_grad():
                for t in test_texts:
                    inp = tokenizer(t, return_tensors='pt', truncation=True, max_length=128)
                    _ = model(**{k: v.to(device) for k,v in inp.items()})
            hh2.remove()

            pc, ac = [], []
            for q, k, v in te_qkv:
                with torch.no_grad():
                    kc = proj_k.compress(k); vc = proj_v.compress(v)
                    kp = proj_k.decompress(kc); vp = proj_v.decompress(vc)
                    ck, cv = adapter(kc, vc)
                    o = compute_attention_output(q, k, v, num_heads, sc)
                    pc.append(compute_cosine_sim(o, compute_attention_output(q, kp, vp, num_heads, sc)))
                    ac.append(compute_cosine_sim(o, compute_attention_output(q, kp+ck, vp+cv, num_heads, sc)))
            r = {'pca_cos': float(np.mean(pc)), 'ada_cos': float(np.mean(ac))}
            print(f"  L{l:2d}: PCA={r['pca_cos']:.4f}  Joint={r['ada_cos']:.4f}  dt={time.time()-t0:.1f}s")
            layer_results.append(r)
        avg_p = np.mean([rr['pca_cos'] for rr in layer_results])
        avg_j = np.mean([rr['ada_cos'] for rr in layer_results])
        print(f"  AVG: PCA={avg_p:.4f}  Joint={avg_j:.4f}")
        print(f"  Compare Task 1 separate: k=9 Ada=0.752, k=3 Ada=0.694")

    print(f"\nTask 6: Joint adapter. Compare to separate adapter results from Task 1.")
    json.dump({}, open(OUT_DIR / "sweep_task6.json", 'w'), indent=2)
    return None


# ---- Task 7: Warm-Start Init ----
def run_task7(device="cpu", epochs=10):
    """Warm-start adapter from PCA residual mean."""
    print("=" * 60)
    print("TASK 7: Warm-Start vs Random Init (out-of-sample, 3 layers)")
    print("=" * 60)
    model, tokenizer = load_gpt2(device)
    train_texts, test_texts = get_train_test_texts()
    num_heads = model.config.n_head; hd = model.config.n_embd // num_heads; sc = 1.0 / math.sqrt(hd)
    acts = collect_activations_all_layers(model, tokenizer, train_texts + test_texts, device)

    for k_dim in [9, 3]:
        print(f"\n--- k={k_dim} ---")
        for l in range(3):
            t0 = time.time()
            proj_k = EigenProjector(768, k_dim).to(device); proj_k.init_from_pca(acts[l]['k'])
            proj_v = EigenProjector(768, k_dim).to(device); proj_v.init_from_pca(acts[l]['v'])

            all_qkv = []
            def hk(m, i, o):
                sz = o.shape[-1]//3; all_qkv.append((o[...,:sz].detach(), o[...,sz:2*sz].detach(), o[...,2*sz:].detach()))
            hh = model.transformer.h[l].attn.c_attn.register_forward_hook(hk)
            with torch.no_grad():
                for t in train_texts + test_texts:
                    inp = tokenizer(t, return_tensors='pt', truncation=True, max_length=128)
                    _ = model(**{k: v.to(device) for k,v in inp.items()})
            hh.remove()
            tr = all_qkv[:len(train_texts)]; te = all_qkv[len(train_texts):]

            for label, warm in [("random", False), ("warm", True)]:
                ak = LowRankAdapter(k=k_dim, hidden=768, bottleneck=64, seed=42).to(device)
                av = LowRankAdapter(k=k_dim, hidden=768, bottleneck=64, seed=43).to(device)
                ak.set_residual_subspace(proj_k.get_pca_vectors())
                av.set_residual_subspace(proj_v.get_pca_vectors())
                if warm:
                    ak.alpha.data.fill_(0.5); av.alpha.data.fill_(0.5)
                    ak.W1.data.zero_(); ak.W2.data.zero_(); av.W1.data.zero_(); av.W2.data.zero_()
                    # Add tiny noise to W2 so GELU gradients can flow
                    ak.W2.data.normal_(0, 0.01 / math.sqrt(64))
                    av.W2.data.normal_(0, 0.01 / math.sqrt(64))

                opt = torch.optim.Adam(list(ak.parameters())+list(av.parameters()), lr=1e-3)
                for _ in range(epochs):
                    for q, kv, vv in tr:
                        opt.zero_grad()
                        kc = proj_k.compress(kv); kp = proj_k.decompress(kc)
                        vc = proj_v.compress(vv); vp = proj_v.decompress(vc)
                        ka = ak(kc, kp); va = av(vc, vp)
                        o = compute_attention_output_train(q, kv, vv, num_heads, sc)
                        a = compute_attention_output_train(q, ka, va, num_heads, sc)
                        F.mse_loss(a, o).backward(); opt.step()

                pc, ac = [], []
                for q, kv, vv in te:
                    with torch.no_grad():
                        kc = proj_k.compress(kv); kp = proj_k.decompress(kc)
                        vc = proj_v.compress(vv); vp = proj_v.decompress(vc)
                        ka = ak(kc, kp); va = av(vc, vp)
                        o = compute_attention_output(q, kv, vv, num_heads, sc)
                        pc.append(compute_cosine_sim(o, compute_attention_output(q, kp, vp, num_heads, sc)))
                        ac.append(compute_cosine_sim(o, compute_attention_output(q, ka, va, num_heads, sc)))
                avg_p = float(np.mean(pc)); avg_a = float(np.mean(ac))
                print(f"  L{l} {label:6s}: PCA={avg_p:.4f} Ada={avg_a:.4f} delta={avg_a-avg_p:+.4f}")

    print(f"\nTask 7: warm-start zeros weights vs random. Compare deltas.")
    json.dump({}, open(OUT_DIR / "sweep_task7.json", 'w'), indent=2)
    return None


# ---- Task 8: Direct Decoder (No PCA) ----
class DirectDecoder(nn.Module):
    def __init__(self, k, hidden=768, bottleneck=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, bottleneck), nn.GELU(),
            nn.Linear(bottleneck, bottleneck), nn.GELU(),
            nn.Linear(bottleneck, hidden))

    def forward(self, z): return self.net(z)


def run_task8(device="cpu", epochs=10):
    """Direct decoder: no PCA basis needed at inference time."""
    print("=" * 60)
    print("TASK 8: Direct Decoder — Bypass PCA (out-of-sample)")
    print("=" * 60)
    model, tokenizer = load_gpt2(device)
    train_texts, test_texts = get_train_test_texts()
    n_layers = model.config.n_layer
    num_heads = model.config.n_head; hd = model.config.n_embd // num_heads; sc = 1.0 / math.sqrt(hd)

    for k_dim in [9, 3]:
        comp = 768 / k_dim
        dp = (k_dim*128 + 128*128 + 128*768) * 2 // 1000
        print(f"\n--- k={k_dim} ({comp:.1f}x) decoder={dp}K params/layer ---")
        layer_results = []
        for l in range(n_layers):
            t0 = time.time()
            all_kv = []
            def hk(m, i, o):
                sz = o.shape[-1]//3; all_kv.append((o[...,sz:2*sz].detach(), o[...,2*sz:].detach()))
            hh = model.transformer.h[l].attn.c_attn.register_forward_hook(hk)
            with torch.no_grad():
                for t in train_texts + test_texts:
                    inp = tokenizer(t, return_tensors='pt', truncation=True, max_length=128)
                    _ = model(**{k: v.to(device) for k,v in inp.items()})
            hh.remove()
            tr_kv = all_kv[:len(train_texts)]; te_kv = all_kv[len(train_texts):]

            acts_l = collect_activations_all_layers(model, tokenizer, train_texts, device)
            pk = EigenProjector(768, k_dim).to(device); pk.init_from_pca(acts_l[l]['k'])
            pv = EigenProjector(768, k_dim).to(device); pv.init_from_pca(acts_l[l]['v'])

            test_qkv = []
            def hk2(m, i, o):
                sz = o.shape[-1]//3; test_qkv.append((o[...,:sz].detach(), o[...,sz:2*sz].detach(), o[...,2*sz:].detach()))
            hh2 = model.transformer.h[l].attn.c_attn.register_forward_hook(hk2)
            with torch.no_grad():
                for t in test_texts:
                    inp = tokenizer(t, return_tensors='pt', truncation=True, max_length=128)
                    _ = model(**{k: v.to(device) for k,v in inp.items()})
            hh2.remove()

            dk = DirectDecoder(k_dim, 768, 128).to(device)
            dv = DirectDecoder(k_dim, 768, 128).to(device)
            opt = torch.optim.Adam(list(dk.parameters())+list(dv.parameters()), lr=1e-3)
            for _ in range(epochs):
                for kv, vv in tr_kv:
                    opt.zero_grad()
                    kc = pk.compress(kv); vc = pv.compress(vv)
                    kr = dk(kc); vr = dv(vc)
                    F.mse_loss(kr, kv.reshape(-1,768)).backward(retain_graph=True)
                    F.mse_loss(vr, vv.reshape(-1,768)).backward()
                    opt.step()

            pc, dc = [], []
            for q, kv, vv in test_qkv:
                with torch.no_grad():
                    kc = pk.compress(kv); vc = pv.compress(vv)
                    kr = dk(kc); vr = dv(vc)
                    kp = pk.decompress(kc); vp = pv.decompress(vc)
                    o = compute_attention_output(q, kv, vv, num_heads, sc)
                    pc.append(compute_cosine_sim(o, compute_attention_output(q, kp, vp, num_heads, sc)))
                    dc.append(compute_cosine_sim(o, compute_attention_output(q, kr, vr, num_heads, sc)))
            r = {'pca_cos': float(np.mean(pc)), 'dec_cos': float(np.mean(dc))}
            print(f"  L{l:2d}: PCA={r['pca_cos']:.4f}  Dec={r['dec_cos']:.4f}  dt={time.time()-t0:.1f}s")
            layer_results.append(r)
        avg_p = np.mean([rr['pca_cos'] for rr in layer_results])
        avg_d = np.mean([rr['dec_cos'] for rr in layer_results])
        print(f"  AVG: PCA={avg_p:.4f}  Decoder={avg_d:.4f}")
        print(f"  Compare: PCA+adapter k=9 Ada=0.752, k=3 Ada=0.694 (Task 1)")

    json.dump({}, open(OUT_DIR / "sweep_task8.json", 'w'), indent=2)
    return None


if __name__ == '__main__':
    main()
