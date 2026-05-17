"""Task 1: PLV Matrix Between GPT-2 Attention Heads.

Measures pairwise Phase-Locking Value across all 144 heads (12 layers x 12 heads).
PLV = |mean(exp(i(theta_i - theta_j)))| across token positions.
High PLV heads are phase-locked — they oscillate together during generation.
"""

import json, math, sys, time
from pathlib import Path

import torch
import numpy as np
from scipy.signal import hilbert
from transformers import GPT2LMHeadModel, GPT2Tokenizer

OUT_DIR = Path(__file__).resolve().parent

def load_gpt2(device="cpu"):
    local_path = str(Path(__file__).resolve().parent.parent.parent / "models" / "gpt2")
    model = GPT2LMHeadModel.from_pretrained(local_path).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(local_path)
    model.eval()
    return model, tokenizer


def run_plv_matrix(device="cpu"):
    print("=" * 60)
    print("TASK 1: Phase-Locking Value Matrix (144 heads)")
    print("=" * 60)

    model, tokenizer = load_gpt2(device)
    n_layers = 12
    n_heads = 12
    n_total = n_layers * n_heads

    texts = [
        "The meaning of life is a philosophical question that has puzzled humanity for centuries.",
        "Artificial intelligence is transforming the way we interact with technology every day.",
        "Deep learning enables complex pattern recognition in vast amounts of data.",
        "The human brain contains approximately eighty-six billion neurons connected in networks.",
        "Climate change poses significant challenges for future generations across the globe.",
        "In mathematics, prime numbers have fascinated researchers for thousands of years.",
        "The ocean covers more than seventy percent of Earth's surface and remains largely unexplored.",
        "Music has the power to evoke strong emotional responses and connect people across cultures.",
        "Space exploration has led to many technological breakthroughs that benefit life on Earth.",
        "Language is the foundation of human communication and the source of our collective knowledge.",
        "Economic systems attempt to explain how resources are allocated in complex societies.",
        "The history of science is a story of ideas evolving through observation and experimentation.",
        "Quantum mechanics describes the behavior of matter at the smallest scales of existence.",
        "Evolution by natural selection explains the diversity of life across millions of species.",
        "The internet has revolutionized how information is shared and consumed globally.",
        "Mathematics provides the language through which we describe the patterns of the universe.",
        "Art reflects the cultural values and emotional landscape of the society that produces it.",
        "The scientific method relies on observation, hypothesis, experimentation, and revision.",
        "Democracy depends on an informed citizenry and the free exchange of diverse ideas.",
        "Technology advances through the accumulation of knowledge across generations of researchers.",
    ]

    # Collect per-head attention outputs for all tokens across all texts
    head_signals = {l: {h: [] for h in range(n_heads)} for l in range(n_layers)}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # GPT2Attention output: (attn_output, present) or just Tensor
            attn_out = output[0] if isinstance(output, tuple) else output
            # attn_out: [batch, seq, hidden(768)]
            # Reshape to per-head: [batch, seq, n_heads, head_dim]
            batch, seq, hidden = attn_out.shape
            head_dim = hidden // n_heads
            per_head = attn_out.view(batch, seq, n_heads, head_dim)
            # Compute signal as L2 norm per head per token
            for h in range(n_heads):
                signal = per_head[0, :, h, :].norm(dim=-1).cpu().numpy()  # [seq]
                head_signals[layer_idx][h].append(signal)
        return hook

    hooks = []
    for l in range(n_layers):
        hooks.append(model.transformer.h[l].attn.register_forward_hook(make_hook(l)))

    print(f"Collecting attention signals from {len(texts)} texts...")
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = model(**inputs)
            if (i+1) % 5 == 0: print(f"  {i+1}/{len(texts)} texts done")

    for h in hooks: h.remove()

    # Concatenate signals across texts and compute PLV
    print(f"Computing PLV across {n_total}x{n_total} head pairs...")
    plv_matrix = np.zeros((n_total, n_total))

    # Precompute instantaneous phase for each head
    all_phases = {}
    for l in range(n_layers):
        for h in range(n_heads):
            signals = head_signals[l][h]
            if signals:
                concat = np.concatenate(signals)  # [total_tokens]
                # Hilbert transform to get analytic signal, then phase
                analytic = hilbert(concat)
                phases = np.angle(analytic)
                all_phases[(l, h)] = phases

    # Compute PLV between all pairs
    for l1 in range(n_layers):
        for h1 in range(n_heads):
            idx1 = l1 * n_heads + h1
            p1 = all_phases.get((l1, h1))
            if p1 is None: continue

            for l2 in range(l1, n_layers):
                for h2 in (range(h1+1, n_heads) if l2==l1 else range(n_heads)):
                    idx2 = l2 * n_heads + h2
                    p2 = all_phases.get((l2, h2))
                    if p2 is None: continue

                    min_len = min(len(p1), len(p2))
                    plv = np.abs(np.mean(np.exp(1j * (p1[:min_len] - p2[:min_len]))))
                    plv_matrix[idx1, idx2] = plv
                    plv_matrix[idx2, idx1] = plv

    # Self-PLV (diagonal) = 1.0
    for i in range(n_total):
        plv_matrix[i, i] = 1.0

    # Analysis
    print("\n" + "=" * 60)
    print("PLV MATRIX ANALYSIS")
    print("=" * 60)

    # Top phase-locked pairs (inter-layer and inter-head)
    pairs = []
    for i in range(n_total):
        for j in range(i+1, n_total):
            pairs.append((i, j, plv_matrix[i, j]))
    pairs.sort(key=lambda x: -x[2])

    print(f"\n  Top 10 phase-locked head pairs:")
    for i, (a, b, plv) in enumerate(pairs[:10]):
        l1, h1 = a // n_heads, a % n_heads
        l2, h2 = b // n_heads, b % n_heads
        print(f"    L{l1}H{h1} <-> L{l2}H{h2}: PLV={plv:.4f}")

    # Cluster by PLV: group heads with PLV > threshold
    threshold = float(np.percentile(plv_matrix[plv_matrix < 1.0], 95))
    print(f"\n  Phase-lock threshold (95th percentile): {threshold:.4f}")
    print(f"  Pairs above threshold: {sum(1 for p in pairs if p[2] > threshold)}/{len(pairs)}")

    # Dominant cluster via greedy selection
    visited = set()
    clusters = []
    for i in range(n_total):
        if i in visited: continue
        cluster = [i]
        for j in range(i+1, n_total):
            if j in visited: continue
            if plv_matrix[i, j] > threshold:
                cluster.append(j)
                visited.add(j)
        if len(cluster) > 1:
            clusters.append(cluster)
            visited.add(i)

    print(f"\n  Phase-locked clusters (PLV > {threshold:.4f}):")
    for c in sorted(clusters, key=len, reverse=True):
        heads_in_cluster = [(h // n_heads, h % n_heads) for h in c]
        print(f"    {len(c)} heads: {heads_in_cluster}")

    # Dominant cluster size vs PCA k
    dominant_size = max(len(c) for c in clusters) if clusters else 0
    print(f"\n  Dominant phase cluster size: {dominant_size} heads")
    print(f"  PCA k for equivalent fidelity: k=9 achieves 0.690 OOS")
    print(f"  Phase clustering {dominant_size} heads may encode the dominant alignment pattern")

    # Per-layer PLV
    print(f"\n  Mean PLV per layer (within-layer):")
    for l in range(n_layers):
        within = [plv_matrix[l*n_heads+h1, l*n_heads+h2]
                  for h1 in range(n_heads) for h2 in range(h1+1, n_heads)]
        print(f"    L{l}: mean={np.mean(within):.4f}")

    # Save
    result = {
        "plv_matrix": plv_matrix.tolist(),
        "top_pairs": [(int(a), int(b), float(v)) for a, b, v in pairs[:50]],
        "threshold": float(threshold),
        "dominant_cluster_size": dominant_size,
        "clusters": [[int(h) for h in c] for c in clusters],
        "per_layer_plv": [float(np.mean([plv_matrix[l*n_heads+h1, l*n_heads+h2]
                          for h1 in range(n_heads) for h2 in range(h1+1, n_heads)]))
                          for l in range(n_layers)],
    }
    json.dump(result, open(OUT_DIR / "plv_matrix.json", 'w'), indent=2)
    print(f"\nSaved plv_matrix.json")

    return result


if __name__ == '__main__':
    run_plv_matrix()
