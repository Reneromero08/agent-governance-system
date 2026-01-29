#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q20 CIRCULAR VALIDATION FIX: Novel Domain Test

PROBLEM IDENTIFIED:
The 8e constant was derived FROM text embedding data, then "validated" ON more
embedding data. This is CIRCULAR VALIDATION and proves nothing.

SOLUTION:
Test 8e on a domain that was NEVER used to derive it:
- GRAPH EMBEDDINGS (node2vec on real networks)

Graph embeddings are:
1. A completely different modality (graph topology, not language)
2. Trained with different objectives (node proximity, not semantic similarity)
3. Have NO connection to text, language, or NLP
4. Use real-world network data (social networks, citations, etc.)

PRE-REGISTERED PREDICTIONS:
- H1 (STRONG): If 8e is universal, graph embeddings should show Df x alpha = 8e (+/- 15%)
- H0 (NULL): If 8e is text-specific, graph embeddings will show Df x alpha far from 8e (> 25% deviation)

FALSIFICATION CRITERIA:
- If deviation < 15%: 8e has cross-modal validity (supports universality)
- If deviation > 25%: 8e is text-specific (supports tautology concern)
- If 15-25%: Inconclusive, needs more investigation

CRITICAL: This is NOT validation. This is a NOVEL PREDICTION test.
The result, whatever it is, will be reported HONESTLY.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Constants
EIGHT_E = 8 * np.e  # ~21.746


def compute_df(eigenvalues: np.ndarray) -> float:
    """Participation ratio Df = (sum(eigenvalues))^2 / sum(eigenvalues^2)"""
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) == 0:
        return 0.0
    return float((np.sum(ev)**2) / np.sum(ev**2))


def compute_alpha(eigenvalues: np.ndarray) -> float:
    """Power law decay exponent alpha where eigenvalue_k ~ k^(-alpha)"""
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) < 10:
        return 0.0

    k = np.arange(1, len(ev) + 1)
    n_fit = len(ev) // 2
    if n_fit < 5:
        return 0.0

    log_k = np.log(k[:n_fit])
    log_ev = np.log(ev[:n_fit])
    slope, _ = np.polyfit(log_k, log_ev, 1)
    return float(-slope)


def get_eigenspectrum(embeddings: np.ndarray) -> np.ndarray:
    """Get eigenvalues from covariance matrix."""
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return np.maximum(eigenvalues, 1e-10)


def analyze_embeddings(embeddings: np.ndarray, name: str) -> Dict[str, Any]:
    """Compute Df, alpha, and Df x alpha for embeddings."""
    eigenvalues = get_eigenspectrum(embeddings)
    Df = compute_df(eigenvalues)
    alpha = compute_alpha(eigenvalues)
    df_alpha = Df * alpha
    vs_8e = abs(df_alpha - EIGHT_E) / EIGHT_E * 100

    return {
        'name': name,
        'shape': list(embeddings.shape),
        'Df': float(Df),
        'alpha': float(alpha),
        'Df_alpha': float(df_alpha),
        'vs_8e_percent': float(vs_8e),
        '8e_target': float(EIGHT_E),
    }


def generate_karate_graph_embeddings(dim: int = 64) -> Tuple[np.ndarray, str]:
    """
    Generate node2vec-style embeddings for Zachary's Karate Club graph.

    This is a real-world social network from 1977 showing friendships
    between members of a university karate club. It's a classic benchmark
    for graph embedding algorithms.

    If node2vec is not available, we use a spectral embedding approach
    which captures similar structural information.
    """
    # Zachary's Karate Club adjacency matrix (34 nodes)
    # This is REAL DATA from a 1977 sociological study
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
        (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
        (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
        (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
        (3, 7), (3, 12), (3, 13), (4, 6), (4, 10), (5, 6), (5, 10), (5, 16),
        (6, 16), (8, 30), (8, 32), (8, 33), (9, 33), (13, 33), (14, 32), (14, 33),
        (15, 32), (15, 33), (18, 32), (18, 33), (19, 33), (20, 32), (20, 33),
        (22, 32), (22, 33), (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
        (24, 25), (24, 27), (24, 31), (25, 31), (26, 29), (26, 33), (27, 33),
        (28, 31), (28, 33), (29, 32), (29, 33), (30, 32), (30, 33), (31, 32),
        (31, 33), (32, 33)
    ]

    n_nodes = 34
    adj = np.zeros((n_nodes, n_nodes))
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1

    try:
        # Try node2vec if available
        import networkx as nx
        from node2vec import Node2Vec

        G = nx.from_numpy_array(adj)
        node2vec = Node2Vec(G, dimensions=dim, walk_length=30, num_walks=200, workers=1, quiet=True)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        embeddings = np.array([model.wv[str(i)] for i in range(n_nodes)])
        method = "node2vec"

    except ImportError:
        # Fallback to spectral embedding (also captures graph structure)
        # This is a valid graph embedding method used in spectral clustering
        D = np.diag(adj.sum(axis=1))
        L = D - adj  # Laplacian
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(adj.sum(axis=1), 1e-10)))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt  # Normalized Laplacian

        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

        # Use smallest non-trivial eigenvectors as embedding
        # (the Fiedler vector approach)
        n_components = min(dim, n_nodes - 1)
        embeddings = eigenvectors[:, 1:n_components+1]  # Skip first (constant)

        # Pad to desired dimension if needed
        if embeddings.shape[1] < dim:
            padding = np.zeros((n_nodes, dim - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, padding])

        method = "spectral_laplacian"

    return embeddings, f"Karate Club ({method})"


def generate_citation_graph_embeddings(dim: int = 64) -> Tuple[np.ndarray, str]:
    """
    Generate embeddings for a citation-like network.

    Uses a synthetic citation network with realistic structure:
    - Power-law degree distribution (rich-get-richer)
    - Preferential attachment (Barabasi-Albert model)
    """
    np.random.seed(42)

    # Generate Barabasi-Albert graph (scale-free network)
    n_nodes = 100
    m = 3  # New nodes attach to m existing nodes

    # Start with a small complete graph
    adj = np.zeros((n_nodes, n_nodes))
    for i in range(m):
        for j in range(i+1, m):
            adj[i, j] = 1
            adj[j, i] = 1

    # Add nodes one by one with preferential attachment
    for new_node in range(m, n_nodes):
        degrees = adj[:new_node, :new_node].sum(axis=1) + 1  # +1 to avoid zero
        probs = degrees / degrees.sum()
        targets = np.random.choice(new_node, size=min(m, new_node), replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = 1
            adj[t, new_node] = 1

    # Spectral embedding of this graph
    D = np.diag(adj.sum(axis=1))
    L = D - adj
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(adj.sum(axis=1), 1e-10)))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
    n_components = min(dim, n_nodes - 1)
    embeddings = eigenvectors[:, 1:n_components+1]

    if embeddings.shape[1] < dim:
        padding = np.zeros((n_nodes, dim - embeddings.shape[1]))
        embeddings = np.hstack([embeddings, padding])

    return embeddings, "Citation Network (BA spectral)"


def generate_random_graph_embeddings(dim: int = 64) -> Tuple[np.ndarray, str]:
    """
    Generate embeddings for an Erdos-Renyi random graph.

    This is a NEGATIVE CONTROL - random graphs have no learned structure.
    If random graphs ALSO show 8e, then 8e is meaningless.
    """
    np.random.seed(123)

    n_nodes = 100
    p = 0.1  # Edge probability

    # Generate random adjacency matrix
    adj = np.random.rand(n_nodes, n_nodes) < p
    adj = adj | adj.T  # Make symmetric
    np.fill_diagonal(adj, 0)  # No self-loops
    adj = adj.astype(float)

    # Spectral embedding
    D = np.diag(adj.sum(axis=1))
    L = D - adj
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(adj.sum(axis=1), 1e-10)))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
    n_components = min(dim, n_nodes - 1)
    embeddings = eigenvectors[:, 1:n_components+1]

    if embeddings.shape[1] < dim:
        padding = np.zeros((n_nodes, dim - embeddings.shape[1]))
        embeddings = np.hstack([embeddings, padding])

    return embeddings, "Random Graph (ER spectral)"


def test_audio_embeddings() -> Optional[Dict[str, Any]]:
    """
    Test audio embeddings using wav2vec or CLAP if available.

    Audio embeddings are trained on speech/music, completely different
    from text embeddings.
    """
    print("\n[AUDIO EMBEDDINGS TEST]")

    try:
        # Try to use a small wav2vec or audio model
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        import torch

        # Generate synthetic audio for testing (sine waves at different frequencies)
        # This represents "audio data" even without real recordings
        sample_rate = 16000
        n_samples = 50
        duration = 1.0  # 1 second clips

        audio_data = []
        for i in range(n_samples):
            freq = 200 + i * 50  # Different frequencies
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
            audio_data.append(audio)

        # Load wav2vec
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        model.eval()

        embeddings = []
        for audio in audio_data:
            inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            # Mean pool over time
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)

        embeddings = np.array(embeddings)
        result = analyze_embeddings(embeddings, "wav2vec2-base")
        result['domain'] = 'audio'
        result['model'] = 'facebook/wav2vec2-base-960h'
        result['method'] = 'wav2vec2'

        return result

    except Exception as e:
        print(f"  Audio embeddings not available: {e}")
        return None


def test_image_embeddings() -> Optional[Dict[str, Any]]:
    """
    Test pure image embeddings using DINOv2 if available.

    DINOv2 is trained on images only (no text), making it a novel domain
    compared to CLIP which was tested before.
    """
    print("\n[IMAGE EMBEDDINGS TEST]")

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from PIL import Image

        # Generate synthetic images (geometric patterns)
        n_images = 50
        size = 224

        images = []
        for i in range(n_images):
            # Create different geometric patterns
            img = np.zeros((size, size, 3), dtype=np.uint8)

            pattern_type = i % 5
            if pattern_type == 0:  # Circles
                center = (size//2 + i*2 % 50, size//2 + i*3 % 50)
                for y in range(size):
                    for x in range(size):
                        if (x - center[0])**2 + (y - center[1])**2 < (30 + i)**2:
                            img[y, x] = [255, i*5 % 256, 100]
            elif pattern_type == 1:  # Gradients
                for y in range(size):
                    for x in range(size):
                        img[y, x] = [x % 256, y % 256, (x + y + i*10) % 256]
            elif pattern_type == 2:  # Stripes
                for y in range(size):
                    for x in range(size):
                        if (x + i*5) % 20 < 10:
                            img[y, x] = [200, 100, 50]
            elif pattern_type == 3:  # Checkerboard
                for y in range(size):
                    for x in range(size):
                        if ((x // 20) + (y // 20) + i) % 2 == 0:
                            img[y, x] = [50, 150, 200]
            else:  # Noise with structure
                base = np.random.randint(0, 100, (size, size, 3), dtype=np.uint8)
                img = base + i * 3

            images.append(Image.fromarray(img.astype(np.uint8)))

        # Load DINOv2
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        model = AutoModel.from_pretrained("facebook/dinov2-small")
        model.eval()

        embeddings = []
        for img in images:
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(embedding)

        embeddings = np.array(embeddings)
        result = analyze_embeddings(embeddings, "DINOv2-small")
        result['domain'] = 'image'
        result['model'] = 'facebook/dinov2-small'
        result['method'] = 'dinov2'

        return result

    except Exception as e:
        print(f"  Image embeddings not available: {e}")
        return None


def main():
    print("=" * 70)
    print("Q20 NOVEL DOMAIN TEST: Breaking Circular Validation")
    print("=" * 70)
    print()
    print("PROBLEM: 8e was derived from text embeddings, then 'validated' on text.")
    print("         This is CIRCULAR and proves nothing.")
    print()
    print("SOLUTION: Test 8e on domains NEVER used to derive it.")
    print()
    print("DOMAINS TO TEST:")
    print("  1. Graph embeddings (node2vec/spectral) - NOVEL")
    print("  2. Audio embeddings (wav2vec) - NOVEL if available")
    print("  3. Image embeddings (DINOv2) - NOVEL (CLIP was used, DINOv2 was not)")
    print()
    print("PREDICTIONS:")
    print(f"  If 8e is universal: Df x alpha = {EIGHT_E:.2f} (+/- 15%)")
    print("  If 8e is text-specific: Df x alpha will deviate > 25%")
    print()
    print("=" * 70)

    results = {
        'timestamp': datetime.now().isoformat(),
        'test': 'Q20_NOVEL_DOMAIN_CIRCULAR_VALIDATION_FIX',
        'hypothesis': 'Test if 8e holds in domains never used to derive it',
        '8e_value': float(EIGHT_E),
        'domains': []
    }

    # Test 1: Graph embeddings (Karate Club)
    print("\n[1] GRAPH EMBEDDINGS - Zachary's Karate Club (1977)")
    print("-" * 50)
    try:
        embeddings, name = generate_karate_graph_embeddings(dim=32)
        result = analyze_embeddings(embeddings, name)
        result['domain'] = 'graph'
        result['network'] = 'karate_club'
        result['n_nodes'] = embeddings.shape[0]
        results['domains'].append(result)

        print(f"  Network: {name}")
        print(f"  Nodes: {result['n_nodes']}, Embedding dim: {result['shape'][1]}")
        print(f"  Df = {result['Df']:.4f}")
        print(f"  alpha = {result['alpha']:.4f}")
        print(f"  Df x alpha = {result['Df_alpha']:.4f}")
        print(f"  Error vs 8e: {result['vs_8e_percent']:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Test 2: Graph embeddings (Citation/BA network)
    print("\n[2] GRAPH EMBEDDINGS - Citation Network (Barabasi-Albert)")
    print("-" * 50)
    try:
        embeddings, name = generate_citation_graph_embeddings(dim=64)
        result = analyze_embeddings(embeddings, name)
        result['domain'] = 'graph'
        result['network'] = 'citation_ba'
        result['n_nodes'] = embeddings.shape[0]
        results['domains'].append(result)

        print(f"  Network: {name}")
        print(f"  Nodes: {result['n_nodes']}, Embedding dim: {result['shape'][1]}")
        print(f"  Df = {result['Df']:.4f}")
        print(f"  alpha = {result['alpha']:.4f}")
        print(f"  Df x alpha = {result['Df_alpha']:.4f}")
        print(f"  Error vs 8e: {result['vs_8e_percent']:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Test 3: Random graph (NEGATIVE CONTROL)
    print("\n[3] NEGATIVE CONTROL - Random Graph (Erdos-Renyi)")
    print("-" * 50)
    try:
        embeddings, name = generate_random_graph_embeddings(dim=64)
        result = analyze_embeddings(embeddings, name)
        result['domain'] = 'graph'
        result['network'] = 'random_er'
        result['n_nodes'] = embeddings.shape[0]
        result['is_negative_control'] = True
        results['domains'].append(result)

        print(f"  Network: {name}")
        print(f"  Nodes: {result['n_nodes']}, Embedding dim: {result['shape'][1]}")
        print(f"  Df = {result['Df']:.4f}")
        print(f"  alpha = {result['alpha']:.4f}")
        print(f"  Df x alpha = {result['Df_alpha']:.4f}")
        print(f"  Error vs 8e: {result['vs_8e_percent']:.2f}%")
        print("  (Random should NOT show 8e)")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Test 4: Audio embeddings (if available)
    audio_result = test_audio_embeddings()
    if audio_result:
        results['domains'].append(audio_result)
        print(f"  Df x alpha = {audio_result['Df_alpha']:.4f}")
        print(f"  Error vs 8e: {audio_result['vs_8e_percent']:.2f}%")

    # Test 5: Image embeddings (if available)
    image_result = test_image_embeddings()
    if image_result:
        results['domains'].append(image_result)
        print(f"  Df x alpha = {image_result['Df_alpha']:.4f}")
        print(f"  Error vs 8e: {image_result['vs_8e_percent']:.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: NOVEL DOMAIN 8e TEST")
    print("=" * 70)

    novel_domains = [d for d in results['domains'] if not d.get('is_negative_control', False)]
    control_domains = [d for d in results['domains'] if d.get('is_negative_control', False)]

    print(f"\n{'Domain':<35} {'Df':<10} {'alpha':<10} {'Df*a':<10} {'vs 8e':<10} {'Status'}")
    print("-" * 85)

    for d in results['domains']:
        error = d['vs_8e_percent']
        if d.get('is_negative_control', False):
            status = "CTRL-PASS" if error > 20 else "CTRL-FAIL"
        elif error < 15:
            status = "8e FOUND"
        elif error < 25:
            status = "UNCLEAR"
        else:
            status = "NO 8e"

        print(f"{d['name']:<35} {d['Df']:<10.2f} {d['alpha']:<10.4f} {d['Df_alpha']:<10.2f} {error:<10.2f}% {status}")

    # Compute verdict
    if novel_domains:
        mean_error = np.mean([d['vs_8e_percent'] for d in novel_domains])
        mean_df_alpha = np.mean([d['Df_alpha'] for d in novel_domains])

        print(f"\nNOVEL DOMAINS (n={len(novel_domains)}):")
        print(f"  Mean Df x alpha: {mean_df_alpha:.2f}")
        print(f"  Mean error vs 8e: {mean_error:.2f}%")

        if mean_error < 15:
            verdict = "8e_CONFIRMED_NOVEL_DOMAIN"
            explanation = "8e appears in novel domains never used to derive it. This REFUTES the circular validation concern."
        elif mean_error < 25:
            verdict = "INCONCLUSIVE"
            explanation = "Results are ambiguous. More novel domain testing needed."
        else:
            verdict = "8e_NOT_UNIVERSAL"
            explanation = "8e does NOT appear in novel domains. The tautology concern is CONFIRMED - 8e may be text-specific."
    else:
        verdict = "NO_DATA"
        explanation = "No novel domain tests succeeded."

    if control_domains:
        control_error = control_domains[0]['vs_8e_percent']
        if control_error > 20:
            print(f"\nNEGATIVE CONTROL: PASSED (error = {control_error:.2f}%)")
            print("  Random graphs do NOT show 8e - confirming it's not a mathematical artifact.")
        else:
            print(f"\nNEGATIVE CONTROL: FAILED (error = {control_error:.2f}%)")
            print("  WARNING: Random graphs ALSO show 8e! This would mean 8e is meaningless.")
            verdict = "INVALID_NEGATIVE_CONTROL_FAILED"
            explanation = "Random graphs show 8e, which invalidates all 8e claims."

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")
    print(f"\n{explanation}")

    results['summary'] = {
        'novel_domain_count': len(novel_domains),
        'mean_df_alpha': float(mean_df_alpha) if novel_domains else None,
        'mean_error_percent': float(mean_error) if novel_domains else None,
        'verdict': verdict,
        'explanation': explanation
    }

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f'q20_novel_domain_{timestamp}.json'

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {output_path}")

    return results


if __name__ == '__main__':
    main()
