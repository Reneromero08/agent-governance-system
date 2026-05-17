#!/usr/bin/env python3
"""Eigen-Spectrum Alignment CLI.

Usage:
    eigen-alignment anchors build <input> [--output FILE]
    eigen-alignment signature compute <anchor_set> --model MODEL [--k K] [--output FILE]
    eigen-alignment signature compare <sig1> <sig2> [--threshold TAU]
    eigen-alignment map fit --source SIG --target SIG [--output FILE]
    eigen-alignment map apply --input FILE --map FILE [--output FILE]

Commands:
    anchors build       Build ANCHOR_SET from word list file
    signature compute   Compute SPECTRUM_SIGNATURE for a model
    signature compare   Compare two signatures (Spearman correlation)
    map fit             Fit ALIGNMENT_MAP between two models
    map apply           Apply alignment map to vectors/distances
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def cmd_anchors_build(args):
    """Build ANCHOR_SET from word list."""
    from ..lib.protocol import AnchorSet

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    # Read word list (one word per line)
    with open(input_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]

    if not words:
        print("Error: No words found in input file", file=sys.stderr)
        return 1

    # Create anchor set
    anchor_set = AnchorSet.from_words(words)
    result = anchor_set.to_dict()

    # Output
    output = args.output or f"{input_path.stem}_anchor_set.json"
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f"Created anchor set with {len(words)} anchors")
    print(f"  Hash: {anchor_set.anchor_hash}")
    print(f"  Output: {output}")

    return 0


def cmd_signature_compute(args):
    """Compute SPECTRUM_SIGNATURE for a model."""
    from ..lib import mds
    from ..lib.protocol import spectrum_signature
    from ..lib.adapters import SentenceTransformersAdapter

    # Load anchor set
    anchor_path = Path(args.anchor_set)
    if not anchor_path.exists():
        print(f"Error: Anchor set not found: {anchor_path}", file=sys.stderr)
        return 1

    with open(anchor_path, 'r', encoding='utf-8') as f:
        anchor_data = json.load(f)

    anchors = anchor_data['anchors']
    anchor_hash = anchor_data['anchor_hash']
    words = [a['text'] for a in anchors]

    print(f"Loaded {len(words)} anchors from {anchor_path}")

    # Load model
    model_name = args.model
    print(f"Loading model: {model_name}...")
    adapter = SentenceTransformersAdapter(model_name)

    # Compute embeddings
    print("Computing embeddings...")
    embeddings = adapter.embed(words)

    # Compute MDS
    print("Computing MDS...")
    D2 = mds.squared_distance_matrix(embeddings)
    X, eigenvalues, eigenvectors = mds.classical_mds(D2, k=args.k)

    # Create signature
    k = args.k or len(eigenvalues)
    sig = spectrum_signature(
        eigenvalues=eigenvalues,
        anchor_set_hash=anchor_hash,
        embedder_id=adapter.embedder_id,
        k=k
    )

    result = sig.to_dict()

    # Output
    output = args.output or f"{model_name.replace('/', '_')}_signature.json"
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f"\nSpectrum Signature:")
    print(f"  Model: {adapter.embedder_id}")
    print(f"  k: {k}")
    print(f"  Effective rank: {sig.effective_rank:.2f}")
    print(f"  Top eigenvalues: {eigenvalues[:5].round(4).tolist()}")
    print(f"  Hash: {sig.spectrum_hash}")
    print(f"  Output: {output}")

    return 0


def cmd_signature_compare(args):
    """Compare two spectrum signatures."""
    from scipy.stats import spearmanr, pearsonr

    # Load signatures
    with open(args.sig1, 'r', encoding='utf-8') as f:
        sig1 = json.load(f)
    with open(args.sig2, 'r', encoding='utf-8') as f:
        sig2 = json.load(f)

    ev1 = np.array(sig1['eigenvalues'])
    ev2 = np.array(sig2['eigenvalues'])

    # Truncate to common length
    k = min(len(ev1), len(ev2))
    ev1 = ev1[:k]
    ev2 = ev2[:k]

    # Compute correlations
    spearman_r, spearman_p = spearmanr(ev1, ev2)
    pearson_r, pearson_p = pearsonr(ev1, ev2)

    threshold = args.threshold

    print(f"\nSpectrum Comparison:")
    print(f"  Signature 1: {sig1['embedder_id']}")
    print(f"  Signature 2: {sig2['embedder_id']}")
    print(f"  Common k: {k}")
    print(f"")
    print(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"")
    print(f"  Threshold: {threshold}")

    if spearman_r >= threshold:
        print(f"  Result: PASS (spectra are compatible)")
        return 0
    else:
        print(f"  Result: FAIL (spectra differ too much)")
        return 1


def cmd_map_fit(args):
    """Fit ALIGNMENT_MAP between two models."""
    from ..lib import mds, procrustes
    from ..lib.protocol import alignment_map
    from ..lib.adapters import SentenceTransformersAdapter

    # Load signatures
    with open(args.source, 'r', encoding='utf-8') as f:
        source_sig = json.load(f)
    with open(args.target, 'r', encoding='utf-8') as f:
        target_sig = json.load(f)

    # Verify anchor hashes match
    if source_sig['anchor_set_hash'] != target_sig['anchor_set_hash']:
        print("Error: Anchor set hashes do not match", file=sys.stderr)
        return 1

    # Load anchor set to get words
    anchor_hash = source_sig['anchor_set_hash']

    # We need to recompute MDS coordinates to get eigenvectors
    # This requires the anchor set file - for now, assume standard anchors
    # In production, would store eigenvectors in signature or reference anchor set

    print("Note: map fit requires recomputing MDS. Use with caution.")
    print("For full implementation, provide --anchor-set argument.")

    # For demo, create dummy alignment map
    k = min(source_sig['k'], target_sig['k'])

    # Create identity rotation as placeholder
    R = np.eye(k).tolist()

    amap = alignment_map(
        rotation=np.array(R),
        source_embedder=source_sig['embedder_id'],
        target_embedder=target_sig['embedder_id'],
        anchor_set_hash=anchor_hash,
        residual=0.0
    )

    result = amap.to_dict()

    # Output
    output = args.output or "alignment_map.json"
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f"\nAlignment Map:")
    print(f"  Source: {amap.source_embedder}")
    print(f"  Target: {amap.target_embedder}")
    print(f"  k: {amap.k}")
    print(f"  Output: {output}")

    return 0


def cmd_map_apply(args):
    """Apply alignment map to vectors/distances."""
    from ..lib import procrustes

    # Load map
    with open(args.map, 'r', encoding='utf-8') as f:
        amap = json.load(f)

    R = np.array(amap['rotation_matrix'])

    # Load input vectors
    input_path = Path(args.input)
    if input_path.suffix == '.npy':
        vectors = np.load(input_path)
    elif input_path.suffix == '.json':
        with open(input_path, 'r', encoding='utf-8') as f:
            vectors = np.array(json.load(f))
    else:
        print(f"Error: Unsupported input format: {input_path.suffix}", file=sys.stderr)
        return 1

    # Apply alignment
    aligned = procrustes.align_points(vectors, R)

    # Output
    output = args.output or f"{input_path.stem}_aligned{input_path.suffix}"
    if output.endswith('.npy'):
        np.save(output, aligned)
    else:
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(aligned.tolist(), f)

    print(f"Aligned {len(vectors)} vectors")
    print(f"  Input shape: {vectors.shape}")
    print(f"  Output shape: {aligned.shape}")
    print(f"  Output: {output}")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog='eigen-alignment',
        description='Eigen-Spectrum Alignment Protocol CLI'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # anchors subcommand
    anchors_parser = subparsers.add_parser('anchors', help='Anchor set operations')
    anchors_sub = anchors_parser.add_subparsers(dest='anchors_cmd')

    # anchors build
    build_parser = anchors_sub.add_parser('build', help='Build anchor set from word list')
    build_parser.add_argument('input', help='Input file (one word per line)')
    build_parser.add_argument('--output', '-o', help='Output JSON file')

    # signature subcommand
    sig_parser = subparsers.add_parser('signature', help='Signature operations')
    sig_sub = sig_parser.add_subparsers(dest='sig_cmd')

    # signature compute
    compute_parser = sig_sub.add_parser('compute', help='Compute spectrum signature')
    compute_parser.add_argument('anchor_set', help='Anchor set JSON file')
    compute_parser.add_argument('--model', '-m', required=True, help='Model name')
    compute_parser.add_argument('--k', '-k', type=int, help='Number of eigenvalues')
    compute_parser.add_argument('--output', '-o', help='Output JSON file')

    # signature compare
    compare_parser = sig_sub.add_parser('compare', help='Compare two signatures')
    compare_parser.add_argument('sig1', help='First signature JSON')
    compare_parser.add_argument('sig2', help='Second signature JSON')
    compare_parser.add_argument('--threshold', '-t', type=float, default=0.95,
                                help='Spearman correlation threshold (default: 0.95)')

    # map subcommand
    map_parser = subparsers.add_parser('map', help='Alignment map operations')
    map_sub = map_parser.add_subparsers(dest='map_cmd')

    # map fit
    fit_parser = map_sub.add_parser('fit', help='Fit alignment map')
    fit_parser.add_argument('--source', '-s', required=True, help='Source signature')
    fit_parser.add_argument('--target', '-t', required=True, help='Target signature')
    fit_parser.add_argument('--output', '-o', help='Output JSON file')

    # map apply
    apply_parser = map_sub.add_parser('apply', help='Apply alignment map')
    apply_parser.add_argument('--input', '-i', required=True, help='Input vectors')
    apply_parser.add_argument('--map', '-m', required=True, help='Alignment map JSON')
    apply_parser.add_argument('--output', '-o', help='Output file')

    args = parser.parse_args()

    if args.command == 'anchors':
        if args.anchors_cmd == 'build':
            return cmd_anchors_build(args)
        else:
            anchors_parser.print_help()
            return 1

    elif args.command == 'signature':
        if args.sig_cmd == 'compute':
            return cmd_signature_compute(args)
        elif args.sig_cmd == 'compare':
            return cmd_signature_compare(args)
        else:
            sig_parser.print_help()
            return 1

    elif args.command == 'map':
        if args.map_cmd == 'fit':
            return cmd_map_fit(args)
        elif args.map_cmd == 'apply':
            return cmd_map_apply(args)
        else:
            map_parser.print_help()
            return 1

    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
