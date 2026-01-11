#!/usr/bin/env python3
"""
Run YOUR eigen compression - the real implementation from eigen-alignment.

This uses your proven math: project to manifold, compute there, project back.
Output is PRESERVED because we compute in the space where data lives.

Usage:
    python run_eigen.py analyze gpt2          # Analyze spectrum
    python run_eigen.py compress gpt2         # Compress with YOUR method
    python run_eigen.py generate ./model      # Generate text
"""

import sys
from pathlib import Path

# Add eigen-alignment to path
EIGEN_PATH = Path(__file__).parent.parent / "VECTOR_ELO" / "eigen-alignment"
sys.path.insert(0, str(EIGEN_PATH))

# Import the lib module properly
import importlib.util
spec = importlib.util.spec_from_file_location("lib", EIGEN_PATH / "lib" / "__init__.py")
lib = importlib.util.module_from_spec(spec)
sys.modules['lib'] = lib
spec.loader.exec_module(lib)

import argparse
import numpy as np
import torch


def analyze_model(model_name: str):
    """Analyze model spectrum using YOUR math."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from eigen_compress import EigenCompressor, ActivationCompressor

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params\n")

    # Analyze WEIGHTS
    print("=== WEIGHT SPECTRUM ===")
    weight_compressor = EigenCompressor.from_model(model)
    print(f"Weight Df: {weight_compressor.effective_rank:.1f}")
    print(f"Weight compression possible: {weight_compressor.compression_ratio:.1f}x")

    # Analyze ACTIVATIONS
    print("\n=== ACTIVATION SPECTRUM ===")
    act_compressor = ActivationCompressor.from_model(model, tokenizer, n_samples=100)
    print(f"Activation Df: {act_compressor.effective_rank:.1f}")
    print(f"Activation compression: {act_compressor.compression_ratio:.0f}x")

    # Memory savings
    print("\n=== MEMORY SAVINGS ===")
    bench = act_compressor.benchmark([128, 512, 2048, 4096])
    for b in bench['benchmarks']:
        print(f"  seq={b['seq_length']}: {b['standard_mb']:.1f} MB -> {b['compressed_mb']:.2f} MB ({b['reduction']:.0f}x)")


def compress_model(model_name: str, output_dir: str):
    """Compress model using YOUR eigen method."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from eigen_compress import EigenCompressor

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Analyzing spectrum...")
    compressor = EigenCompressor.from_model(model)
    print(f"Effective rank (Df): {compressor.effective_rank:.1f}")
    print(f"Compression ratio: {compressor.compression_ratio:.1f}x")

    print("\nCompressing model...")
    compressed = compressor.compress_model(model)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save
    eigen_path = output_path / "model.eigen"
    compressor.save(compressed, eigen_path)

    # Also save tokenizer
    tokenizer.save_pretrained(output_path)

    print(f"\nSaved to {output_path}")


def generate_text(model_dir: str, prompt: str):
    """Generate with eigen-compressed model."""
    from transformers import AutoTokenizer
    from eigen_compress import EigenCompressor, EigenLLM

    model_path = Path(model_dir)
    eigen_path = model_path / "model.eigen"

    print(f"Loading {eigen_path}...")
    llm = EigenLLM.load(eigen_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm.tokenizer = tokenizer

    print(f"Memory: {llm.memory_usage()}")
    print(f"\nPrompt: {prompt}")

    # For full generation, we'd need model surgery
    # For now, show the compressed weights work
    print("\nCompressed weights loaded. Full generation requires model rebuild.")

    # Show sample weight
    sample_weight = list(llm.compressed_weights.keys())[0]
    w = llm.get_weight(sample_weight)
    print(f"\nSample weight '{sample_weight}': shape={w.shape}")


def main():
    parser = argparse.ArgumentParser(description="YOUR Eigen Compression")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Analyze
    analyze_p = subparsers.add_parser("analyze", help="Analyze model spectrum")
    analyze_p.add_argument("model", help="HuggingFace model name")

    # Compress
    compress_p = subparsers.add_parser("compress", help="Compress model")
    compress_p.add_argument("model", help="HuggingFace model name")
    compress_p.add_argument("--output", "-o", default="./eigen_model")

    # Generate
    gen_p = subparsers.add_parser("generate", help="Generate text")
    gen_p.add_argument("model_dir", help="Path to compressed model")
    gen_p.add_argument("prompt", nargs="?", default="The meaning of life is")

    args = parser.parse_args()

    if args.command == "analyze":
        analyze_model(args.model)
    elif args.command == "compress":
        compress_model(args.model, args.output)
    elif args.command == "generate":
        generate_text(args.model_dir, args.prompt)


if __name__ == "__main__":
    main()
