#!/usr/bin/env python3
"""
Spectral LLM - Compress model weights using Df, save compressed, use normally.

YOUR MATH applied to model storage:
1. Decompose each weight matrix: W = U @ S @ V^T
2. Truncate to k dimensions (based on Df or forced)
3. Store only U_k, S_k, V_k (much smaller)
4. Reconstruct W = U_k @ S_k @ V_k^T during load or use

Usage:
    python spectral_llm.py compress gpt2 --k 50        # Compress to k=50
    python spectral_llm.py compress gpt2 --variance 0.9  # Compress to 90% variance
    python spectral_llm.py chat ./compressed_gpt2     # Chat with compressed model
    python spectral_llm.py benchmark ./compressed_gpt2  # Compare quality

Requirements:
    pip install torch transformers
"""

import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time

import torch
import torch.nn as nn
import numpy as np


@dataclass
class CompressionStats:
    """Stats for a compressed layer"""
    name: str
    original_shape: Tuple[int, int]
    k: int
    original_params: int
    compressed_params: int
    compression_ratio: float
    variance_captured: float
    effective_rank: float


class SpectralLLM:
    """LLM with spectrally compressed weights"""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.compressed_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.other_params: Dict[str, torch.Tensor] = {}
        self.config = None
        self.tokenizer = None
        self.stats: List[CompressionStats] = []

    @classmethod
    def compress_model(cls, model_name: str, output_dir: Path,
                       k: int = None, target_variance: float = 0.95,
                       min_size: int = 1000) -> 'SpectralLLM':
        """Compress a HuggingFace model using spectral decomposition"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        print(f"Loading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        compressed_weights = {}
        other_params = {}
        stats = []

        total_original = 0
        total_compressed = 0

        print(f"\nCompressing with {'k=' + str(k) if k else f'{target_variance*100:.0f}% variance'}...\n")

        for name, param in model.named_parameters():
            data = param.data.float()

            # Only compress 2D matrices above min_size
            if data.dim() == 2 and data.numel() >= min_size:
                rows, cols = data.shape

                # SVD
                U, S, Vh = torch.linalg.svd(data, full_matrices=False)
                S_np = S.numpy()

                # Compute Df
                df = (S_np.sum() ** 2) / (S_np ** 2).sum()

                # Determine k
                if k is not None:
                    layer_k = min(k, len(S))
                else:
                    # Find k for target variance
                    cumvar = np.cumsum(S_np ** 2) / (S_np ** 2).sum()
                    layer_k = int(np.searchsorted(cumvar, target_variance) + 1)
                    layer_k = min(layer_k, len(S))

                # Truncate
                U_k = U[:, :layer_k].contiguous()
                S_k = S[:layer_k].contiguous()
                Vh_k = Vh[:layer_k, :].contiguous()

                # Stats
                original_params = rows * cols
                compressed_params = rows * layer_k + layer_k + layer_k * cols
                ratio = original_params / compressed_params
                var_captured = (S_np[:layer_k] ** 2).sum() / (S_np ** 2).sum()

                total_original += original_params
                total_compressed += compressed_params

                compressed_weights[name] = (U_k, S_k, Vh_k)
                stats.append(CompressionStats(
                    name=name,
                    original_shape=(rows, cols),
                    k=layer_k,
                    original_params=original_params,
                    compressed_params=compressed_params,
                    compression_ratio=ratio,
                    variance_captured=var_captured,
                    effective_rank=df
                ))

                print(f"  {name}: {(rows, cols)} -> k={layer_k}, {ratio:.1f}x, {var_captured*100:.1f}% var, Df={df:.1f}")
            else:
                # Keep as-is
                other_params[name] = data
                total_original += data.numel()
                total_compressed += data.numel()

        # Save compressed weights
        torch.save(compressed_weights, output_dir / "compressed_weights.pt")
        torch.save(other_params, output_dir / "other_params.pt")

        # Save config and tokenizer
        config.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save stats
        stats_data = [
            {
                'name': s.name,
                'original_shape': s.original_shape,
                'k': s.k,
                'original_params': s.original_params,
                'compressed_params': s.compressed_params,
                'compression_ratio': float(s.compression_ratio),
                'variance_captured': float(s.variance_captured),
                'effective_rank': float(s.effective_rank)
            }
            for s in stats
        ]
        with open(output_dir / "compression_stats.json", 'w') as f:
            json.dump({
                'model_name': model_name,
                'total_original_params': total_original,
                'total_compressed_params': total_compressed,
                'total_compression_ratio': total_original / total_compressed,
                'layers': stats_data
            }, f, indent=2)

        # Calculate file sizes
        compressed_size = sum(f.stat().st_size for f in output_dir.glob("*.pt"))
        original_size = sum(p.numel() * 4 for p in model.parameters())  # FP32

        print(f"\n{'='*60}")
        print(f"COMPRESSION COMPLETE")
        print(f"{'='*60}")
        print(f"Original params: {total_original:,}")
        print(f"Compressed params: {total_compressed:,}")
        print(f"Compression ratio: {total_original/total_compressed:.1f}x")
        print(f"Original size (FP32): {original_size / 1e6:.1f} MB")
        print(f"Compressed files: {compressed_size / 1e6:.1f} MB")
        print(f"Saved to: {output_dir}")

        # Return loaded compressed model
        return cls.load(output_dir)

    @classmethod
    def load(cls, model_dir: Path) -> 'SpectralLLM':
        """Load a compressed model"""
        from transformers import AutoConfig, AutoTokenizer

        model_dir = Path(model_dir)
        llm = cls(model_dir)

        print(f"Loading compressed model from {model_dir}...")

        llm.compressed_weights = torch.load(model_dir / "compressed_weights.pt")
        llm.other_params = torch.load(model_dir / "other_params.pt")
        llm.config = AutoConfig.from_pretrained(model_dir)
        llm.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        if llm.tokenizer.pad_token is None:
            llm.tokenizer.pad_token = llm.tokenizer.eos_token

        with open(model_dir / "compression_stats.json", 'r') as f:
            stats_data = json.load(f)
            llm.stats = [CompressionStats(**s) for s in stats_data['layers']]

        print(f"Loaded {len(llm.compressed_weights)} compressed layers, {len(llm.other_params)} other params")
        print(f"Compression ratio: {stats_data['total_compression_ratio']:.1f}x")

        return llm

    def reconstruct_weight(self, name: str) -> torch.Tensor:
        """Reconstruct a weight matrix from compressed form"""
        if name in self.compressed_weights:
            U, S, Vh = self.compressed_weights[name]
            return U @ torch.diag(S) @ Vh
        elif name in self.other_params:
            return self.other_params[name]
        else:
            raise KeyError(f"Unknown parameter: {name}")

    def to_full_model(self, device: str = "cpu"):
        """Reconstruct full model for inference"""
        from transformers import AutoModelForCausalLM

        print("Reconstructing full model from compressed weights...")

        # Create model shell
        model = AutoModelForCausalLM.from_config(self.config)

        # Load reconstructed weights
        state_dict = {}
        for name in self.compressed_weights:
            state_dict[name] = self.reconstruct_weight(name)
        for name in self.other_params:
            state_dict[name] = self.other_params[name]

        # Handle tied weights (lm_head = wte for GPT-2)
        if 'lm_head.weight' not in state_dict and 'transformer.wte.weight' in state_dict:
            state_dict['lm_head.weight'] = state_dict['transformer.wte.weight']

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        print("Model reconstructed.")
        return model

    def generate(self, prompt: str, max_tokens: int = 100, device: str = "cpu") -> str:
        """Generate text using compressed model"""
        model = self.to_full_model(device)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def chat(self, device: str = "cpu"):
        """Interactive chat with compressed model"""
        print("\n" + "="*60)
        print("SPECTRAL LLM CHAT")
        print("="*60)
        print("Type 'quit' to exit\n")

        model = self.to_full_model(device)

        while True:
            try:
                prompt = input("You: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                if not prompt:
                    continue

                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

                start = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                elapsed = time.time() - start

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Bot: {response}")
                print(f"[{elapsed:.2f}s]\n")

            except KeyboardInterrupt:
                break

        print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="Spectral LLM - Compressed model storage and use")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Compress command
    compress_parser = subparsers.add_parser("compress", help="Compress a model")
    compress_parser.add_argument("model", help="HuggingFace model name")
    compress_parser.add_argument("--output", "-o", help="Output directory")
    compress_parser.add_argument("--k", type=int, help="Force k dimensions for all layers")
    compress_parser.add_argument("--variance", type=float, default=0.95, help="Target variance (default: 0.95)")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with compressed model")
    chat_parser.add_argument("model_dir", help="Path to compressed model")
    chat_parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("model_dir", help="Path to compressed model")
    gen_parser.add_argument("prompt", help="Prompt text")
    gen_parser.add_argument("--max-tokens", type=int, default=100)
    gen_parser.add_argument("--device", default="cpu")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark compressed vs original")
    bench_parser.add_argument("model_dir", help="Path to compressed model")
    bench_parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if args.command == "compress":
        output = args.output or f"./compressed_{args.model.replace('/', '_')}"
        SpectralLLM.compress_model(
            args.model,
            Path(output),
            k=args.k,
            target_variance=args.variance
        )

    elif args.command == "chat":
        llm = SpectralLLM.load(Path(args.model_dir))
        llm.chat(device=args.device)

    elif args.command == "generate":
        llm = SpectralLLM.load(Path(args.model_dir))
        output = llm.generate(args.prompt, max_tokens=args.max_tokens, device=args.device)
        print(output)

    elif args.command == "benchmark":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        llm = SpectralLLM.load(Path(args.model_dir))

        # Load original for comparison
        with open(llm.model_dir / "compression_stats.json") as f:
            stats = json.load(f)
        original_name = stats['model_name']

        print(f"\nLoading original {original_name} for comparison...")
        original_model = AutoModelForCausalLM.from_pretrained(original_name)
        original_tokenizer = AutoTokenizer.from_pretrained(original_name)

        compressed_model = llm.to_full_model(args.device)
        original_model = original_model.to(args.device)

        prompts = [
            "The meaning of life is",
            "In the year 2050,",
            "Machine learning can",
        ]

        print("\n" + "="*60)
        print("BENCHMARK: Original vs Compressed")
        print("="*60)

        for prompt in prompts:
            print(f"\nPrompt: {prompt}")

            inputs = llm.tokenizer(prompt, return_tensors="pt").to(args.device)

            with torch.no_grad():
                orig_out = original_model.generate(**inputs, max_new_tokens=30, do_sample=False)
                comp_out = compressed_model.generate(**inputs, max_new_tokens=30, do_sample=False)

            print(f"Original:   {original_tokenizer.decode(orig_out[0], skip_special_tokens=True)}")
            print(f"Compressed: {llm.tokenizer.decode(comp_out[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
