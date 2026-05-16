#!/usr/bin/env python3
"""
Spectral Model Compression - Practical Implementation

Downloads a HuggingFace model, applies spectral compression (Df method),
and creates a usable compressed version.

Usage:
    python spectral_compress.py                     # Compress TinyLlama-1.1B
    python spectral_compress.py --model Qwen/Qwen2.5-0.5B
    python spectral_compress.py --analyze-only      # Just show Df, no compression
    python spectral_compress.py --test              # Test compressed model

The Math:
    Weight matrices W have effective rank Df = (sum(sigma))^2 / sum(sigma^2)
    If Df << min(rows, cols), we can compress W = U @ S @ V^T to k=ceil(Df) components
    Compression ratio: (rows * cols) / (rows * k + k + k * cols) ~ min(rows,cols) / k

Requirements:
    pip install torch transformers accelerate
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
import gc

import torch
import numpy as np


@dataclass
class LayerSpectrum:
    """Spectrum analysis for a single layer"""
    name: str
    shape: Tuple[int, int]
    effective_rank: float  # Df
    singular_values: List[float]  # Top 20 only (for storage)
    variance_at_k: Dict[int, float]  # {k: cumulative_variance}
    recommended_k: int  # k for 95% variance
    compression_ratio: float  # original_size / compressed_size at recommended_k


@dataclass
class ModelSpectrum:
    """Full model spectrum analysis"""
    model_name: str
    total_params: int
    total_params_mb: float
    layers: List[LayerSpectrum]
    mean_effective_rank: float
    median_effective_rank: float
    total_compression_ratio: float
    compressed_size_mb: float


class SpectralCompressor:
    """Compress model weights using spectral decomposition"""

    def __init__(self, target_variance: float = 0.95):
        self.target_variance = target_variance
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def analyze_matrix(self, name: str, weight: torch.Tensor) -> Optional[LayerSpectrum]:
        """Analyze a single weight matrix"""
        if weight.dim() != 2:
            return None

        rows, cols = weight.shape
        if min(rows, cols) < 4:  # Skip tiny matrices
            return None

        # Move to CPU for SVD (more stable)
        w = weight.detach().float().cpu()

        try:
            # Compute singular values only (faster than full SVD)
            S = torch.linalg.svdvals(w)
            S = S.numpy()

            # Filter numerical noise
            S = S[S > 1e-10]
            if len(S) == 0:
                return None

            # Effective rank (Df)
            df = (S.sum() ** 2) / (S ** 2).sum()

            # Cumulative variance
            total_var = (S ** 2).sum()
            cumvar = np.cumsum(S ** 2) / total_var

            # Find k for target variance
            k = int(np.searchsorted(cumvar, self.target_variance) + 1)
            k = max(k, int(np.ceil(df)))  # At least ceil(Df)
            k = min(k, len(S))  # Can't exceed rank

            # Compression ratio at k
            original_size = rows * cols
            compressed_size = rows * k + k + k * cols  # U(m,k) + S(k) + V(k,n)
            ratio = original_size / compressed_size

            # Store variance at key k values
            variance_at_k = {}
            for test_k in [1, 2, 3, 5, 10, 20, 50, 100, k]:
                if test_k <= len(cumvar):
                    variance_at_k[test_k] = float(cumvar[test_k - 1])

            return LayerSpectrum(
                name=name,
                shape=(rows, cols),
                effective_rank=float(df),
                singular_values=S[:20].tolist(),  # Top 20 only
                variance_at_k=variance_at_k,
                recommended_k=k,
                compression_ratio=ratio
            )

        except Exception as e:
            print(f"  Warning: Failed to analyze {name}: {e}")
            return None

    def analyze_model(self, model) -> ModelSpectrum:
        """Analyze all weight matrices in a model"""
        print("\nAnalyzing model spectrum...")

        layers = []
        total_original = 0
        total_compressed = 0

        for name, param in model.named_parameters():
            if param.dim() == 2 and param.numel() > 1000:  # Only 2D matrices > 1K params
                spectrum = self.analyze_matrix(name, param)
                if spectrum:
                    layers.append(spectrum)

                    rows, cols = spectrum.shape
                    k = spectrum.recommended_k
                    total_original += rows * cols
                    total_compressed += rows * k + k + k * cols

                    print(f"  {name}: {spectrum.shape} -> Df={spectrum.effective_rank:.1f}, k={k}, {spectrum.compression_ratio:.1f}x")

        # Summary stats
        dfs = [l.effective_rank for l in layers]
        total_params = sum(p.numel() for p in model.parameters())

        return ModelSpectrum(
            model_name=model.config._name_or_path if hasattr(model, 'config') else "unknown",
            total_params=total_params,
            total_params_mb=total_params * 4 / (1024 ** 2),  # FP32
            layers=layers,
            mean_effective_rank=float(np.mean(dfs)) if dfs else 0,
            median_effective_rank=float(np.median(dfs)) if dfs else 0,
            total_compression_ratio=total_original / total_compressed if total_compressed > 0 else 1,
            compressed_size_mb=total_compressed * 4 / (1024 ** 2)
        )

    def compress_matrix(self, weight: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress a weight matrix to rank k using SVD"""
        w = weight.detach().float()

        # Full SVD
        U, S, Vh = torch.linalg.svd(w, full_matrices=False)

        # Truncate to k components
        U_k = U[:, :k]
        S_k = S[:k]
        Vh_k = Vh[:k, :]

        return U_k, S_k, Vh_k

    def decompress_matrix(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor) -> torch.Tensor:
        """Reconstruct weight matrix from compressed form"""
        return U @ torch.diag(S) @ Vh

    def compress_model(self, model, spectrum: ModelSpectrum) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compress all weight matrices in model"""
        print("\nCompressing model weights...")

        compressed = {}

        for layer_spec in spectrum.layers:
            name = layer_spec.name
            k = layer_spec.recommended_k

            # Get the parameter
            parts = name.split('.')
            obj = model
            for part in parts:
                obj = getattr(obj, part)

            weight = obj.data
            U, S, Vh = self.compress_matrix(weight, k)
            compressed[name] = (U, S, Vh)

            print(f"  {name}: {layer_spec.shape} -> k={k}")

        return compressed

    def save_compressed(self, compressed: Dict, spectrum: ModelSpectrum, output_dir: Path):
        """Save compressed model"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save compressed weights
        weights_file = output_dir / "compressed_weights.pt"
        torch.save(compressed, weights_file)

        # Save spectrum (metadata)
        spectrum_file = output_dir / "spectrum.json"
        with open(spectrum_file, 'w') as f:
            # Convert to serializable format
            data = asdict(spectrum)
            json.dump(data, f, indent=2)

        # Calculate sizes
        weights_size = weights_file.stat().st_size / (1024 ** 2)

        print(f"\nSaved to: {output_dir}")
        print(f"  Weights: {weights_size:.1f} MB")
        print(f"  Spectrum: {spectrum_file.stat().st_size / 1024:.1f} KB")

        return output_dir

    def load_compressed(self, model_dir: Path) -> Tuple[Dict, ModelSpectrum]:
        """Load compressed model"""
        weights = torch.load(model_dir / "compressed_weights.pt")

        with open(model_dir / "spectrum.json", 'r') as f:
            data = json.load(f)
            # Reconstruct LayerSpectrum objects
            layers = [LayerSpectrum(**l) for l in data['layers']]
            data['layers'] = layers
            spectrum = ModelSpectrum(**data)

        return weights, spectrum


class CompressedModel:
    """Wrapper for running inference on compressed model"""

    def __init__(self, original_model, compressed_weights: Dict, spectrum: ModelSpectrum):
        self.model = original_model
        self.compressed = compressed_weights
        self.spectrum = spectrum
        self.device = next(original_model.parameters()).device

        # Replace weight matrices with compressed versions
        self._apply_compression()

    def _apply_compression(self):
        """Apply compressed weights to model"""
        for name, (U, S, Vh) in self.compressed.items():
            # Reconstruct weight
            weight = U @ torch.diag(S) @ Vh

            # Find and replace parameter
            parts = name.split('.')
            obj = self.model
            for part in parts[:-1]:
                obj = getattr(obj, part)

            # Set the weight
            param = getattr(obj, parts[-1])
            param.data = weight.to(param.device).to(param.dtype)

    def generate(self, tokenizer, prompt: str, max_tokens: int = 100) -> str:
        """Generate text with compressed model"""
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)


def download_and_analyze(model_name: str, compressor: SpectralCompressor) -> Tuple:
    """Download model and analyze spectrum"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Downloading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # Analyze spectrum
    spectrum = compressor.analyze_model(model)

    return model, tokenizer, spectrum


def main():
    parser = argparse.ArgumentParser(description="Spectral Model Compression")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       help="HuggingFace model to compress")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze spectrum, don't compress")
    parser.add_argument("--variance", type=float, default=0.95,
                       help="Target variance to capture (default: 0.95)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for compressed model")
    parser.add_argument("--test", action="store_true",
                       help="Test the compressed model with generation")

    args = parser.parse_args()

    print("=" * 60)
    print("SPECTRAL MODEL COMPRESSION")
    print("=" * 60)

    compressor = SpectralCompressor(target_variance=args.variance)

    # Download and analyze
    model, tokenizer, spectrum = download_and_analyze(args.model, compressor)

    # Print summary
    print("\n" + "=" * 60)
    print("SPECTRUM SUMMARY")
    print("=" * 60)
    print(f"Model: {spectrum.model_name}")
    print(f"Total params: {spectrum.total_params:,} ({spectrum.total_params_mb:.1f} MB in FP32)")
    print(f"Analyzed layers: {len(spectrum.layers)}")
    print(f"Mean Df: {spectrum.mean_effective_rank:.1f}")
    print(f"Median Df: {spectrum.median_effective_rank:.1f}")
    print(f"Compression ratio: {spectrum.total_compression_ratio:.1f}x")
    print(f"Compressed size: {spectrum.compressed_size_mb:.1f} MB")

    if args.analyze_only:
        print("\n(Analysis only - skipping compression)")
        return

    # Compress
    compressed = compressor.compress_model(model, spectrum)

    # Save
    output_dir = Path(args.output) if args.output else Path(f"./models/{spectrum.model_name.replace('/', '_')}_compressed")
    compressor.save_compressed(compressed, spectrum, output_dir)

    # Test
    if args.test:
        print("\n" + "=" * 60)
        print("TESTING COMPRESSED MODEL")
        print("=" * 60)

        compressed_model = CompressedModel(model, compressed, spectrum)

        test_prompts = [
            "The quick brown fox",
            "What is machine learning?",
            "Write a haiku about compression:",
        ]

        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            response = compressed_model.generate(tokenizer, prompt, max_tokens=50)
            print(f"Response: {response}")

    print("\n" + "=" * 60)
    print("COMPRESSION COMPLETE")
    print("=" * 60)
    print(f"Original: {spectrum.total_params_mb:.1f} MB")
    print(f"Compressed: {spectrum.compressed_size_mb:.1f} MB")
    print(f"Ratio: {spectrum.total_compression_ratio:.1f}x")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
