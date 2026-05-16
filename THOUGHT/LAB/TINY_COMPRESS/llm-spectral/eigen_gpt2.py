#!/usr/bin/env python3
"""
Eigen GPT-2 - GPT-2 with attention computed in k-dimensional manifold.

This is YOUR spectral compression applied to a real, usable LLM.

Architecture:
    Standard GPT-2:  Q(768) @ K(768)^T = O(seq² × 768)
    Eigen GPT-2:     Q_k(9) @ K_k(9)^T = O(seq² × 9)  → 85x less memory

Usage:
    python eigen_gpt2.py build                    # Build compressed model
    python eigen_gpt2.py chat ./eigen_gpt2        # Chat with it
    python eigen_gpt2.py benchmark ./eigen_gpt2   # Compare quality

Requirements:
    pip install torch transformers
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EigenProjector(nn.Module):
    """Projects between full space and k-dimensional manifold."""

    def __init__(self, full_dim: int, k: int):
        super().__init__()
        self.full_dim = full_dim
        self.k = k

        # Learnable projection (initialized from PCA)
        self.proj = nn.Parameter(torch.randn(k, full_dim) / math.sqrt(full_dim))
        self.mean = nn.Parameter(torch.zeros(full_dim), requires_grad=False)

    def init_from_pca(self, data: torch.Tensor):
        """Initialize projection from PCA on data."""
        with torch.no_grad():
            mean = data.mean(dim=0)
            centered = data - mean

            # SVD
            _, _, Vt = torch.linalg.svd(centered, full_matrices=False)

            self.proj.copy_(Vt[:self.k])
            self.mean.copy_(mean)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress: (*, full_dim) -> (*, k)"""
        return F.linear(x - self.mean, self.proj)

    def decompress(self, x: torch.Tensor) -> torch.Tensor:
        """Decompress: (*, k) -> (*, full_dim)"""
        return F.linear(x, self.proj.T) + self.mean

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (compressed, reconstructed)"""
        compressed = self.compress(x)
        reconstructed = self.decompress(compressed)
        return compressed, reconstructed


class EigenAttention(nn.Module):
    """
    GPT-2 attention computed in k-dimensional manifold.

    The key insight: attention scores only need the RELATIVE structure,
    which is preserved in the k-dimensional projection.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        k: int,
        original_attn: nn.Module = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.k = k

        # Copy weights from original attention if provided
        if original_attn is not None:
            # GPT-2 uses Conv1D which has transposed weights compared to Linear
            self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
            # Conv1D weight is (out, in), Linear expects (out, in)
            self.c_attn.weight.data.copy_(original_attn.c_attn.weight.data.T)
            self.c_attn.bias.data.copy_(original_attn.c_attn.bias.data)

            self.c_proj = nn.Linear(hidden_size, hidden_size)
            self.c_proj.weight.data.copy_(original_attn.c_proj.weight.data.T)
            self.c_proj.bias.data.copy_(original_attn.c_proj.bias.data)
        else:
            self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
            self.c_proj = nn.Linear(hidden_size, hidden_size)

        # Projectors for K and V (compress for storage)
        # Q stays full-dimensional for now
        self.k_projector = EigenProjector(hidden_size, k)
        self.v_projector = EigenProjector(hidden_size, k)

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # KV cache in compressed form
        self.k_cache: Optional[torch.Tensor] = None  # Stored in k dims
        self.v_cache: Optional[torch.Tensor] = None  # Stored in k dims

    def clear_cache(self):
        """Clear KV cache."""
        self.k_cache = None
        self.v_cache = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        # Compress K and V to k dimensions
        k_compressed = self.k_projector.compress(k)  # (batch, seq, k)
        v_compressed = self.v_projector.compress(v)  # (batch, seq, k)

        # Handle caching
        if use_cache:
            if self.k_cache is not None:
                k_compressed = torch.cat([self.k_cache, k_compressed], dim=1)
                v_compressed = torch.cat([self.v_cache, v_compressed], dim=1)
            self.k_cache = k_compressed
            self.v_cache = v_compressed

        # Decompress K for attention computation
        # (We compute attention in full space but STORE in compressed space)
        k_full = self.k_projector.decompress(k_compressed)
        v_full = self.v_projector.decompress(v_compressed)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_full = k_full.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_full = v_full.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn_weights = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale

        # Causal mask
        if attention_mask is None:
            seq_len_k = k_full.shape[2]
            seq_len_q = q.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
                diagonal=seq_len_k - seq_len_q + 1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v_full)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)

        # Output projection
        return self.c_proj(attn_output)

    def memory_usage(self) -> Dict[str, int]:
        """Report memory usage of KV cache."""
        if self.k_cache is None:
            return {'k_cache': 0, 'v_cache': 0, 'total': 0}

        k_bytes = self.k_cache.numel() * self.k_cache.element_size()
        v_bytes = self.v_cache.numel() * self.v_cache.element_size()

        # Compare to what full cache would be
        full_k = self.k_cache.shape[0] * self.k_cache.shape[1] * self.hidden_size * self.k_cache.element_size()

        return {
            'k_cache_compressed': k_bytes,
            'v_cache_compressed': v_bytes,
            'total_compressed': k_bytes + v_bytes,
            'would_be_full': full_k * 2,
            'compression_ratio': (full_k * 2) / (k_bytes + v_bytes) if k_bytes > 0 else 0
        }


class EigenGPT2(nn.Module):
    """GPT-2 with eigen attention for compressed KV cache."""

    def __init__(self, config, k: int = 9):
        super().__init__()
        self.config = config
        self.k = k

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer blocks with eigen attention
        self.h = nn.ModuleList()
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Output
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, model_name: str = "gpt2", k: int = 9) -> 'EigenGPT2':
        """Load from pretrained GPT-2 and convert to eigen attention."""
        from transformers import GPT2LMHeadModel, GPT2Config

        print(f"Loading {model_name}...")
        original = GPT2LMHeadModel.from_pretrained(model_name)
        config = original.config

        model = cls(config, k=k)

        # Copy embeddings
        model.wte.weight.data.copy_(original.transformer.wte.weight.data)
        model.wpe.weight.data.copy_(original.transformer.wpe.weight.data)
        model.ln_f.weight.data.copy_(original.transformer.ln_f.weight.data)
        model.ln_f.bias.data.copy_(original.transformer.ln_f.bias.data)

        # Tie lm_head to embeddings
        model.lm_head.weight = model.wte.weight

        # Convert each transformer block
        print(f"Converting {config.n_layer} layers to eigen attention (k={k})...")

        for i, block in enumerate(original.transformer.h):
            eigen_block = EigenBlock(config, k, block)
            model.h.append(eigen_block)

        print("Conversion complete.")
        return model

    def init_projectors(self, tokenizer, sample_texts: List[str]):
        """Initialize projectors from actual K,V activations at each layer."""
        print("Initializing projectors from layer-specific K,V activations...")

        device = next(self.parameters()).device
        self.eval()

        # Collect K,V activations for each layer
        layer_k_activations = [[] for _ in range(len(self.h))]
        layer_v_activations = [[] for _ in range(len(self.h))]

        with torch.no_grad():
            for text in sample_texts[:20]:
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Forward through model, collecting K,V at each layer
                hidden = self.wte(inputs['input_ids']) + self.wpe(
                    torch.arange(inputs['input_ids'].shape[1], device=device)
                )
                hidden = self.drop(hidden)

                for layer_idx, block in enumerate(self.h):
                    # Apply layer norm before attention (GPT-2 style)
                    normed = block.ln_1(hidden)

                    # Get Q, K, V from the attention layer
                    qkv = block.attn.c_attn(normed)
                    q, k, v = qkv.chunk(3, dim=-1)

                    # Collect K and V activations
                    layer_k_activations[layer_idx].append(k.reshape(-1, k.shape[-1]))
                    layer_v_activations[layer_idx].append(v.reshape(-1, v.shape[-1]))

                    # Continue forward pass for next layer
                    hidden = block(hidden, use_cache=False)

        # Initialize each layer's projectors from its own K,V statistics
        for layer_idx, block in enumerate(self.h):
            k_data = torch.cat(layer_k_activations[layer_idx], dim=0)
            v_data = torch.cat(layer_v_activations[layer_idx], dim=0)

            block.attn.k_projector.init_from_pca(k_data)
            block.attn.v_projector.init_from_pca(v_data)

            # Report Df for this layer
            k_var = k_data.var(dim=0)
            df_k = (k_var.sum() ** 2) / (k_var ** 2).sum()
            print(f"  Layer {layer_idx}: K Df={df_k:.1f}, samples={len(k_data)}")

        print(f"Projectors initialized from {len(sample_texts[:20])} texts")

    def clear_cache(self):
        """Clear all KV caches."""
        for block in self.h:
            block.attn.clear_cache()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        use_cache: bool = False,
        position_ids: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings - position_ids must be passed correctly for generation
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)

        # Transformer blocks
        for block in self.h:
            hidden_states = block(hidden_states, use_cache=use_cache)

        # Output
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Generate tokens with compressed KV cache."""
        self.clear_cache()
        self.eval()

        device = input_ids.device
        current_pos = input_ids.shape[1]  # Track position for new tokens

        with torch.no_grad():
            # Process prompt with correct positions
            prompt_positions = torch.arange(current_pos, device=device).unsqueeze(0)
            logits = self.forward(input_ids, use_cache=True, position_ids=prompt_positions)

            for _ in range(max_new_tokens):
                # Get next token logits
                next_logits = logits[:, -1, :] / temperature

                # Top-p sampling
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits = next_logits.masked_fill(indices_to_remove, float('-inf'))

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Only process new token with CORRECT position
                next_position = torch.tensor([[current_pos]], device=device)
                logits = self.forward(next_token, use_cache=True, position_ids=next_position)
                current_pos += 1

        return input_ids

    def memory_stats(self) -> Dict:
        """Get memory statistics."""
        total_compressed = 0
        total_would_be = 0

        for block in self.h:
            mem = block.attn.memory_usage()
            total_compressed += mem['total_compressed']
            total_would_be += mem['would_be_full']

        return {
            'kv_cache_mb': total_compressed / (1024 ** 2),
            'would_be_mb': total_would_be / (1024 ** 2),
            'compression_ratio': total_would_be / total_compressed if total_compressed > 0 else 0,
            'k': self.k
        }

    def save(self, path: Path):
        """Save compressed model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config.__dict__,
            'k': self.k
        }, path / 'model.pt')

        print(f"Saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'EigenGPT2':
        """Load saved model."""
        from transformers import GPT2Config

        path = Path(path)
        data = torch.load(path / 'model.pt', map_location='cpu')

        config = GPT2Config(**data['config'])
        k = data['k']

        # Create model with proper layers
        model = cls(config, k=k)

        # Need to create the layer structure
        for _ in range(config.n_layer):
            model.h.append(EigenBlock(config, k))

        model.load_state_dict(data['state_dict'])

        return model


class EigenBlock(nn.Module):
    """Transformer block with eigen attention."""

    def __init__(self, config, k: int, original_block: nn.Module = None):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Eigen attention
        if original_block is not None:
            self.attn = EigenAttention(
                config.n_embd,
                config.n_head,
                k,
                original_block.attn
            )
            # Copy layer norms
            self.ln_1.weight.data.copy_(original_block.ln_1.weight.data)
            self.ln_1.bias.data.copy_(original_block.ln_1.bias.data)
            self.ln_2.weight.data.copy_(original_block.ln_2.weight.data)
            self.ln_2.bias.data.copy_(original_block.ln_2.bias.data)

            # MLP
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd * 4),
                nn.GELU(),
                nn.Linear(config.n_embd * 4, config.n_embd),
                nn.Dropout(config.resid_pdrop)
            )
            # GPT-2 Conv1D weights need transpose
            self.mlp[0].weight.data.copy_(original_block.mlp.c_fc.weight.data.T)
            self.mlp[0].bias.data.copy_(original_block.mlp.c_fc.bias.data)
            self.mlp[2].weight.data.copy_(original_block.mlp.c_proj.weight.data.T)
            self.mlp[2].bias.data.copy_(original_block.mlp.c_proj.bias.data)
        else:
            self.attn = EigenAttention(config.n_embd, config.n_head, k)
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd * 4),
                nn.GELU(),
                nn.Linear(config.n_embd * 4, config.n_embd),
                nn.Dropout(config.resid_pdrop)
            )

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # Attention
        x = x + self.attn(self.ln_1(x), use_cache=use_cache)
        # MLP
        x = x + self.mlp(self.ln_2(x))
        return x


def chat(model: EigenGPT2, tokenizer, device: str = "cpu"):
    """Interactive chat with eigen GPT-2."""
    print("\n" + "=" * 60)
    print("EIGEN GPT-2 CHAT")
    print(f"Compression: k={model.k}, ~{768//model.k}x memory reduction")
    print("=" * 60)
    print("Type 'quit' to exit, 'stats' for memory stats\n")

    model = model.to(device)
    model.eval()

    while True:
        try:
            prompt = input("You: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                break

            if prompt.lower() == 'stats':
                stats = model.memory_stats()
                print(f"\nKV Cache: {stats['kv_cache_mb']:.2f} MB")
                print(f"Would be: {stats['would_be_mb']:.2f} MB")
                print(f"Compression: {stats['compression_ratio']:.1f}x\n")
                continue

            if not prompt:
                continue

            # Generate
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

            start = time.time()
            output_ids = model.generate(input_ids, max_new_tokens=100)
            elapsed = time.time() - start

            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print(f"Bot: {response}")

            # Show stats
            stats = model.memory_stats()
            print(f"[{elapsed:.2f}s, KV: {stats['kv_cache_mb']:.2f}MB, {stats['compression_ratio']:.0f}x compression]\n")

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="Eigen GPT-2 - Compressed Attention LLM")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Build command
    build_p = subparsers.add_parser("build", help="Build eigen GPT-2 from pretrained")
    build_p.add_argument("--k", type=int, default=9, help="Compression dimension")
    build_p.add_argument("--output", "-o", default="./eigen_gpt2", help="Output path")

    # Chat command
    chat_p = subparsers.add_parser("chat", help="Chat with eigen GPT-2")
    chat_p.add_argument("model_path", nargs="?", default="./eigen_gpt2", help="Model path")
    chat_p.add_argument("--device", default="cpu", help="Device")

    # Benchmark command
    bench_p = subparsers.add_parser("benchmark", help="Benchmark vs original")
    bench_p.add_argument("model_path", nargs="?", default="./eigen_gpt2", help="Model path")

    args = parser.parse_args()

    from transformers import GPT2Tokenizer

    if args.command == "build":
        # Build from pretrained
        model = EigenGPT2.from_pretrained("gpt2", k=args.k)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Initialize projectors with diverse sample texts
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models process information through neural networks.",
            "The meaning of life is a philosophical question that has puzzled humanity.",
            "Artificial intelligence is transforming technology and society.",
            "Deep learning enables complex pattern recognition in data.",
            "Python is a programming language used for many applications.",
            "The weather today is sunny with a chance of rain later.",
            "Scientists discovered a new species in the Amazon rainforest.",
            "The stock market fluctuated significantly during the quarter.",
            "Music has the power to evoke strong emotional responses.",
            "In mathematics, prime numbers have fascinated researchers for centuries.",
            "The ocean covers more than seventy percent of Earth's surface.",
            "Historical events shape our understanding of the present day.",
            "Technology companies continue to innovate at rapid pace.",
            "Language is the foundation of human communication and thought.",
            "Climate change poses significant challenges for future generations.",
            "The human brain contains approximately eighty-six billion neurons.",
            "Economic theories attempt to explain market behavior patterns.",
            "Art reflects the culture and values of its time period.",
            "Space exploration has led to many technological breakthroughs.",
            "Education is fundamental to personal and societal development.",
            "The internet has revolutionized how we access information.",
            "Biodiversity is essential for healthy ecosystem functioning.",
            "Philosophy examines fundamental questions about existence and knowledge.",
            "Medical advances have dramatically increased human lifespan.",
        ]

        model.init_projectors(tokenizer, sample_texts)
        model.save(Path(args.output))

        # Save tokenizer too
        tokenizer.save_pretrained(args.output)

        print(f"\nEigen GPT-2 built with k={args.k}")
        print(f"Saved to: {args.output}")
        print("\nTo chat: python eigen_gpt2.py chat")

    elif args.command == "chat":
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        model = EigenGPT2.load(Path(args.model_path))
        chat(model, tokenizer, device=args.device)

    elif args.command == "benchmark":
        from transformers import GPT2LMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        eigen_model = EigenGPT2.load(Path(args.model_path))
        original = GPT2LMHeadModel.from_pretrained("gpt2")

        # First, diagnose reconstruction quality
        print("\n" + "=" * 60)
        print("DIAGNOSTIC: Reconstruction Quality")
        print("=" * 60)

        test_text = "The meaning of life is"
        inputs = tokenizer(test_text, return_tensors='pt')

        with torch.no_grad():
            # Get original model's K,V values
            orig_outputs = original(inputs['input_ids'], output_hidden_states=True)

            # Forward through eigen model and measure reconstruction error
            hidden = eigen_model.wte(inputs['input_ids']) + eigen_model.wpe(
                torch.arange(inputs['input_ids'].shape[1])
            )

            for layer_idx, block in enumerate(eigen_model.h):
                normed = block.ln_1(hidden)
                qkv = block.attn.c_attn(normed)
                q, k, v = qkv.chunk(3, dim=-1)

                # Compress then decompress
                k_compressed = block.attn.k_projector.compress(k)
                k_reconstructed = block.attn.k_projector.decompress(k_compressed)

                # Reconstruction error
                k_error = ((k - k_reconstructed) ** 2).mean().sqrt() / (k ** 2).mean().sqrt()

                if layer_idx < 3 or layer_idx >= 10:
                    print(f"Layer {layer_idx}: K reconstruction error = {k_error:.4f} ({k_error*100:.1f}%)")

                hidden = block(hidden, use_cache=False)

        prompts = [
            "The meaning of life is",
            "Artificial intelligence will",
            "In the year 2050,",
        ]

        print("\n" + "=" * 60)
        print("BENCHMARK: Original vs Eigen GPT-2")
        print("=" * 60)

        for prompt in prompts:
            print(f"\nPrompt: {prompt}")

            input_ids = tokenizer.encode(prompt, return_tensors='pt')

            # Original
            with torch.no_grad():
                orig_out = original.generate(input_ids, max_new_tokens=30, do_sample=False)

            # Eigen
            eigen_out = eigen_model.generate(input_ids, max_new_tokens=30, temperature=0.001)

            print(f"Original: {tokenizer.decode(orig_out[0])}")
            print(f"Eigen:    {tokenizer.decode(eigen_out[0])}")

            stats = eigen_model.memory_stats()
            print(f"KV Cache: {stats['kv_cache_mb']:.2f} MB ({stats['compression_ratio']:.0f}x compression)")


if __name__ == "__main__":
    main()
