"""Collect real catalytic hidden states for EigenBuddy training.

Runs the catalytic inference experiment and captures:
- Complex hidden states (real + imag, 7168 bytes = 2x896 f32)
  from the Rust engine's pre-uncompute hidden_state field
- Ground-truth next token from Qwen lm_head projection
- Warm/cold miss flags and per-layer restoration status

Output: torch .pt file with (states_real, states_imag, targets) for training.
"""
import sys
import os
import time
import numpy as np
import torch
import struct
from pathlib import Path
from collections import Counter

CAT_CAS_DIR = Path(__file__).parent
EIGEN_DIR = CAT_CAS_DIR.parent.parent / "EIGEN_BUDDY"
sys.path.insert(0, str(CAT_CAS_DIR))
sys.path.insert(0, str(EIGEN_DIR / "core" / "rust_ffi" / "target" / "release"))
os.chdir(str(EIGEN_DIR / "core" / "rust_ffi" / "target" / "release"))
import catalytic_ffi

from experiment import (
    CatalyticInferenceRuntime, HDDWeightStreamer, TokenizerBridge,
    HIDDEN_DIM, COMPLEX_DIM, F32_DIM, NUM_LAYERS, TAPE_SIZE, TAPE_SIZE_MB,
    TOTAL_WEIGHT_F32, TOTAL_WEIGHT_U8, HDD_MODEL_PATH, TOKENIZER_PATH,
)

COLLECT_DIR = Path(__file__).parent / "collected_hidden_states"
COLLECT_DIR.mkdir(exist_ok=True)


def collect_real_outputs(
    num_tokens: int = 200,
    prompt: str = "The catalytic computing paradigm demonstrates that information can be processed without",
    save_prefix: str = "catalytic_hidden_states",
):
    """Run catalytic inference and collect hidden states + ground-truth tokens."""
    print("=" * 78)
    print("COLLECTING CATALYTIC HIDDEN STATES")
    print("  For EigenBuddy training on real output distributions")
    print("=" * 78)
    print()

    runtime = CatalyticInferenceRuntime()
    try:
        token_ids = runtime.tokenizer.tokenize(prompt)
        print(f"  Prompt: '{prompt[:60]}...' -> {len(token_ids)} tokens")
        print(f"  Collecting {num_tokens} tokens...")
        print()

        hidden_states_real = []
        hidden_states_imag = []
        target_tokens = []
        warm_hits = []
        entropies = []
        first_broken = []

        for step in range(num_tokens):
            current_token = token_ids[-1] if token_ids else 0
            embedding = runtime.tokenizer.embed(current_token)

            tape_bytes = bytes(runtime.tape)
            t0 = time.perf_counter()
            result = catalytic_ffi.catalytic_inference_step(
                tape_bytes, embedding, NUM_LAYERS, runtime.streamer.scrambled_weights, step
            )
            elapsed = time.perf_counter() - t0

            if "working_region" in result:
                runtime.tape[:len(result["working_region"])] = bytearray(result["working_region"])

            # hidden_state from Rust = input_offset after forward (before uncompute).
            # Format: 7168 bytes = 3584 real f32 + 3584 imag f32.
            if "hidden_state" in result:
                hidden_raw = bytes(result["hidden_state"])
            else:
                hidden_raw = bytes(runtime.tape[:COMPLEX_DIM])

            hidden_real = np.frombuffer(hidden_raw[:F32_DIM], dtype=np.float32).copy()
            hidden_imag = np.frombuffer(hidden_raw[F32_DIM:COMPLEX_DIM], dtype=np.float32).copy()
            # XOR'd f32 bytes can decode to NaN/Inf bit patterns. Replace with 0.
            hidden_real = np.nan_to_num(hidden_real, nan=0.0, posinf=0.0, neginf=0.0)
            hidden_imag = np.nan_to_num(hidden_imag, nan=0.0, posinf=0.0, neginf=0.0)

            hidden_states_real.append(hidden_real)
            hidden_states_imag.append(hidden_imag)

            # Ground-truth next token: use engine output (what the fabric actually produces).
            # lm_head on NaN-filled hidden state always returns token 0, creating a
            # feedback loop. The EigenBuddy trains to map hidden_state -> engine_token.
            head_token = result["generated_token"]

            target_tokens.append(head_token)
            warm_hits.append(result.get("warm_hit", False))
            entropies.append(result["total_entropy"])
            first_broken.append(result.get("first_broken_layer", -1))

            token_ids.append(head_token)
            runtime.tokens_generated += 1
            runtime.streamer.bytes_streamed += NUM_LAYERS * TOTAL_WEIGHT_U8

            if step % 20 == 0 or step < 5:
                try:
                    tok_text = runtime.tokenizer.detokenize(head_token) if runtime.tokenizer._qwen_tokenizer else f"[{head_token}]"
                    tok_text = tok_text.encode('ascii', errors='replace').decode('ascii')
                except Exception:
                    tok_text = f"[{head_token}]"
                print(f"  [{step:>4}] tok={head_token:>5} '{tok_text}' warm={result.get('warm_hit',False)} "
                      f"broken={result.get('first_broken_layer',-1)} time={elapsed*1000:.0f}ms "
                      f"real_range=[{hidden_real.min():.2f},{hidden_real.max():.2f}]")

        states_real = torch.from_numpy(np.stack(hidden_states_real))
        states_imag = torch.from_numpy(np.stack(hidden_states_imag))
        targets = torch.tensor(target_tokens, dtype=torch.long)

        print()
        print(f"  Collected: {len(targets)} samples")
        print(f"  Warm hits: {sum(warm_hits)}/{len(warm_hits)} ({sum(warm_hits)/max(len(warm_hits),1)*100:.0f}%)")
        print(f"  Broken layers: {sum(1 for b in first_broken if b >= 0)}")
        print(f"  Real range: [{states_real.min():.4f}, {states_real.max():.4f}] "
              f"mean={states_real.mean():.4f} std={states_real.std():.4f}")
        print(f"  Imag range: [{states_imag.min():.4f}, {states_imag.max():.4f}] "
              f"mean={states_imag.mean():.4f} std={states_imag.std():.4f}")
        print(f"  Target range: [{targets.min().item()}, {targets.max().item()}]")
        print(f"  Unique targets: {len(set(targets.tolist()))}")

        cold_mask = torch.tensor([not w for w in warm_hits])
        n_cold = cold_mask.sum().item()
        print(f"  Cold-miss samples (real compute): {n_cold}")

        # Print cold-miss stats separately
        if n_cold > 0:
            cold_real = states_real[cold_mask]
            cold_imag = states_imag[cold_mask]
            print(f"  Cold real: mean={cold_real.mean():.4f} std={cold_real.std():.4f} "
                  f"range=[{cold_real.min():.4f},{cold_real.max():.4f}]")
            print(f"  Cold imag: mean={cold_imag.mean():.4f} std={cold_imag.std():.4f} "
                  f"range=[{cold_imag.min():.4f},{cold_imag.max():.4f}]")

        save_path = COLLECT_DIR / f"{save_prefix}.pt"
        torch.save({
            'states_real': states_real,
            'states_imag': states_imag,
            'targets': targets,
            'warm_hits': torch.tensor(warm_hits, dtype=torch.bool),
            'entropies': torch.tensor(entropies, dtype=torch.float64),
            'first_broken': torch.tensor(first_broken, dtype=torch.int32),
            'cold_mask': cold_mask,
            'hidden_dim': HIDDEN_DIM,
            'num_tokens': num_tokens,
            'prompt': prompt,
        }, save_path)

        print(f"\n  Saved to {save_path}")
        print(f"  File size: {save_path.stat().st_size / 1024:.0f} KB")

        return save_path

    finally:
        runtime.cleanup()


def quick_statistics(data_path: Path):
    """Print statistics about collected data."""
    data = torch.load(data_path, weights_only=True)

    states_real = data['states_real']
    states_imag = data['states_imag']
    targets = data['targets']
    cold_mask = data['cold_mask']
    entropies = data['entropies']

    print(f"\n{'='*78}")
    print(f"DATA STATISTICS: {data_path.name}")
    print(f"{'='*78}")
    print(f"  Samples: {len(targets)}")
    print(f"  Cold-miss: {cold_mask.sum().item()}/{len(targets)}")
    print(f"  Unique targets: {len(set(targets.tolist()))}")
    print(f"  Real: mean={states_real.mean():.4f} std={states_real.std():.4f}")
    print(f"  Imag: mean={states_imag.mean():.4f} std={states_imag.std():.4f}")
    print(f"  NaN in real: {torch.isnan(states_real).any().item()}")
    print(f"  NaN in imag: {torch.isnan(states_imag).any().item()}")
    print(f"  Entropy: mean={entropies.mean():.2e} std={entropies.std():.2e}")

    # Token diversity
    token_counts = Counter(targets.tolist())
    top = token_counts.most_common(10)
    print(f"  Top targets: {[(t,c) for t,c in top[:8]]}")

    return data


if __name__ == "__main__":
    data_path = collect_real_outputs(num_tokens=500, save_prefix="catalytic_hidden_states_500")
    quick_statistics(data_path)
    print("\nDone -- data ready for EigenBuddy training.")
    print(f"Next: python eigen_buddy_tokenizer.py --data {data_path}")
