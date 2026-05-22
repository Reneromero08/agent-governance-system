"""Collect real catalytic hidden states for EigenBuddy training.

Runs the catalytic inference experiment and captures:
- Complex hidden states (real + imag, 2×896 = 1792 dims) from the tape after each token's forward pass
- Ground-truth next token from the Qwen lm_head projection
- Warm/cold miss flags, entropy, and per-layer restoration status

Output: torch .pt file pairs of (hidden_states, target_tokens) for training.
"""
import sys
import os
import time
import hashlib
import struct
import mmap
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

# Path setup
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

# Output directory
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
    
    # Initialize runtime
    runtime = CatalyticInferenceRuntime()
    try:
        token_ids = runtime.tokenizer.tokenize(prompt)
        print(f"  Prompt: '{prompt[:60]}...' -> {len(token_ids)} tokens")
        print(f"  Collecting {num_tokens} tokens...")
        print()
        
        # Storage
        hidden_states_real = []  # real component of hidden state (float32)
        hidden_states_imag = []  # imag component of hidden state
        target_tokens = []       # ground-truth next token from lm_head
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
            
            # Sync working region back from Rust
            if "working_region" in result:
                runtime.tape[:len(result["working_region"])] = bytearray(result["working_region"])
            
            # Capture hidden state from the warm-tape cache slot that was just written.
            # Rust stores hidden state at warm_tape_offset + slot*warm_tape_stride + 4
            # (4-byte hash prefix, then COMPLEX_DIM bytes of XY complex output).
            # The output was stored AFTER the layers ran but BEFORE embedding XOR-out,
            # so it represents the pure model output for this token.
            
            # Compute embedding hash (FNV-1a, same as Rust)
            emb_hash = 2166136261
            for b in embedding:
                emb_hash = (emb_hash ^ b) & 0xFFFFFFFF
                emb_hash = (emb_hash * 16777619) & 0xFFFFFFFF
            emb_hash_bytes = struct.pack('<I', emb_hash)
            
            WARM_TAPE_SLOTS = 256
            warm_tape_stride = 4 + COMPLEX_DIM
            # warm_tape_offset calculation matching Rust layout
            weight_offset = COMPLEX_DIM
            scratch_base = weight_offset + NUM_LAYERS * TOTAL_WEIGHT_F32
            temp_offset = scratch_base
            pre_gate_base = temp_offset + COMPLEX_DIM
            saved_outputs_offset = pre_gate_base + NUM_LAYERS * COMPLEX_DIM
            warm_tape_offset = saved_outputs_offset + NUM_LAYERS * COMPLEX_DIM
            
            # The slot index = hash % WARM_TAPE_SLOTS (Rust uses emb_hash as usize)
            slot = emb_hash % WARM_TAPE_SLOTS
            slot_base = warm_tape_offset + slot * warm_tape_stride
            
            # Verify the hash matches
            stored_hash = runtime.tape[slot_base:slot_base + 4]
            if stored_hash == emb_hash_bytes:
                # Read hidden state: bytes 4..4+COMPLEX_DIM
                output_start = slot_base + 4
                output_raw = bytes(runtime.tape[output_start:output_start + COMPLEX_DIM])
                hidden_real = np.frombuffer(output_raw[:F32_DIM], dtype=np.float32).copy()
                hidden_imag = np.frombuffer(output_raw[F32_DIM:COMPLEX_DIM], dtype=np.float32).copy()
            else:
                # Scan all slots for the right hash
                found = False
                for s in range(WARM_TAPE_SLOTS):
                    sbase = warm_tape_offset + s * warm_tape_stride
                    if runtime.tape[sbase:sbase+4] == emb_hash_bytes:
                        output_raw = bytes(runtime.tape[sbase+4:sbase+4+COMPLEX_DIM])
                        hidden_real = np.frombuffer(output_raw[:F32_DIM], dtype=np.float32).copy()
                        hidden_imag = np.frombuffer(output_raw[F32_DIM:COMPLEX_DIM], dtype=np.float32).copy()
                        found = True
                        break
                if not found:
                    hidden_real = np.zeros(HIDDEN_DIM, dtype=np.float32)
                    hidden_imag = np.zeros(HIDDEN_DIM, dtype=np.float32)
            
            hidden_states_real.append(hidden_real)
            hidden_states_imag.append(hidden_imag)
            
            # Ground-truth token from lm_head projection
            if runtime._real_embedding_np is not None:
                hidden_f32 = hidden_real  # use real component for lm_head
                logits = runtime._real_embedding_np @ hidden_f32
                head_token = int(np.argmax(logits))
            else:
                head_token = result["generated_token"]
            
            target_tokens.append(head_token)
            warm_hits.append(result.get("warm_hit", False))
            entropies.append(result["total_entropy"])
            first_broken.append(result.get("first_broken_layer", -1))
            
            # Advance
            token_ids.append(head_token)
            runtime.tokens_generated += 1
            runtime.streamer.bytes_streamed += NUM_LAYERS * TOTAL_WEIGHT_U8
            
            if step % 20 == 0:
                tok_text = runtime.tokenizer.detokenize(head_token) if runtime.tokenizer._qwen_tokenizer else f"[{head_token}]"
                print(f"  [{step:>4}] tok={head_token:>5} '{tok_text}' warm={result.get('warm_hit',False)} "
                      f"broken={result.get('first_broken_layer',-1)} time={elapsed*1000:.0f}ms")
        
        # Stack into tensors
        states_real = torch.from_numpy(np.stack(hidden_states_real))
        states_imag = torch.from_numpy(np.stack(hidden_states_imag))
        targets = torch.tensor(target_tokens, dtype=torch.long)
        
        print()
        print(f"  Collected: {len(targets)} samples")
        print(f"  Warm hits: {sum(warm_hits)}/{len(warm_hits)} ({sum(warm_hits)/max(len(warm_hits),1)*100:.0f}%)")
        print(f"  Broken layers: {sum(1 for b in first_broken if b >= 0)}")
        print(f"  Real range: [{states_real.min():.4f}, {states_real.max():.4f}]")
        print(f"  Imag range: [{states_imag.min():.4f}, {states_imag.max():.4f}]")
        print(f"  Target range: [{targets.min().item()}, {targets.max().item()}]")
        
        # Split: cold-miss samples are the interesting ones (engine actually computed them)
        cold_mask = torch.tensor([not w for w in warm_hits])
        n_cold = cold_mask.sum().item()
        print(f"  Cold-miss samples (real compute): {n_cold}")
        
        # Save
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
    
    print(f"\n{'='*78}")
    print(f"DATA STATISTICS: {data_path.name}")
    print(f"{'='*78}")
    print(f"  Samples: {len(targets)}")
    print(f"  Cold-miss (real compute): {cold_mask.sum().item()}")
    
    # Per-token diversity
    unique_tokens = len(set(targets.tolist()))
    print(f"  Unique target tokens: {unique_tokens}/{len(targets)} ({unique_tokens/max(len(targets),1)*100:.1f}%)")
    
    # Cold vs warm hidden state statistics
    cold_states_real = states_real[cold_mask]
    cold_states_imag = states_imag[cold_mask]
    warm_states_real = states_real[~cold_mask]
    warm_states_imag = states_imag[~cold_mask]
    
    if len(cold_states_real) > 0:
        cr_mean = cold_states_real.mean().item()
        cr_std = cold_states_real.std().item()
        cr_min = cold_states_real.min().item()
        cr_max = cold_states_real.max().item()
        print(f"  Cold real: mean={cr_mean:.4f} std={cr_std:.4f} range=[{cr_min:.4f}, {cr_max:.4f}]")
        
        ci_mean = cold_states_imag.mean().item()
        ci_std = cold_states_imag.std().item()
        ci_min = cold_states_imag.min().item()
        ci_max = cold_states_imag.max().item()
        print(f"  Cold imag: mean={ci_mean:.4f} std={ci_std:.4f} range=[{ci_min:.4f}, {ci_max:.4f}]")
    
    if len(warm_states_real) > 0:
        wr_mean = warm_states_real.mean().item()
        wr_std = warm_states_real.std().item()
        print(f"  Warm real: mean={wr_mean:.4f} std={wr_std:.4f}")
    
    # Token frequency
    from collections import Counter
    token_counts = Counter(targets.tolist())
    top_tokens = token_counts.most_common(10)
    print(f"  Top targets: {[(t, c) for t, c in top_tokens[:5]]}")
    
    # Entropy distribution
    entropies = data['entropies']
    print(f"  Entropy: μ={entropies.mean():.2e} σ={entropies.std():.2e}")


if __name__ == "__main__":
    # Collect data
    data_path = collect_real_outputs(num_tokens=200)
    
    # Statistics
    quick_statistics(data_path)
    
    print("\nDone — data ready for EigenBuddy training.")
    print(f"Next: python eigen_buddy_tokenizer.py --data {data_path}")
