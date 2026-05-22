"""Generate gold-standard training data: catalytic hidden states + Qwen gold tokens.

Runs Qwen 0.5B in torch (fast, 100+ tok/s) to get gold next-token predictions.
Teacher-forces the catalytic engine with the SAME gold tokens as input, collecting
hidden states. Output: (catalytic_hidden_state, qwen_gold_next_token) pairs for
EigenBuddy oracle training.

Architecture (20.2 oracle+verifier pattern):
  Qwen(torch) → gold_tokens (fast, oracle)
  Catalytic engine → hidden_states (slow, verifier)
  EigenBuddy learns: catalytic_XOR'd_state → qwen_gold_token
"""
import sys
import os
import time
import json
import struct
import mmap
import hashlib
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

CAT_CAS_DIR = Path(__file__).parent
EIGEN_DIR = CAT_CAS_DIR.parent.parent / "EIGEN_BUDDY"
sys.path.insert(0, str(CAT_CAS_DIR))
sys.path.insert(0, str(EIGEN_DIR / "core" / "rust_ffi" / "target" / "release"))
os.chdir(str(EIGEN_DIR / "core" / "rust_ffi" / "target" / "release"))
import catalytic_ffi

from experiment import (
    CatalyticInferenceRuntime, HDDWeightStreamer,
    HIDDEN_DIM, COMPLEX_DIM, F32_DIM, NUM_LAYERS,
    TOTAL_WEIGHT_F32, TOTAL_WEIGHT_U8,
    HDD_MODEL_PATH, TOKENIZER_PATH,
)

MODEL_DIR = CAT_CAS_DIR / "gemini_update" / "qwen_0.5b"
GOLD_DIR = CAT_CAS_DIR / "gold_training_data"
GOLD_DIR.mkdir(exist_ok=True)

PROMPTS = [
    "The catalytic computing paradigm demonstrates that",
    "Artificial intelligence research has shown that",
    "The fundamental laws of physics suggest that",
    "Recent advances in quantum computing indicate that",
    "The relationship between information theory and thermodynamics reveals that",
    "A comprehensive analysis of the data shows that",
    "The most important discovery in computer science is that",
    "When we examine the mathematical foundations of",
    "The key insight that emerged from decades of research is that",
    "Scientists have long hypothesized that the nature of",
]


def load_qwen_model():
    """Load Qwen 0.5B in torch for fast gold-token generation."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("  Loading Qwen tokenizer...")
        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR), local_files_only=True, trust_remote_code=True
        )
        print(f"  Tokenizer loaded: {time.perf_counter()-t0:.1f}s")
        t0 = time.perf_counter()
        print("  Loading Qwen model to CUDA (float16)...")
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR), local_files_only=True, dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        model.eval()
        print(f"  Model loaded: {time.perf_counter()-t0:.1f}s (device: {model.device})")
        return model, tokenizer
    except Exception as e:
        print(f"  [WARN] Could not load Qwen: {e}")
        print(f"  Falling back to engine-only mode")
        return None, None


def generate_gold_sequence(model, tokenizer, prompt: str, max_tokens: int = 50):
    """Generate gold-standard token sequence using Qwen in torch."""
    if model is None:
        return None
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    full_tokens = outputs[0].tolist()
    gold_tokens = full_tokens[prompt_len:]  # just the generated part
    return gold_tokens


def collect_gold_pairs(
    model, tokenizer,
    num_sequences: int = 10,
    tokens_per_seq: int = 50,
    save_prefix: str = "gold_pairs",
):
    """Run catalytic engine with Qwen gold tokens as teacher forcing input.
    All sequences feed through ONE catalytic runtime to maximize warm cache.
    Collect (catalytic_hidden_state, qwen_gold_next_token) pairs.
    """
    print("=" * 78)
    print("GOLD DATA GENERATOR: Oracle + Verifier Training")
    print("  Qwen (torch) = Oracle (100+ tok/s)")
    print("  Catalytic Engine = Verifier (collects hidden states)")
    print("=" * 78)
    print()

    has_qwen = model is not None

    all_hidden_real = []
    all_hidden_imag = []
    all_gold_tokens = []
    timings = []

    # 1. Generate ALL gold tokens from Qwen first (fast, CUDA)
    all_prompts = []
    all_gold_sequences = []
    print("--- Phase 1: Qwen Oracle (gold tokens) ---")
    t_phase1 = time.perf_counter()
    for seq_idx in range(num_sequences):
        prompt = PROMPTS[seq_idx % len(PROMPTS)]
        if has_qwen:
            gold_tokens = generate_gold_sequence(model, tokenizer, prompt, tokens_per_seq)
        else:
            gold_tokens = None

        if gold_tokens is None:
            gold_tokens = [0] * tokens_per_seq

        all_prompts.append(prompt)
        all_gold_sequences.append(gold_tokens)
        gold_text = ""
        if tokenizer:
            try:
                gold_text = tokenizer.decode(gold_tokens[:10])
                gold_text = gold_text.encode('ascii', errors='replace').decode('ascii')[:60]
            except:
                pass
        print(f"  [{seq_idx+1}] '{prompt[:40]}...' -> {len(gold_tokens)} tokens: '{gold_text}...'")
    print(f"  Oracle time: {time.perf_counter()-t_phase1:.1f}s")
    print()

    # 2. Teacher-force catalytic engine through all gold tokens in ONE continuous run
    print("--- Phase 2: Catalytic Verifier (hidden states) ---")
    t_phase2 = time.perf_counter()
    runtime = CatalyticInferenceRuntime()
    try:
        runtime.tokenizer._real_embedding_table = runtime.streamer.embedding_table
        runtime._real_embedding_np = runtime.streamer.embedding_np

        total_tokens = 0
        total_warm = 0
        total_cold = 0

        for seq_idx, (prompt, gold_tokens) in enumerate(zip(all_prompts, all_gold_sequences)):
            # Tokenize prompt
            prompt_ids = runtime.tokenizer.tokenize(prompt)
            all_input_ids = prompt_ids + gold_tokens

            for i, token_id in enumerate(all_input_ids):
                embedding = runtime.tokenizer.embed(token_id)
                tape_bytes = bytes(runtime.tape)
                t0 = time.perf_counter()
                result = catalytic_ffi.catalytic_inference_step(
                    tape_bytes, embedding, NUM_LAYERS,
                    runtime.streamer.scrambled_weights, total_tokens
                )
                elapsed = time.perf_counter() - t0

                if "working_region" in result:
                    runtime.tape[:len(result["working_region"])] = bytearray(result["working_region"])

                is_warm = result.get("warm_hit", False)
                if is_warm:
                    total_warm += 1
                else:
                    total_cold += 1

                # Capture hidden state
                if "hidden_state" in result:
                    hidden_raw = bytes(result["hidden_state"])
                else:
                    hidden_raw = bytes(runtime.tape[:COMPLEX_DIM])

                hidden_real = np.frombuffer(hidden_raw[:F32_DIM], dtype=np.float32).copy()
                hidden_imag = np.frombuffer(hidden_raw[F32_DIM:COMPLEX_DIM], dtype=np.float32).copy()
                hidden_real = np.nan_to_num(hidden_real, nan=0.0, posinf=0.0, neginf=0.0)
                hidden_imag = np.nan_to_num(hidden_imag, nan=0.0, posinf=0.0, neginf=0.0)

                all_hidden_real.append(hidden_real)
                all_hidden_imag.append(hidden_imag)

                # Gold target: next token in the Qwen-generated sequence
                if i + 1 < len(all_input_ids):
                    gold_next = all_input_ids[i + 1]
                else:
                    gold_next = token_id
                all_gold_tokens.append(gold_next)
                timings.append(elapsed)
                total_tokens += 1

                if total_tokens <= 5 or total_tokens % 30 == 0:
                    label = "WARM" if is_warm else "COLD"
                    gold_text = f"[{gold_next}]"
                    if tokenizer:
                        try:
                            gold_text = tokenizer.decode([gold_next])
                            gold_text = gold_text.encode('ascii', errors='replace').decode('ascii')
                        except:
                            pass
                    print(f"    [{total_tokens:>4}] tok={token_id:>5} gold_next={gold_next:>5} "
                          f"'{gold_text}' {label} {elapsed*1000:.0f}ms "
                          f"(total: warm={total_warm} cold={total_cold})")

        t_phase2 = time.perf_counter() - t_phase2
        print(f"\n  Phase 2 time: {t_phase2:.1f}s ({total_warm} warm, {total_cold} cold)")

    finally:
        runtime.cleanup()

    # Stack into tensors
    states_real = torch.from_numpy(np.stack(all_hidden_real))
    states_imag = torch.from_numpy(np.stack(all_hidden_imag))
    targets = torch.tensor(all_gold_tokens, dtype=torch.long)

    N = len(targets)
    unique_tokens = len(set(targets.tolist()))

    print(f"\n{'='*78}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*78}")
    print(f"  Total pairs: {N}")
    print(f"  Unique gold tokens: {unique_tokens}")
    print(f"  Total warm: {total_warm}/{N} ({total_warm/max(N,1)*100:.0f}%)")
    print(f"  Total cold: {total_cold}/{N} ({total_cold/max(N,1)*100:.0f}%)")
    print(f"  Avg time/pair: {np.mean(timings)*1000:.0f}ms")
    print(f"  Real range: [{states_real.min():.4e}, {states_real.max():.4e}]")
    print(f"  Imag range: [{states_imag.min():.4e}, {states_imag.max():.4e}]")
    print(f"  NaN in real: {torch.isnan(states_real).any().item()}")
    print(f"  NaN in imag: {torch.isnan(states_imag).any().item()}")

    # Save
    save_path = GOLD_DIR / f"{save_prefix}.pt"
    torch.save({
        'states_real': states_real,
        'states_imag': states_imag,
        'targets': targets,
        'hidden_dim': HIDDEN_DIM,
        'num_pairs': N,
        'unique_tokens': unique_tokens,
    }, save_path)
    print(f"\n  Saved to {save_path}")
    print(f"  Size: {save_path.stat().st_size / 1024:.0f} KB")

    return save_path


if __name__ == "__main__":
    model, tokenizer = load_qwen_model()
    data_path = collect_gold_pairs(model, tokenizer, num_sequences=2, tokens_per_seq=30)
    print(f"\nNext: python eigen_buddy_tokenizer.py --data {data_path} --oracle")
