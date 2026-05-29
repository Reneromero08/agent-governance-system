# Agent Summary - GGUF Backend Integrated into Phase 3.5

**Updated:** 2026-05-18
**Location:** `extensions/03_flat_llm/` (v4 Phase 3.5)

## Structure

```
extensions/03_flat_llm/
  gguf_backend.py         -- LFM2.5 + Qwen3.6 GGUF CUDA backend (llama-cpp-python JamePeng fork)
  flat_llm_adapter.py     -- gguf-demo + qwen-demo subcommands
  train_adapter.py        -- unchanged (HF GPT-2 path for per-layer K/V training)
  REPORT.md               -- Low-rank adapter benchmark results (GPT-2)
  benchmark_results.json, train_results.json, ...  -- existing results
```

## What Was Done

### GGUF Backend (Phase 3.5)

1. **`gguf_backend.py`** -- Self-contained module with CUDA bootstrap. Supports two models:
   - **LFM2.5** (1.2B, Q8_0, 0.0.3.23 abetlen compatible): 16 layers, full GPU offload
   - **Qwen3.6-35B-A3B-MTP** (35B MoE, Q4_K_XL, JamePeng 0.3.39): 41 layers, 18/42 GPU offload

   API:
   - `generate()`, `chat()` -- text/chat completion (reuses instance, no reload needed)
   - `get_logits()` -- full vocabulary logits `(n_tokens, n_vocab)`
   - `get_embedding()` -- per-token embeddings `(n_tokens, n_embd)`

2. **`flat_llm_adapter.py`** -- Added `gguf-demo` (LFM2.5) and `qwen-demo` (Qwen3.6)
   subcommands demonstrating GGUF inference.

3. **JamePeng llama-cpp-python v0.3.39** -- Fork supporting `qwen35moe` arch.
   Known C-level bug: `eval()` pollutes model state -- can't call after `create_completion`.
   Workaround: Use fresh Llama instance for `eval()`-based ops (logits, embeddings).

4. **CUDA bootstrap** -- cuBLAS/cudart DLLs from Ollama `cuda_v12`; fake root at
   `%TEMP%/_cuda_v12_fake`. No nvcc or full CUDA toolkit needed.

## Architecture Detail (Qwen3.6)
- 35B parameters (A3B active per token), 262K context, `qwen35moe` arch
- 41 transformer layers, 256 experts (8 active), hybrid SSM + Attention
- 18 GPU layers: ~11.8 GB VRAM, ~3 tok/s generation
- Requires `KMP_DUPLICATE_LIB_OK=TRUE` when creating multiple Llama instances in one process

## Remaining
- Per-layer K/V hidden states: need HF safetensors or llama.cpp C API
- GGUF-backed adapter training blocked without per-layer signal access

## Usage
```bash
python flat_llm_adapter.py gguf-demo           # LFM2.5 demo
python flat_llm_adapter.py qwen-demo            # Qwen3.6 35B MoE demo
python flat_llm_adapter.py benchmark            # GPT-2 PCA benchmark
python train_adapter.py --k 9 --epochs 20       # GPT-2 adapter training
```
