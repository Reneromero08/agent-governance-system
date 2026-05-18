# Agent Summary - GGUF Backend Integrated into Phase 3.5

**Updated:** 2026-05-18
**Location:** `extensions/03_flat_llm/` (v4 Phase 3.5)

## Structure

```
extensions/03_flat_llm/
  gguf_backend.py         -- NEW: LFM2.5 GGUF + CUDA backend (llama-cpp-python)
  flat_llm_adapter.py     -- UPDATED: added gguf-demo subcommand
  train_adapter.py        -- unchanged (HF GPT-2 path for per-layer K/V training)
  REPORT.md               -- UPDATED: includes GGUF backend integration section
  benchmark_results.json, train_results.json, ...  -- existing results
```

## What Was Done

1. **`gguf_backend.py`** -- Self-contained module with CUDA bootstrap, loads LFM2.5
   GGUF Q8_0 via llama-cpp-python with full GPU offload. Provides:
   - `generate()`, `chat()` -- text/chat completion
   - `get_logits()` -- full vocabulary logits `(1, 65536)`
   - `get_embedding()` -- per-token embeddings `(n_tokens, 2048)`

2. **`flat_llm_adapter.py`** -- Added `gguf-demo` subcommand demonstrating the
   GGUF backend alongside the existing HF GPT-2 benchmark pipeline.

3. **CUDA bootstrap** -- cuBLAS/cudart DLLs sourced from Ollama's `cuda_v12`
   bundle; fake CUDA root at `%TEMP%/_cuda_v12_fake`. No nvcc or full CUDA
   toolkit needed at runtime.

## Architecture Detail
- LFM2.5: 16 layers, 6 attn layers (2,5,8,10,12,14), 10 conv layers, 2048 embd, 65536 vocab
- KV cache: 6 layers on CUDA0, 10 filtered (conv layers have no KV)
- Recurrent state (R/S) buffers on CUDA0 for conv layers
- RTX 3060, 1186 MiB VRAM for model, 6 MiB KV cache, 139 MiB compute buffer

## Remaining
- For per-layer K/V hidden states: need HF safetensors path (transformers >=4.57.2 for LFM2.5)
  or llama.cpp C API bindings
- GGUF-backed adapter training blocked without per-layer signal access

## Usage
```bash
python flat_llm_adapter.py gguf-demo           # LFM2.5 inference demo
python flat_llm_adapter.py benchmark            # GPT-2 PCA benchmark (existing)
python train_adapter.py --k 9 --epochs 20       # GPT-2 adapter training (existing)
```
