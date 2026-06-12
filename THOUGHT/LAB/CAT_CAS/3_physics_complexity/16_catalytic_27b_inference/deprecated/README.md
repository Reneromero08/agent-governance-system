# Deprecated: u8 Quantization Inference Engine

## Status: DEPRECATED — replaced by f32 tape engine

The `lib_u8.rs.bak` file is the complete u8 quantization inference engine that was active through commit 4c6d915b.

### What it did
- 48-layer complex-plane inference (36 DeltaNet + 12 Gated Attention)
- Real Qwen 0.5B safetensors weights
- 16-head complex dot-product attention with KV cache
- SPN-scrambled weight buffer with dynamic per-layer decatalysis
- 100% tape restoration, 38% warm-hit rate
- 3.35 tok/s

### Why it's deprecated
The u8 quantization path destroyed weight precision. Raw BF16→uint8 conversion without calibration produced random subword output. The entire paradigm requires REAL weight values flowing through the catalytic fabric — quantization defeats the purpose.

The replacement uses f32 tape storage (4 bytes per complex dimension) with byte-level XOR fabric for the catalytic substrate. Real BF16 weights pass through without precision loss.

### Migration path
- `lib.rs` now contains the f32 tape inference engine
- `tape_read_f32()` / `tape_write_f32()` replace all `tape[offset] as f32 * FP8_SCALE` patterns
- HIDDEN_DIM = 896 (Qwen 0.5B), F32_BYTES = 4
- All offsets multiply by F32_BYTES for correct addressing
