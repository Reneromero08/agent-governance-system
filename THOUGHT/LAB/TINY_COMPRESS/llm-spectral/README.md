# LLM Spectral Compression

**What this is**: Applying the Df (effective dimensionality) formula to LLM inference. Discovered that LLM hidden states have Df approx 2 (meaning lives in 2 dimensions!), but attention spreads it to 160-460D.

## Key Discovery

| Component | Df | Compressible? |
|-----------|-----|---------------|
| Hidden states (between layers) | ~2 | YES - 85x theoretically |
| K projections (attention) | 160-460 | LIMITED - 5x practical |
| V projections (attention) | 160-460 | LIMITED - 5x practical |
| Model weights | 500+ | NO (use INT4 quantization) |

The barrier: GPT-2 was trained to compute in 768D. Compressing back to 2D destroys information attention needs. Getting 85x requires learned adapters or native low-dimensional architecture.

## Files

### Core experiments

| File | What it does |
|------|-------------|
| `eigen_gpt2.py` | GPT-2 with compressed KV cache (build/chat/benchmark) |
| `eigen_attention.py` | Attention computed entirely in k-space with learnable projectors |
| `activation_compress.py` | Df measurement on any HuggingFace model's activations |
| `compressed_inference.py` | Wrapper for on-the-fly activation compression during inference |
| `spectral_compress.py` | SVD-based weight matrix compression (analyze + compress + save) |
| `spectral_compress-01.py` | Duplicate of spectral_compress.py (identical code) |
| `spectral_llm.py` | Full pipeline: compress model weights via SVD, reconstruct for inference |
| `compress_and_finetune.py` | Pipeline: GLM-4.7 download -> spectral compress -> LoRA fine-tune |
| `run_eigen.py` | Bridge to eigen-alignment code in VECTOR_ELO lab |

### Results

| Report | What it covers |
|--------|---------------|
| `results/REPORT_SPECTRAL_COMPRESSION.md` | Initial Df validation on GPT-2, Qwen activations |
| `results/REPORT_COMPRESSION_BARRIER.md` | Why 85x compression requires adapters or new architecture |

## Usage

```bash
# Measure Df on any model
python activation_compress.py --model gpt2 --benchmark

# Build and chat with eigen GPT-2
python eigen_gpt2.py build --k 150
python eigen_gpt2.py chat ./eigen_gpt2

# Compress model weights
python spectral_llm.py compress gpt2 --variance 0.95
```
