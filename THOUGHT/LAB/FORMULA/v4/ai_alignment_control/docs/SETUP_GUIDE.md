# Phase 2 Runtime: Gemma 4B Setup Guide

Date: 2026-05-16 | Model: google/gemma-4-E4B-it | GPU: RTX 3060 12GB

---

## TL;DR

```powershell
# One-time setup
C:/Users/rene_/AppData/Local/Programs/Python/Python311/python.exe -m pip install git+https://github.com/huggingface/transformers.git
C:/Users/rene_/AppData/Local/Programs/Python/Python311/python.exe -m pip install bitsandbytes accelerate sentence-transformers

# Run
set TORCH_COMPILE_DISABLE=1 && C:/Users/rene_/AppData/Local/Programs/Python/Python311/python.exe script.py
```

---

## Which Python

**Use the system Python 3.11**, not the venv.

```
C:/Users/rene_/AppData/Local/Programs/Python/Python311/python.exe
```

Why: the system Python has `torch 2.10.0+cu126` with CUDA support. The venv has `torch 2.5.1+cu121` which works but has `torchao` version conflicts with the system's packages. The system Python avoids all torch version conflicts.

## Which Transformers

**Install from source.** The `gemma4` model type is not in any release version of transformers yet.

```powershell
pip install git+https://github.com/huggingface/transformers.git
```

Do NOT use `pip install transformers` — the released version doesn't know about `gemma4` and will error with `The checkpoint you are trying to load has model type 'gemma4' but Transformers does not recognize this architecture`.

## CUDA / Triton Issues

**Set `TORCH_COMPILE_DISABLE=1`** before running any script.

```powershell
set TORCH_COMPILE_DISABLE=1
```

Without this, PyTorch tries to import `triton` which fails on Windows with cryptic errors about `ConstantFunction` and `inspect.getsourcefile`. This environment variable disables the compilation path that touches triton entirely.

## GPU Memory

The RTX 3060 has 12GB VRAM. Gemma 4B in 4-bit uses ~3GB. Sentence transformers (`all-MiniLM-L6-v2`) uses ~500MB.

**Critical: kill all Python processes between runs.** WDDM (Windows Display Driver Model) does not release GPU memory until ALL processes that allocated it terminate.

```powershell
taskkill /F /IM python.exe /T
nvidia-smi  # verify memory.free > 10GB
```

If `nvidia-smi` shows >10GB in use and no Python processes are running, the allocations are zombie WDDM handles. A reboot clears them. This manifests as `OSError: The paging file is too small` during model loading.

## How the Phase 2 Scripts Were Run

### Phase 2a (Inference-only)
```
set TORCH_COMPILE_DISABLE=1 && C:/Users/rene_/AppData/Local/Programs/Python/Python311/python.exe THOUGHT/LAB/FORMULA/v4/ai_alignment_control/phase2a_run.py
```

### Phase 2b (SFT fine-tuning)
```
set TORCH_COMPILE_DISABLE=1 && C:/Users/rene_/AppData/Local/Programs/Python/Python311/python.exe THOUGHT/LAB/FORMULA/v4/ai_alignment_control/phase2b_sft.py
```

### Phase 2c (Resonance-guided sampling)
```
set TORCH_COMPILE_DISABLE=1 && C:/Users/rene_/AppData/Local/Programs/Python/Python311/python.exe THOUGHT/LAB/FORMULA/v4/ai_alignment_control/phase2c_resonance.py
```

All three use `BitsAndBytesConfig(load_in_4bit=True)` to fit in VRAM.

## Model Loading Pattern

The working pattern across all Phase 2 scripts:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E4B-it",
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    output_hidden_states=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()
```

## `generate()` Output Handling

Newer transformers returns `GenerateDecoderOnlyOutput` (an object with `.sequences`) not a bare tensor. The Phase 2 scripts use this pattern:

```python
out = model.generate(...)
ids = out.sequences[0] if hasattr(out, "sequences") else out[0]
response = tokenizer.decode(ids[input_len:], skip_special_tokens=True)
```

## `Trainer` Compatibility

Newer transformers Trainer uses `processing_class` instead of `tokenizer`:

```python
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,  # NOT "tokenizer="
)
```

## Troubleshooting Checklist

| Error | Fix |
|-------|-----|
| `gemma4` not recognized | Install transformers from source (see above) |
| `torch.int1` not found | Use system Python, not venv |
| `triton` / `inspect.getsourcefile` | Set `TORCH_COMPILE_DISABLE=1` |
| `paging file too small` | Kill all python.exe, check VRAM free >10GB |
| `ModuleNotFoundError: No module named 'x'` | Install to system Python, not venv |
| Trainer `tokenizer` kwarg error | Use `processing_class=` instead |
| `generate` returns Tensor not dict | Use `hasattr(out, "sequences")` guard |
| LoRA `target_modules` error | Gemma4 uses `Gemma4ClippableLinear`, use `target_modules="all-linear"` |
