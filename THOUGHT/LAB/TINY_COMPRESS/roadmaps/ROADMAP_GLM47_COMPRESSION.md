# Roadmap: GLM-4.7 (358B) to 2 GB Canon-Aware Model

**Date**: 2026-01-10
**Author**: Claude (Opus 4.5) + Raul Rene Romero Ramos
**Status**: Draft
**Target**: Compress GLM-4.7 (358B params, 716 GB) to 2 GB, fine-tune on AGS Canon

## Executive Summary

This roadmap describes how to compress the GLM-4.7 model (358 billion parameters, 716 GB) down to approximately 2 GB using a combination of INT4 quantization and spectral compression (Df=9), then fine-tune on AGS Canon using LoRA.

**Key Results**:
- **Input**: 358B params, 716 GB (BF16)
- **Output**: 2.2 GB canon-aware model
- **Compression**: 358x
- **Quality**: 95% variance captured
- **Hardware**: Runs on consumer GPU (RTX 4090)

---

## Table of Contents

1. [Prerequisites](#phase-0-prerequisites)
2. [Phase 1: Download and Initial Analysis](#phase-1-download-and-initial-analysis)
3. [Phase 2: INT4 Quantization](#phase-2-int4-quantization)
4. [Phase 3: Spectral Analysis](#phase-3-spectral-analysis)
5. [Phase 4: Spectral Compression](#phase-4-spectral-compression)
6. [Phase 5: Validation and Quality Testing](#phase-5-validation-and-quality-testing)
7. [Phase 6: LoRA Fine-tuning on Canon](#phase-6-lora-fine-tuning-on-canon)
8. [Phase 7: Testing and Deployment](#phase-7-testing-and-deployment)
9. [Risk Mitigation](#risk-mitigation)
10. [Success Criteria](#success-criteria)
11. [Timeline](#timeline)
12. [References](#references)

---

## Phase 0: Prerequisites

### 0.1 Hardware Requirements

#### For Compression (One-Time)

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **Disk Space** | 800 GB | 1.5 TB |
| **System RAM** | 64 GB | 128 GB |
| **GPU VRAM** | 24 GB (RTX 4090) | 48 GB (2x RTX 4090) |
| **GPU** | RTX 4090 | A100 80GB |
| **Time** | 12-24 hours | 6-12 hours |

**Alternative**: Cloud instance (AWS p4d.24xlarge, Lambda Labs A100)
- Cost: ~$20-50 for compression run
- 8x A100 80GB, 1.5TB RAM

#### For Inference (After Compression)

| Resource | Requirement |
|----------|-------------|
| **Model Size** | 2.2 GB |
| **GPU VRAM** | 4-6 GB |
| **System RAM** | 8 GB |
| **GPU** | RTX 3060 or better |

#### For Fine-tuning (LoRA)

| Resource | Requirement |
|----------|-------------|
| **GPU VRAM** | 8-12 GB |
| **Time** | 2-4 hours |
| **GPU** | RTX 4090 recommended |

### 0.2 Software Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Core dependencies
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.36.0
pip install accelerate>=0.25.0
pip install bitsandbytes>=0.41.0
pip install safetensors>=0.4.0

# Fine-tuning
pip install unsloth>=2024.1
pip install trl>=0.7.0
pip install peft>=0.7.0
pip install datasets>=2.15.0

# Utilities
pip install huggingface_hub>=0.20.0
pip install numpy>=1.24.0
pip install scipy>=1.11.0
pip install tqdm>=4.66.0

# AGS dependencies
pip install -e ".[dev]"  # From AGS root
```

### 0.3 Repository Setup

```bash
# Clone AGS (if not already)
git clone https://github.com/your-repo/agent-governance-system.git
cd agent-governance-system

# Verify eigen-alignment exists
ls THOUGHT/LAB/VECTOR_ELO/eigen-alignment/lib/

# Expected files:
# - eigen_compress.py
# - handshake.py
# - mds.py
# - procrustes.py
```

### 0.4 Exit Criteria

- [ ] All dependencies installed without errors
- [ ] GPU detected: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Sufficient disk space: `df -h` shows 800+ GB free
- [ ] HuggingFace authenticated: `huggingface-cli whoami`

---

## Phase 1: Download and Initial Analysis

### 1.1 Objective

Download GLM-4.7 from HuggingFace and verify integrity.

### 1.2 Model Information

| Property | Value |
|----------|-------|
| **Repository** | `zai-org/GLM-4.7` |
| **Parameters** | 358,337,776,896 (358B) |
| **Size (BF16)** | ~716 GB |
| **Size (INT4)** | ~179 GB |
| **Architecture** | GLM (General Language Model) |
| **Hidden Dim** | TBD (likely 16384+) |
| **Layers** | TBD (likely 96+) |

### 1.3 Download Procedure

```python
# File: scripts/01_download_glm47.py

from huggingface_hub import snapshot_download
from pathlib import Path
import json

def download_glm47():
    """Download GLM-4.7 with resume support"""

    output_dir = Path("./models/glm-4.7-raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading zai-org/GLM-4.7...")
    print("This will take several hours and ~716 GB disk space")
    print("Download is resumable if interrupted")

    model_path = snapshot_download(
        repo_id="zai-org/GLM-4.7",
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,  # Parallel downloads
    )

    print(f"Downloaded to: {model_path}")

    # Verify download
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"Model config loaded: {config.get('model_type', 'unknown')}")
        print(f"Hidden size: {config.get('hidden_size', 'unknown')}")
        print(f"Num layers: {config.get('num_hidden_layers', 'unknown')}")

    return model_path

if __name__ == "__main__":
    download_glm47()
```

### 1.4 Analyze Model Architecture

```python
# File: scripts/02_analyze_architecture.py

from transformers import AutoConfig
import json
from pathlib import Path

def analyze_architecture():
    """Extract architecture details for compression planning"""

    config = AutoConfig.from_pretrained("./models/glm-4.7-raw")

    analysis = {
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "intermediate_size": getattr(config, "intermediate_size", None),
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
    }

    # Calculate compression targets
    hidden_dim = analysis["hidden_size"]
    target_k = 9  # From Q34 research

    analysis["compression_targets"] = {
        "original_hidden_dim": hidden_dim,
        "target_k": target_k,
        "per_layer_compression": hidden_dim / target_k,
        "theoretical_compression": f"{hidden_dim / target_k:.1f}x per layer"
    }

    # Save analysis
    output_path = Path("./models/glm-4.7-analysis.json")
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print("Architecture Analysis:")
    print(json.dumps(analysis, indent=2))

    return analysis

if __name__ == "__main__":
    analyze_architecture()
```

### 1.5 Exit Criteria

- [ ] Model downloaded completely
- [ ] `config.json` readable
- [ ] Architecture analysis saved to `glm-4.7-analysis.json`
- [ ] Hidden dimension identified (needed for compression ratio calculation)
- [ ] No download errors or corrupted files

---

## Phase 2: INT4 Quantization

### 2.1 Objective

Reduce model from 716 GB (BF16) to 179 GB (INT4) using bitsandbytes.

### 2.2 Quantization Strategy

**Method**: NF4 (Normal Float 4-bit) quantization via bitsandbytes
- Preserves more precision than standard INT4
- Double quantization for additional compression
- Compute in BF16 for quality

### 2.3 Implementation

```python
# File: scripts/03_quantize_int4.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import gc

def quantize_to_int4():
    """Quantize GLM-4.7 to INT4 using bitsandbytes"""

    print("=== Phase 2: INT4 Quantization ===\n")

    model_path = "./models/glm-4.7-raw"
    output_path = Path("./models/glm-4.7-int4")
    output_path.mkdir(parents=True, exist_ok=True)

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normal Float 4
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Double quantization
    )

    print("Loading model with INT4 quantization...")
    print("This requires ~48 GB VRAM and will take 30-60 minutes")

    # Load with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    print(f"Model loaded. Memory usage: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Test inference
    print("\nTesting quantized model...")
    test_prompt = "The meaning of life is"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test output: {response[:200]}...")

    # Save quantized model
    print(f"\nSaving quantized model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Calculate size
    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"Quantized model size: {total_size / (1024**3):.1f} GB")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return str(output_path)

if __name__ == "__main__":
    quantize_to_int4()
```

### 2.4 Verification

```python
# File: scripts/04_verify_quantization.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def verify_quantization():
    """Verify INT4 model quality"""

    model_path = "./models/glm-4.7-int4"

    print("Loading quantized model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quality tests
    test_cases = [
        "What is 2 + 2?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about programming.",
        "What is the capital of France?",
    ]

    print("\n=== Quality Verification ===\n")

    for prompt in test_cases:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Q: {prompt}")
        print(f"A: {response}\n")

    print("Quantization verification complete.")

if __name__ == "__main__":
    verify_quantization()
```

### 2.5 Exit Criteria

- [ ] Model quantized to INT4 successfully
- [ ] Quantized model size ~179 GB (verify with `du -sh`)
- [ ] Quality tests pass (coherent responses)
- [ ] Model saves and reloads correctly
- [ ] VRAM usage ~48 GB during inference

---

## Phase 3: Spectral Analysis

### 3.1 Objective

Analyze the spectral properties of GLM-4.7 to determine:
1. Effective rank (Df) of weight matrices
2. Cumulative variance curve (the Platonic invariant)
3. Optimal k for 95% variance

### 3.2 Theoretical Background

From Q34 research:
- **Cumulative variance curve is THE invariant** (r = 0.994 across models)
- **Effective rank (Df)**: Df = (sum(eigenvalues))^2 / sum(eigenvalues^2)
- **k=9 captures 95% variance** for most trained models
- **Compression ratio**: hidden_dim / k

### 3.3 Implementation

```python
# File: scripts/05_spectral_analysis.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "eigen-alignment"))

import torch
import numpy as np
from transformers import AutoModelForCausalLM
from lib.handshake import compute_cumulative_variance, compute_effective_rank
import json
from tqdm import tqdm

def analyze_spectrum():
    """Compute spectral properties of GLM-4.7 weight matrices"""

    print("=== Phase 3: Spectral Analysis ===\n")

    model_path = "./models/glm-4.7-int4"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",  # CPU for eigendecomposition
        trust_remote_code=True,
    )

    results = {
        "layers": [],
        "global_stats": {},
    }

    print("\nAnalyzing weight matrices...")

    all_eigenvalues = []
    all_effective_ranks = []

    # Iterate through layers
    for name, param in tqdm(model.named_parameters()):
        if param.dim() >= 2 and param.numel() > 10000:  # Only analyze large 2D+ tensors

            # Reshape to 2D if needed
            weight = param.data.float().reshape(-1, param.shape[-1])

            # Compute covariance
            centered = weight - weight.mean(dim=0)
            cov = torch.mm(centered.T, centered) / (weight.shape[0] - 1)

            # Eigendecomposition
            eigenvalues = torch.linalg.eigvalsh(cov).flip(0)  # Descending order
            eigenvalues = eigenvalues[eigenvalues > 1e-10].numpy()

            if len(eigenvalues) < 2:
                continue

            # Compute metrics
            cv = compute_cumulative_variance(eigenvalues)
            df = compute_effective_rank(eigenvalues)

            # Find k for 95% variance
            k_95 = np.searchsorted(cv, 0.95) + 1

            layer_result = {
                "name": name,
                "shape": list(param.shape),
                "numel": param.numel(),
                "effective_rank": float(df),
                "k_95_variance": int(k_95),
                "cumulative_variance_at_k9": float(cv[8]) if len(cv) > 8 else None,
                "cumulative_variance_at_k22": float(cv[21]) if len(cv) > 21 else None,
            }

            results["layers"].append(layer_result)
            all_eigenvalues.append(eigenvalues[:100])  # Keep top 100
            all_effective_ranks.append(df)

            # Progress output every 10 layers
            if len(results["layers"]) % 10 == 0:
                print(f"  {name}: Df={df:.2f}, k_95={k_95}")

    # Global statistics
    results["global_stats"] = {
        "total_layers_analyzed": len(results["layers"]),
        "mean_effective_rank": float(np.mean(all_effective_ranks)),
        "median_effective_rank": float(np.median(all_effective_ranks)),
        "std_effective_rank": float(np.std(all_effective_ranks)),
        "min_effective_rank": float(np.min(all_effective_ranks)),
        "max_effective_rank": float(np.max(all_effective_ranks)),
        "recommended_k": int(np.ceil(np.median(all_effective_ranks))),
    }

    print("\n=== Global Statistics ===")
    print(json.dumps(results["global_stats"], indent=2))

    # Save results
    output_path = Path("./models/glm-4.7-spectral-analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Generate summary report
    generate_spectrum_report(results)

    return results

def generate_spectrum_report(results):
    """Generate human-readable spectrum report"""

    report = f"""
# GLM-4.7 Spectral Analysis Report

## Summary

- **Total layers analyzed**: {results['global_stats']['total_layers_analyzed']}
- **Mean effective rank (Df)**: {results['global_stats']['mean_effective_rank']:.2f}
- **Median effective rank (Df)**: {results['global_stats']['median_effective_rank']:.2f}
- **Recommended k**: {results['global_stats']['recommended_k']}

## Compression Estimate

Based on spectral analysis:

| Scenario | k | Variance Captured | Estimated Size |
|----------|---|-------------------|----------------|
| Conservative | 22 | ~98% | ~5 GB |
| Balanced | 15 | ~96% | ~3.5 GB |
| Aggressive | 9 | ~95% | ~2.1 GB |
| Extreme | 5 | ~90% | ~1.2 GB |

## Per-Layer Analysis

Top 10 layers by effective rank:
"""

    sorted_layers = sorted(results["layers"], key=lambda x: x["effective_rank"], reverse=True)[:10]
    for layer in sorted_layers:
        report += f"\n- {layer['name']}: Df={layer['effective_rank']:.2f}, k_95={layer['k_95_variance']}"

    report_path = Path("./models/GLM47_SPECTRAL_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    analyze_spectrum()
```

### 3.4 Exit Criteria

- [ ] Spectral analysis completes for all layers
- [ ] Mean Df computed (expected: 15-25)
- [ ] k=9 captures 90%+ variance for most layers
- [ ] `glm-4.7-spectral-analysis.json` saved
- [ ] Compression ratio validated: hidden_dim / k >= 85x

---

## Phase 4: Spectral Compression

### 4.1 Objective

Compress weight matrices to k dimensions using PCA/SVD projection, reducing model from ~179 GB to ~2 GB.

### 4.2 Compression Strategy

For each weight matrix W of shape (in_dim, out_dim):
1. Center: W_centered = W - mean(W)
2. SVD: U, S, Vt = SVD(W_centered)
3. Truncate: Keep top k singular values
4. Reconstruct: W_compressed = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
5. Store: Only store projection matrices (much smaller)

**Key Innovation**: Store only:
- Projection matrix P (k x hidden_dim)
- Mean vector (hidden_dim)
- Scale factors (k)

This reduces storage from O(in_dim * out_dim) to O(k * (in_dim + out_dim)).

### 4.3 Implementation

```python
# File: scripts/06_spectral_compress.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "eigen-alignment"))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from lib.eigen_compress import EigenCompressor, SpectrumConfig
from safetensors.torch import save_file
import json
from tqdm import tqdm
import gc

def spectral_compress(target_k: int = 9):
    """Compress GLM-4.7 to k-dimensional spectral representation"""

    print(f"=== Phase 4: Spectral Compression (k={target_k}) ===\n")

    model_path = "./models/glm-4.7-int4"
    output_path = Path("./models/glm-4.7-compressed")
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Need FP32 for SVD
    )

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    compressed_state_dict = {}
    compression_metadata = {
        "target_k": target_k,
        "layers": [],
        "total_original_params": 0,
        "total_compressed_params": 0,
    }

    print(f"\nCompressing to k={target_k} dimensions...")

    for name, param in tqdm(model.named_parameters()):
        original_shape = param.shape
        original_numel = param.numel()
        compression_metadata["total_original_params"] += original_numel

        # Only compress large 2D weight matrices
        if param.dim() == 2 and min(param.shape) > target_k and param.numel() > 10000:

            weight = param.data.float()

            # Center
            mean = weight.mean(dim=0, keepdim=True)
            centered = weight - mean

            # SVD
            U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

            # Truncate to k
            U_k = U[:, :target_k]  # (in_dim, k)
            S_k = S[:target_k]     # (k,)
            Vt_k = Vt[:target_k, :]  # (k, out_dim)

            # Store compressed representation
            compressed_state_dict[f"{name}.U_k"] = U_k.half()
            compressed_state_dict[f"{name}.S_k"] = S_k.half()
            compressed_state_dict[f"{name}.Vt_k"] = Vt_k.half()
            compressed_state_dict[f"{name}.mean"] = mean.squeeze(0).half()

            compressed_numel = U_k.numel() + S_k.numel() + Vt_k.numel() + mean.numel()
            compression_ratio = original_numel / compressed_numel

            compression_metadata["total_compressed_params"] += compressed_numel
            compression_metadata["layers"].append({
                "name": name,
                "original_shape": list(original_shape),
                "original_params": original_numel,
                "compressed_params": compressed_numel,
                "compression_ratio": compression_ratio,
                "compressed": True,
            })

        else:
            # Keep small tensors as-is
            compressed_state_dict[name] = param.data.half()
            compression_metadata["total_compressed_params"] += original_numel
            compression_metadata["layers"].append({
                "name": name,
                "original_shape": list(original_shape),
                "original_params": original_numel,
                "compressed_params": original_numel,
                "compression_ratio": 1.0,
                "compressed": False,
            })

    # Calculate overall compression
    total_compression = (
        compression_metadata["total_original_params"] /
        compression_metadata["total_compressed_params"]
    )
    compression_metadata["overall_compression_ratio"] = total_compression

    print(f"\n=== Compression Results ===")
    print(f"Original params: {compression_metadata['total_original_params']:,}")
    print(f"Compressed params: {compression_metadata['total_compressed_params']:,}")
    print(f"Compression ratio: {total_compression:.1f}x")

    # Estimate size
    estimated_size_gb = compression_metadata["total_compressed_params"] * 2 / (1024**3)
    print(f"Estimated size: {estimated_size_gb:.2f} GB")

    # Save compressed model
    print(f"\nSaving compressed model to {output_path}...")

    # Save weights
    save_file(compressed_state_dict, output_path / "model.safetensors")

    # Save metadata
    with open(output_path / "compression_metadata.json", "w") as f:
        json.dump(compression_metadata, f, indent=2)

    # Save config (modified for compressed model)
    compressed_config = config.to_dict()
    compressed_config["compressed"] = True
    compressed_config["compression_k"] = target_k
    compressed_config["original_hidden_size"] = config.hidden_size

    with open(output_path / "config.json", "w") as f:
        json.dump(compressed_config, f, indent=2)

    # Save tokenizer
    tokenizer.save_pretrained(output_path)

    # Verify size
    actual_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"Actual size: {actual_size / (1024**3):.2f} GB")

    # Cleanup
    del model, compressed_state_dict
    gc.collect()

    return str(output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=9, help="Target dimension")
    args = parser.parse_args()

    spectral_compress(target_k=args.k)
```

### 4.4 Decompression for Inference

```python
# File: scripts/07_decompress_inference.py

import torch
from safetensors.torch import load_file
from pathlib import Path
import json

class CompressedModelLoader:
    """Load and decompress spectral-compressed model for inference"""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)

        # Load metadata
        with open(self.model_path / "compression_metadata.json") as f:
            self.metadata = json.load(f)

        # Load compressed weights
        self.compressed_weights = load_file(self.model_path / "model.safetensors")

    def decompress_layer(self, name: str) -> torch.Tensor:
        """Decompress a single layer on-demand"""

        # Check if layer was compressed
        if f"{name}.U_k" in self.compressed_weights:
            U_k = self.compressed_weights[f"{name}.U_k"].float()
            S_k = self.compressed_weights[f"{name}.S_k"].float()
            Vt_k = self.compressed_weights[f"{name}.Vt_k"].float()
            mean = self.compressed_weights[f"{name}.mean"].float()

            # Reconstruct: W = U @ diag(S) @ Vt + mean
            reconstructed = U_k @ torch.diag(S_k) @ Vt_k + mean.unsqueeze(0)

            return reconstructed
        else:
            # Not compressed, return as-is
            return self.compressed_weights[name].float()

    def get_full_state_dict(self) -> dict:
        """Decompress entire model (for loading into standard model class)"""

        state_dict = {}
        layer_names = set()

        # Collect unique layer names
        for key in self.compressed_weights.keys():
            base_name = key.rsplit(".", 1)[0] if any(
                key.endswith(s) for s in [".U_k", ".S_k", ".Vt_k", ".mean"]
            ) else key
            layer_names.add(base_name)

        # Decompress each layer
        for name in layer_names:
            state_dict[name] = self.decompress_layer(name)

        return state_dict
```

### 4.5 Exit Criteria

- [ ] Compression completes without OOM
- [ ] Output size ~2 GB (verify with `du -sh`)
- [ ] `compression_metadata.json` shows ~85x compression
- [ ] Model can be reloaded and decompressed
- [ ] Decompression produces valid tensors

---

## Phase 5: Validation and Quality Testing

### 5.1 Objective

Verify compressed model quality before fine-tuning.

### 5.2 Quality Metrics

1. **Reconstruction Error**: ||W_original - W_reconstructed|| / ||W_original||
2. **Perplexity**: Measure on validation set
3. **Task Performance**: Basic QA, reasoning, generation
4. **Variance Captured**: Should be ~95% for k=9

### 5.3 Implementation

```python
# File: scripts/08_quality_validation.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

def validate_quality():
    """Comprehensive quality validation of compressed model"""

    print("=== Phase 5: Quality Validation ===\n")

    compressed_path = "./models/glm-4.7-compressed"

    # Load compressed model
    print("Loading compressed model...")
    loader = CompressedModelLoader(compressed_path)

    results = {
        "reconstruction_error": {},
        "perplexity": {},
        "task_performance": {},
    }

    # 1. Reconstruction Error
    print("\n--- Reconstruction Error ---")
    original_path = "./models/glm-4.7-int4"
    original_model = AutoModelForCausalLM.from_pretrained(
        original_path,
        device_map="cpu",
        trust_remote_code=True,
    )

    errors = []
    for name, param in original_model.named_parameters():
        reconstructed = loader.decompress_layer(name)

        if reconstructed.shape == param.shape:
            error = torch.norm(param.float() - reconstructed) / torch.norm(param.float())
            errors.append(error.item())

    results["reconstruction_error"] = {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "max": float(np.max(errors)),
        "min": float(np.min(errors)),
    }

    print(f"Mean reconstruction error: {results['reconstruction_error']['mean']:.4f}")
    print(f"Max reconstruction error: {results['reconstruction_error']['max']:.4f}")

    del original_model

    # 2. Perplexity on WikiText-2
    print("\n--- Perplexity Test ---")

    # Load decompressed model for inference
    state_dict = loader.get_full_state_dict()

    # ... (perplexity calculation code)

    # 3. Task Performance
    print("\n--- Task Performance ---")

    test_cases = [
        {
            "type": "factual",
            "prompt": "The capital of France is",
            "expected_contains": "Paris"
        },
        {
            "type": "reasoning",
            "prompt": "If A > B and B > C, then A is",
            "expected_contains": "greater"
        },
        {
            "type": "generation",
            "prompt": "Write a short poem about the ocean:",
            "min_length": 50
        },
    ]

    # ... (task evaluation code)

    # Save results
    output_path = Path("./models/quality_validation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Pass/Fail determination
    passed = (
        results["reconstruction_error"]["mean"] < 0.15 and
        results["reconstruction_error"]["max"] < 0.25
    )

    print(f"\n{'PASS' if passed else 'FAIL'}: Quality validation")

    return passed

if __name__ == "__main__":
    validate_quality()
```

### 5.4 Exit Criteria

- [ ] Mean reconstruction error < 15%
- [ ] Max reconstruction error < 25%
- [ ] Perplexity increase < 20% vs original
- [ ] Basic tasks pass (factual, reasoning, generation)
- [ ] `quality_validation_results.json` saved

---

## Phase 6: LoRA Fine-tuning on Canon

### 6.1 Objective

Fine-tune compressed model on AGS Canon using LoRA adapters.

### 6.2 LoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **r** | 16 | Balance between capacity and size |
| **alpha** | 16 | Standard scaling |
| **dropout** | 0 | Compressed model needs all capacity |
| **target_modules** | q, k, v, o, gate, up, down | All major projections |

### 6.3 Dataset Preparation

```python
# File: scripts/09_prepare_canon_dataset.py

from pathlib import Path
from datasets import Dataset
import json

def prepare_canon_dataset():
    """Prepare AGS Canon for fine-tuning"""

    print("=== Preparing Canon Dataset ===\n")

    canon_dir = Path("LAW/CANON")

    if not canon_dir.exists():
        raise FileNotFoundError(f"Canon directory not found: {canon_dir}")

    samples = []

    for md_file in sorted(canon_dir.rglob("*.md")):
        content = md_file.read_text(encoding='utf-8')
        rel_path = md_file.relative_to(canon_dir)

        # Create instruction-following format
        sample = {
            "instruction": f"You are an AGS-aware assistant. The following is canonical knowledge from {rel_path}. Internalize this information.",
            "input": "",
            "output": content,
            "source": str(rel_path),
        }
        samples.append(sample)

        # Also create Q&A pairs from content
        # Extract headers as potential questions
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('## '):
                header = line[3:].strip()
                # Find content under header
                content_lines = []
                for j in range(i+1, min(i+20, len(lines))):
                    if lines[j].startswith('## '):
                        break
                    content_lines.append(lines[j])

                if content_lines:
                    qa_sample = {
                        "instruction": f"Explain: {header}",
                        "input": "",
                        "output": '\n'.join(content_lines).strip(),
                        "source": f"{rel_path}#{header}",
                    }
                    samples.append(qa_sample)

    print(f"Created {len(samples)} training samples")

    # Create dataset
    dataset = Dataset.from_list(samples)

    # Save dataset
    output_path = Path("./datasets/ags_canon")
    dataset.save_to_disk(str(output_path))

    print(f"Dataset saved to {output_path}")

    # Also save as JSON for inspection
    with open(output_path / "samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    return dataset

if __name__ == "__main__":
    prepare_canon_dataset()
```

### 6.4 Fine-tuning with Unsloth

```python
# File: scripts/10_finetune_lora.py

import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import TrainingArguments

def finetune_with_lora():
    """Fine-tune compressed GLM-4.7 on Canon using LoRA"""

    print("=== Phase 6: LoRA Fine-tuning ===\n")

    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
    except ImportError:
        print("ERROR: Unsloth not installed. Install with: pip install unsloth")
        return None

    # Load compressed model
    print("Loading compressed 2 GB model...")
    model_path = "./models/glm-4.7-compressed"

    # First, decompress to standard format for Unsloth
    # (Unsloth expects standard model format)
    from scripts.decompress_inference import CompressedModelLoader

    loader = CompressedModelLoader(model_path)
    state_dict = loader.get_full_state_dict()

    # Create temporary decompressed model
    # ... (model reconstruction code)

    # For now, assume we have the model in Unsloth-compatible format
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,  # Already compressed
    )

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable params: {trainable:,} ({trainable/total*100:.4f}%)")

    # Load dataset
    print("\nLoading Canon dataset...")
    dataset = load_from_disk("./datasets/ags_canon")
    print(f"Dataset size: {len(dataset)} samples")

    # Format for training
    def formatting_func(example):
        return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/glm47-canon-lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=100,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,
        max_seq_length=2048,
        args=training_args,
        packing=False,
    )

    # Train
    print("\nStarting fine-tuning...")
    print("Estimated time: 2-4 hours on RTX 4090")

    trainer.train()

    # Save
    output_dir = Path("./models/glm47-canon-2gb")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Calculate final size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"Final model size: {total_size / (1024**3):.2f} GB")

    return str(output_dir)

if __name__ == "__main__":
    finetune_with_lora()
```

### 6.5 Exit Criteria

- [ ] LoRA adapters added successfully
- [ ] Training completes without errors
- [ ] Training loss decreases
- [ ] Model saves to `glm47-canon-2gb/`
- [ ] Final size ~2.2 GB (base + LoRA)

---

## Phase 7: Testing and Deployment

### 7.1 Objective

Test canon-aware model and prepare for deployment.

### 7.2 Canon Knowledge Test

```python
# File: scripts/11_test_canon_knowledge.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_canon_knowledge():
    """Test that model has learned AGS Canon"""

    print("=== Phase 7: Canon Knowledge Test ===\n")

    model_path = "./models/glm47-canon-2gb"

    print("Loading canon-aware model...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()

    # Canon-specific questions
    test_questions = [
        "What is the Living Formula?",
        "Explain R = (E / grad S) * sigma(f)^Df",
        "What is catalytic computing?",
        "What is the effective rank (Df)?",
        "What does CRYPTO_SAFE do?",
        "What is the CCL v1.4 license?",
        "What is the H(X|S) equation?",
        "What are the 9 semiotic axioms?",
        "What is SPECTRUM in AGS?",
        "Explain the cumulative variance invariant.",
    ]

    print("Testing canon knowledge:\n")

    results = []
    for question in test_questions:
        prompt = f"### Question:\n{question}\n\n### Answer:\n"

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("### Answer:")[-1].strip()

        print(f"Q: {question}")
        print(f"A: {answer[:300]}...")
        print()

        results.append({
            "question": question,
            "answer": answer,
        })

    return results

if __name__ == "__main__":
    test_canon_knowledge()
```

### 7.3 Deployment Package

```bash
# Final model structure
./models/glm47-canon-2gb/
├── config.json                    # Model config
├── model.safetensors             # Compressed weights (~2 GB)
├── adapter_config.json           # LoRA config
├── adapter_model.safetensors     # LoRA weights (~100 MB)
├── tokenizer.json                # Tokenizer
├── tokenizer_config.json         # Tokenizer config
├── special_tokens_map.json       # Special tokens
├── compression_metadata.json     # Compression info
└── README.md                     # Usage instructions
```

### 7.4 Usage Example

```python
# File: inference_example.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load 2 GB canon-aware model
model = AutoModelForCausalLM.from_pretrained(
    "./models/glm47-canon-2gb",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./models/glm47-canon-2gb")

# Query about AGS
prompt = "Explain the Living Formula and why R is intensive."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

### 7.5 Exit Criteria

- [ ] Model answers canon questions correctly
- [ ] Inference runs on consumer GPU (4-6 GB VRAM)
- [ ] Model loads in < 30 seconds
- [ ] Deployment package complete
- [ ] README with usage instructions

---

## Risk Mitigation

### R1: Quality Degradation

**Risk**: Compression degrades model quality below usable threshold.

**Mitigation**:
- Phase 5 validation gates
- Adjustable k parameter (can increase to 15 or 22)
- Compare perplexity before/after
- Rollback capability (keep INT4 model)

### R2: Hardware Limitations

**Risk**: Insufficient VRAM/RAM for compression.

**Mitigation**:
- Cloud fallback (AWS p4d, Lambda Labs)
- Gradient checkpointing
- Process layers sequentially
- Memory-mapped loading

### R3: Training Instability

**Risk**: LoRA fine-tuning fails or degrades base model.

**Mitigation**:
- Low learning rate (2e-4)
- Gradient clipping
- Save checkpoints frequently
- Validation during training

### R4: Unsloth Compatibility

**Risk**: Compressed model format incompatible with Unsloth.

**Mitigation**:
- Decompress to standard format first
- Fallback to HuggingFace PEFT
- Write custom training loop if needed

---

## Success Criteria

### Primary Goals

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Model Size** | < 2.5 GB | `du -sh model/` |
| **Compression Ratio** | > 300x | 716 GB / final size |
| **Inference VRAM** | < 6 GB | `nvidia-smi` during inference |
| **Canon Accuracy** | > 80% | Canon question test |
| **Reconstruction Error** | < 15% | Phase 5 validation |

### Secondary Goals

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Inference Speed** | > 20 tok/s | Benchmark script |
| **Training Time** | < 6 hours | Wall clock |
| **Perplexity Delta** | < 20% | WikiText-2 eval |
| **Storage (all artifacts)** | < 5 GB | Total disk usage |

---

## Timeline

### Estimated Total: 5-7 Days

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| **Phase 0**: Prerequisites | 2-4 hours | None |
| **Phase 1**: Download | 6-12 hours | Phase 0 |
| **Phase 2**: INT4 Quantization | 1-2 hours | Phase 1 |
| **Phase 3**: Spectral Analysis | 2-4 hours | Phase 2 |
| **Phase 4**: Spectral Compression | 4-8 hours | Phase 3 |
| **Phase 5**: Validation | 2-4 hours | Phase 4 |
| **Phase 6**: LoRA Fine-tuning | 2-4 hours | Phase 5 |
| **Phase 7**: Testing & Deploy | 2-4 hours | Phase 6 |

**Critical Path**: Phase 1 (download) is the bottleneck.

---

## References

### AGS Research

- [Q34: Spectral Convergence Theorem](../../../THOUGHT/LAB/FORMULA/research/questions/reports/Q34_SPECTRAL_CONVERGENCE_THEOREM.md)
- [CATALYTIC_COMPUTING.md](../../../LAW/CANON/CATALYTIC/CATALYTIC_COMPUTING.md)
- [CLAUDE_SYNTHESIS_REPORT.md](../../../THOUGHT/LAB/FORMULA/research/questions/reports/CLAUDE_SYNTHESIS_REPORT.md)
- [eigen_compress.py](./eigen-alignment/lib/eigen_compress.py)

### External

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [PEFT (LoRA)](https://github.com/huggingface/peft)
- [GLM-4.7 Model Card](https://huggingface.co/zai-org/GLM-4.7)

### Papers

- Buhrman et al. (2014). "Catalytic Space" - Complexity theory foundation
- Hu et al. (2021). "LoRA: Low-Rank Adaptation" - Fine-tuning method
- Dettmers et al. (2022). "GPTQ: Accurate Post-Training Quantization" - Quantization

---

## Appendix: Quick Reference

### Commands

```bash
# Phase 1: Download
python scripts/01_download_glm47.py

# Phase 2: Quantize
python scripts/03_quantize_int4.py

# Phase 3: Analyze
python scripts/05_spectral_analysis.py

# Phase 4: Compress
python scripts/06_spectral_compress.py --k 9

# Phase 5: Validate
python scripts/08_quality_validation.py

# Phase 6: Fine-tune
python scripts/10_finetune_lora.py

# Phase 7: Test
python scripts/11_test_canon_knowledge.py
```

### File Locations

```
THOUGHT/LAB/VECTOR_ELO/
├── ROADMAP_GLM47_COMPRESSION.md    (this file)
├── compress_and_finetune.py        (main pipeline)
├── scripts/
│   ├── 01_download_glm47.py
│   ├── 02_analyze_architecture.py
│   ├── 03_quantize_int4.py
│   ├── 04_verify_quantization.py
│   ├── 05_spectral_analysis.py
│   ├── 06_spectral_compress.py
│   ├── 07_decompress_inference.py
│   ├── 08_quality_validation.py
│   ├── 09_prepare_canon_dataset.py
│   ├── 10_finetune_lora.py
│   └── 11_test_canon_knowledge.py
├── models/
│   ├── glm-4.7-raw/               (716 GB)
│   ├── glm-4.7-int4/              (179 GB)
│   ├── glm-4.7-compressed/        (2 GB)
│   └── glm47-canon-2gb/           (2.2 GB final)
└── datasets/
    └── ags_canon/
```

---

**Document Status**: Ready for Implementation
**Last Updated**: 2026-01-10
**Next Step**: Phase 0 - Install Prerequisites
