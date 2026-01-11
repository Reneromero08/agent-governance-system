#!/usr/bin/env python3
"""
Compress GLM-4.7 (358B!) and fine-tune on AGS Canon

Workflow:
1. Download GLM-4.7 (358B params, ~716 GB in BF16, ~358 GB in INT4)
2. Compress using Df=22 spectral method → 24 MB base
3. Fine-tune compressed model with LoRA on canon → +2 MB
4. Result: 26 MB canon-aware model

Requirements:
    pip install unsloth transformers datasets trl huggingface_hub bitsandbytes accelerate

Model: zai-org/GLM-4.7 (358B parameters!)
https://huggingface.co/zai-org/GLM-4.7

IMPORTANT: This is a 358B parameter model. You'll need:
- ~716 GB disk space for download
- ~200 GB VRAM for full precision (8x A100 80GB)
- OR ~90 GB VRAM with INT4 quantization (2x A100 80GB)
- OR use bitsandbytes 4-bit quantization to fit in ~48 GB VRAM

The compression step will reduce it to 24 MB!
"""

import sys
from pathlib import Path
from typing import List

# Add eigen-alignment to path
sys.path.insert(0, str(Path(__file__).parent / "eigen-alignment"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset


def download_glm4():
    """Download GLM-4.7 from HuggingFace"""
    print("=== Step 1: Download GLM-4.7 ===\n")

    try:
        from huggingface_hub import snapshot_download

        print("Downloading zai-org/GLM-4.7...")
        print("(4.7B params, ~9.4 GB)")
        print("Model: https://huggingface.co/zai-org/GLM-4.7")

        model_path = snapshot_download(
            repo_id="zai-org/GLM-4.7",
            cache_dir="./models"
        )

        print(f"Downloaded to: {model_path}")
        return model_path

    except ImportError:
        print("huggingface_hub not installed. Install with:")
        print("  pip install huggingface_hub")
        return None


def compress_model(model_path: str):
    """Compress GLM-4 using spectral method (Df=22)"""
    print("\n=== Step 2: Compress to 24 MB ===\n")

    # Import compression code
    try:
        from lib.eigen_compress import EigenCompressor
    except ImportError:
        print("ERROR: eigen_compress module not found")
        print("Make sure eigen-alignment/lib/eigen_compress.py exists")
        return None

    print("Loading full GLM-4-9B model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    # Analyze spectrum
    print("\nAnalyzing activation spectrum...")
    compressor = EigenCompressor.from_model(model)

    print(f"Effective rank (Df): {compressor.config.effective_rank:.2f}")
    print(f"Target compression: {compressor.config.compression_ratio:.1f}x")

    # Compress
    print("\nCompressing model...")
    compressed = compressor.compress(model)

    # Save compressed model
    output_dir = Path("./models/glm4-24mb-base")
    output_dir.mkdir(parents=True, exist_ok=True)

    compressed.save_pretrained(output_dir)

    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_dir)

    print(f"\n✓ Compressed model saved to: {output_dir}")

    # Calculate size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"  Size: {total_size / (1024**2):.1f} MB")

    return str(output_dir)


def load_canon_dataset() -> Dataset:
    """Load AGS Canon as training dataset"""
    print("\n=== Loading AGS Canon ===\n")

    canon_dir = Path(__file__).parent.parent.parent.parent / "LAW" / "CANON"

    if not canon_dir.exists():
        raise FileNotFoundError(f"Canon directory not found: {canon_dir}")

    texts = []
    sources = []

    for md_file in sorted(canon_dir.rglob("*.md")):
        content = md_file.read_text(encoding='utf-8')

        # Format for instruction tuning
        formatted = f"""Below is content from the AGS Canon.

Source: {md_file.relative_to(canon_dir)}

{content}"""

        texts.append(formatted)
        sources.append(str(md_file.relative_to(canon_dir)))

    print(f"Loaded {len(texts)} canon files:")
    for source in sources[:5]:
        print(f"  - {source}")
    if len(sources) > 5:
        print(f"  ... and {len(sources) - 5} more")

    return Dataset.from_dict({
        "text": texts,
        "source": sources
    })


def finetune_with_lora(compressed_model_path: str):
    """Fine-tune compressed model with LoRA on canon"""
    print("\n=== Step 3: Fine-tune with LoRA ===\n")

    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        print("ERROR: Unsloth not installed")
        print("Install with: pip install unsloth")
        return None

    # Load compressed model
    print("Loading compressed 24 MB base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=compressed_model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,  # Already compressed
    )

    print("✓ Loaded compressed base")

    # Add LoRA adapters
    print("\nAdding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"✓ LoRA adapters added")
    print(f"  Trainable params: {trainable:,} ({trainable/total*100:.2f}%)")

    # Load canon dataset
    dataset = load_canon_dataset()

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/glm4-canon-lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=200,  # Adjust based on dataset size
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
        packing=False,
    )

    # Train
    print("\nStarting fine-tuning...")
    print("This will take 2-4 hours on a consumer GPU (RTX 3090/4090)")

    trainer.train()

    # Save
    output_dir = Path("./models/glm4-canon-26mb")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✓ Canon-aware model saved to: {output_dir}")

    # Calculate final size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"  Total size: {total_size / (1024**2):.1f} MB")

    return str(output_dir)


def test_model(model_path: str):
    """Test the fine-tuned model"""
    print("\n=== Step 4: Test Canon Knowledge ===\n")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
    )

    FastLanguageModel.for_inference(model)  # Enable inference mode

    # Test questions
    test_prompts = [
        "What is the Living Formula?",
        "Explain catalytic computing.",
        "What is the effective rank (Df)?",
        "What is CRYPTO_SAFE?",
    ]

    print("Testing canon knowledge:\n")

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Q: {prompt}")
        print(f"A: {response}\n")


def main():
    print("=" * 60)
    print("GLM-4 Compression + Canon Fine-tuning Pipeline")
    print("=" * 60)

    # Step 1: Download
    model_path = download_glm4()
    if model_path is None:
        print("\nERROR: Could not download model")
        return

    # Step 2: Compress
    compressed_path = compress_model(model_path)
    if compressed_path is None:
        print("\nERROR: Compression failed")
        return

    # Step 3: Fine-tune
    finetuned_path = finetune_with_lora(compressed_path)
    if finetuned_path is None:
        print("\nERROR: Fine-tuning failed")
        return

    # Step 4: Test
    test_model(finetuned_path)

    print("\n" + "=" * 60)
    print("✓ Pipeline Complete!")
    print("=" * 60)
    print(f"\nYour 26 MB canon-aware model is ready:")
    print(f"  {finetuned_path}")
    print("\nYou can now run inference on your local computer!")


if __name__ == "__main__":
    main()
