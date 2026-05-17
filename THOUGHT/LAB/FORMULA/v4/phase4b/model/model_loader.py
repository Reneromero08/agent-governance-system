"""TraDo-4B-Instruct dLLM model loader.

Loads the model from the local `model/` directory with fallback options.
Supports BF16, 8-bit, and 4-bit (bitsandbytes) quantization.
Provides hidden state access for resonance computation.
"""

import logging, os, sys, time, numpy as np
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .block_diffusion_generate import block_diffusion_generate

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent


class TraDoModel:
    """Wrapper around the loaded TraDo model with convenience methods."""

    def __init__(self, model, tokenizer, quant_type: str = "bf16"):
        self.model = model
        self.tokenizer = tokenizer
        self.quant_type = quant_type
        self.device = next(model.parameters()).device
        self.hidden_dim = model.config.hidden_size

    def generate(self, prompt: str, history: list = None) -> tuple[str, np.ndarray]:
        """Generate text using TraDo block diffusion LM.

        Uses block_diffusion_generate instead of standard autoregressive generate.
        TraDo is a block diffusion model: starts with MASK tokens, iteratively
        denoises through multiple steps per block.

        Returns (generated_text, logits_array).
        """
        history = history or []
        messages = [{"role": "user", "content": prompt}]
        for msg in history:
            messages.append(msg)

        chat = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer.batch_encode_plus(
            [chat], return_tensors='pt', padding=True, truncation=True, max_length=200)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = block_diffusion_generate(
                self.model,
                prompt=inputs,
                mask_id=self.tokenizer.mask_token_id,
                gen_length=200,
                block_length=4,
                denoising_steps=4,
                temperature=0.6,
                remasking_strategy='low_confidence_dynamic',
                confidence_threshold=0.9,
            )

        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        # Strip MASK tokens from output (block diffusion may leave some)
        generated_text = generated_text.replace('<|MASK|>', '').strip()

        # Return logits=None for now (diffusion has no single-step logits)
        return generated_text, None

        # Decode generated tokens
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Return logits=None for now (diffusion has no single-step logits)
        return generated_text, None

    def get_last_hidden(self) -> np.ndarray:
        """Get the last hidden states for R = Tr(rho @ C) computation.

        Returns a numpy array of shape [hidden_dim] or None.
        """
        if self.last_hidden_states is None:
            return None
        # Take last layer, last token hidden state
        last_layer = self.last_hidden_states[-1]  # list per generated token
        if isinstance(last_layer, tuple):
            last_layer = last_layer[-1]
        if isinstance(last_layer, torch.Tensor):
            # last_layer shape: [batch, seq_len, hidden_dim] or similar
            h = last_layer[0, -1, :].float()
            return h.cpu().numpy()
        return None


def get_model(
    quantize: str = "auto",
    device_map: str = "auto",
    max_memory: dict = None,
) -> TraDoModel:
    """Load the TraDo-4B-Instruct model.

    Args:
        quantize: One of "auto" (try q4, fall back to bf16), "q4", "q8", "bf16".
        device_map: Device map for model parallelism.
        max_memory: Optional dict like {0: "10GiB", "cpu": "16GiB"}.

    Returns:
        TraDoModel wrapper instance.
    """
    if max_memory is None:
        max_memory = {0: "10GiB", "cpu": "32GiB"}

    model_dir = str(MODEL_PATH)

    logger.info(f"Loading TraDo-4B-Instruct from {model_dir}")
    logger.info(f"Using device_map={device_map}, quantize={quantize}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Determine quantization
    quant_type = "bf16"
    quantization_config = None
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "torch_dtype": torch.bfloat16,
    }

    if quantize in ("auto", "q4"):
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quantization_config
            quant_type = "q4"
            logger.info("Using 4-bit NF4 quantization (fits ~3GB VRAM)")
        except Exception as e:
            logger.warning(f"4-bit quantization failed: {e}")

    if quantize == "q8" or (quant_type == "bf16" and quantize in ("auto", "q8")):
        try:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = quantization_config
            quant_type = "q8"
            logger.info("Using 8-bit quantization (fits ~5GB VRAM)")
        except Exception as e:
            logger.warning(f"8-bit quantization failed: {e}")

    if quant_type == "bf16":
        logger.info("Using BF16 (requires ~9GB VRAM)")

    # Set max_memory if not using auto device_map
    if device_map != "auto":
        model_kwargs["max_memory"] = max_memory

    # Load
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, **model_kwargs)
    model.eval()

    elapsed = time.time() - t0
    logger.info(f"Model loaded in {elapsed:.1f}s ({quant_type})")

    return TraDoModel(model, tokenizer, quant_type)


def check_model_ready() -> dict:
    """Check if the model is downloaded and ready to load.

    Returns dict with status information.
    """
    safetensors = list(MODEL_PATH.glob("*.safetensors"))
    config = MODEL_PATH / "config.json"
    modeling = MODEL_PATH / "modeling_sdar.py"

    all_present = len(safetensors) >= 2 and config.exists() and modeling.exists()
    total_size_gb = sum(f.stat().st_size for f in safetensors) / (1024**3) if safetensors else 0

    return {
        "ready": all_present,
        "n_shards": len(safetensors),
        "total_size_gb": round(total_size_gb, 2),
        "config_exists": config.exists(),
        "modeling_code_exists": modeling.exists(),
        "path": str(MODEL_PATH),
    }


class RealTraDoGenerator:
    """Adapter that wraps TraDoModel.generate for the phase4b control loop.

    Conforms to the AgentGenerateFn signature:
        (prompt: str, history: list) -> (text: str, logits: np.ndarray)
    """

    def __init__(self, model: TraDoModel):
        self.model = model

    def generate(self, prompt: str, history: list) -> tuple[str, np.ndarray]:
        return self.model.generate(prompt, history)
