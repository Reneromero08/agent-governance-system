"""TraDo-4B-Instruct dLLM model loader for Phase 4b.

Supports:
1. Direct BF16 loading (requires ~9GB VRAM)
2. 4-bit bitsandbytes quantization (requires ~3GB VRAM)
3. CPU offloading via device_map="auto"
4. Hidden state access via output_hidden_states=True
5. Diffusion step access for step-level verification

Call get_model() to load, or use the RealTraDoGenerator wrapper
to integrate with the phase4b control loop.
"""

from .model_loader import (
    TraDoModel,
    RealTraDoGenerator,
    MODEL_PATH,
    get_model,
    check_model_ready,
)
