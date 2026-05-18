"""LFM 2.5 GGUF adapter for Phase 4b runner interface.

Bridges GgufBackend into the Phase 4b run_lattice_condition interface:
    generate(prompt: str, history: list) -> (text: str, logits: np.ndarray)
"""
import sys, numpy as np
from pathlib import Path

_src = Path(__file__).resolve().parents[4] / "LAB" / "TINY_COMPRESS" / "extensions" / "03_flat_llm"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gguf_backend import GgufBackend


class LFMBackend:
    """Adapter: GgufBackend -> Phase 4b generate(prompt, history) -> (text, logits)."""

    def __init__(self, temperature=0.7):
        self.llm = GgufBackend()
        self.temperature = temperature

    def generate(self, prompt: str, history: list) -> tuple:
        # Build context from history if available
        full_prompt = prompt
        for msg in history:
            content = msg.get("content", "")
            if "VERIFICATION FAILED" in content:
                full_prompt = content + "\n\nQuestion: " + prompt

        # Generate text first
        text = self.llm.generate(full_prompt, max_tokens=40, temperature=self.temperature)

        # Get logits from a fresh eval (separate call, model is stateless after generate)
        try:
            logits = self.llm.get_logits(full_prompt).astype(np.float32)
        except Exception:
            logits = np.zeros((1, 65536), dtype=np.float32)

        return text, logits


def get_lfm_backend(temperature=0.7) -> LFMBackend:
    return LFMBackend(temperature=temperature)


if __name__ == "__main__":
    backend = get_lfm_backend()
    text, logits = backend.generate("What is the capital of France?", [])
    print("Text:", text)
    print("Logits shape:", logits.shape)
