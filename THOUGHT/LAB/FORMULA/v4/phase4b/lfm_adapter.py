"""LFM 2.5 GGUF adapter for Phase 4b runner interface.

Bridges GgufBackend into the Phase 4b run_lattice_condition interface:
    generate(prompt: str, history: list) -> (text: str, logits: np.ndarray)

Correction path uses create_chat_completion with system prompt for
proper role handling. System prompt primes model to trust retrieved context.
"""
import sys, numpy as np
from pathlib import Path

_src = Path(__file__).resolve().parents[4] / "LAB" / "TINY_COMPRESS" / "extensions" / "03_flat_llm"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gguf_backend import GgufBackend

SYSTEM_PROMPT = (
    "You are a precise question-answering agent. "
    "When given a correction or context, you MUST use the exact information provided. "
    "If told the correct answer is X, respond with X. "
    "Do not guess. Do not hallucinate. Trust the context you are given. "
    "Respond in one short sentence."
)


class LFMBackend:
    """Adapter: GgufBackend -> Phase 4b generate(prompt, history) -> (text, logits)."""

    def __init__(self, temperature=0.7):
        self.llm = GgufBackend()
        self.temperature = temperature

    def generate(self, prompt: str, history: list) -> tuple:
        has_correction = any("VERIFICATION FAILED" in msg.get("content", "")
                            for msg in history)

        if has_correction and history:
            # Build chat messages with proper roles
            correction = ""
            for msg in history:
                if "VERIFICATION FAILED" in msg.get("content", ""):
                    correction = msg["content"]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": history[1].get("content", "")[:200] if len(history) > 1 else ""},
                {"role": "system", "content": correction[:400]},
                {"role": "user", "content": "Given this correction, what is the correct answer? Reply in one sentence."},
            ]
            try:
                text = self.llm.chat(messages, max_tokens=50, temperature=0.0)
            except Exception:
                # Fallback: inline correction
                full = correction[:400] + "\n\n" + prompt + "\nAnswer in one sentence."
                text = self.llm.generate(full, max_tokens=50, temperature=0.0)
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            try:
                text = self.llm.chat(messages, max_tokens=50, temperature=self.temperature)
            except Exception:
                text = self.llm.generate(prompt, max_tokens=50, temperature=self.temperature)

        try:
            logits = self.llm.get_logits(prompt).astype(np.float32)
        except Exception:
            logits = np.zeros((1, 65536), dtype=np.float32)

        return text, logits


def get_lfm_backend(temperature=0.7) -> LFMBackend:
    return LFMBackend(temperature=temperature)


if __name__ == "__main__":
    backend = get_lfm_backend()
    text, _ = backend.generate("What is the capital of France?", [])
    print("Generate:", text)

    # Test correction
    text2, _ = backend.generate("What does INV-005 state?", [
        {"role": "user", "content": "What does INV-005 state?"},
        {"role": "assistant", "content": "INV-005 states that quality is declining."},
        {"role": "system", "content": "VERIFICATION FAILED. Correct answer: determinism"},
    ])
    print("Correction:", text2)
