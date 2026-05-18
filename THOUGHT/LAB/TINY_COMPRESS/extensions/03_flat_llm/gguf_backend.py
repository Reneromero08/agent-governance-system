"""GGUF backend for LFM2.5 with CUDA -- Phase 3.5 KV Cache Compression.

Provides a LlamaModel wrapper that loads LFM2.5 GGUF via llama-cpp-python
with full GPU offload, giving access to logits and embeddings.

Usage:
    from gguf_backend import GgufBackend
    backend = GgufBackend()
    text = backend.generate("The capital of France is")
    logits = backend.get_logits("Hello world")
    emb = backend.get_embedding("Hello world")
"""

import os, sys, shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# CUDA DLL bootstrap -- must run before "import llama_cpp"
# ---------------------------------------------------------------------------
_CUDA_DLL_SRC = Path(
    r"C:\Users\rene_\AppData\Local\Programs\Ollama\lib\ollama\cuda_v12"
)
_FAKE_CUDA = Path(os.environ.get("TEMP", ".")) / "_cuda_v12_fake"
_BIN = _FAKE_CUDA / "bin"
_LIB = _FAKE_CUDA / "lib"

if not _BIN.is_dir() or not list(_BIN.glob("*.dll")):
    _BIN.mkdir(parents=True, exist_ok=True)
    _LIB.mkdir(exist_ok=True)
    for dll in _CUDA_DLL_SRC.glob("*.dll"):
        shutil.copy2(dll, _BIN / dll.name)

os.environ["CUDA_PATH"] = str(_FAKE_CUDA)
if str(_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_BIN) + os.pathsep + os.environ.get("PATH", "")
# ---------------------------------------------------------------------------

import numpy as np
from llama_cpp import Llama

GGUF_PATH = r"D:\Reneshizzle\Apps\LM Studio\lmstudio-community\LFM2.5-1.2B-Instruct-GGUF\LFM2.5-1.2B-Instruct-Q8_0.gguf"


class GgufBackend:
    """Wraps a GGUF model loaded via llama-cpp-python with CUDA offload.

    Provides the signals available from llama.cpp: logits and final embeddings.
    Per-layer K/V states are NOT exposed by this backend -- use the HF
    transformers path (via the original flat_llm_adapter) for adapter training.

    Attributes:
        model_path: Path to the GGUF file.
        n_ctx: Context window.
        n_embd: Embedding dimension (2048 for LFM2.5 1.2B).
        n_vocab: Vocabulary size (65536 for LFM2.5).
    """

    def __init__(self, gguf_path: str = GGUF_PATH, n_ctx: int = 2048,
                 n_gpu_layers: int = -1, verbose: bool = False):
        self.model_path = gguf_path

        self.llm = Llama(
            model_path=gguf_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
            embedding=True,
        )
        # Model metadata
        self.n_ctx = n_ctx
        self.n_embd = 2048
        self.n_vocab = 65536
        self.n_layers = 16

    def generate(self, prompt: str, max_tokens: int = 64,
                 temperature: float = 0.0, **kwargs) -> str:
        """Generate text completion."""
        out = self.llm(prompt, max_tokens=max_tokens,
                       temperature=temperature, echo=False, **kwargs)
        return out["choices"][0]["text"]

    def chat(self, messages: list, max_tokens: int = 128,
             temperature: float = 0.0) -> str:
        """Chat completion using the model's chat template."""
        out = self.llm.create_chat_completion(
            messages, max_tokens=max_tokens, temperature=temperature
        )
        return out["choices"][0]["message"]["content"]

    def get_logits(self, text: str) -> np.ndarray:
        """Get logits for each token in text. Shape (n_tokens, n_vocab)."""
        tokens = self.llm.tokenize(text.encode("utf-8"))
        self.llm.reset()
        self.llm.eval(tokens)
        logits = np.array(self.llm.eval_logits)
        return logits

    def tokenize(self, text: str) -> list:
        """Tokenize text to token IDs."""
        return self.llm.tokenize(text.encode("utf-8"))

    def detokenize(self, token_id: int) -> str:
        """Detokenize a single token ID to string."""
        return self.llm.detokenize([token_id]).decode("utf-8", errors="replace")

    def vocab_size(self) -> int:
        return self.n_vocab

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding (pooled hidden state). Shape (1, n_embd)."""
        emb = self.llm.create_embedding(text)
        return np.array(emb["data"][0]["embedding"], dtype=np.float64)

    def info(self) -> dict:
        return {
            "model": "LFM2.5-1.2B-Instruct-Q8_0",
            "arch": "lfm2",
            "n_embd": self.n_embd,
            "n_layers": self.n_layers,
            "n_vocab": self.n_vocab,
            "n_ctx": self.n_ctx,
            "gpu": "RTX 3060 12 GB (CUDA 12.4)",
            "layers_offloaded": 17,
        }

    def close(self):
        del self.llm


if __name__ == "__main__":
    print("=== GgufBackend smoke test ===")
    b = GgufBackend(verbose=False)
    print(f"  Info: {b.info()}")

    gen = b.generate("The capital of France is", max_tokens=16)
    print(f"  Generate: {repr(gen)}")

    logits = b.get_logits("Hello world")
    print(f"  Logits shape: {logits.shape}")

    emb = b.get_embedding("Test embedding.")
    print(f"  Embedding shape: {emb.shape}")

    chat = b.chat([
        {"role": "user", "content": "Say hello in one word."}
    ], max_tokens=16)
    print(f"  Chat: {repr(chat)}")

    b.close()
    print("=== OK ===")
