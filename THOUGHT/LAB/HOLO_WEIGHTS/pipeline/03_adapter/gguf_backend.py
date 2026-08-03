"""GGUF backend for LFM2.5 / Qwen3.6 with CUDA -- Phase 3.5 KV Cache Compression.

Provides a GgufBackend class that loads any GGUF model via llama-cpp-python
with automatic GPU offload, giving access to generate, logits, and embeddings.

Usage:
    from gguf_backend import GgufBackend
    backend = GgufBackend()
    text = backend.generate("The capital of France is")
    logits = backend.get_logits("Hello world")
    emb = backend.get_embedding("Hello world")
"""

import os, sys, shutil, re
from pathlib import Path

# ---------------------------------------------------------------------------
# CUDA DLL bootstrap -- must run before "import llama_cpp"
# ---------------------------------------------------------------------------
_CUDA_DLL_SRC = Path(
    r"C:\Users\rene_\AppData\Local\Programs\Ollama\lib\ollama\cuda_v12"
)
_FAKE_CUDA = Path(os.environ.get("TEMP", ".")) / "_cuda_v12_fake"
_BIN = _FAKE_CUDA / "bin"

if not _BIN.is_dir() or not list(_BIN.glob("*.dll")):
    _BIN.mkdir(parents=True, exist_ok=True)
    for dll in _CUDA_DLL_SRC.glob("*.dll"):
        shutil.copy2(dll, _BIN / dll.name)

os.environ["CUDA_PATH"] = str(_FAKE_CUDA)
if str(_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_BIN) + os.pathsep + os.environ.get("PATH", "")
# ---------------------------------------------------------------------------

import numpy as np
import llama_cpp
from llama_cpp import Llama

# Pre-configured model paths
MODELS = {
    "lfm2": r"D:\Reneshizzle\Apps\LM Studio\lmstudio-community\LFM2.5-1.2B-Instruct-GGUF\LFM2.5-1.2B-Instruct-Q8_0.gguf",
    "qwen": r"D:\Reneshizzle\Apps\LM Studio\unsloth\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
}
VRAM_12GB_LAYERS = {"lfm2": -1, "qwen": 18}  # full offload for lfm2, partial for qwen


class GgufBackend:
    """Wraps a GGUF model loaded via llama-cpp-python with CUDA offload.

    Provides logits and final embeddings. Per-layer K/V states are NOT exposed
    by llama-cpp-python -- use the HF transformers path for adapter training.

    Attributes:
        model_path: Path to the GGUF file.
        n_ctx: Context window.
        n_embd: Embedding dimension.
        n_vocab: Vocabulary size.
        n_layers: Number of transformer blocks.
        arch: Model architecture name.
    """

    def __init__(self, gguf_path: str = None, model_key: str = None,
                 n_ctx: int = 2048, n_gpu_layers: int = None,
                 verbose: bool = False):
        # Resolve model path
        if gguf_path is None and model_key is None:
            model_key = "lfm2"
        if gguf_path is None:
            gguf_path = MODELS[model_key]
            if n_gpu_layers is None:
                n_gpu_layers = VRAM_12GB_LAYERS.get(model_key, -1)
        else:
            # Auto-dectect model key from path
            for key, path in MODELS.items():
                if path == gguf_path:
                    model_key = key
                    break
            if n_gpu_layers is None:
                n_gpu_layers = VRAM_12GB_LAYERS.get(model_key, -1) if model_key else -1

        if n_gpu_layers is None:
            n_gpu_layers = -1

        self.model_path = gguf_path
        self.model_key = model_key or "custom"
        self._n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.llm = Llama(
            model_path=gguf_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
            embeddings=True,
        )

        # Detect model architecture from metadata
        meta = self.llm.metadata if hasattr(self.llm, 'metadata') else {}
        self.arch = meta.get('general.architecture', 'unknown')
        self.n_embd = int(meta.get(f'{self.arch}.embedding_length', 0)) or self.llm.n_embd()
        self.n_vocab = self.llm.n_vocab()
        n_blocks = meta.get(f'{self.arch}.block_count', None)
        self.n_layers = int(n_blocks) if n_blocks else (41 if 'qwen' in self.arch else 16)
        self._version = getattr(llama_cpp, '__version__', '0.3.39')

    def _fresh_llm(self):
        """Create a fresh Llama instance for single-use operations.

        Workaround for JamePeng 0.3.39 bug where eval() can't be called
        twice on the same instance (C-level state not cleared by reset()).
        """
        return Llama(
            model_path=self.model_path,
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=False,
            embeddings=True,
        )

    def generate(self, prompt: str, max_tokens: int = 64,
                 temperature: float = 0.0) -> str:
        """Generate text completion. Can be called multiple times on same instance."""
        out = self.llm.create_completion(
            prompt, max_tokens=max_tokens, temperature=temperature, echo=False
        )
        text = out["choices"][0]["text"]
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        return text

    def chat(self, messages: list, max_tokens: int = 128,
             temperature: float = 0.0) -> str:
        """Chat completion using the model's chat template. Can be called multiple times."""
        out = self.llm.create_chat_completion(
            messages, max_tokens=max_tokens, temperature=temperature
        )
        text = out["choices"][0]["message"]["content"]
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        return text

    def get_logits(self, text: str) -> np.ndarray:
        """Get logits for next token. Shape (1, n_vocab)."""
        instance = self._fresh_llm()
        tokens = instance.tokenize(text.encode("utf-8"))
        instance.eval(tokens)
        raw = instance.eval_logits
        del instance
        if isinstance(raw, np.ndarray):
            return raw
        last = raw[-1]
        if isinstance(last, list):
            return np.array(last, dtype=np.float32).reshape(1, -1)
        return np.array(last, dtype=np.float32).reshape(1, -1)

    def tokenize(self, text: str) -> list:
        """Tokenize text to token IDs."""
        return self.llm.tokenize(text.encode("utf-8"))

    def detokenize(self, token_id: int) -> str:
        """Detokenize a single token ID to string."""
        return self.llm.detokenize([token_id]).decode("utf-8", errors="replace")

    def vocab_size(self) -> int:
        return self.n_vocab

    def get_embedding(self, text: str) -> np.ndarray:
        """Get per-token embeddings. Shape (n_tokens, n_embd)."""
        instance = self._fresh_llm()
        tokens = instance.tokenize(text.encode("utf-8"))
        instance.eval(tokens)
        from llama_cpp.llama_cpp import llama_get_embeddings
        import ctypes
        ctx = instance.ctx
        ne = self.n_embd
        emb_ptr = llama_get_embeddings(ctx)
        result = None
        if emb_ptr:
            total = ne * len(tokens)
            arr = ctypes.cast(emb_ptr, ctypes.POINTER(ctypes.c_float * total))
            result = np.array(arr[0], dtype=np.float32).reshape(len(tokens), ne)
        del instance
        if result is None:
            raise RuntimeError("llama_get_embeddings returned NULL")
        return result

    def info(self) -> dict:
        return {
            "model": str(Path(self.model_path).name),
            "key": self.model_key,
            "arch": self.arch,
            "n_embd": self.n_embd,
            "n_layers": self.n_layers,
            "n_vocab": self.n_vocab,
            "n_ctx": self.n_ctx,
            "gpu": "RTX 3060 12 GB (CUDA 12.4)",
        }

    def close(self):
        del self.llm


# ---------------------------------------------------------------------------
# Model-specific wrappers for convenience
# ---------------------------------------------------------------------------

class Lfm2Backend(GgufBackend):
    """LFM2.5 1.2B backend -- full GPU offload."""
    def __init__(self, n_ctx: int = 2048, verbose: bool = False):
        super().__init__(model_key="lfm2", n_ctx=n_ctx, n_gpu_layers=-1, verbose=verbose)


class QwenBackend(GgufBackend):
    """Qwen3.6 35B-A3B MoE backend -- partial GPU offload (18/42 layers)."""
    def __init__(self, n_ctx: int = 512, verbose: bool = False):
        super().__init__(model_key="qwen", n_ctx=n_ctx, n_gpu_layers=18, verbose=verbose)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default="lfm2")
    args = parser.parse_args()

    print(f"=== GgufBackend smoke test: {args.model} ===")

    # Single instance -- eval methods FIRST, generate/chat SECOND
    b = GgufBackend(model_key=args.model, verbose=False)
    info = b.info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    # 1. Eval-based operations first
    logits = b.get_logits("Hello world")
    print(f"  Logits shape: {logits.shape}")

    emb = b.get_embedding("Hello world")
    print(f"  Embedding shape: {emb.shape}")

    # 2. High-level operations (work after eval)
    gen = b.generate("The capital of France is", max_tokens=16)
    print(f"  Generate: {repr(gen)}")

    chat = b.chat([{"role": "user", "content": "Say hello in one word."}], max_tokens=16)
    print(f"  Chat: {repr(chat)}")

    b.close()
    print("=== OK ===")
