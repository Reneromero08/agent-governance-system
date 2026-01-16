"""
Model Client - Connects to model_server.py for embeddings.

Falls back to local loading if model server isn't running.

Performance Features:
- LRU cache for embedding results (configurable size from config.json)
- Avoids recomputing embeddings for identical texts
"""

import requests
import numpy as np
import hashlib
import json
from pathlib import Path
from typing import List, Union, Optional, Dict, Tuple
from collections import OrderedDict

MODEL_SERVER_URL = "http://localhost:8421"

# LRU Cache for embeddings - stores (hash -> embedding) with max size
_embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
_cache_max_size = 500  # Default, updated from config.json
_cache_hits = 0
_cache_misses = 0


def _get_cache_size_from_config() -> int:
    """Read embedding_lru_size from config.json if available."""
    try:
        config_path = Path(__file__).parent.parent / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            return cfg.get('cache', {}).get('embedding_lru_size', 500)
    except Exception:
        pass
    return 500


def _text_hash(text: str) -> str:
    """Create a hash key for a text string."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _cache_get(text: str) -> Optional[np.ndarray]:
    """Get embedding from cache if exists."""
    global _cache_hits, _cache_misses
    key = _text_hash(text)
    if key in _embedding_cache:
        # Move to end (most recently used)
        _embedding_cache.move_to_end(key)
        _cache_hits += 1
        return _embedding_cache[key]
    _cache_misses += 1
    return None


def _cache_put(text: str, embedding: np.ndarray):
    """Store embedding in cache, evicting oldest if full."""
    global _cache_max_size
    key = _text_hash(text)
    _embedding_cache[key] = embedding
    _embedding_cache.move_to_end(key)
    # Evict oldest if over size
    while len(_embedding_cache) > _cache_max_size:
        _embedding_cache.popitem(last=False)


def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    return {
        'size': len(_embedding_cache),
        'max_size': _cache_max_size,
        'hits': _cache_hits,
        'misses': _cache_misses,
        'hit_rate': _cache_hits / max(1, _cache_hits + _cache_misses)
    }


def clear_embedding_cache():
    """Clear the embedding cache."""
    global _cache_hits, _cache_misses
    _embedding_cache.clear()
    _cache_hits = 0
    _cache_misses = 0


def is_model_server_running() -> bool:
    """Check if the model server is running."""
    try:
        resp = requests.get(f"{MODEL_SERVER_URL}/health", timeout=1)
        return resp.status_code == 200
    except:
        return False


class RemoteEmbedder:
    """Embedder that uses the model server with LRU caching."""

    def __init__(self):
        global _cache_max_size
        self._check_server()
        # Initialize cache size from config
        _cache_max_size = _get_cache_size_from_config()

    def _check_server(self):
        if not is_model_server_running():
            raise ConnectionError(
                "Model server not running! Start it with:\n"
                "  python model_server.py\n"
                "Or set USE_MODEL_SERVER=false to load locally."
            )

    def encode(self, texts: Union[str, List[str]], convert_to_numpy: bool = True) -> np.ndarray:
        """Encode texts using the model server with LRU caching.

        Single texts are cached individually. For batches, each text
        is checked against the cache and only uncached texts are sent
        to the server.
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Check cache for each text
        results: Dict[int, np.ndarray] = {}
        uncached_texts: List[Tuple[int, str]] = []

        for i, text in enumerate(texts):
            cached = _cache_get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_texts.append((i, text))

        # Fetch uncached embeddings from server
        if uncached_texts:
            texts_to_embed = [t for _, t in uncached_texts]
            resp = requests.post(
                f"{MODEL_SERVER_URL}/embed",
                json={"texts": texts_to_embed},
                timeout=30
            )
            resp.raise_for_status()

            server_embeddings = np.array(resp.json()["embeddings"])

            # Store in cache and results
            for j, (orig_idx, text) in enumerate(uncached_texts):
                embedding = server_embeddings[j]
                _cache_put(text, embedding)
                results[orig_idx] = embedding

        # Reconstruct ordered result array
        embeddings = np.array([results[i] for i in range(len(texts))])

        return embeddings


def get_embedder(use_remote: Optional[bool] = None):
    """
    Get an embedder instance.

    If use_remote is None, auto-detect based on model server availability.
    If use_remote is True, require model server.
    If use_remote is False, always load locally.
    """
    import os

    # Check environment variable
    env_val = os.environ.get("USE_MODEL_SERVER", "").lower()
    if env_val == "false":
        use_remote = False
    elif env_val == "true":
        use_remote = True

    # Auto-detect if not specified
    if use_remote is None:
        use_remote = is_model_server_running()

    if use_remote:
        return RemoteEmbedder()
    else:
        # Fall back to local loading
        print("Model server not running, loading transformers locally...")
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
