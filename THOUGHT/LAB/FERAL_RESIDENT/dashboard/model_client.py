"""
Model Client - Connects to model_server.py for embeddings.

Falls back to local loading if model server isn't running.
"""

import requests
import numpy as np
from typing import List, Union, Optional

MODEL_SERVER_URL = "http://localhost:8421"


def is_model_server_running() -> bool:
    """Check if the model server is running."""
    try:
        resp = requests.get(f"{MODEL_SERVER_URL}/health", timeout=1)
        return resp.status_code == 200
    except:
        return False


class RemoteEmbedder:
    """Embedder that uses the model server."""

    def __init__(self):
        self._check_server()

    def _check_server(self):
        if not is_model_server_running():
            raise ConnectionError(
                "Model server not running! Start it with:\n"
                "  python model_server.py\n"
                "Or set USE_MODEL_SERVER=false to load locally."
            )

    def encode(self, texts: Union[str, List[str]], convert_to_numpy: bool = True) -> np.ndarray:
        """Encode texts using the model server."""
        if isinstance(texts, str):
            texts = [texts]

        resp = requests.post(
            f"{MODEL_SERVER_URL}/embed",
            json={"texts": texts},
            timeout=30
        )
        resp.raise_for_status()

        embeddings = np.array(resp.json()["embeddings"])
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
