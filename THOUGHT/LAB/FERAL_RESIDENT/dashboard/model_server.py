"""
Model Server - Loads transformers ONCE and serves embeddings via HTTP.

Run this in the background:
    python model_server.py

Then start feral_server.py which will connect to this.
No more waiting for transformers to load every time!
"""

import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

import sys
from pathlib import Path

# Add FERAL_RESIDENT to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "THOUGHT" / "LAB" / "FERAL_RESIDENT"))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import numpy as np

# Suppress warnings
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

print("Loading sentence-transformers... (this only happens once)")
from sentence_transformers import SentenceTransformer

# Load model once at startup
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
print(f"Model '{MODEL_NAME}' loaded and ready!")

app = FastAPI(title="Feral Model Server")


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    """Generate embeddings for a list of texts."""
    embeddings = model.encode(request.texts, convert_to_numpy=True)
    return EmbedResponse(embeddings=embeddings.tolist())


@app.post("/embed_single")
def embed_single(text: str):
    """Generate embedding for a single text (convenience endpoint)."""
    embedding = model.encode([text], convert_to_numpy=True)[0]
    return {"embedding": embedding.tolist()}


if __name__ == "__main__":
    print("\n" + "="*50)
    print("FERAL MODEL SERVER")
    print("="*50)
    print(f"Model: {MODEL_NAME}")
    print(f"Endpoint: http://localhost:8421")
    print("="*50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8421, log_level="warning")
