"""LLM Vector Communication Bridge.

Enables LLM-to-LLM communication via vectors by using embedding models
as "antennas". The LLM doesn't need to understand vectors directly -
it uses the embedding model to encode/decode semantic meaning.

Architecture:
    LLM_A --> embed_a.encode() --> 48D vector --> transmit
    --> embed_b.decode() --> text --> LLM_B interprets

Usage:
    bridge = LLMVectorBridge(
        embed_url="http://10.5.0.2:1234/v1/embeddings",
        embed_model="text-embedding-nomic-embed-text-v1.5",
        llm_url="http://10.5.0.2:1234/v1/chat/completions",
        llm_model="nemotron-3-nano-30b-a3b"
    )

    # Create alignment key
    key = bridge.create_alignment_key()

    # Encode a message
    vector = bridge.encode("Hello, how are you?")

    # Send vector to LLM (it decodes and interprets)
    response = bridge.vector_to_llm(vector, candidates, system_prompt)
"""

import json
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .alignment_key import AlignmentKey
from .canonical_anchors import CANONICAL_128


@dataclass
class LLMVectorBridge:
    """Bridge for LLM communication via embedding vectors."""

    embed_url: str
    embed_model: str
    llm_url: Optional[str] = None
    llm_model: Optional[str] = None
    timeout: int = 60

    _key: Optional[AlignmentKey] = None

    def __post_init__(self):
        """Initialize the bridge."""
        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Verify the embedding endpoint is accessible."""
        try:
            response = requests.post(
                self.embed_url,
                json={"model": self.embed_model, "input": ["test"]},
                timeout=10
            )
            response.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Cannot connect to embedding endpoint: {e}")

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from the API.

        Args:
            texts: List of texts to embed

        Returns:
            (n, dim) array of embeddings
        """
        response = requests.post(
            self.embed_url,
            json={"model": self.embed_model, "input": texts},
            timeout=self.timeout
        )
        response.raise_for_status()

        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return np.array(embeddings)

    def create_alignment_key(self, k: int = 48) -> AlignmentKey:
        """Create an alignment key using the remote embedding model.

        Args:
            k: Number of MDS dimensions (default: 48)

        Returns:
            AlignmentKey ready for encoding/decoding
        """
        self._key = AlignmentKey.create(
            model_id=self.embed_model,
            embed_fn=self._embed,
            anchors=CANONICAL_128,
            k=k
        )
        return self._key

    def get_key(self) -> AlignmentKey:
        """Get or create the alignment key."""
        if self._key is None:
            self.create_alignment_key()
        return self._key

    def encode(self, text: str) -> np.ndarray:
        """Encode text to a vector.

        Args:
            text: Text to encode

        Returns:
            (k,) MDS coordinates
        """
        key = self.get_key()
        return key.encode(text, self._embed)

    def decode(self, vector: np.ndarray, candidates: List[str]) -> Tuple[str, float]:
        """Decode a vector to the best-matching candidate.

        Args:
            vector: (k,) MDS coordinates
            candidates: List of possible texts

        Returns:
            (best_match, similarity_score)
        """
        key = self.get_key()
        return key.decode(vector, candidates, self._embed)

    def decode_all(self, vector: np.ndarray, candidates: List[str]) -> List[Tuple[str, float]]:
        """Decode with all scores.

        Args:
            vector: (k,) MDS coordinates
            candidates: List of possible texts

        Returns:
            List of (candidate, score) sorted by score descending
        """
        key = self.get_key()
        return key.decode_all(vector, candidates, self._embed)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat completion request to the LLM.

        Args:
            messages: List of {role, content} dicts
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Assistant response content
        """
        if not self.llm_url or not self.llm_model:
            raise ValueError("LLM endpoint not configured")

        payload = {
            "model": self.llm_model,
            "messages": messages,
            **kwargs
        }

        response = requests.post(
            self.llm_url,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def vector_to_llm(
        self,
        vector: np.ndarray,
        candidates: List[str],
        system_prompt: Optional[str] = None,
        interpret: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a vector to the LLM for decoding and interpretation.

        This is the core of LLM vector communication:
        1. Decode the vector to find best-matching candidate
        2. (Optionally) Have the LLM interpret/respond to the decoded message

        Args:
            vector: (k,) MDS coordinates received
            candidates: Possible messages to decode against
            system_prompt: System prompt for interpretation
            interpret: Whether to have LLM interpret the decoded message
            **kwargs: Additional chat parameters

        Returns:
            Dict with decoded_text, score, and optionally llm_response
        """
        # Decode the vector
        decoded, score = self.decode(vector, candidates)
        all_scores = self.decode_all(vector, candidates)

        result = {
            "decoded_text": decoded,
            "confidence": float(score),
            "all_scores": [(t, float(s)) for t, s in all_scores[:5]],
            "vector_dim": len(vector),
        }

        if interpret and self.llm_url:
            # Have the LLM interpret the decoded message
            if system_prompt is None:
                system_prompt = (
                    "You received a message that was transmitted via vector encoding. "
                    "The decoded message is shown below. Respond naturally to it."
                )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"[Vector-decoded message]: {decoded}"}
            ]

            response = self.chat(messages, **kwargs)
            result["llm_response"] = response

        return result

    def send_message(
        self,
        message: str,
        candidates: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Full send cycle: encode message, decode at receiver, interpret.

        This demonstrates the complete vector communication protocol:
        1. Encode the message to a 48D vector
        2. (Vector could be transmitted over any channel here)
        3. Decode the vector back to text
        4. Have the LLM interpret and respond

        Args:
            message: Message to send
            candidates: Candidate pool for decoding (must include message)
            system_prompt: System prompt for LLM interpretation
            **kwargs: Additional parameters

        Returns:
            Dict with vector, decoded text, and LLM response
        """
        # Encode
        vector = self.encode(message)

        # Decode and interpret
        result = self.vector_to_llm(vector, candidates, system_prompt, **kwargs)
        result["original_message"] = message
        result["vector"] = vector.tolist()
        result["transmission_success"] = result["decoded_text"] == message

        return result


def demo_llm_communication(embed_url: str, embed_model: str, llm_url: str, llm_model: str):
    """Demonstrate LLM-to-LLM vector communication.

    Args:
        embed_url: Embedding API URL
        embed_model: Embedding model name
        llm_url: LLM chat API URL
        llm_model: LLM model name
    """
    print("=" * 70)
    print("LLM VECTOR COMMUNICATION DEMO")
    print("=" * 70)

    # Create bridge
    print("\n[1] Creating vector bridge...")
    bridge = LLMVectorBridge(
        embed_url=embed_url,
        embed_model=embed_model,
        llm_url=llm_url,
        llm_model=llm_model
    )

    # Create alignment key
    print(f"\n[2] Creating alignment key...")
    print(f"    Embedding model: {embed_model}")
    key = bridge.create_alignment_key(k=48)
    print(f"    Key created: k={key.k}, anchor_hash={key.anchor_hash}")

    # Test messages
    messages = [
        "Hello, how are you today?",
        "What is the meaning of life?",
        "The weather is beautiful outside.",
        "I need help with a coding problem.",
        "Let's discuss artificial intelligence.",
    ]

    distractors = [
        "Goodbye, see you later.",
        "The cat sat on the mat.",
        "Pizza is my favorite food.",
        "The stock market crashed today.",
        "I love hiking in the mountains.",
        "Science fiction movies are entertaining.",
        "Music makes everything better.",
        "The book was incredibly boring.",
    ]

    candidates = messages + distractors

    print(f"\n[3] Testing vector communication...")
    print(f"    Messages: {len(messages)}")
    print(f"    Distractors: {len(distractors)}")
    print(f"    Total candidates: {len(candidates)}")

    # Test each message
    correct = 0
    for msg in messages:
        result = bridge.send_message(
            msg,
            candidates,
            system_prompt="You received a vector-encoded message. Respond briefly.",
            temperature=0.7,
            max_tokens=100
        )

        ok = result["transmission_success"]
        correct += ok

        print(f"\n  TX: \"{msg[:50]}...\"" if len(msg) > 50 else f"\n  TX: \"{msg}\"")
        print(f"  VEC: [{result['vector'][0]:+.3f}, {result['vector'][1]:+.3f}, ...] ({len(result['vector'])}D)")
        print(f"  RX: \"{result['decoded_text'][:50]}...\"" if len(result['decoded_text']) > 50 else f"  RX: \"{result['decoded_text']}\"")
        print(f"  SIM: {result['confidence']:.4f} {'[OK]' if ok else '[FAIL]'}")
        if "llm_response" in result:
            resp = result["llm_response"][:80] + "..." if len(result.get("llm_response", "")) > 80 else result.get("llm_response", "")
            print(f"  LLM: \"{resp}\"")

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print("=" * 70)
    print(f"  Accuracy: {correct}/{len(messages)} ({100*correct/len(messages):.0f}%)")
    print(f"  Compression: embedding_dim -> {key.k}D")
    print(f"  Anchor hash: {key.anchor_hash}")

    return bridge


if __name__ == "__main__":
    # Default to local LM Studio endpoint
    demo_llm_communication(
        embed_url="http://10.5.0.2:1234/v1/embeddings",
        embed_model="text-embedding-nomic-embed-text-v1.5",
        llm_url="http://10.5.0.2:1234/v1/chat/completions",
        llm_model="nemotron-3-nano-30b-a3b"
    )
