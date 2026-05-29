"""
LLM Client for CAT Chat
=======================

Configurable LLM backend with OpenAI-compatible API support.

Configuration loaded from _generated/llm_config.json.

Usage:
    from catalytic_chat.llm_client import get_llm_client, llm_generate

    # Get configured client
    client = get_llm_client()

    # Generate response
    response = client.generate("What is authentication?", context="OAuth is...")

    # Or use the simple function interface
    response = llm_generate("What is auth?", ["OAuth docs...", "JWT docs..."])
"""

import json
import os
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    name: str
    base_url: str
    model: str
    api_key: str
    max_tokens: int = 4096
    temperature: float = 0.7

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        """Create config from dictionary."""
        # Handle api_key vs api_key_env
        api_key = data.get("api_key", "")
        if "api_key_env" in data:
            api_key = os.environ.get(data["api_key_env"], "")

        return cls(
            name=data.get("name", "Unknown"),
            base_url=data.get("base_url", ""),
            model=data.get("model", ""),
            api_key=api_key,
            max_tokens=data.get("max_tokens", 4096),
            temperature=data.get("temperature", 0.7),
        )


def load_llm_config(provider: Optional[str] = None) -> LLMConfig:
    """
    Load LLM configuration from _generated/llm_config.json.

    Args:
        provider: Provider name (e.g., "local", "openai", "anthropic").
                  If None, uses the "default" provider from config.

    Returns:
        LLMConfig for the specified provider
    """
    config_path = Path(__file__).parent.parent / "_generated" / "llm_config.json"

    if not config_path.exists():
        # Return default local config if file doesn't exist
        return LLMConfig(
            name="Local LLM (default)",
            base_url="http://10.5.0.2:1234",
            model="zai-org/glm-4.7-flash",
            api_key="not-needed",
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    # Get provider name
    if provider is None:
        provider = config_data.get("default", "local")

    # Get provider config
    providers = config_data.get("providers", {})
    if provider not in providers:
        raise ValueError(f"Unknown LLM provider: {provider}. Available: {list(providers.keys())}")

    return LLMConfig.from_dict(providers[provider])


# =============================================================================
# LLM Client
# =============================================================================

class LLMClient:
    """
    LLM client with OpenAI-compatible API support.

    Supports any OpenAI-compatible endpoint (OpenAI, local LLMs, etc.)
    """

    def __init__(self, config: Optional[LLMConfig] = None, provider: Optional[str] = None):
        """
        Initialize LLM client.

        Args:
            config: LLMConfig instance (if provided, overrides provider)
            provider: Provider name from llm_config.json
        """
        self.config = config or load_llm_config(provider)

    def generate(
        self,
        query: str,
        context: str = "",
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a response using the configured LLM.

        Args:
            query: User query
            context: Context to include (prepended to query)
            system_prompt: System prompt
            max_tokens: Override max tokens
            temperature: Override temperature

        Returns:
            Generated response text
        """
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Combine context and query
        if context:
            user_content = f"{context}\n\n{query}"
        else:
            user_content = query

        messages.append({"role": "user", "content": user_content})

        # Make request
        return self._chat_completion(
            messages=messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
        )

    def generate_with_context_list(
        self,
        query: str,
        context_docs: List[str],
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        """
        Generate response with a list of context documents.

        Args:
            query: User query
            context_docs: List of context documents
            system_prompt: System prompt

        Returns:
            Generated response text
        """
        # Join context docs
        if context_docs:
            context = "\n\n---\n\n".join(context_docs)
        else:
            context = ""

        return self.generate(query, context=context, system_prompt=system_prompt)

    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Make chat completion request to OpenAI-compatible API.

        Args:
            messages: List of message dicts with role and content
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        url = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"

        # Handle base_url that already includes /v1
        if "/v1/v1/" in url:
            url = url.replace("/v1/v1/", "/v1/")

        headers = {
            "Content-Type": "application/json",
        }

        # Add auth header if API key is provided and not "not-needed"
        if self.config.api_key and self.config.api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"LLM request failed: {e}") from e
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Invalid LLM response format: {e}") from e

    def get_generate_fn(self) -> Callable[[str, List[str]], str]:
        """
        Get a generate function compatible with GeometricChat.respond().

        Returns:
            Function(query, context_docs) -> response
        """
        def generate_fn(query: str, context_docs: List[str]) -> str:
            return self.generate_with_context_list(query, context_docs)
        return generate_fn

    def get_catalytic_generate_fn(self) -> Callable[[str, str], str]:
        """
        Get a generate function compatible with GeometricChat.respond_catalytic().

        Returns:
            Function(system_prompt, context_with_query) -> response
        """
        def generate_fn(system_prompt: str, context_with_query: str) -> str:
            return self.generate(context_with_query, system_prompt=system_prompt)
        return generate_fn


# =============================================================================
# Convenience Functions
# =============================================================================

_default_client: Optional[LLMClient] = None


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """
    Get LLM client instance.

    Args:
        provider: Provider name (uses default if None)

    Returns:
        LLMClient instance
    """
    global _default_client

    if provider is None and _default_client is not None:
        return _default_client

    client = LLMClient(provider=provider)

    if provider is None:
        _default_client = client

    return client


def llm_generate(query: str, context_docs: List[str]) -> str:
    """
    Generate response using default LLM client.

    This is a convenience function compatible with GeometricChat.respond().

    Args:
        query: User query
        context_docs: List of context documents

    Returns:
        Generated response text
    """
    client = get_llm_client()
    return client.generate_with_context_list(query, context_docs)


def llm_generate_catalytic(system_prompt: str, context_with_query: str) -> str:
    """
    Generate response using default LLM client (catalytic format).

    This is a convenience function compatible with GeometricChat.respond_catalytic().

    Args:
        system_prompt: System prompt
        context_with_query: Combined context and query

    Returns:
        Generated response text
    """
    client = get_llm_client()
    return client.generate(context_with_query, system_prompt=system_prompt)


# =============================================================================
# CLI / Testing
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=== LLM Client Test ===\n")

    # Load config
    try:
        config = load_llm_config()
        print(f"Provider: {config.name}")
        print(f"Model: {config.model}")
        print(f"Base URL: {config.base_url}")
        print()
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)

    # Test generation
    print("Testing generation...")
    try:
        client = LLMClient(config=config)
        response = client.generate(
            "Say 'Hello from CAT Chat!' in exactly 5 words.",
            max_tokens=50,
        )
        print(f"Response: {response}")
        print("\nSuccess!")
    except Exception as e:
        print(f"Generation failed: {e}")
        print("\nMake sure the LLM server is running at the configured URL.")
        sys.exit(1)
