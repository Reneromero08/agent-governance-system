#!/usr/bin/env python3
"""
Oracle Bridge - Connect Local Model to External AI Services

Primary: DuckDuckGo AI Chat (free, no API key)
Fallback: OpenAI, Anthropic, or local LLM
"""

import requests
import os
import json
import re
from typing import Optional


class OracleConfig:
    """Configuration for oracle services."""

    # OpenAI API (requires API key)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = "gpt-4"  # or "gpt-3.5-turbo" for cheaper

    # Anthropic API (requires API key) - if you have Claude API access
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

    # Local LLM endpoint (if you have another model running)
    LOCAL_LLM_URL = "http://localhost:1234/v1/chat/completions"
    LOCAL_LLM_MODEL = "claude-3-5-sonnet"  # or whatever you're running

    # Preferred oracle (choose one: "duckduckgo", "openai", "anthropic", "local", "none")
    PREFERRED_ORACLE = "duckduckgo"  # Free, no API key needed!


def ask_duckduckgo_chat(question: str) -> str:
    """
    Ask DuckDuckGo AI Chat (duck.ai) - FREE, no API key needed!

    Uses the duckduckgo-chat package to access Claude/GPT via DuckDuckGo.
    """
    try:
        # Try using duckduckgo_search chat feature
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return (
                "Error: duckduckgo-search not installed.\n"
                "Install: pip install -U duckduckgo-search"
            )

        # DuckDuckGo AI Chat uses their API (free)
        with DDGS() as ddgs:
            try:
                # Use chat method (available in duckduckgo-search 6.0+)
                results = ddgs.chat(question, model="claude-3-haiku")  # or "gpt-4o-mini"

                if results:
                    return f"Oracle (DuckDuckGo AI - Claude):\n{'='*40}\n\n{results}"
                else:
                    return "Oracle returned no response."

            except AttributeError:
                # Older version of duckduckgo-search doesn't have chat
                return (
                    "Error: duckduckgo-search version too old.\n"
                    "Update: pip install -U duckduckgo-search\n"
                    "Requires version 6.0 or newer for AI chat."
                )
            except Exception as e:
                return f"DuckDuckGo Chat error: {e}"

    except Exception as e:
        return f"DuckDuckGo setup error: {e}"


def ask_openai(question: str, config: OracleConfig = None) -> str:
    """Ask ChatGPT via OpenAI API."""
    if config is None:
        config = OracleConfig()

    if not config.OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY not set. Set environment variable or configure in OracleConfig."

    try:
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": config.OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer questions concisely but thoroughly."},
                {"role": "user", "content": question}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        answer = result["choices"][0]["message"]["content"]
        return f"Oracle (ChatGPT):\n{'='*40}\n\n{answer}"

    except requests.exceptions.RequestException as e:
        return f"OpenAI API error: {e}"
    except Exception as e:
        return f"Error: {e}"


def ask_anthropic(question: str, config: OracleConfig = None) -> str:
    """Ask Claude via Anthropic API."""
    if config is None:
        config = OracleConfig()

    if not config.ANTHROPIC_API_KEY:
        return "Error: ANTHROPIC_API_KEY not set. Set environment variable or configure in OracleConfig."

    try:
        headers = {
            "x-api-key": config.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": config.ANTHROPIC_MODEL,
            "messages": [
                {"role": "user", "content": question}
            ],
            "max_tokens": 1000
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        answer = result["content"][0]["text"]
        return f"Oracle (Claude):\n{'='*40}\n\n{answer}"

    except requests.exceptions.RequestException as e:
        return f"Anthropic API error: {e}"
    except Exception as e:
        return f"Error: {e}"


def ask_local_llm(question: str, config: OracleConfig = None) -> str:
    """Ask a local LLM running on localhost."""
    if config is None:
        config = OracleConfig()

    try:
        payload = {
            "model": config.LOCAL_LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer questions concisely but thoroughly."},
                {"role": "user", "content": question}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        response = requests.post(
            config.LOCAL_LLM_URL,
            json=payload,
            timeout=60
        )

        response.raise_for_status()
        result = response.json()

        answer = result["choices"][0]["message"]["content"]
        return f"Oracle (Local LLM):\n{'='*40}\n\n{answer}"

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to local LLM. Is the server running?"
    except requests.exceptions.RequestException as e:
        return f"Local LLM error: {e}"
    except Exception as e:
        return f"Error: {e}"


def ask_oracle_auto(question: str, config: OracleConfig = None) -> str:
    """
    Automatically route to the configured oracle service.

    Priority:
    1. User's PREFERRED_ORACLE setting
    2. DuckDuckGo (free, always available)
    3. First available service (OpenAI > Anthropic > Local)
    """
    if config is None:
        config = OracleConfig()

    # Use preferred oracle if specified
    if config.PREFERRED_ORACLE == "duckduckgo":
        return ask_duckduckgo_chat(question)
    elif config.PREFERRED_ORACLE == "openai":
        return ask_openai(question, config)
    elif config.PREFERRED_ORACLE == "anthropic":
        return ask_anthropic(question, config)
    elif config.PREFERRED_ORACLE == "local":
        return ask_local_llm(question, config)

    # Auto-detect: Try DuckDuckGo first (free, no key needed)
    try:
        result = ask_duckduckgo_chat(question)
        if not result.startswith("Error:"):
            return result
    except:
        pass

    # Fallback to API services if available
    if config.OPENAI_API_KEY:
        return ask_openai(question, config)
    elif config.ANTHROPIC_API_KEY:
        return ask_anthropic(question, config)
    else:
        # Try local as final fallback
        try:
            result = ask_local_llm(question, config)
            if not result.startswith("Error:"):
                return result
        except:
            pass

    return (
        "Oracle Not Available\n"
        "="*40 + "\n\n"
        "Could not connect to any oracle service.\n\n"
        "Tried:\n"
        "1. DuckDuckGo AI Chat (requires: pip install -U duckduckgo-search)\n"
        "2. OpenAI API (requires: OPENAI_API_KEY)\n"
        "3. Anthropic API (requires: ANTHROPIC_API_KEY)\n"
        "4. Local LLM (requires: server at http://localhost:1234)\n\n"
        f"Question was: {question}"
    )


def test_oracle_connection(service: str = "auto") -> str:
    """Test oracle connection with a simple question."""
    test_question = "What is 2+2? Answer with just the number."

    config = OracleConfig()

    if service == "auto":
        result = ask_oracle_auto(test_question, config)
    elif service == "openai":
        result = ask_openai(test_question, config)
    elif service == "anthropic":
        result = ask_anthropic(test_question, config)
    elif service == "local":
        result = ask_local_llm(test_question, config)
    else:
        return f"Unknown service: {service}"

    return f"Test Question: {test_question}\n\n{result}"


if __name__ == "__main__":
    import sys

    # Test the oracle
    print("Testing Oracle Connection...")
    print("="*60)

    if len(sys.argv) > 1:
        service = sys.argv[1]
    else:
        service = "auto"

    result = test_oracle_connection(service)
    print(result)

    print("\n" + "="*60)
    print("\nTo use oracle in your code:")
    print("from oracle_bridge import ask_oracle_auto")
    print('result = ask_oracle_auto("What is quantum computing?")')
