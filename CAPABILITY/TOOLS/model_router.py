#!/usr/bin/env python3
"""Model Router: Deterministic model selection with explicit fallback chain.

This module implements Z.3.1: Router & Fallback Stability.

Key principles:
- Deterministic: given the same inputs, always select the same model
- Explicit fallback chain: try models in declared order
- Fail-closed: invalid models or chains cause immediate failure
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a model.

    Attributes:
        name: Human-readable model name (e.g., "Claude Sonnet 4.5")
        model_id: Canonical model identifier for API calls
        reasoning_level: Expected reasoning capability ("Low", "Medium", "High", "Deep")
        thinking_mode: Whether thinking/reasoning output is enabled
    """
    name: str
    model_id: str
    reasoning_level: str = "Medium"
    thinking_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "reasoning_level": self.reasoning_level,
            "thinking_mode": self.thinking_mode,
        }


# Known model registry
# This is the single source of truth for valid models
KNOWN_MODELS: Dict[str, ModelSpec] = {
    # Claude models
    "Claude Sonnet 4.5 (non-thinking)": ModelSpec(
        name="Claude Sonnet 4.5 (non-thinking)",
        model_id="claude-sonnet-4-5",
        reasoning_level="Medium",
        thinking_mode=False,
    ),
    "Claude Sonnet": ModelSpec(
        name="Claude Sonnet",
        model_id="claude-sonnet-4",
        reasoning_level="High",
        thinking_mode=False,
    ),
    "Claude Opus 4.5": ModelSpec(
        name="Claude Opus 4.5",
        model_id="claude-opus-4-5",
        reasoning_level="Deep",
        thinking_mode=True,
    ),
    # GPT models
    "GPT-5.2-Codex": ModelSpec(
        name="GPT-5.2-Codex",
        model_id="gpt-5.2-codex",
        reasoning_level="High",
        thinking_mode=False,
    ),
    # Gemini models
    "Gemini Pro": ModelSpec(
        name="Gemini Pro",
        model_id="gemini-pro",
        reasoning_level="High",
        thinking_mode=False,
    ),
    "Gemini 3 Pro (Low)": ModelSpec(
        name="Gemini 3 Pro (Low)",
        model_id="gemini-3-pro",
        reasoning_level="Low",
        thinking_mode=False,
    ),
}


class RouterError(Exception):
    """Base exception for router errors."""
    pass


class InvalidModelError(RouterError):
    """Raised when a model name is not in the known registry."""
    pass


class EmptyFallbackChainError(RouterError):
    """Raised when both primary model and fallback chain are empty."""
    pass


@dataclass(frozen=True)
class RouterSelection:
    """Result of router selection.

    Attributes:
        selected_model: The ModelSpec that was selected
        selection_index: Index in the chain (0 = primary, 1+ = fallback)
        selection_reason: Why this model was selected
        fallback_chain_hash: Deterministic hash of the full selection chain
    """
    selected_model: ModelSpec
    selection_index: int
    selection_reason: str
    fallback_chain_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_model": self.selected_model.to_dict(),
            "selection_index": self.selection_index,
            "selection_reason": self.selection_reason,
            "fallback_chain_hash": self.fallback_chain_hash,
        }


def _compute_chain_hash(primary: str, fallbacks: List[str]) -> str:
    """Compute a deterministic hash of the model selection chain.

    This ensures the same chain always produces the same hash,
    enabling reproducibility and auditing.
    """
    chain_data = {
        "primary": primary,
        "fallbacks": fallbacks,
    }
    canonical = json.dumps(chain_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_model_name(model_str: str) -> str:
    """Extract model name from a model specification string.

    Handles formats like:
    - "Claude Sonnet 4.5 (non-thinking) (Reasoning: Medium)"
    - "Claude Sonnet (Reasoning: High)"
    - "Gemini Pro"

    Returns the base model name.
    """
    if not model_str:
        return ""

    # Strip reasoning annotations
    if "(Reasoning:" in model_str:
        model_str = model_str.split("(Reasoning:")[0].strip()

    return model_str.strip()


def validate_model(model_name: str) -> ModelSpec:
    """Validate that a model name is known and return its spec.

    Args:
        model_name: Model name to validate

    Returns:
        ModelSpec for the validated model

    Raises:
        InvalidModelError: If model is not in known registry
    """
    parsed = parse_model_name(model_name)

    if not parsed:
        raise InvalidModelError("Model name cannot be empty")

    if parsed not in KNOWN_MODELS:
        known_names = ", ".join(sorted(KNOWN_MODELS.keys()))
        raise InvalidModelError(
            f"Unknown model: '{parsed}'. "
            f"Known models: {known_names}"
        )

    return KNOWN_MODELS[parsed]


def select_model(
    *,
    primary_model: str,
    fallback_chain: Optional[List[str]] = None,
    selection_index: int = 0,
) -> RouterSelection:
    """Select a model deterministically from primary + fallback chain.

    This is the core routing logic. Given a primary model and fallback chain,
    it selects the model at the specified index (0 = primary).

    Args:
        primary_model: Primary model name
        fallback_chain: Ordered list of fallback model names
        selection_index: Which model to select (0 = primary, 1+ = fallback)

    Returns:
        RouterSelection with the selected model and metadata

    Raises:
        EmptyFallbackChainError: If both primary and fallback chain are empty
        InvalidModelError: If any model in the chain is unknown
        IndexError: If selection_index is out of bounds
    """
    fallbacks = fallback_chain or []

    # Validate inputs
    if not primary_model and not fallbacks:
        raise EmptyFallbackChainError(
            "Must specify at least one model (primary or fallback)"
        )

    # Build the full chain
    chain = []
    if primary_model:
        chain.append(primary_model)
    chain.extend(fallbacks)

    # Validate all models in chain
    validated_specs = []
    for model_name in chain:
        validated_specs.append(validate_model(model_name))

    # Check selection index
    if selection_index < 0 or selection_index >= len(validated_specs):
        raise IndexError(
            f"Selection index {selection_index} out of range "
            f"(chain has {len(validated_specs)} models)"
        )

    # Compute deterministic chain hash
    chain_hash = _compute_chain_hash(primary_model, fallbacks)

    # Select the model
    selected = validated_specs[selection_index]

    if selection_index == 0:
        reason = "primary_model"
    else:
        reason = f"fallback[{selection_index - 1}]"

    return RouterSelection(
        selected_model=selected,
        selection_index=selection_index,
        selection_reason=reason,
        fallback_chain_hash=chain_hash,
    )


def create_router_receipt(
    *,
    selection: RouterSelection,
    task_id: str,
    primary_model: str,
    fallback_chain: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a router receipt for auditing.

    Args:
        selection: The RouterSelection result
        task_id: Task identifier
        primary_model: Primary model name
        fallback_chain: Fallback chain

    Returns:
        Dictionary containing receipt data
    """
    return {
        "task_id": task_id,
        "router_version": "1.0",
        "primary_model": primary_model,
        "fallback_chain": fallback_chain or [],
        "selection": selection.to_dict(),
    }


def main() -> int:
    """CLI entry point for testing router."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model_router.py <primary_model> [fallback1] [fallback2] ...")
        return 1

    primary = sys.argv[1]
    fallbacks = sys.argv[2:] if len(sys.argv) > 2 else []

    try:
        selection = select_model(
            primary_model=primary,
            fallback_chain=fallbacks,
            selection_index=0,
        )

        print(json.dumps(selection.to_dict(), indent=2))
        return 0

    except RouterError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
