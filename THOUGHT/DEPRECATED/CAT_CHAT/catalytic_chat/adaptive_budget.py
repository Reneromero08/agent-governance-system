"""
Adaptive Budget Discovery
=========================

Discovers model context window from configuration and computes adaptive budgets
for the auto-controlled context loop.

Key Design Principle: No arbitrary limits. Budget is derived from:
- Real model context window (discovered at runtime)
- System prompt size (measured)
- Response reserve percentage (configurable, default 25%)

Phase C.1 of Auto-Controlled Context Loop implementation.
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Callable


# =============================================================================
# Exceptions
# =============================================================================

class BudgetExceededError(Exception):
    """
    Raised when working set exceeds available budget.

    This is a hard invariant violation (INV-CATALYTIC-04: Clean Space Bound).
    The system must fail-closed when this occurs.
    """
    def __init__(
        self,
        budget_available: int,
        budget_used: int,
        item_count: int,
        message: str = "Working set exceeds budget"
    ):
        self.budget_available = budget_available
        self.budget_used = budget_used
        self.item_count = item_count
        super().__init__(
            f"{message}: used {budget_used} tokens "
            f"(available: {budget_available}, items: {item_count})"
        )


class BudgetConfigError(Exception):
    """Raised when budget configuration is invalid or missing."""
    pass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AdaptiveBudget:
    """
    Adaptive budget computed from model context window.

    All values are in tokens.

    Attributes:
        context_window: Total model context window (e.g., 40961 for nemotron)
        system_prompt_tokens: Tokens used by fixed system prompt
        response_reserve_pct: Percentage reserved for model response (0.0-1.0)
        model_id: Identifier for the model (for logging/debugging)
    """
    context_window: int
    system_prompt_tokens: int = 0
    response_reserve_pct: float = 0.25
    model_id: str = "unknown"

    def __post_init__(self):
        """Validate budget parameters."""
        if self.context_window <= 0:
            raise BudgetConfigError(
                f"Invalid context_window: {self.context_window} (must be positive)"
            )
        if not (0.0 <= self.response_reserve_pct < 1.0):
            raise BudgetConfigError(
                f"Invalid response_reserve_pct: {self.response_reserve_pct} "
                f"(must be in [0.0, 1.0))"
            )
        if self.system_prompt_tokens < 0:
            raise BudgetConfigError(
                f"Invalid system_prompt_tokens: {self.system_prompt_tokens} "
                f"(must be non-negative)"
            )

    @property
    def response_reserve_tokens(self) -> int:
        """Tokens reserved for model response."""
        return int(self.context_window * self.response_reserve_pct)

    @property
    def available_for_working_set(self) -> int:
        """
        Tokens available for working set (clean space).

        Formula: context_window - system_prompt - response_reserve
        """
        available = (
            self.context_window
            - self.system_prompt_tokens
            - self.response_reserve_tokens
        )
        return max(0, available)

    def check_invariant(self, tokens_used: int, item_count: int = 0) -> None:
        """
        Check budget invariant (INV-CATALYTIC-04).

        Raises BudgetExceededError if tokens_used > available_for_working_set.
        """
        if tokens_used > self.available_for_working_set:
            raise BudgetExceededError(
                budget_available=self.available_for_working_set,
                budget_used=tokens_used,
                item_count=item_count
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "context_window": self.context_window,
            "system_prompt_tokens": self.system_prompt_tokens,
            "response_reserve_pct": self.response_reserve_pct,
            "response_reserve_tokens": self.response_reserve_tokens,
            "available_for_working_set": self.available_for_working_set,
            "model_id": self.model_id,
        }

    def __repr__(self) -> str:
        return (
            f"AdaptiveBudget(context={self.context_window}, "
            f"available={self.available_for_working_set}, "
            f"reserve={self.response_reserve_pct:.0%}, "
            f"model={self.model_id})"
        )


@dataclass
class BudgetConfig:
    """
    Configuration for budget discovery.

    This is the JSON schema for model configuration files.
    """
    model_context_window: int
    response_reserve_pct: float = 0.25
    model_id: str = "default"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BudgetConfig":
        """Create from dictionary (parsed JSON)."""
        return cls(
            model_context_window=data["model_context_window"],
            response_reserve_pct=data.get("response_reserve_pct", 0.25),
            model_id=data.get("model_id", "default"),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> "BudgetConfig":
        """Load from JSON file."""
        if not path.exists():
            raise BudgetConfigError(f"Config file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise BudgetConfigError(f"Invalid JSON in {path}: {e}")
        except KeyError as e:
            raise BudgetConfigError(f"Missing required field in {path}: {e}")


# =============================================================================
# Model Budget Discovery
# =============================================================================

class ModelBudgetDiscovery:
    """
    Discovers model budget from configuration.

    Usage:
        discovery = ModelBudgetDiscovery(config_path="model_config.json")
        budget = discovery.discover(system_prompt="You are a helpful assistant.")

        # Or with explicit context window
        budget = ModelBudgetDiscovery.from_context_window(40961)
    """

    # Common model context windows for fallback discovery
    KNOWN_MODELS: Dict[str, int] = {
        "nemotron": 40961,
        "nemotron-mini": 40961,
        "llama-3.1-70b": 131072,
        "llama-3.1-8b": 131072,
        "llama-3.2-3b": 131072,
        "qwen-2.5-72b": 131072,
        "qwen-2.5-14b": 131072,
        "qwen-2.5-7b": 131072,
        "mistral-large": 131072,
        "gpt-4o": 128000,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
    }

    def __init__(
        self,
        config_path: Optional[Path] = None,
        token_estimator: Optional[Callable[[str], int]] = None
    ):
        """
        Initialize budget discovery.

        Args:
            config_path: Path to JSON config file (optional)
            token_estimator: Function to estimate tokens from text (default: len//4)
        """
        self.config_path = Path(config_path) if config_path else None
        self.token_estimator = token_estimator or (lambda s: len(s) // 4)
        self._config: Optional[BudgetConfig] = None

    def load_config(self) -> Optional[BudgetConfig]:
        """Load configuration from file if available."""
        if self._config is not None:
            return self._config

        if self.config_path and self.config_path.exists():
            self._config = BudgetConfig.from_json_file(self.config_path)

        return self._config

    def discover(
        self,
        system_prompt: str = "",
        model_id: Optional[str] = None,
        context_window: Optional[int] = None,
        response_reserve_pct: Optional[float] = None
    ) -> AdaptiveBudget:
        """
        Discover adaptive budget.

        Priority for context window:
        1. Explicit context_window parameter
        2. Config file (if loaded)
        3. Known model lookup (if model_id provided)
        4. Raise BudgetConfigError if none found

        Args:
            system_prompt: System prompt text to measure
            model_id: Model identifier for known model lookup
            context_window: Explicit context window override
            response_reserve_pct: Override response reserve percentage

        Returns:
            AdaptiveBudget with computed values
        """
        # Determine context window
        final_context_window = context_window
        final_model_id = model_id or "unknown"
        final_reserve_pct = response_reserve_pct

        # Try config file
        config = self.load_config()
        if config:
            if final_context_window is None:
                final_context_window = config.model_context_window
            if final_model_id == "unknown":
                final_model_id = config.model_id
            if final_reserve_pct is None:
                final_reserve_pct = config.response_reserve_pct

        # Try known models
        if final_context_window is None and model_id:
            model_key = model_id.lower().replace("_", "-")
            for known_id, known_window in self.KNOWN_MODELS.items():
                if known_id in model_key or model_key in known_id:
                    final_context_window = known_window
                    break

        # Validate we have a context window
        if final_context_window is None:
            raise BudgetConfigError(
                "Cannot discover context window. Provide one of: "
                "config file, context_window parameter, or recognized model_id"
            )

        # Set default reserve if still None
        if final_reserve_pct is None:
            final_reserve_pct = 0.25

        # Measure system prompt
        system_prompt_tokens = self.token_estimator(system_prompt) if system_prompt else 0

        return AdaptiveBudget(
            context_window=final_context_window,
            system_prompt_tokens=system_prompt_tokens,
            response_reserve_pct=final_reserve_pct,
            model_id=final_model_id,
        )

    @classmethod
    def from_context_window(
        cls,
        context_window: int,
        system_prompt: str = "",
        response_reserve_pct: float = 0.25,
        model_id: str = "custom",
        token_estimator: Optional[Callable[[str], int]] = None
    ) -> AdaptiveBudget:
        """
        Create budget directly from known context window.

        Convenience method for when you already know the model's context window.
        """
        estimator = token_estimator or (lambda s: len(s) // 4)
        system_prompt_tokens = estimator(system_prompt) if system_prompt else 0

        return AdaptiveBudget(
            context_window=context_window,
            system_prompt_tokens=system_prompt_tokens,
            response_reserve_pct=response_reserve_pct,
            model_id=model_id,
        )


# =============================================================================
# Budget Snapshot for Determinism
# =============================================================================

@dataclass
class BudgetSnapshot:
    """
    Immutable snapshot of budget state for deterministic replay.

    Captures the exact budget configuration at a point in time,
    including working set token usage.
    """
    budget: AdaptiveBudget
    tokens_used: int
    item_count: int
    timestamp: str  # ISO format

    @property
    def tokens_remaining(self) -> int:
        """Tokens still available in working set."""
        return max(0, self.budget.available_for_working_set - self.tokens_used)

    @property
    def utilization_pct(self) -> float:
        """Budget utilization as percentage."""
        if self.budget.available_for_working_set == 0:
            return 1.0 if self.tokens_used > 0 else 0.0
        return self.tokens_used / self.budget.available_for_working_set

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "budget": self.budget.to_dict(),
            "tokens_used": self.tokens_used,
            "tokens_remaining": self.tokens_remaining,
            "item_count": self.item_count,
            "utilization_pct": self.utilization_pct,
            "timestamp": self.timestamp,
        }

    def compute_hash(self) -> str:
        """Compute deterministic hash of snapshot for chain linking."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


# =============================================================================
# Default Configuration Template
# =============================================================================

DEFAULT_CONFIG_TEMPLATE = """{
    "model_context_window": 40961,
    "response_reserve_pct": 0.25,
    "model_id": "nemotron"
}
"""

def create_default_config(path: Path) -> None:
    """Create a default configuration file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")


if __name__ == "__main__":
    # Quick sanity test
    print("Adaptive Budget Discovery - Sanity Test")
    print("=" * 50)

    # Test with explicit context window
    budget = ModelBudgetDiscovery.from_context_window(
        context_window=40961,
        system_prompt="You are a helpful assistant for catalytic computing.",
        model_id="nemotron"
    )

    print(f"Context Window: {budget.context_window:,} tokens")
    print(f"System Prompt: {budget.system_prompt_tokens:,} tokens")
    print(f"Response Reserve: {budget.response_reserve_tokens:,} tokens ({budget.response_reserve_pct:.0%})")
    print(f"Available for Working Set: {budget.available_for_working_set:,} tokens")
    print(f"\n{budget}")

    # Test invariant check
    print("\n--- Invariant Check ---")
    try:
        budget.check_invariant(tokens_used=10000, item_count=5)
        print("Check passed: 10,000 tokens within budget")
    except BudgetExceededError as e:
        print(f"Check failed: {e}")

    try:
        budget.check_invariant(tokens_used=50000, item_count=10)
        print("Check passed: 50,000 tokens within budget")
    except BudgetExceededError as e:
        print(f"Check failed (expected): {e}")
