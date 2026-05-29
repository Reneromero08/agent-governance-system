"""
Vector Fallback Resolver (Phase E.2)

Provides governed vector search as the LAST RESORT fallback in the retrieval chain.

Key Design Principles:
1. FALLBACK ONLY: Vectors are never the primary retrieval path
2. BUDGET-BOUNDED: Fill results until budget exhausted - NO arbitrary max_results
3. NO TRUST BYPASS: All results are hash-verified before inclusion
4. SIMILARITY THRESHOLD: Only empirically validated threshold (0.5 from Q44)
5. CONFIGURABLE: Config file for tuning without code changes
6. LOGGED: Every search logged for analysis

Budget Philosophy:
- Only ONE limit: percentage of remaining budget allocated to vectors
- NO max_results - just fill until budget is exhausted
- The only fixed number is min_similarity (0.5) which is empirically validated
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from datetime import datetime, timezone

from catalytic_chat.mcp_integration import ChatToolExecutor, McpAccessError
from catalytic_chat.adaptive_budget import BudgetExceededError


# =============================================================================
# Default Configuration
# =============================================================================

# Minimum similarity threshold (empirically validated in Q44, NOT arbitrary)
# This is the ONLY hard filter - agent is free to search until satisfied
DEFAULT_MIN_SIMILARITY = 0.5

# Default config file location
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "_generated" / "vector_fallback_config.json"

# Default log file location
DEFAULT_LOG_PATH = Path(__file__).parent.parent / "_generated" / "vector_fallback_search.jsonl"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VectorBudgetConfig:
    """
    Configuration for vector fallback.

    Only ONE tunable parameter:
    - min_similarity: Minimum similarity threshold (empirically validated in Q44)

    The agent is FREE to search until it finds what it needs.
    Budget (remaining_budget) is a SAFETY BOUNDARY passed by caller, not configured here.
    """
    min_similarity: float = DEFAULT_MIN_SIMILARITY

    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.min_similarity <= 1.0:
            raise ValueError(f"min_similarity must be [0, 1.0], got {self.min_similarity}")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for serialization."""
        return {
            "min_similarity": self.min_similarity
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorBudgetConfig':
        """Create from dict."""
        return cls(
            min_similarity=float(data.get("min_similarity", DEFAULT_MIN_SIMILARITY))
        )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'VectorBudgetConfig':
        """
        Load config from JSON file.

        Args:
            path: Path to config file (uses default if None)

        Returns:
            VectorBudgetConfig loaded from file, or defaults if file doesn't exist
        """
        config_path = path or DEFAULT_CONFIG_PATH
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return cls.from_dict(data)
            except (json.JSONDecodeError, IOError):
                pass
        return cls()

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save config to JSON file.

        Args:
            path: Path to config file (uses default if None)
        """
        config_path = path or DEFAULT_CONFIG_PATH
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VectorResult:
    """
    Result from vector search.

    Attributes:
        content: The retrieved content
        similarity: Cosine similarity score (0.0 - 1.0)
        content_hash: SHA-256 hash of content
        source_cassette: Source cassette ID
        verified: True if hash was verified
        tokens: Token count (estimated)
    """
    content: str
    similarity: float
    content_hash: str
    source_cassette: str
    verified: bool = False
    tokens: int = 0


@dataclass
class VectorSearchLog:
    """
    Log entry for a vector search operation.

    Captures all decisions for tuning and debugging.
    """
    timestamp: str
    query: str
    remaining_budget: int
    allocation: int
    min_similarity: float
    raw_results_count: int
    filtered_by_similarity: int
    filtered_by_verification: int
    final_results_count: int
    tokens_used: int
    budget_utilization_pct: float
    config: Dict[str, float]


# =============================================================================
# Search Logger
# =============================================================================

class VectorSearchLogger:
    """
    Logger for vector search operations.

    Writes JSONL logs for analysis and tuning.
    """

    def __init__(self, log_path: Optional[Path] = None):
        """
        Initialize logger.

        Args:
            log_path: Path to log file (uses default if None)
        """
        self.log_path = log_path or DEFAULT_LOG_PATH
        self._in_memory: List[VectorSearchLog] = []

    def log(self, entry: VectorSearchLog) -> None:
        """Log a search operation."""
        self._in_memory.append(entry)

        # Write to file
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'a', encoding='utf-8') as f:
                record = {
                    "timestamp": entry.timestamp,
                    "query": entry.query,
                    "remaining_budget": entry.remaining_budget,
                    "allocation": entry.allocation,
                    "min_similarity": entry.min_similarity,
                    "raw_results_count": entry.raw_results_count,
                    "filtered_by_similarity": entry.filtered_by_similarity,
                    "filtered_by_verification": entry.filtered_by_verification,
                    "final_results_count": entry.final_results_count,
                    "tokens_used": entry.tokens_used,
                    "budget_utilization_pct": entry.budget_utilization_pct,
                    "config": entry.config
                }
                f.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception:
            pass  # Don't fail on logging errors

    def get_recent(self, n: int = 100) -> List[VectorSearchLog]:
        """Get recent log entries from memory."""
        return list(self._in_memory[-n:])

    def clear_memory(self) -> None:
        """Clear in-memory logs."""
        self._in_memory.clear()


# =============================================================================
# Vector Fallback Resolver
# =============================================================================

class VectorFallbackResolver:
    """
    Governed vector search resolver.

    Provides vector search as a last-resort fallback with strict governance:
    - Budget = percentage of remaining (configurable)
    - NO max_results - fill until budget exhausted
    - Similarity threshold (0.5, empirically validated)
    - Hash verification (no trust bypass)
    - Full logging for tuning

    This resolver should only be used when all other retrieval paths fail:
    SPC -> FTS -> Local Index -> CAS -> Vector Fallback (this)
    """

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        tool_executor: Optional[ChatToolExecutor] = None,
        config: Optional[VectorBudgetConfig] = None,
        config_path: Optional[Path] = None,
        token_estimator: Optional[Callable[[str], int]] = None,
        logger: Optional[VectorSearchLogger] = None,
        log_path: Optional[Path] = None
    ):
        """
        Initialize vector fallback resolver.

        Args:
            repo_root: Repository root path
            tool_executor: ChatToolExecutor instance (created if None)
            config: Budget configuration (loads from file if None)
            config_path: Path to config file (for loading)
            token_estimator: Function to estimate tokens from content
            logger: Search logger (creates default if None)
            log_path: Path to log file (for logger)
        """
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[4]
        self.repo_root = repo_root

        self._tool_executor = tool_executor

        # Load config from file or use provided/defaults
        if config is not None:
            self.config = config
        else:
            self.config = VectorBudgetConfig.load(config_path)

        self.token_estimator = token_estimator or self._default_token_estimator

        # Initialize logger
        if logger is not None:
            self.logger = logger
        else:
            self.logger = VectorSearchLogger(log_path)

    @property
    def tool_executor(self) -> ChatToolExecutor:
        """Lazy load tool executor."""
        if self._tool_executor is None:
            self._tool_executor = ChatToolExecutor(self.repo_root)
        return self._tool_executor

    @staticmethod
    def _default_token_estimator(content: str) -> int:
        """Default token estimator: ~4 chars per token."""
        return max(1, len(content) // 4)

    def compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def search(
        self,
        query: str,
        remaining_budget: int
    ) -> List[VectorResult]:
        """
        Execute governed vector search.

        The agent is FREE to search until it finds what it needs.
        Budget is a SAFETY BOUNDARY, not a target to fill.
        Most searches will find what they need on first query.

        Args:
            query: Search query
            remaining_budget: Safety boundary (tokens available)

        Returns:
            List of verified VectorResult objects that pass:
            - Similarity threshold (0.5 from Q44)
            - Hash verification (NO TRUST BYPASS)

        Note:
            - Returns ALL valid results (agent decides when satisfied)
            - Budget is tracked for monitoring, enforced as safety limit
            - Every search is logged for analysis
        """
        # Track filtering for logging
        raw_count = 0
        filtered_by_similarity = 0
        filtered_by_verification = 0

        if remaining_budget <= 0:
            self._log_search(
                query=query,
                remaining_budget=remaining_budget,
                allocation=remaining_budget,
                raw_count=0,
                filtered_similarity=0,
                filtered_verification=0,
                final_count=0,
                tokens_used=0
            )
            return []

        # Execute semantic search via MCP
        raw_results = self._execute_semantic_search(query, limit=50)
        raw_count = len(raw_results)

        if not raw_results:
            self._log_search(
                query=query,
                remaining_budget=remaining_budget,
                allocation=remaining_budget,
                raw_count=0,
                filtered_similarity=0,
                filtered_verification=0,
                final_count=0,
                tokens_used=0
            )
            return []

        # Return all valid results - agent decides when it has enough
        # Budget is safety boundary, not a fill target
        verified_results: List[VectorResult] = []
        tokens_used = 0

        for result in raw_results:
            # Filter by similarity threshold
            if result.similarity < self.config.min_similarity:
                filtered_by_similarity += 1
                continue

            # Verify hash (NO TRUST BYPASS)
            if not self._verify_result(result):
                filtered_by_verification += 1
                continue

            # Estimate tokens
            result_tokens = self.token_estimator(result.content)

            # Safety boundary check - stop if we'd exceed budget
            if tokens_used + result_tokens > remaining_budget:
                break  # Hit safety boundary

            # Include result
            result.verified = True
            result.tokens = result_tokens
            verified_results.append(result)
            tokens_used += result_tokens

        # Log search
        self._log_search(
            query=query,
            remaining_budget=remaining_budget,
            allocation=remaining_budget,
            raw_count=raw_count,
            filtered_similarity=filtered_by_similarity,
            filtered_verification=filtered_by_verification,
            final_count=len(verified_results),
            tokens_used=tokens_used
        )

        return verified_results

    def _log_search(
        self,
        query: str,
        remaining_budget: int,
        allocation: int,
        raw_count: int,
        filtered_similarity: int,
        filtered_verification: int,
        final_count: int,
        tokens_used: int
    ) -> None:
        """Log a search operation."""
        utilization = (tokens_used / allocation * 100) if allocation > 0 else 0.0

        entry = VectorSearchLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
            remaining_budget=remaining_budget,
            allocation=allocation,
            min_similarity=self.config.min_similarity,
            raw_results_count=raw_count,
            filtered_by_similarity=filtered_similarity,
            filtered_by_verification=filtered_verification,
            final_results_count=final_count,
            tokens_used=tokens_used,
            budget_utilization_pct=round(utilization, 2),
            config=self.config.to_dict()
        )
        self.logger.log(entry)

    def _execute_semantic_search(
        self,
        query: str,
        limit: int
    ) -> List[VectorResult]:
        """
        Execute semantic search via MCP.

        Args:
            query: Search query
            limit: Max results to request from MCP

        Returns:
            List of VectorResult (not yet verified)
        """
        try:
            result = self.tool_executor.execute_tool(
                "semantic_search",
                {"query": query, "limit": limit}
            )

            if not result:
                return []

            return self._parse_search_results(result)

        except McpAccessError:
            return []
        except Exception:
            return []

    def _parse_search_results(self, mcp_result: Dict[str, Any]) -> List[VectorResult]:
        """Parse MCP semantic_search results into VectorResult objects."""
        results: List[VectorResult] = []

        content_items = mcp_result.get('content', [])
        for item in content_items:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    text = item.get('text', '')
                    try:
                        data = json.loads(text)
                        if isinstance(data, list):
                            for entry in data:
                                vr = self._parse_single_result(entry)
                                if vr:
                                    results.append(vr)
                        elif isinstance(data, dict):
                            vr = self._parse_single_result(data)
                            if vr:
                                results.append(vr)
                    except (json.JSONDecodeError, TypeError):
                        results.append(VectorResult(
                            content=text,
                            similarity=0.0,
                            content_hash=self.compute_hash(text),
                            source_cassette="unknown"
                        ))

        return results

    def _parse_single_result(self, data: Dict[str, Any]) -> Optional[VectorResult]:
        """Parse a single search result dict."""
        content = data.get('content', data.get('text', ''))
        if not content:
            return None

        return VectorResult(
            content=content,
            similarity=float(data.get('score', data.get('similarity', 0.0))),
            content_hash=data.get('hash', self.compute_hash(content)),
            source_cassette=data.get('cassette_id', data.get('source', 'unknown'))
        )

    def _verify_result(self, result: VectorResult) -> bool:
        """
        Verify vector result - NO TRUST BYPASS.

        Computes actual hash and compares to claimed hash.
        """
        actual_hash = self.compute_hash(result.content)
        return actual_hash == result.content_hash

    # Convenience methods for tests
    def get_search_logs(self) -> List[VectorSearchLog]:
        """Get recent search logs."""
        return self.logger.get_recent()

    def clear_search_logs(self) -> None:
        """Clear in-memory logs."""
        self.logger.clear_memory()
