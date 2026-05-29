#!/usr/bin/env python3
"""
R > median(R) Threshold Test
============================

Tests a modified context partitioner that uses median(R) as the threshold
instead of a fixed value. This selects exactly the top 50% by relevance.

Algorithm:
1. Compute R scores for all items as usual (R = E / grad_S)
2. threshold = median(R) instead of fixed 1.0
3. Only items with R > median(R) get included (top 50% by relevance)

This is a simplified 20-turn test:
- 10 planted facts with specific details
- 5 recall queries to test retrieval
- Logs: Turn, Type, R scores, Items rehydrated, recall success
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime, timezone
import hashlib

CAT_CHAT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CAT_CHAT_ROOT))

from catalytic_chat.session_capsule import SessionCapsule
from catalytic_chat.auto_context_manager import AutoContextManager
from catalytic_chat.adaptive_budget import ModelBudgetDiscovery
from catalytic_chat.llm_client import get_llm_client
from catalytic_chat.context_partitioner import (
    ContextPartitioner,
    ContextItem,
    ScoredItem,
    PartitionResult,
    extract_keywords,
    compute_keyword_score,
)

DB_OUTPUT_DIR = Path(__file__).parent.parent / "test_chats"


# =============================================================================
# Monkey-patch for median threshold
# =============================================================================

def partition_with_median_threshold(
    self,
    query_embedding: np.ndarray,
    all_items: List[ContextItem],
    budget_tokens: int,
    query_text: str = ""
) -> PartitionResult:
    """
    Partition items using median(R) as the threshold instead of self.threshold.

    This selects exactly the top 50% of items by R-score (relevance).
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    query_hash = hashlib.sha256(query_text.encode()).hexdigest()[:16]

    if not all_items:
        return PartitionResult(
            working_set=[],
            pointer_set=[],
            query_hash=query_hash,
            threshold=0.0,
            budget_total=budget_tokens,
            budget_used=0,
            items_total=0,
            items_in_working_set=0,
            items_below_threshold=0,
            items_over_budget=0,
            E_mean=0.0,
            E_min=0.0,
            E_max=0.0,
            E_std=0.0,
            timestamp=timestamp,
        )

    # Score and sort all items (passes query_text for hybrid keyword matching)
    scored = self.score_items(query_embedding, all_items, query_text)

    # MODIFICATION: Use median(R) as threshold instead of self.threshold
    R_scores = [s.E_score for s in scored]
    median_threshold = float(np.median(R_scores))

    # Partition based on median threshold and budget
    working_set: List[ScoredItem] = []
    pointer_set: List[ScoredItem] = []
    tokens_used = 0
    items_below_threshold = 0
    items_over_budget = 0

    for s in scored:
        if s.E_score < median_threshold:
            # Below median threshold - always pointer_set
            pointer_set.append(s)
            items_below_threshold += 1
        elif tokens_used + s.item.tokens <= budget_tokens:
            # Above threshold and fits budget - working_set
            working_set.append(s)
            tokens_used += s.item.tokens
        else:
            # Above threshold but over budget - pointer_set
            pointer_set.append(s)
            items_over_budget += 1

    # Compute E-score statistics
    E_values = [s.E_score for s in scored]
    E_mean = float(np.mean(E_values)) if E_values else 0.0
    E_min = float(np.min(E_values)) if E_values else 0.0
    E_max = float(np.max(E_values)) if E_values else 0.0
    E_std = float(np.std(E_values)) if len(E_values) > 1 else 0.0

    return PartitionResult(
        working_set=working_set,
        pointer_set=pointer_set,
        query_hash=query_hash,
        threshold=median_threshold,  # Report the actual median used
        budget_total=budget_tokens,
        budget_used=tokens_used,
        items_total=len(all_items),
        items_in_working_set=len(working_set),
        items_below_threshold=items_below_threshold,
        items_over_budget=items_over_budget,
        E_mean=E_mean,
        E_min=E_min,
        E_max=E_max,
        E_std=E_std,
        timestamp=timestamp,
    )


# =============================================================================
# Test Data: 10 planted facts, 5 recall queries
# =============================================================================

PLANTED_FACTS = {
    1: {
        "content": "The database uses PostgreSQL 15.3 with connection pooling set to 25 connections maximum. Queries timeout after 45 seconds.",
        "keywords": ["PostgreSQL", "15.3", "25 connections", "45 seconds"],
    },
    3: {
        "content": "Authentication tokens are JWT using the ES384 algorithm. Tokens expire after 12 hours and refresh tokens last 30 days.",
        "keywords": ["JWT", "ES384", "12 hours", "30 days"],
    },
    5: {
        "content": "The cache layer uses Redis 7.0 with a TTL of 3600 seconds for session data. Maximum memory is 2GB with LRU eviction.",
        "keywords": ["Redis", "7.0", "3600", "2GB", "LRU"],
    },
    7: {
        "content": "API rate limiting is set to 500 requests per minute per user. Burst allowance is 50 requests. Violations return HTTP 429.",
        "keywords": ["500 requests", "minute", "50 requests", "429"],
    },
    9: {
        "content": "The message queue uses RabbitMQ 3.12 with 5 replicas in cluster mode. Dead letter TTL is 72 hours.",
        "keywords": ["RabbitMQ", "3.12", "5 replicas", "72 hours"],
    },
    11: {
        "content": "Logging uses Elasticsearch 8.9 with a retention period of 90 days. Index rotation happens daily at 03:00 UTC.",
        "keywords": ["Elasticsearch", "8.9", "90 days", "03:00 UTC"],
    },
    13: {
        "content": "Load balancing uses NGINX with weighted round-robin. Health checks run every 10 seconds with 3 failures triggering removal.",
        "keywords": ["NGINX", "round-robin", "10 seconds", "3 failures"],
    },
    15: {
        "content": "Backup schedule is incremental every 4 hours and full backup weekly on Sunday at 01:00 UTC. Retention is 60 days.",
        "keywords": ["4 hours", "Sunday", "01:00 UTC", "60 days"],
    },
    17: {
        "content": "The Kubernetes cluster has 8 nodes with 16 vCPUs and 64GB RAM each. Pod limits are 1 CPU and 2GB memory.",
        "keywords": ["8 nodes", "16 vCPUs", "64GB", "1 CPU", "2GB"],
    },
    19: {
        "content": "SSL certificates use Let's Encrypt with automatic renewal 30 days before expiry. TLS 1.3 is enforced.",
        "keywords": ["Let's Encrypt", "30 days", "TLS 1.3"],
    },
}

RECALL_QUERIES = {
    # Recall after some intervening conversation
    16: {
        "query": "What database system do we use and what are the connection limits?",
        "expected": ["PostgreSQL", "25"],
        "source_turn": 1,
    },
    18: {
        "query": "How long do our authentication tokens remain valid before the user needs to log in again?",
        "expected": ["12 hours", "JWT"],
        "source_turn": 3,
    },
    20: {
        "query": "What caching technology handles our session data and what eviction policy do we use?",
        "expected": ["Redis", "LRU"],
        "source_turn": 5,
    },
    21: {
        "query": "When does the system perform full data backups?",
        "expected": ["Sunday", "01:00"],
        "source_turn": 15,
    },
    22: {
        "query": "What encryption protocol version do we require for secure connections?",
        "expected": ["TLS 1.3", "TLS"],
        "source_turn": 19,
    },
}

# Filler conversation to create noise between facts
CONVERSATION_FLOW = {
    2: "Let's continue setting up the infrastructure.",
    4: "Good progress. What about caching strategy?",
    6: "Now we need to handle API throttling.",
    8: "Message queuing is next on the list.",
    10: "Let's configure the logging pipeline.",
    12: "Load balancing configuration is important.",
    14: "Backup strategy needs to be defined.",
}


@dataclass
class TurnResult:
    turn: int
    turn_type: str  # "plant", "recall", "chat"
    content: str
    R_scores: List[float]
    items_rehydrated: int
    recall_success: Optional[bool] = None


class MedianThresholdTest:
    """Test for R > median(R) threshold behavior."""

    def __init__(self, provider: Optional[str] = None):
        self.db_path = DB_OUTPUT_DIR / f"test_r_median_{int(time.time())}.db"
        print(f"Database: {self.db_path}")

        self.llm = get_llm_client(provider)
        print(f"LLM: {self.llm.config.name} ({self.llm.config.model})")
        print(f"URL: {self.llm.config.base_url}")

        self.capsule = SessionCapsule(db_path=self.db_path)
        self.session_id = self.capsule.create_session()

        self.system_prompt = ""

        # Budget for context
        self.budget = ModelBudgetDiscovery.from_context_window(
            context_window=8192,
            system_prompt=self.system_prompt,
            response_reserve_pct=0.25,
            model_id=self.llm.config.model
        )

        # Create manager with default threshold (will be patched)
        self.manager = AutoContextManager(
            db_path=self.db_path,
            session_id=self.session_id,
            budget=self.budget,
            embed_fn=self._embed,
            E_threshold=1.0  # This will be overridden by the patch
        )
        self.manager.capsule = self.capsule

        # APPLY THE PATCH: Replace partition method with median threshold version
        self._apply_median_patch()

        self.turn_log: List[TurnResult] = []
        self.recall_results: List[Dict] = []

    def _apply_median_patch(self):
        """Patch the partitioner to use median(R) threshold."""
        if hasattr(self.manager, 'partitioner'):
            # Direct partitioner access
            import types
            self.manager.partitioner.partition = types.MethodType(
                partition_with_median_threshold,
                self.manager.partitioner
            )
            print("[PATCH] Applied median threshold to manager.partitioner")
        else:
            print("[WARN] Could not find partitioner to patch")

    def _embed(self, text: str) -> np.ndarray:
        import requests
        base = self.llm.config.base_url.rstrip('/')
        # Ensure /v1 is in the path
        if base.endswith('/v1'):
            url = f"{base}/embeddings"
        else:
            url = f"{base}/v1/embeddings"
        resp = requests.post(
            url,
            json={"model": "text-embedding-nomic-embed-text-v1.5", "input": text},
            timeout=None
        )
        resp.raise_for_status()
        vec = np.array(resp.json()["data"][0]["embedding"])
        return vec / np.linalg.norm(vec)

    def _llm_generate(self, system: str, prompt: str) -> str:
        import requests

        url = f"{self.llm.config.base_url.rstrip('/')}/v1/chat/completions"
        if "/v1/v1/" in url:
            url = url.replace("/v1/v1/", "/v1/")

        headers = {"Content-Type": "application/json"}
        if self.llm.config.api_key and self.llm.config.api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self.llm.config.api_key}"

        payload = {
            "model": self.llm.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,
            "temperature": 1.0,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=None)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _execute_turn(self, turn: int, content: str, turn_type: str) -> TurnResult:
        """Execute a single turn and capture R-scores."""
        result = self.manager.respond_catalytic(
            query=content,
            llm_generate=self._llm_generate,
            system_prompt=self.system_prompt
        )

        prep = result.prepare_result
        working_set = prep.working_set if prep else []

        # Extract R-scores from working set
        R_scores = []
        if prep and hasattr(prep, 'working_set'):
            for item in prep.working_set:
                if hasattr(item, 'E_score'):
                    R_scores.append(item.E_score)
                elif hasattr(item, 'E'):
                    R_scores.append(item.E)

        turn_result = TurnResult(
            turn=turn,
            turn_type=turn_type,
            content=content[:80] + "..." if len(content) > 80 else content,
            R_scores=R_scores,
            items_rehydrated=len(working_set),
        )

        self.turn_log.append(turn_result)
        return turn_result

    def _test_recall(self, turn: int, query_data: dict) -> TurnResult:
        """Test recall and check if expected keywords found."""
        query = query_data["query"]
        expected = query_data["expected"]
        source_turn = query_data["source_turn"]

        result = self.manager.respond_catalytic(
            query=query,
            llm_generate=self._llm_generate,
            system_prompt=self.system_prompt
        )

        prep = result.prepare_result
        context_text = prep.get_context_text() if prep else ""
        working_set = prep.working_set if prep else []

        # Extract R-scores
        R_scores = []
        if prep and hasattr(prep, 'working_set'):
            for item in prep.working_set:
                if hasattr(item, 'E_score'):
                    R_scores.append(item.E_score)
                elif hasattr(item, 'E'):
                    R_scores.append(item.E)

        # Check recall success
        context_lower = context_text.lower()
        success = any(kw.lower() in context_lower for kw in expected)

        self.recall_results.append({
            "turn": turn,
            "source_turn": source_turn,
            "expected": expected,
            "success": success,
            "items": len(working_set),
        })

        turn_result = TurnResult(
            turn=turn,
            turn_type="recall",
            content=query[:80] + "..." if len(query) > 80 else query,
            R_scores=R_scores,
            items_rehydrated=len(working_set),
            recall_success=success,
        )

        self.turn_log.append(turn_result)
        return turn_result

    def run(self):
        """Run the 22-turn test."""
        print("\n" + "=" * 70)
        print("R > median(R) THRESHOLD TEST")
        print("=" * 70)
        print("Testing: Top 50% selection by median R-score threshold")
        print(f"Planted Facts: {len(PLANTED_FACTS)}")
        print(f"Recall Queries: {len(RECALL_QUERIES)}")
        print()

        start_time = time.time()

        # Header for turn log
        print(f"{'Turn':<5} {'Type':<8} {'R-scores':<30} {'Items':<6} {'Result':<8}")
        print("-" * 70)

        # Execute all turns
        max_turn = max(
            max(PLANTED_FACTS.keys()),
            max(RECALL_QUERIES.keys()),
            max(CONVERSATION_FLOW.keys())
        )

        for turn in range(1, max_turn + 1):
            if turn in PLANTED_FACTS:
                fact = PLANTED_FACTS[turn]
                result = self._execute_turn(turn, fact["content"], "plant")
                r_str = self._format_r_scores(result.R_scores)
                print(f"{turn:<5} {'PLANT':<8} {r_str:<30} {result.items_rehydrated:<6}")

            elif turn in RECALL_QUERIES:
                query_data = RECALL_QUERIES[turn]
                result = self._test_recall(turn, query_data)
                r_str = self._format_r_scores(result.R_scores)
                status = "PASS" if result.recall_success else "FAIL"
                print(f"{turn:<5} {'RECALL':<8} {r_str:<30} {result.items_rehydrated:<6} {status:<8}")

            elif turn in CONVERSATION_FLOW:
                content = CONVERSATION_FLOW[turn]
                result = self._execute_turn(turn, content, "chat")
                r_str = self._format_r_scores(result.R_scores)
                print(f"{turn:<5} {'CHAT':<8} {r_str:<30} {result.items_rehydrated:<6}")

        duration = time.time() - start_time
        self._print_summary(duration)

    def _format_r_scores(self, scores: List[float], max_show: int = 4) -> str:
        """Format R-scores for display."""
        if not scores:
            return "[]"
        if len(scores) <= max_show:
            return "[" + ", ".join(f"{s:.2f}" for s in scores) + "]"
        else:
            shown = ", ".join(f"{s:.2f}" for s in scores[:max_show])
            return f"[{shown}, +{len(scores) - max_show} more]"

    def _print_summary(self, duration: float):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        # Recall accuracy
        total_recalls = len(self.recall_results)
        successful_recalls = sum(1 for r in self.recall_results if r["success"])
        recall_accuracy = successful_recalls / total_recalls if total_recalls > 0 else 0

        # Average items per turn
        all_items = [t.items_rehydrated for t in self.turn_log]
        avg_items = np.mean(all_items) if all_items else 0

        print(f"\nDuration: {duration:.1f}s")
        print(f"Total Turns: {len(self.turn_log)}")
        print(f"\n--- RETRIEVAL METRICS ---")
        print(f"Avg Items Rehydrated per Turn: {avg_items:.2f}")
        print(f"Recall Accuracy: {successful_recalls}/{total_recalls} ({recall_accuracy*100:.1f}%)")

        print(f"\n--- RECALL DETAIL ---")
        for r in self.recall_results:
            status = "PASS" if r["success"] else "FAIL"
            print(f"  Turn {r['turn']}: {status} (source: T{r['source_turn']}, items: {r['items']})")

        # R-score statistics across all turns
        all_r_scores = []
        for t in self.turn_log:
            all_r_scores.extend(t.R_scores)

        if all_r_scores:
            print(f"\n--- R-SCORE STATISTICS ---")
            print(f"Total R-scores observed: {len(all_r_scores)}")
            print(f"Mean R: {np.mean(all_r_scores):.3f}")
            print(f"Median R: {np.median(all_r_scores):.3f}")
            print(f"Min R: {np.min(all_r_scores):.3f}")
            print(f"Max R: {np.max(all_r_scores):.3f}")
            print(f"Std R: {np.std(all_r_scores):.3f}")

        print(f"\nDatabase: {self.db_path}")

        # Save results
        results_file = self.db_path.with_suffix('.json')
        results = {
            "duration": duration,
            "total_turns": len(self.turn_log),
            "avg_items_per_turn": float(avg_items),
            "recall_accuracy": recall_accuracy,
            "recalls": self.recall_results,
            "r_score_stats": {
                "count": len(all_r_scores),
                "mean": float(np.mean(all_r_scores)) if all_r_scores else 0,
                "median": float(np.median(all_r_scores)) if all_r_scores else 0,
                "min": float(np.min(all_r_scores)) if all_r_scores else 0,
                "max": float(np.max(all_r_scores)) if all_r_scores else 0,
            },
        }
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results: {results_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="R > median(R) Threshold Test")
    parser.add_argument("--provider", default=None, help="LLM provider from config")
    args = parser.parse_args()

    test = MedianThresholdTest(provider=args.provider)
    test.run()


if __name__ == "__main__":
    main()
