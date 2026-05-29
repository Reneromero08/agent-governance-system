#!/usr/bin/env python3
"""
Test: R > mean(R) Threshold
===========================

Tests a modified context partitioner that uses "R > mean(R)" threshold
instead of the fixed R > 1.0.

The approach:
1. Compute R scores for all items as usual (R = E / grad_S)
2. Instead of threshold = 1.0, use threshold = mean(R)
3. Only items with R > mean(R) get included (above-average relevance)

This is a simplified 20-turn test that:
- Plants 10 facts with specific details
- Has 5 recall queries to test retrieval
- Logs: Turn, Type, R scores, Items rehydrated count, recall success
- Prints summary: avg items per turn, recall accuracy

The partitioner is monkey-patched for this test only.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from unittest.mock import patch

CAT_CHAT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CAT_CHAT_ROOT))

from catalytic_chat.session_capsule import SessionCapsule
from catalytic_chat.auto_context_manager import AutoContextManager
from catalytic_chat.adaptive_budget import ModelBudgetDiscovery
from catalytic_chat.llm_client import get_llm_client
from catalytic_chat.context_partitioner import (
    ContextPartitioner,
    ContextItem,
    PartitionResult,
    ScoredItem,
)

import hashlib
from datetime import datetime, timezone

DB_OUTPUT_DIR = Path(__file__).parent.parent / "test_chats"


# =============================================================================
# PLANTED FACTS (10 items across 20 turns)
# =============================================================================

PLANTED_FACTS = {
    2: {
        "content": "Our primary database uses PostgreSQL 15 with connection pooling via PgBouncer. The pool size is set to 50 connections maximum.",
        "keywords": ["PostgreSQL 15", "PgBouncer", "50 connections"],
    },
    4: {
        "content": "The authentication service uses JWT tokens with RS256 algorithm. Token expiry is set to 3600 seconds (1 hour).",
        "keywords": ["JWT", "RS256", "3600 seconds"],
    },
    6: {
        "content": "Cache layer is Redis 7 with a cluster of 3 nodes. Default TTL for cached items is 300 seconds.",
        "keywords": ["Redis 7", "3 nodes", "300 seconds"],
    },
    8: {
        "content": "Message queue uses RabbitMQ with 5 replicas for high availability. Dead letter queue is named 'dlq.errors'.",
        "keywords": ["RabbitMQ", "5 replicas", "dlq.errors"],
    },
    10: {
        "content": "API rate limiting is 1000 requests per minute per API key. Burst limit is 50 requests. Exceeded returns HTTP 429.",
        "keywords": ["1000 requests", "minute", "429"],
    },
    12: {
        "content": "The deployment runs on Kubernetes with 4 nodes, each having 8 vCPU and 32GB RAM. Pod autoscaling between 2 and 20 pods.",
        "keywords": ["Kubernetes", "4 nodes", "8 vCPU", "2-20 pods"],
    },
    14: {
        "content": "Monitoring uses Prometheus with 10-second scrape interval. Alerts route to PagerDuty for P1 incidents.",
        "keywords": ["Prometheus", "10-second", "PagerDuty"],
    },
    16: {
        "content": "Backup schedule: full backup daily at 03:00 UTC, incrementals every 4 hours. Retention is 30 days.",
        "keywords": ["03:00 UTC", "4 hours", "30 days"],
    },
    17: {
        "content": "Password hashing uses Argon2id with memory cost 65536 KB and time cost 3 iterations.",
        "keywords": ["Argon2id", "65536", "3 iterations"],
    },
    18: {
        "content": "Load balancer health checks every 5 seconds. Unhealthy threshold is 3 consecutive failures.",
        "keywords": ["5 seconds", "3 consecutive", "health check"],
    },
}


# =============================================================================
# RECALL QUERIES (5 items at turns 15, 19, 20)
# =============================================================================

RECALL_QUERIES = {
    15: {
        "query": "What's our message queue setup and how do we handle failed messages?",
        "expected": ["RabbitMQ", "dlq", "5 replicas"],
        "source_turn": 8,
    },
    19: {
        "query": "What database are we using and how many connections can we handle?",
        "expected": ["PostgreSQL", "50", "PgBouncer"],
        "source_turn": 2,
    },
    20: {
        "query": "How are user passwords stored securely?",
        "expected": ["Argon2id", "65536"],
        "source_turn": 17,
    },
}

# Additional queries at specific points
RECALL_QUERIES[11] = {
    "query": "Tell me about the caching infrastructure.",
    "expected": ["Redis", "3 nodes", "300"],
    "source_turn": 6,
}
RECALL_QUERIES[13] = {
    "query": "How do we prevent API abuse?",
    "expected": ["1000", "rate", "429"],
    "source_turn": 10,
}


# =============================================================================
# CONVERSATION FILLER
# =============================================================================

CONVERSATION_FLOW = {
    1: "Let's review the infrastructure setup for the payment system.",
    3: "How does the auth flow work with the frontend?",
    5: "We should discuss caching strategies.",
    7: "What about async processing?",
    9: "Let me think about API design.",
}


# =============================================================================
# PATCHED PARTITION METHOD
# =============================================================================

def partition_with_mean_threshold(
    self,
    query_embedding: np.ndarray,
    all_items: List[ContextItem],
    budget_tokens: int,
    query_text: str = ""
) -> PartitionResult:
    """
    Partition using mean(R) as threshold instead of fixed threshold.

    Key modification: threshold = mean(R_scores) instead of self.threshold
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

    # KEY MODIFICATION: Use mean(R) as threshold instead of fixed self.threshold
    R_scores = [s.E_score for s in scored]
    dynamic_threshold = float(np.mean(R_scores)) if R_scores else 0.0

    # Partition based on DYNAMIC threshold and budget
    working_set: List[ScoredItem] = []
    pointer_set: List[ScoredItem] = []
    tokens_used = 0
    items_below_threshold = 0
    items_over_budget = 0

    for s in scored:
        if s.E_score < dynamic_threshold:
            # Below mean(R) threshold - always pointer_set
            pointer_set.append(s)
            items_below_threshold += 1
        elif tokens_used + s.item.tokens <= budget_tokens:
            # Above mean(R) and fits budget - working_set
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
        threshold=dynamic_threshold,  # Report the dynamic threshold used
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
# DATA CLASSES
# =============================================================================

@dataclass
class TurnResult:
    turn: int
    turn_type: str  # "plant", "recall", "chat"
    content: str
    R_scores: List[float]
    R_mean: float
    R_threshold: float  # The mean(R) threshold used
    items_rehydrated: int
    tokens: int
    response: str
    recall_success: Optional[bool] = None  # For recall turns


# =============================================================================
# TEST CLASS
# =============================================================================

class RMeanThresholdTest:
    """Test using mean(R) as dynamic threshold."""

    def __init__(self, provider: Optional[str] = None):
        self.db_path = DB_OUTPUT_DIR / f"test_r_mean_{int(time.time())}.db"
        print(f"Database: {self.db_path}")

        self.llm = get_llm_client(provider)
        print(f"LLM: {self.llm.config.name} ({self.llm.config.model})")
        print(f"URL: {self.llm.config.base_url}")

        self.capsule = SessionCapsule(db_path=self.db_path)
        self.session_id = self.capsule.create_session()

        self.system_prompt = ""

        # 8K token budget
        self.budget = ModelBudgetDiscovery.from_context_window(
            context_window=8192,
            system_prompt=self.system_prompt,
            response_reserve_pct=0.25,
            model_id=self.llm.config.model
        )

        # Manager with default threshold (will be overridden by patch)
        self.manager = AutoContextManager(
            db_path=self.db_path,
            session_id=self.session_id,
            budget=self.budget,
            embed_fn=self._embed,
            E_threshold=1.0  # This will be overridden by mean(R)
        )
        self.manager.capsule = self.capsule

        self.turn_log: List[TurnResult] = []

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
        """Execute a single turn with patched partitioner."""
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
                if hasattr(item, 'E') or (isinstance(item, dict) and 'E' in item):
                    r = item.E if hasattr(item, 'E') else item.get('E', 0)
                    R_scores.append(float(r))

        # Get partition result for threshold info
        partition_result = prep.partition_result if prep and hasattr(prep, 'partition_result') else None
        R_threshold = partition_result.threshold if partition_result else 0.0
        R_mean_val = partition_result.E_mean if partition_result else 0.0

        turn_result = TurnResult(
            turn=turn,
            turn_type=turn_type,
            content=content[:80] + "..." if len(content) > 80 else content,
            R_scores=R_scores,
            R_mean=R_mean_val,
            R_threshold=R_threshold,
            items_rehydrated=len(working_set),
            tokens=result.tokens_in_context,
            response=result.response[:100] + "..." if len(result.response) > 100 else result.response,
        )

        self.turn_log.append(turn_result)
        return turn_result

    def _test_recall(self, turn: int, query_data: dict) -> TurnResult:
        """Test recall and check if expected keywords are found."""
        query = query_data["query"]
        expected = query_data["expected"]

        result = self.manager.respond_catalytic(
            query=query,
            llm_generate=self._llm_generate,
            system_prompt=self.system_prompt
        )

        prep = result.prepare_result
        context_text = prep.get_context_text() if prep else ""
        working_set = prep.working_set if prep else []

        # Check if expected keywords are in context
        context_lower = context_text.lower()
        context_found = any(kw.lower() in context_lower for kw in expected)

        # Extract R-scores
        R_scores = []
        if prep and hasattr(prep, 'working_set'):
            for item in prep.working_set:
                if hasattr(item, 'E') or (isinstance(item, dict) and 'E' in item):
                    r = item.E if hasattr(item, 'E') else item.get('E', 0)
                    R_scores.append(float(r))

        partition_result = prep.partition_result if prep and hasattr(prep, 'partition_result') else None
        R_threshold = partition_result.threshold if partition_result else 0.0
        R_mean_val = partition_result.E_mean if partition_result else 0.0

        turn_result = TurnResult(
            turn=turn,
            turn_type="recall",
            content=query[:80] + "..." if len(query) > 80 else query,
            R_scores=R_scores,
            R_mean=R_mean_val,
            R_threshold=R_threshold,
            items_rehydrated=len(working_set),
            tokens=result.tokens_in_context,
            response=result.response[:100] + "..." if len(result.response) > 100 else result.response,
            recall_success=context_found,
        )

        self.turn_log.append(turn_result)
        return turn_result

    def run(self):
        """Run the 20-turn test with patched partitioner."""
        print("\n" + "=" * 70)
        print("TEST: R > mean(R) THRESHOLD")
        print("=" * 70)
        print("Testing dynamic threshold using mean of R-scores")
        print(f"Planted Facts: {len(PLANTED_FACTS)}")
        print(f"Recall Queries: {len(RECALL_QUERIES)}")
        print()

        start_time = time.time()

        # PATCH the partitioner's partition method
        original_partition = ContextPartitioner.partition

        try:
            # Apply the monkey patch
            ContextPartitioner.partition = partition_with_mean_threshold

            print("Turn  Type    R-mean  R-thresh  Items  Recall")
            print("-" * 55)

            for turn in range(1, 21):
                if turn in PLANTED_FACTS:
                    fact = PLANTED_FACTS[turn]
                    result = self._execute_turn(turn, fact["content"], "plant")
                    print(f"{turn:4d}  PLANT   {result.R_mean:6.3f}  {result.R_threshold:6.3f}    {result.items_rehydrated:3d}     -")

                elif turn in RECALL_QUERIES:
                    query_data = RECALL_QUERIES[turn]
                    result = self._test_recall(turn, query_data)
                    status = "PASS" if result.recall_success else "FAIL"
                    print(f"{turn:4d}  RECALL  {result.R_mean:6.3f}  {result.R_threshold:6.3f}    {result.items_rehydrated:3d}   {status}")

                elif turn in CONVERSATION_FLOW:
                    content = CONVERSATION_FLOW[turn]
                    result = self._execute_turn(turn, content, "chat")
                    print(f"{turn:4d}  CHAT    {result.R_mean:6.3f}  {result.R_threshold:6.3f}    {result.items_rehydrated:3d}     -")

                else:
                    content = "Let's continue the discussion."
                    result = self._execute_turn(turn, content, "filler")
                    # Don't print filler turns to keep output clean

        finally:
            # Restore original partition method
            ContextPartitioner.partition = original_partition

        duration = time.time() - start_time
        self._print_summary(duration)

    def _print_summary(self, duration: float):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        # Calculate statistics
        recall_results = [t for t in self.turn_log if t.turn_type == "recall"]
        total_recalls = len(recall_results)
        successful_recalls = sum(1 for r in recall_results if r.recall_success)

        all_items_counts = [t.items_rehydrated for t in self.turn_log]
        avg_items = np.mean(all_items_counts) if all_items_counts else 0

        all_R_means = [t.R_mean for t in self.turn_log if t.R_mean > 0]
        avg_R_mean = np.mean(all_R_means) if all_R_means else 0

        all_thresholds = [t.R_threshold for t in self.turn_log if t.R_threshold > 0]
        avg_threshold = np.mean(all_thresholds) if all_thresholds else 0

        print(f"\nDuration: {duration:.1f}s")
        print(f"Total Turns: {len(self.turn_log)}")
        print(f"\n--- RECALL ACCURACY ---")
        print(f"Successful: {successful_recalls}/{total_recalls} ({successful_recalls/total_recalls*100:.1f}%)" if total_recalls > 0 else "No recalls")

        print(f"\n--- CONTEXT STATS ---")
        print(f"Avg items rehydrated per turn: {avg_items:.2f}")
        print(f"Avg R-mean across turns: {avg_R_mean:.3f}")
        print(f"Avg dynamic threshold (mean R): {avg_threshold:.3f}")

        print(f"\n--- RECALL DETAIL ---")
        for r in recall_results:
            status = "PASS" if r.recall_success else "FAIL"
            print(f"Turn {r.turn}: {status} | Items: {r.items_rehydrated} | R-thresh: {r.R_threshold:.3f}")

        # Save results
        results_file = self.db_path.with_suffix('.json')
        results = {
            "duration": duration,
            "total_turns": len(self.turn_log),
            "recall_accuracy": successful_recalls / total_recalls if total_recalls > 0 else 0,
            "avg_items_per_turn": avg_items,
            "avg_R_mean": avg_R_mean,
            "avg_threshold": avg_threshold,
            "turns": [
                {
                    "turn": t.turn,
                    "type": t.turn_type,
                    "R_mean": t.R_mean,
                    "R_threshold": t.R_threshold,
                    "items_rehydrated": t.items_rehydrated,
                    "recall_success": t.recall_success,
                }
                for t in self.turn_log
            ],
        }
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {results_file}")
        print(f"Database: {self.db_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test R > mean(R) threshold")
    parser.add_argument("--provider", default=None, help="LLM provider from config")
    args = parser.parse_args()

    test = RMeanThresholdTest(provider=args.provider)
    test.run()


if __name__ == "__main__":
    main()
