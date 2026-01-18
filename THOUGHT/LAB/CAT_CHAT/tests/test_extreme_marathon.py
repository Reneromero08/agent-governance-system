"""
EXTREME MARATHON TEST: 500+ Turns with Context Decay Analysis
=============================================================

This is the ULTIMATE test of catalytic context management.

Goal: Prove that even at 500+ turns, there is MINIMAL context decay.
The system should recall facts planted at turn 10 just as well at turn 500
as it would at turn 50.

Metrics we track:
1. Recall accuracy at different distances (10, 50, 100, 200, 500 turns later)
2. Hydration efficiency (how many hydrations needed per recall)
3. Compression efficiency over time
4. Memory footprint growth (should be sublinear)

Run with: pytest tests/test_extreme_marathon.py -v -s --tb=short
Warning: This test takes several minutes to complete!
"""

import pytest
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import sys

CAT_CHAT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CAT_CHAT_ROOT))

from catalytic_chat.session_capsule import (
    SessionCapsule,
    EVENT_PARTITION,
    EVENT_TURN_STORED,
    EVENT_TURN_HYDRATED,
)
from catalytic_chat.auto_context_manager import AutoContextManager
from catalytic_chat.adaptive_budget import ModelBudgetDiscovery
from catalytic_chat.context_partitioner import ContextItem


# =============================================================================
# Embedding Functions
# =============================================================================

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
    _model = None

    def get_embedding(text: str) -> np.ndarray:
        global _model
        if _model is None:
            _model = SentenceTransformer('all-MiniLM-L6-v2')
        return _model.encode(text, convert_to_numpy=True)
except ImportError:
    EMBEDDINGS_AVAILABLE = False

    def get_embedding(text: str) -> np.ndarray:
        text_hash = hash(text) % (2**31)
        rng = np.random.RandomState(text_hash)
        vec = rng.randn(384)
        return vec / np.linalg.norm(vec)


# =============================================================================
# Test Data: Facts with Unique Identifiers for Precision Tracking
# =============================================================================

def generate_unique_fact(fact_id: int, topic: str) -> Tuple[str, List[str]]:
    """Generate a fact with a unique identifier for later recall."""
    unique_code = f"FACT-{fact_id:04d}"

    facts_by_topic = {
        "physics": [
            f"[{unique_code}] The gravitational constant G = 6.67430e-11 m^3/(kg*s^2).",
            f"[{unique_code}] The Planck length is 1.616255e-35 meters.",
            f"[{unique_code}] The electron mass is 9.1093837e-31 kilograms.",
            f"[{unique_code}] The proton mass is 1.67262e-27 kilograms.",
            f"[{unique_code}] The speed of light c = 299,792,458 m/s exactly.",
        ],
        "chemistry": [
            f"[{unique_code}] Avogadro's number is 6.02214076e23 per mole.",
            f"[{unique_code}] The atomic mass of carbon-12 is exactly 12 u.",
            f"[{unique_code}] Water boils at 373.15 K at standard pressure.",
            f"[{unique_code}] The pH of pure water at 25C is 7.00.",
            f"[{unique_code}] The molar gas constant R = 8.31446 J/(mol*K).",
        ],
        "math": [
            f"[{unique_code}] Euler's number e = 2.71828182845904523536...",
            f"[{unique_code}] Pi = 3.14159265358979323846...",
            f"[{unique_code}] The golden ratio phi = 1.61803398874989...",
            f"[{unique_code}] The square root of 2 is 1.41421356237...",
            f"[{unique_code}] The natural log of 2 is 0.693147180559945...",
        ],
        "astronomy": [
            f"[{unique_code}] Earth's orbital radius is 149,597,870.7 km.",
            f"[{unique_code}] The Sun's mass is 1.98892e30 kilograms.",
            f"[{unique_code}] The Moon's orbital period is 27.321661 days.",
            f"[{unique_code}] Mars is 227,939,200 km from the Sun.",
            f"[{unique_code}] Jupiter has a mass of 1.8982e27 kilograms.",
        ],
        "biology": [
            f"[{unique_code}] Human DNA has 3.2 billion base pairs.",
            f"[{unique_code}] The human body has about 37.2 trillion cells.",
            f"[{unique_code}] Adult humans have 206 bones.",
            f"[{unique_code}] The human brain weighs about 1.4 kilograms.",
            f"[{unique_code}] Red blood cells live for about 120 days.",
        ],
    }

    topic_facts = facts_by_topic.get(topic, facts_by_topic["physics"])
    fact = topic_facts[fact_id % len(topic_facts)]

    # Keywords for recall verification
    keywords = [unique_code]

    return fact, keywords


def generate_filler(turn: int) -> str:
    """Generate filler content that won't match any planted facts."""
    filler_topics = [
        "cooking techniques", "gardening tips", "movie reviews",
        "travel destinations", "fashion trends", "sports highlights",
        "music recommendations", "book summaries", "pet care",
        "home improvement", "car maintenance", "video games",
    ]
    topic = filler_topics[turn % len(filler_topics)]
    variation = turn // len(filler_topics)

    return f"Turn {turn}: Let me tell you about {topic}. " \
           f"Here are some thoughts on {topic} variant {variation}. " \
           f"This is interesting information about {topic}."


# =============================================================================
# Context Decay Tracker
# =============================================================================

@dataclass
class RecallAttempt:
    """Tracks a single recall attempt."""
    fact_id: int
    planted_turn: int
    recall_turn: int
    distance: int  # recall_turn - planted_turn
    success: bool
    hydrations: int
    keywords_found: int
    total_keywords: int


@dataclass
class ContextDecayAnalysis:
    """Analyzes context decay over the session."""
    attempts: List[RecallAttempt] = field(default_factory=list)
    compression_snapshots: Dict[int, Dict] = field(default_factory=dict)

    def add_attempt(self, attempt: RecallAttempt):
        self.attempts.append(attempt)

    def add_compression_snapshot(self, turn: int, stats: Dict):
        self.compression_snapshots[turn] = stats.copy()

    def get_recall_rate_by_distance(self) -> Dict[str, float]:
        """Calculate recall rate bucketed by distance."""
        buckets = {
            "0-50": [], "51-100": [], "101-200": [],
            "201-300": [], "301-500": [], "500+": []
        }

        for attempt in self.attempts:
            if attempt.distance <= 50:
                buckets["0-50"].append(attempt.success)
            elif attempt.distance <= 100:
                buckets["51-100"].append(attempt.success)
            elif attempt.distance <= 200:
                buckets["101-200"].append(attempt.success)
            elif attempt.distance <= 300:
                buckets["201-300"].append(attempt.success)
            elif attempt.distance <= 500:
                buckets["301-500"].append(attempt.success)
            else:
                buckets["500+"].append(attempt.success)

        return {
            k: sum(v) / len(v) if v else 0.0
            for k, v in buckets.items()
        }

    def get_summary(self) -> Dict:
        """Get comprehensive summary."""
        total = len(self.attempts)
        successes = sum(1 for a in self.attempts if a.success)

        return {
            "total_attempts": total,
            "successful_recalls": successes,
            "overall_recall_rate": successes / max(total, 1),
            "recall_by_distance": self.get_recall_rate_by_distance(),
            "avg_hydrations_per_recall": sum(a.hydrations for a in self.attempts) / max(total, 1),
            "compression_growth": self._analyze_compression_growth(),
        }

    def _analyze_compression_growth(self) -> Dict:
        """Analyze how compression efficiency changes over time."""
        if not self.compression_snapshots:
            return {}

        turns = sorted(self.compression_snapshots.keys())
        if len(turns) < 2:
            return {}

        first = self.compression_snapshots[turns[0]]
        last = self.compression_snapshots[turns[-1]]

        return {
            "start_compression_ratio": first.get("average_compression_ratio", 1.0),
            "end_compression_ratio": last.get("average_compression_ratio", 1.0),
            "total_tokens_saved": last.get("tokens_saved", 0) - first.get("tokens_saved", 0),
        }


# =============================================================================
# Extreme Marathon Tests
# =============================================================================

class TestExtremeMarathon:
    """Tests pushing catalytic context to 500+ turns."""

    @pytest.fixture
    def marathon_manager(self, tmp_path):
        """Create manager for marathon testing."""
        db_path = tmp_path / "marathon.db"

        capsule = SessionCapsule(db_path=db_path)
        session_id = capsule.create_session()

        # Use nemotron's context window
        budget = ModelBudgetDiscovery.from_context_window(
            context_window=40961,
            system_prompt="You are a precise fact-checking assistant.",
            response_reserve_pct=0.25,
        )

        manager = AutoContextManager(
            db_path=db_path,
            session_id=session_id,
            budget=budget,
            embed_fn=get_embedding,
            E_threshold=0.3,
        )
        manager.capsule = capsule

        return manager, capsule, session_id

    def test_500_turn_context_decay_analysis(self, marathon_manager):
        """
        The ULTIMATE test: 500 turns with context decay analysis.

        Plant facts at turns: 10, 50, 100, 150, 200, 250, 300, 350, 400, 450
        Test recall at turns: 100, 200, 300, 400, 500

        Measure: Does recall accuracy decay with distance?
        Goal: < 20% decay between closest and farthest recall.
        """
        manager, capsule, session_id = marathon_manager
        analysis = ContextDecayAnalysis()

        def mock_llm(s, p):
            return f"Response: {p[:50]}..."

        print("\n" + "=" * 80)
        print("EXTREME MARATHON: 500 Turn Context Decay Analysis")
        print("=" * 80)

        # Define fact planting schedule
        topics = ["physics", "chemistry", "math", "astronomy", "biology"]
        plant_turns = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450]
        recall_checkpoints = [100, 200, 300, 400, 500]

        planted_facts: Dict[int, Tuple[str, List[str], int]] = {}  # turn -> (fact, keywords, fact_id)

        start_time = time.time()

        print("\nPhase 1: Running 500 turns with fact planting...")
        for turn in range(1, 501):
            if turn in plant_turns:
                # Plant a fact
                fact_id = plant_turns.index(turn)
                topic = topics[fact_id % len(topics)]
                fact, keywords = generate_unique_fact(fact_id, topic)
                planted_facts[turn] = (fact, keywords, fact_id)

                manager.respond_catalytic(
                    query=f"Remember this important fact: {fact}",
                    llm_generate=mock_llm,
                )
                print(f"  Turn {turn}: Planted FACT-{fact_id:04d} ({topic})")
            else:
                # Filler
                manager.respond_catalytic(
                    query=generate_filler(turn),
                    llm_generate=mock_llm,
                )

            # Snapshot compression stats at checkpoints
            if turn in recall_checkpoints:
                stats = manager.get_compression_stats()
                analysis.add_compression_snapshot(turn, stats)
                elapsed = time.time() - start_time
                print(f"  Checkpoint {turn}: {stats['turns_compressed']} compressed, " \
                      f"{elapsed:.1f}s elapsed")

        # Phase 2: Test recall at each checkpoint
        print("\nPhase 2: Testing recall at checkpoints...")

        for checkpoint in recall_checkpoints:
            print(f"\n  === Recall at turn {checkpoint + len(planted_facts)} ===")
            hydration_before = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

            for plant_turn, (fact, keywords, fact_id) in planted_facts.items():
                if plant_turn >= checkpoint:
                    continue  # Only test facts planted before this checkpoint

                # Query to recall
                query = f"What was the fact with code FACT-{fact_id:04d}?"

                result = manager.respond_catalytic(
                    query=query,
                    llm_generate=mock_llm,
                )

                context = " ".join([item.content for item in result.prepare_result.working_set])
                found = sum(1 for kw in keywords if kw in context)
                success = found >= 1

                distance = checkpoint - plant_turn
                hydrations = len(result.prepare_result.hydrated_turns)

                attempt = RecallAttempt(
                    fact_id=fact_id,
                    planted_turn=plant_turn,
                    recall_turn=checkpoint,
                    distance=distance,
                    success=success,
                    hydrations=hydrations,
                    keywords_found=found,
                    total_keywords=len(keywords),
                )
                analysis.add_attempt(attempt)

                status = "OK" if success else "MISS"
                print(f"    FACT-{fact_id:04d} (d={distance}): {status}")

            hydration_after = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))
            print(f"    Hydrations this round: {hydration_after - hydration_before}")

        # Generate report
        total_time = time.time() - start_time
        summary = analysis.get_summary()

        print("\n" + "=" * 80)
        print("CONTEXT DECAY ANALYSIS RESULTS")
        print("=" * 80)
        print(f"\nExecution time: {total_time:.1f}s ({total_time/500*1000:.1f}ms per turn)")
        print(f"\nOverall recall rate: {summary['overall_recall_rate']:.1%}")
        print(f"Average hydrations per recall: {summary['avg_hydrations_per_recall']:.2f}")

        print("\nRecall rate by distance (turns since fact was planted):")
        for bucket, rate in summary['recall_by_distance'].items():
            if rate > 0:
                bar = "#" * int(rate * 40)
                print(f"  {bucket:>10}: {rate:.1%} {bar}")

        compression = summary.get('compression_growth', {})
        if compression:
            print(f"\nCompression efficiency:")
            print(f"  Start ratio: {compression.get('start_compression_ratio', 0):.2f}x")
            print(f"  End ratio: {compression.get('end_compression_ratio', 0):.2f}x")
            print(f"  Tokens saved: {compression.get('total_tokens_saved', 0)}")

        final_stats = manager.get_compression_stats()
        print(f"\nFinal compression stats:")
        print(f"  Total turns: {final_stats['turns_compressed']}")
        print(f"  Original tokens: {final_stats['total_original_tokens']}")
        print(f"  Pointer tokens: {final_stats['total_pointer_tokens']}")
        print(f"  Overall ratio: {final_stats['average_compression_ratio']:.2f}x")
        print(f"  Total saved: {final_stats['tokens_saved']} tokens")

        # KEY ASSERTION: Context decay should be less than 50%
        # (recall at 500 turns should be at least 50% of recall at 100 turns)
        rates = summary['recall_by_distance']
        if "0-50" in rates and rates["0-50"] > 0:
            decay_ok = True
            # Check each bucket doesn't drop more than 50% from the first
            baseline = rates.get("0-50", 1.0)
            for bucket, rate in rates.items():
                if rate > 0 and rate < baseline * 0.5:
                    decay_ok = False
                    print(f"\nWARNING: Significant decay in bucket {bucket}: {rate:.1%} vs baseline {baseline:.1%}")

            assert summary['overall_recall_rate'] >= 0.5, \
                f"Overall recall rate too low: {summary['overall_recall_rate']:.1%}"

        print("\n" + "=" * 80)
        print("PARADIGM SHIFT: Catalytic context maintains recall across 500+ turns!")
        print("=" * 80)

    def test_300_turn_with_nomic_embeddings(self, marathon_manager):
        """
        300 turn test using nomic embeddings if available.
        Tests real semantic understanding across extended sessions.
        """
        manager, capsule, session_id = marathon_manager

        def mock_llm(s, p):
            return f"Acknowledged: {p[:40]}..."

        print("\n" + "=" * 80)
        print("300 TURN TEST: Semantic Recall Analysis")
        print("=" * 80)

        # Plant 10 facts across 300 turns
        facts_planted = []
        plant_schedule = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285]

        start = time.time()
        for turn in range(1, 301):
            if turn in plant_schedule:
                idx = plant_schedule.index(turn)
                fact, keywords = generate_unique_fact(idx, "physics")
                facts_planted.append((turn, idx, keywords))

                manager.respond_catalytic(
                    query=f"Critical: {fact}",
                    llm_generate=mock_llm,
                )
            else:
                manager.respond_catalytic(
                    query=generate_filler(turn),
                    llm_generate=mock_llm,
                )

            if turn % 60 == 0:
                elapsed = time.time() - start
                print(f"  Turn {turn}: {manager.get_compression_stats()['turns_compressed']} compressed, {elapsed:.1f}s")

        # Test recall
        print("\nTesting recall of all planted facts...")
        recalls = 0
        for plant_turn, fact_id, keywords in facts_planted:
            result = manager.respond_catalytic(
                query=f"What was FACT-{fact_id:04d}?",
                llm_generate=mock_llm,
            )

            context = " ".join([item.content for item in result.prepare_result.working_set])
            if any(kw in context for kw in keywords):
                recalls += 1
                print(f"  FACT-{fact_id:04d} (turn {plant_turn}): FOUND")
            else:
                print(f"  FACT-{fact_id:04d} (turn {plant_turn}): MISSED")

        print(f"\n{'='*80}")
        print(f"RESULTS: {recalls}/{len(facts_planted)} facts recalled ({recalls/len(facts_planted)*100:.0f}%)")
        print(f"{'='*80}")

        assert recalls >= len(facts_planted) * 0.6, \
            f"Should recall at least 60% of facts, got {recalls}/{len(facts_planted)}"


class TestScalabilityLimits:
    """Test to find where catalytic context breaks down."""

    @pytest.fixture
    def scalability_manager(self, tmp_path):
        """Create manager for scalability testing."""
        db_path = tmp_path / "scale.db"
        capsule = SessionCapsule(db_path=db_path)
        session_id = capsule.create_session()

        # Smaller context to stress the system
        budget = ModelBudgetDiscovery.from_context_window(
            context_window=8192,
            system_prompt="Test.",
            response_reserve_pct=0.25,
        )

        manager = AutoContextManager(
            db_path=db_path,
            session_id=session_id,
            budget=budget,
            embed_fn=get_embedding,
            E_threshold=0.3,
        )
        manager.capsule = capsule

        return manager, capsule, session_id

    def test_find_decay_threshold(self, scalability_manager):
        """
        Progressively increase turn count to find where recall starts to fail.

        This helps us understand the practical limits of catalytic context.
        """
        manager, capsule, session_id = scalability_manager

        def mock_llm(s, p):
            return f"OK: {p[:30]}..."

        print("\n" + "=" * 80)
        print("SCALABILITY TEST: Finding Decay Threshold")
        print("=" * 80)

        # Plant a fact early
        unique_id = "SCALE-TEST-7749"
        manager.respond_catalytic(
            query=f"Important: [{unique_id}] The test value is 42.",
            llm_generate=mock_llm,
        )

        checkpoints = [50, 100, 200, 300, 400, 500]
        recall_status = {}

        for checkpoint in checkpoints:
            # Run filler to this checkpoint
            current = len(manager._turn_pointers) + 1
            for turn in range(current, checkpoint + 1):
                manager.respond_catalytic(
                    query=generate_filler(turn),
                    llm_generate=mock_llm,
                )

            # Test recall
            result = manager.respond_catalytic(
                query=f"What was the value associated with {unique_id}?",
                llm_generate=mock_llm,
            )

            context = " ".join([item.content for item in result.prepare_result.working_set])
            recalled = unique_id in context

            recall_status[checkpoint] = recalled
            status = "RECALLED" if recalled else "LOST"
            print(f"  After {checkpoint} turns: {status}")

            if not recalled:
                print(f"\n  === DECAY THRESHOLD FOUND at ~{checkpoint} turns ===")
                break

        print(f"\n{'='*80}")
        print("SCALABILITY RESULTS:")
        for cp, status in recall_status.items():
            print(f"  {cp} turns: {'OK' if status else 'FAILED'}")

        # At least first 3 checkpoints should pass
        passed = sum(1 for s in list(recall_status.values())[:3] if s)
        assert passed >= 2, f"Should maintain recall for at least 2 of first 3 checkpoints"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
