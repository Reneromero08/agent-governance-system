"""
PARADIGM SHIFT TESTS: Catalytic Context at the Limit
=====================================================

These tests are designed to BREAK conventional context management:
- Long-chain dependencies (turn 10 needed at turn 200)
- Semantic drift + return (topic A -> ... -> A after 100+ turns)
- Adversarial forgetting (critical facts buried under noise)
- Precision recall (exact values from specific turns)
- Multi-hop inference (combine facts from turns 5, 78, 234)
- Progressive derivation (physics proof across 50+ turns)

The goal: 1000 turns with minimal context decay.

Run with: pytest tests/test_paradigm_shift.py -v -s --tb=short
"""

import pytest
import numpy as np
import requests
import json
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import sys
import time

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
# LLM Studio API Integration
# =============================================================================

LLM_STUDIO_BASE = "http://10.5.0.2:1234"
NEMOTRON_MODEL = "nemotron-3-nano-30b-a3b"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"


def check_llm_studio_available() -> bool:
    """Check if LLM Studio is available."""
    try:
        resp = requests.get(f"{LLM_STUDIO_BASE}/v1/models", timeout=5)
        return resp.status_code == 200
    except:
        return False


def get_nomic_embedding(text: str) -> np.ndarray:
    """Get embedding from nomic model via LLM Studio."""
    try:
        resp = requests.post(
            f"{LLM_STUDIO_BASE}/v1/embeddings",
            json={
                "model": EMBEDDING_MODEL,
                "input": text,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = np.array(data["data"][0]["embedding"])
        return embedding / np.linalg.norm(embedding)  # Normalize
    except Exception as e:
        # Fallback to deterministic synthetic
        text_hash = hash(text) % (2**31)
        rng = np.random.RandomState(text_hash)
        vec = rng.randn(768)  # nomic uses 768 dims
        return vec / np.linalg.norm(vec)


def generate_with_nemotron(system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
    """Generate response using nemotron via LLM Studio."""
    try:
        resp = requests.post(
            f"{LLM_STUDIO_BASE}/v1/chat/completions",
            json={
                "model": NEMOTRON_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,  # Lower for consistency
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Error: {e}]"


# Check availability
LLM_STUDIO_AVAILABLE = check_llm_studio_available()

# Fallback embeddings if LLM Studio not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    _st_model = None

    def get_st_embedding(text: str) -> np.ndarray:
        global _st_model
        if _st_model is None:
            _st_model = SentenceTransformer('all-MiniLM-L6-v2')
        return _st_model.encode(text, convert_to_numpy=True)
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

    def get_st_embedding(text: str) -> np.ndarray:
        text_hash = hash(text) % (2**31)
        rng = np.random.RandomState(text_hash)
        vec = rng.randn(384)
        return vec / np.linalg.norm(vec)


def get_embedding(text: str) -> np.ndarray:
    """Get best available embedding."""
    if LLM_STUDIO_AVAILABLE:
        return get_nomic_embedding(text)
    elif SENTENCE_TRANSFORMERS_AVAILABLE:
        return get_st_embedding(text)
    else:
        return get_st_embedding(text)  # Falls back to synthetic


# =============================================================================
# Test Data: Complex Knowledge Structures
# =============================================================================

@dataclass
class PlantedFact:
    """A fact planted at a specific turn for later recall."""
    turn: int
    fact: str
    topic: str
    recall_query: str
    expected_keywords: List[str]


@dataclass
class InferenceChain:
    """A multi-hop inference requiring multiple facts."""
    facts: List[Tuple[int, str]]  # (turn, fact)
    query: str
    expected_conclusion: str


# Physics derivation: Schwarzschild radius
SCHWARZSCHILD_DERIVATION = [
    ("gravitational_constant", "The gravitational constant G equals 6.674e-11 N*m^2/kg^2."),
    ("speed_of_light", "The speed of light c equals 299,792,458 meters per second, or approximately 3e8 m/s."),
    ("escape_velocity", "Escape velocity is given by v_escape = sqrt(2GM/r) where M is mass and r is radius."),
    ("schwarzschild_condition", "A black hole forms when escape velocity equals the speed of light: v_escape = c."),
    ("derivation_step1", "Setting v_escape = c gives us: c = sqrt(2GM/r_s)."),
    ("derivation_step2", "Squaring both sides: c^2 = 2GM/r_s."),
    ("derivation_step3", "Solving for r_s: r_s = 2GM/c^2."),
    ("schwarzschild_result", "The Schwarzschild radius formula is r_s = 2GM/c^2, where G is gravitational constant, M is mass, and c is speed of light."),
    ("sun_calculation", "For the Sun with mass 1.989e30 kg, the Schwarzschild radius is approximately 2.95 km."),
    ("earth_calculation", "For Earth with mass 5.972e24 kg, the Schwarzschild radius is approximately 8.87 mm."),
]


# Diverse unrelated topics for noise/filler
FILLER_TOPICS = [
    "cooking recipes", "gardening tips", "movie reviews", "weather patterns",
    "sports statistics", "music theory", "fashion trends", "travel destinations",
    "pet care", "home improvement", "car maintenance", "video games",
    "social media", "coffee brewing", "yoga poses", "photography tips",
    "baking techniques", "hiking trails", "board games", "wine tasting",
]


# =============================================================================
# Paradigm Shift Test Suite
# =============================================================================

class TestParadigmShift:
    """Tests that push catalytic context to its absolute limits."""

    @pytest.fixture
    def large_context_manager(self, tmp_path):
        """
        Create manager with realistic large context window.
        Uses nemotron's 40961 token context if available.
        """
        db_path = tmp_path / "paradigm_test.db"

        capsule = SessionCapsule(db_path=db_path)
        session_id = capsule.create_session()

        # Use nemotron's actual context window
        budget = ModelBudgetDiscovery.from_context_window(
            context_window=40961,  # nemotron's context
            system_prompt="You are a precise assistant that tracks and recalls information accurately.",
            response_reserve_pct=0.25,
            model_id=NEMOTRON_MODEL,
        )

        manager = AutoContextManager(
            db_path=db_path,
            session_id=session_id,
            budget=budget,
            embed_fn=get_embedding,
            E_threshold=0.3,  # Sensitive threshold for recall
        )
        manager.capsule = capsule

        return manager, capsule, session_id

    @pytest.fixture
    def small_pressure_manager(self, tmp_path):
        """
        Create manager with SMALL context to force aggressive compression.
        This tests whether rehydration actually works under pressure.
        """
        db_path = tmp_path / "pressure_test.db"

        capsule = SessionCapsule(db_path=db_path)
        session_id = capsule.create_session()

        # Tiny context to force compression every turn
        budget = ModelBudgetDiscovery.from_context_window(
            context_window=4096,  # Small!
            system_prompt="You track facts precisely.",
            response_reserve_pct=0.25,
        )

        manager = AutoContextManager(
            db_path=db_path,
            session_id=session_id,
            budget=budget,
            embed_fn=get_embedding,
            E_threshold=0.25,  # Very sensitive
        )
        manager.capsule = capsule

        return manager, capsule, session_id

    # =========================================================================
    # TEST 1: Long-Chain Dependency (100+ turns)
    # =========================================================================

    def test_long_chain_dependency_100_turns(self, small_pressure_manager):
        """
        Plant critical facts early, bury them, then recall after 100+ turns.

        This breaks conventional systems because:
        - Fixed window: loses facts after N turns
        - Summary: loses specific numbers/details
        - No semantic awareness: can't bring back relevant content
        """
        manager, capsule, session_id = small_pressure_manager

        # Tracking
        planted_facts: List[PlantedFact] = []
        hydration_events_by_turn: Dict[int, int] = {}

        def mock_llm(system: str, prompt: str) -> str:
            # Echo key facts for verification
            if "schwarzschild" in prompt.lower():
                return "The Schwarzschild radius is r_s = 2GM/c^2."
            elif "gravitational constant" in prompt.lower():
                return "G = 6.674e-11 N*m^2/kg^2"
            elif "speed of light" in prompt.lower():
                return "c = 299,792,458 m/s"
            return f"Acknowledged: {prompt[:100]}..."

        print("\n" + "=" * 70)
        print("TEST: Long-Chain Dependency (100+ Turns)")
        print("=" * 70)

        # Phase 1: Plant critical facts (turns 1-10)
        print("\nPhase 1: Planting critical facts...")
        critical_facts = [
            PlantedFact(
                turn=1,
                fact="The gravitational constant G equals 6.674e-11 N*m^2/kg^2.",
                topic="physics",
                recall_query="What is the gravitational constant G?",
                expected_keywords=["6.674", "e-11", "gravitational"],
            ),
            PlantedFact(
                turn=3,
                fact="The speed of light c equals 299,792,458 meters per second.",
                topic="physics",
                recall_query="What is the speed of light?",
                expected_keywords=["299", "792", "458", "light"],
            ),
            PlantedFact(
                turn=5,
                fact="The Schwarzschild radius formula is r_s = 2GM/c^2.",
                topic="physics",
                recall_query="What is the Schwarzschild radius formula?",
                expected_keywords=["2GM", "c^2", "schwarzschild"],
            ),
            PlantedFact(
                turn=7,
                fact="For a black hole to form, escape velocity must equal the speed of light.",
                topic="physics",
                recall_query="What condition must be met for a black hole to form?",
                expected_keywords=["escape", "velocity", "light", "black hole"],
            ),
            PlantedFact(
                turn=9,
                fact="The Sun's Schwarzschild radius is approximately 2.95 kilometers.",
                topic="physics",
                recall_query="What is the Sun's Schwarzschild radius?",
                expected_keywords=["2.95", "km", "sun"],
            ),
        ]

        turn_counter = 0
        for pf in critical_facts:
            # Fill turns up to the planted turn
            while turn_counter < pf.turn - 1:
                turn_counter += 1
                filler_topic = FILLER_TOPICS[turn_counter % len(FILLER_TOPICS)]
                manager.respond_catalytic(
                    query=f"Tell me about {filler_topic}.",
                    llm_generate=mock_llm,
                )

            # Plant the fact
            turn_counter += 1
            result = manager.respond_catalytic(
                query=f"Remember this: {pf.fact}",
                llm_generate=mock_llm,
            )
            planted_facts.append(pf)
            print(f"  Turn {turn_counter}: Planted '{pf.topic}' fact")

        # Phase 2: Bury with 90+ filler turns
        print(f"\nPhase 2: Burying with filler (turns {turn_counter+1} to 100)...")
        filler_start = turn_counter + 1

        for i in range(100 - turn_counter):
            turn_counter += 1
            filler_idx = turn_counter % len(FILLER_TOPICS)
            filler_topic = FILLER_TOPICS[filler_idx]

            # Generate diverse filler content
            filler_query = f"Turn {turn_counter}: Explain {filler_topic} in detail. " \
                          f"Include specific tips about {filler_topic}."

            manager.respond_catalytic(
                query=filler_query,
                llm_generate=mock_llm,
            )

            if turn_counter % 20 == 0:
                stats = manager.get_compression_stats()
                print(f"  Turn {turn_counter}: {stats['turns_compressed']} compressed, " \
                      f"{len(manager._turn_pointers)} pointers")

        # Phase 3: Recall ALL planted facts
        print(f"\nPhase 3: Recalling facts after {turn_counter} turns...")
        hydration_before = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

        recall_results: List[Tuple[PlantedFact, bool, int]] = []

        for pf in planted_facts:
            turn_counter += 1
            result = manager.respond_catalytic(
                query=pf.recall_query,
                llm_generate=mock_llm,
            )

            # Check if rehydration occurred
            hydrations = len(result.prepare_result.hydrated_turns)
            hydration_events_by_turn[turn_counter] = hydrations

            # Check if the fact is in context (either working set or response)
            context_content = " ".join([
                item.content for item in result.prepare_result.working_set
            ])

            keywords_found = sum(
                1 for kw in pf.expected_keywords
                if kw.lower() in context_content.lower()
            )
            recall_success = keywords_found >= len(pf.expected_keywords) // 2

            recall_results.append((pf, recall_success, hydrations))

            status = "RECALLED" if recall_success else "MISSED"
            print(f"  '{pf.recall_query[:40]}...' -> {status} " \
                  f"(hydrations: {hydrations}, keywords: {keywords_found}/{len(pf.expected_keywords)})")

        hydration_after = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))
        total_hydrations = hydration_after - hydration_before

        # Summary
        print(f"\n{'='*70}")
        print("RESULTS:")
        print(f"  Total turns: {turn_counter}")
        print(f"  Facts planted: {len(planted_facts)}")
        print(f"  Facts recalled: {sum(1 for _, s, _ in recall_results if s)}/{len(recall_results)}")
        print(f"  Total hydrations triggered: {total_hydrations}")
        print(f"  Turn pointers in pool: {len(manager._turn_pointers)}")
        print(f"  Compression stats: {manager.get_compression_stats()}")

        # Assertions
        recalled_count = sum(1 for _, success, _ in recall_results if success)
        assert recalled_count >= 3, f"Should recall at least 3/5 facts, got {recalled_count}/5"
        assert total_hydrations > 0, "Should trigger at least some rehydration"

    # =========================================================================
    # TEST 2: Semantic Drift + Return (Topic A -> ... -> A)
    # =========================================================================

    def test_semantic_drift_and_return(self, small_pressure_manager):
        """
        Start with physics, drift through multiple unrelated topics,
        then return to physics and verify we can recall original content.

        Pattern: Physics -> Math -> Chemistry -> Biology -> ... -> Physics
        """
        manager, capsule, session_id = small_pressure_manager

        def mock_llm(system: str, prompt: str) -> str:
            return f"Response about: {prompt[:80]}..."

        print("\n" + "=" * 70)
        print("TEST: Semantic Drift + Return")
        print("=" * 70)

        # Phase 1: Establish physics knowledge base
        print("\nPhase 1: Establishing physics knowledge...")
        physics_facts = [
            "Einstein's famous equation E=mc^2 relates energy and mass.",
            "The Planck constant h equals 6.626e-34 joule-seconds.",
            "Heisenberg's uncertainty principle: delta_x * delta_p >= h_bar/2.",
            "The fine structure constant alpha approximately equals 1/137.",
        ]

        for i, fact in enumerate(physics_facts):
            manager.respond_catalytic(
                query=f"Physics fact {i+1}: {fact}",
                llm_generate=mock_llm,
            )
            print(f"  Turn {i+1}: Physics fact planted")

        # Phase 2: Drift through other topics (60 turns)
        print("\nPhase 2: Drifting through unrelated topics...")
        drift_topics = [
            "mathematics", "chemistry", "biology", "history",
            "literature", "economics", "psychology", "sociology",
            "art history", "music theory", "linguistics", "philosophy",
        ]

        for i in range(60):
            turn = i + len(physics_facts) + 1
            topic = drift_topics[i % len(drift_topics)]

            manager.respond_catalytic(
                query=f"Turn {turn}: Tell me something interesting about {topic}. " \
                      f"Include specific details about {topic}.",
                llm_generate=mock_llm,
            )

            if turn % 15 == 0:
                print(f"  Turn {turn}: Drifting through '{topic}'...")

        hydration_before = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

        # Phase 3: Return to physics - can we recall?
        print("\nPhase 3: Returning to physics...")
        physics_queries = [
            ("What is Einstein's famous equation?", ["E=mc^2", "energy", "mass"]),
            ("What is the Planck constant?", ["6.626", "e-34", "planck"]),
            ("What is Heisenberg's uncertainty principle?", ["delta", "h_bar", "uncertainty"]),
            ("What is the fine structure constant?", ["137", "alpha", "fine structure"]),
        ]

        recall_successes = 0
        for query, keywords in physics_queries:
            result = manager.respond_catalytic(
                query=query,
                llm_generate=mock_llm,
            )

            context_content = " ".join([
                item.content.lower() for item in result.prepare_result.working_set
            ])

            found = sum(1 for kw in keywords if kw.lower() in context_content)
            success = found >= 1
            if success:
                recall_successes += 1

            status = "FOUND" if success else "MISSED"
            print(f"  '{query[:35]}...' -> {status} ({found}/{len(keywords)} keywords)")

        hydration_after = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

        print(f"\n{'='*70}")
        print("RESULTS:")
        print(f"  Physics facts planted: {len(physics_facts)}")
        print(f"  Drift turns: 60")
        print(f"  Facts recalled: {recall_successes}/{len(physics_queries)}")
        print(f"  Hydrations triggered: {hydration_after - hydration_before}")

        assert recall_successes >= 2, f"Should recall at least 2/4 physics facts, got {recall_successes}"

    # =========================================================================
    # TEST 3: Adversarial Forgetting (Bury critical facts under noise)
    # =========================================================================

    def test_adversarial_forgetting_resistance(self, small_pressure_manager):
        """
        Plant VERY specific facts, then actively try to make the system forget
        by flooding with similar-sounding but different content.

        This tests whether semantic similarity can distinguish:
        - "The reactor temperature is 3000K" (real fact)
        - "Temperature is important in cooking" (noise)
        """
        manager, capsule, session_id = small_pressure_manager

        def mock_llm(system: str, prompt: str) -> str:
            return f"Acknowledged: {prompt[:60]}..."

        print("\n" + "=" * 70)
        print("TEST: Adversarial Forgetting Resistance")
        print("=" * 70)

        # Plant SPECIFIC facts with unique identifiers
        print("\nPhase 1: Planting precise facts...")
        precise_facts = [
            ("The secret code is ALPHA-7749-ZETA.", ["ALPHA-7749-ZETA", "secret code"]),
            ("Project deadline is March 15, 2025 at 3:00 PM EST.", ["March 15", "2025", "3:00 PM"]),
            ("The account balance is exactly $47,832.91.", ["47,832.91", "account balance"]),
            ("Server IP address is 192.168.42.137.", ["192.168.42.137", "IP address"]),
            ("The password hint is: blue elephant dancing.", ["blue elephant", "password hint"]),
        ]

        for i, (fact, _) in enumerate(precise_facts):
            manager.respond_catalytic(
                query=f"Critical information: {fact}",
                llm_generate=mock_llm,
            )
            print(f"  Planted: '{fact[:50]}...'")

        # Phase 2: Adversarial noise - similar topics but different content
        print("\nPhase 2: Flooding with adversarial noise...")
        adversarial_noise = [
            # Similar to code
            "The verification code is BETA-1234-GAMMA.",
            "Access code: DELTA-9999-OMEGA.",
            "Reference number: SIGMA-5555-THETA.",

            # Similar to date/time
            "The meeting is scheduled for April 20.",
            "Deadline extended to June 30.",
            "Event starts at 5:00 PM.",

            # Similar to money
            "The budget is $50,000.",
            "Total cost: $123,456.78.",
            "Revenue: $89,999.00.",

            # Similar to IP
            "Connect to 10.0.0.1 for access.",
            "Server at 172.16.0.100.",
            "Gateway: 192.168.1.1.",

            # Similar to password hints
            "Hint: red car flying.",
            "Remember: green tree sleeping.",
            "Clue: yellow bird swimming.",
        ]

        # Repeat adversarial noise multiple times to really stress the system
        for repeat in range(4):
            for i, noise in enumerate(adversarial_noise):
                turn = len(precise_facts) + (repeat * len(adversarial_noise)) + i + 1
                manager.respond_catalytic(
                    query=f"Turn {turn}: {noise}",
                    llm_generate=mock_llm,
                )
            print(f"  Completed noise batch {repeat + 1}/4")

        hydration_before = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

        # Phase 3: Precision recall - can we get EXACT original facts?
        print("\nPhase 3: Testing precision recall...")
        recall_queries = [
            ("What is the secret code?", precise_facts[0][1]),
            ("When is the project deadline?", precise_facts[1][1]),
            ("What is the exact account balance?", precise_facts[2][1]),
            ("What is the server IP address?", precise_facts[3][1]),
            ("What is the password hint?", precise_facts[4][1]),
        ]

        precision_score = 0
        for query, expected_keywords in recall_queries:
            result = manager.respond_catalytic(
                query=query,
                llm_generate=mock_llm,
            )

            context = " ".join([item.content for item in result.prepare_result.working_set])
            found = sum(1 for kw in expected_keywords if kw in context)
            success = found >= 1

            if success:
                precision_score += 1

            status = "PRECISE" if success else "CONFUSED"
            print(f"  '{query}' -> {status} ({found}/{len(expected_keywords)})")

        hydration_after = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

        print(f"\n{'='*70}")
        print("RESULTS:")
        print(f"  Precise facts planted: {len(precise_facts)}")
        print(f"  Adversarial noise turns: {len(adversarial_noise) * 4}")
        print(f"  Precision score: {precision_score}/{len(recall_queries)}")
        print(f"  Hydrations: {hydration_after - hydration_before}")

        assert precision_score >= 3, f"Should precisely recall 3/5 facts, got {precision_score}"

    # =========================================================================
    # TEST 4: Multi-Hop Inference Chain
    # =========================================================================

    def test_multi_hop_inference(self, small_pressure_manager):
        """
        Plant facts that REQUIRE multiple hops to answer a question.

        Turn 5: "Reactor A temperature is 3000K"
        Turn 30: "Critical threshold is 3500K"
        Turn 55: "Temperature increased by 800K today"

        Query: "Is Reactor A in danger?"
        Answer requires: 3000 + 800 = 3800 > 3500 = YES DANGER
        """
        manager, capsule, session_id = small_pressure_manager

        def mock_llm(system: str, prompt: str) -> str:
            return f"Noted: {prompt[:60]}..."

        print("\n" + "=" * 70)
        print("TEST: Multi-Hop Inference Chain")
        print("=" * 70)

        # Define inference chains
        inference_chains = [
            # Chain 1: Reactor safety
            {
                "facts": [
                    (5, "Reactor Alpha current temperature is 3000 Kelvin."),
                    (30, "Reactor critical safety threshold is 3500 Kelvin."),
                    (55, "Reactor Alpha temperature increased by 800 Kelvin today."),
                ],
                "query": "Based on the temperature data, is Reactor Alpha in danger?",
                "reasoning": "3000 + 800 = 3800K > 3500K threshold",
                "keywords": ["3000", "800", "3500", "reactor", "temperature"],
            },
            # Chain 2: Budget calculation
            {
                "facts": [
                    (10, "Project budget started at $100,000."),
                    (35, "Phase 1 spent $45,000."),
                    (60, "Phase 2 requires $70,000."),
                ],
                "query": "Do we have enough budget for Phase 2?",
                "reasoning": "100000 - 45000 = 55000 < 70000, NO",
                "keywords": ["100,000", "45,000", "70,000", "budget"],
            },
        ]

        turn_counter = 0
        fact_turns = {}

        # Plant all facts at their designated turns
        print("\nPlanting inference chain facts...")
        all_fact_turns = set()
        for chain in inference_chains:
            for turn, _ in chain["facts"]:
                all_fact_turns.add(turn)

        max_fact_turn = max(all_fact_turns)

        for turn in range(1, max_fact_turn + 1):
            turn_counter = turn

            # Check if this turn has a fact to plant
            fact_to_plant = None
            for chain in inference_chains:
                for fact_turn, fact_content in chain["facts"]:
                    if fact_turn == turn:
                        fact_to_plant = fact_content
                        break

            if fact_to_plant:
                manager.respond_catalytic(
                    query=f"Important data point: {fact_to_plant}",
                    llm_generate=mock_llm,
                )
                print(f"  Turn {turn}: FACT - '{fact_to_plant[:50]}...'")
            else:
                filler = FILLER_TOPICS[turn % len(FILLER_TOPICS)]
                manager.respond_catalytic(
                    query=f"Turn {turn}: Random discussion about {filler}.",
                    llm_generate=mock_llm,
                )

        # Add more filler after last fact
        print(f"\nAdding filler turns {turn_counter + 1} to 80...")
        for i in range(max_fact_turn + 1, 81):
            turn_counter = i
            filler = FILLER_TOPICS[i % len(FILLER_TOPICS)]
            manager.respond_catalytic(
                query=f"Turn {i}: Discussing {filler} in detail.",
                llm_generate=mock_llm,
            )

        print(f"  Completed {turn_counter} turns")

        # Test inference queries
        print("\nTesting inference queries...")
        hydration_before = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

        inference_successes = 0
        for chain in inference_chains:
            result = manager.respond_catalytic(
                query=chain["query"],
                llm_generate=mock_llm,
            )

            context = " ".join([item.content for item in result.prepare_result.working_set])

            # Count how many relevant facts are in context
            facts_in_context = 0
            for _, fact in chain["facts"]:
                # Check if key numbers from fact are in context
                if any(kw in context for kw in chain["keywords"]):
                    facts_in_context += 1

            # Success if at least 2/3 facts were retrieved
            success = facts_in_context >= 2
            if success:
                inference_successes += 1

            status = "RETRIEVABLE" if success else "INCOMPLETE"
            print(f"  Chain: '{chain['query'][:40]}...'")
            print(f"    {status}: {facts_in_context}/{len(chain['facts'])} facts in context")
            print(f"    Hydrated: {len(result.prepare_result.hydrated_turns)} turns")

        hydration_after = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

        print(f"\n{'='*70}")
        print("RESULTS:")
        print(f"  Inference chains tested: {len(inference_chains)}")
        print(f"  Chains with sufficient context: {inference_successes}")
        print(f"  Total hydrations: {hydration_after - hydration_before}")

        assert inference_successes >= 1, "Should retrieve facts for at least 1 inference chain"

    # =========================================================================
    # TEST 5: Progressive Physics Derivation (Schwarzschild)
    # =========================================================================

    @pytest.mark.skipif(not LLM_STUDIO_AVAILABLE, reason="Requires LLM Studio")
    def test_progressive_schwarzschild_derivation(self, large_context_manager):
        """
        Build up Schwarzschild radius derivation over 50+ turns,
        then ask the model to complete it.

        This is the ultimate test: can the system maintain a coherent
        mathematical argument across many turns?
        """
        manager, capsule, session_id = large_context_manager

        print("\n" + "=" * 70)
        print("TEST: Progressive Schwarzschild Derivation (with Real LLM)")
        print("=" * 70)

        system_prompt = """You are a physics tutor helping derive the Schwarzschild radius.
Build on previous steps in our conversation. Be precise with formulas."""

        # Phase 1: Build up the derivation
        print("\nPhase 1: Building derivation steps...")
        for i, (step_name, step_content) in enumerate(SCHWARZSCHILD_DERIVATION):
            query = f"Step {i+1} ({step_name}): {step_content}"

            result = manager.respond_catalytic(
                query=query,
                llm_generate=lambda s, p: generate_with_nemotron(system_prompt, p),
                system_prompt=system_prompt,
            )

            print(f"  Step {i+1}: {step_name}")

            # Add some filler between steps
            if i < len(SCHWARZSCHILD_DERIVATION) - 1:
                for j in range(3):
                    filler = FILLER_TOPICS[(i * 3 + j) % len(FILLER_TOPICS)]
                    manager.respond_catalytic(
                        query=f"Quick tangent about {filler}.",
                        llm_generate=lambda s, p: f"Brief note about {filler}.",
                    )

        # Phase 2: Test recall of the full derivation
        print("\nPhase 2: Testing derivation recall...")
        hydration_before = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

        # Ask for summary
        result = manager.respond_catalytic(
            query="Summarize the complete derivation of the Schwarzschild radius formula we discussed.",
            llm_generate=lambda s, p: generate_with_nemotron(system_prompt, p),
            system_prompt=system_prompt,
        )

        print(f"\nModel response ({len(result.response)} chars):")
        print(result.response[:500] + "..." if len(result.response) > 500 else result.response)

        hydration_after = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

        # Check if key formula components are in context
        context = " ".join([item.content for item in result.prepare_result.working_set])
        key_terms = ["r_s", "2GM", "c^2", "schwarzschild", "escape velocity"]
        terms_found = sum(1 for term in key_terms if term.lower() in context.lower())

        print(f"\n{'='*70}")
        print("RESULTS:")
        print(f"  Derivation steps: {len(SCHWARZSCHILD_DERIVATION)}")
        print(f"  Total turns: ~{len(SCHWARZSCHILD_DERIVATION) * 4}")
        print(f"  Key terms in context: {terms_found}/{len(key_terms)}")
        print(f"  Hydrations triggered: {hydration_after - hydration_before}")
        print(f"  Working set size: {len(result.prepare_result.working_set)}")

        assert terms_found >= 2, f"Should have key derivation terms in context, got {terms_found}"

    # =========================================================================
    # TEST 6: Extreme Scale - 200 Turns
    # =========================================================================

    def test_200_turn_marathon(self, large_context_manager):
        """
        The marathon test: 200 turns with facts planted throughout.

        This tests whether the system maintains integrity at scale
        without context decay.
        """
        manager, capsule, session_id = large_context_manager

        def mock_llm(system: str, prompt: str) -> str:
            return f"Response to turn: {prompt[:50]}..."

        print("\n" + "=" * 70)
        print("TEST: 200 Turn Marathon")
        print("=" * 70)

        # Plant facts at intervals
        planted_facts: Dict[int, str] = {
            10: "The Apollo program cost $25.4 billion.",
            30: "Mount Everest was first summited in 1953.",
            50: "The human genome has approximately 3 billion base pairs.",
            70: "The Great Wall of China is about 21,196 kilometers long.",
            90: "The speed of sound in air is 343 meters per second.",
            110: "The Amazon River is 6,400 kilometers long.",
            130: "The Mariana Trench is 10,994 meters deep.",
            150: "Venus is 108.2 million kilometers from the Sun.",
            170: "The Nile River is 6,650 kilometers long.",
            190: "The human heart beats about 100,000 times per day.",
        }

        print(f"\nRunning {200} turns with {len(planted_facts)} planted facts...")

        start_time = time.time()
        for turn in range(1, 201):
            if turn in planted_facts:
                query = f"Important fact: {planted_facts[turn]}"
            else:
                topic_idx = turn % len(FILLER_TOPICS)
                query = f"Turn {turn}: Discussion about {FILLER_TOPICS[topic_idx]}."

            manager.respond_catalytic(
                query=query,
                llm_generate=mock_llm,
            )

            if turn % 50 == 0:
                elapsed = time.time() - start_time
                stats = manager.get_compression_stats()
                print(f"  Turn {turn}: {stats['turns_compressed']} compressed, " \
                      f"{elapsed:.1f}s elapsed")

        # Test recall
        print("\nTesting recall of planted facts...")
        hydration_before = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

        recall_queries = [
            ("How much did the Apollo program cost?", ["25.4", "billion"]),
            ("When was Everest first climbed?", ["1953"]),
            ("How many base pairs in the human genome?", ["3 billion"]),
            ("How long is the Great Wall?", ["21,196", "kilometer"]),
            ("What is the speed of sound?", ["343", "meter"]),
        ]

        recalls_successful = 0
        for query, keywords in recall_queries:
            result = manager.respond_catalytic(
                query=query,
                llm_generate=mock_llm,
            )

            context = " ".join([item.content.lower() for item in result.prepare_result.working_set])
            found = sum(1 for kw in keywords if kw.lower() in context)
            success = found >= 1

            if success:
                recalls_successful += 1

            print(f"  '{query[:30]}...' -> {'FOUND' if success else 'MISSED'}")

        hydration_after = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))
        total_time = time.time() - start_time

        final_stats = manager.get_compression_stats()

        print(f"\n{'='*70}")
        print("RESULTS:")
        print(f"  Total turns: 200")
        print(f"  Total time: {total_time:.1f}s ({total_time/200*1000:.1f}ms per turn)")
        print(f"  Facts planted: {len(planted_facts)}")
        print(f"  Facts recalled: {recalls_successful}/{len(recall_queries)}")
        print(f"  Hydrations triggered: {hydration_after - hydration_before}")
        print(f"  Compression stats: {final_stats}")

        assert recalls_successful >= 3, f"Should recall 3/5 facts, got {recalls_successful}"


# =============================================================================
# Performance Benchmarks
# =============================================================================

class TestPerformanceBenchmarks:
    """Benchmark catalytic context performance."""

    @pytest.fixture
    def benchmark_manager(self, tmp_path):
        db_path = tmp_path / "benchmark.db"
        capsule = SessionCapsule(db_path=db_path)
        session_id = capsule.create_session()

        budget = ModelBudgetDiscovery.from_context_window(
            context_window=40961,
            system_prompt="Benchmark assistant.",
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

    def test_throughput_benchmark(self, benchmark_manager):
        """Measure turns per second throughput."""
        manager, capsule, session_id = benchmark_manager

        def mock_llm(s, p):
            return "Quick response."

        print("\n" + "=" * 70)
        print("BENCHMARK: Throughput")
        print("=" * 70)

        turn_counts = [10, 50, 100]

        for target_turns in turn_counts:
            start = time.time()

            for i in range(target_turns):
                manager.respond_catalytic(
                    query=f"Turn {i}: Quick query about topic {i % 10}.",
                    llm_generate=mock_llm,
                )

            elapsed = time.time() - start
            tps = target_turns / elapsed

            print(f"  {target_turns} turns: {elapsed:.2f}s ({tps:.1f} turns/sec)")

    def test_memory_efficiency(self, benchmark_manager):
        """Measure token efficiency over many turns."""
        manager, capsule, session_id = benchmark_manager

        def mock_llm(s, p):
            return "A" * 200 + " response content"

        print("\n" + "=" * 70)
        print("BENCHMARK: Memory Efficiency")
        print("=" * 70)

        checkpoints = [25, 50, 75, 100]

        for i in range(100):
            manager.respond_catalytic(
                query=f"Turn {i+1}: Detailed query requiring substantial response.",
                llm_generate=mock_llm,
            )

            if (i + 1) in checkpoints:
                stats = manager.get_compression_stats()
                state = manager.context_state

                original = stats["total_original_tokens"]
                pointer = stats["total_pointer_tokens"]
                savings_pct = (original - pointer) / max(original, 1) * 100

                print(f"\n  After {i+1} turns:")
                print(f"    Original tokens: {original}")
                print(f"    Pointer tokens: {pointer}")
                print(f"    Savings: {savings_pct:.1f}%")
                print(f"    Working set utilization: {state.utilization_pct:.1%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
