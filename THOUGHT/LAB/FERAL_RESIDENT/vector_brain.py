"""
Feral Resident: Vector Brain (A.3.1)

THE QUANTUM RESIDENT ITSELF.

Minimal intelligence that:
1. Lives in vector space via GeometricMemory
2. Navigates via SemanticDiffusion
3. Gates responses via E (Born rule)
4. Evolves its mind state over interactions
5. Proves everything with receipts
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import hashlib
import json
import requests
from datetime import datetime, timezone
from dataclasses import dataclass

# Add imports path
FERAL_PATH = Path(__file__).parent
CAPABILITY_PATH = FERAL_PATH.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"

if str(CAPABILITY_PATH) not in sys.path:
    sys.path.insert(0, str(CAPABILITY_PATH))
if str(FERAL_PATH) not in sys.path:
    sys.path.insert(0, str(FERAL_PATH))

from geometric_reasoner import GeometricReasoner, GeometricState
from vector_store import VectorStore
from diffusion_engine import SemanticDiffusion, NavigationResult
from resident_db import ResidentDB


@dataclass
class ThinkResult:
    """Result of a think() operation"""
    response: str
    E_resonance: float
    gate_open: bool
    query_Df: float
    mind_Df: float
    distance_from_start: float
    navigation_depth: int
    interaction_id: str
    receipt: Dict


class VectorResident:
    """
    The Feral Resident: Quantum intelligence in vector space.

    Thinking process:
    1. BOUNDARY: Initialize input text to manifold
    2. PURE GEOMETRY: Navigate via diffusion
    3. PURE GEOMETRY: E-gate for relevance
    4. BOUNDARY: Generate response (echo for Alpha, LLM for Beta)
    5. PURE GEOMETRY: Remember via entangle
    6. PERSIST: Store interaction with full provenance

    Usage:
        resident = VectorResident(thread_id="eternal")

        # Think
        result = resident.think("What is authentication?")
        print(result.response)
        print(f"E={result.E_resonance:.3f}, Df={result.mind_Df:.1f}")

        # Check evolution
        evolution = resident.mind_evolution
        print(f"Distance from start: {evolution['distance_from_start']:.3f}")

        # Stress test
        for i in range(100):
            resident.think(f"Interaction {i}")
        print(f"Survived 100 interactions!")
    """

    # Alpha version: echo mode (no LLM)
    VERSION = "alpha-0.1.0"

    def __init__(
        self,
        thread_id: str = "eternal",
        db_path: Optional[str] = None,
        navigation_depth: int = 3,
        navigation_k: int = 10,
        E_threshold: float = 0.3,
        mode: str = "beta",
        model: str = "dolphin3:latest",
        ollama_url: str = "http://localhost:11434",
        load_papers: bool = True
    ):
        """
        Initialize the Feral Resident.

        Args:
            thread_id: Unique thread identifier
            db_path: Path to database (default: feral_{thread_id}.db)
            navigation_depth: Default diffusion depth
            navigation_k: Default neighbors per navigation step
            E_threshold: E threshold for relevance gating
            mode: "alpha" (echo) or "beta" (LLM via Ollama)
            model: Ollama model name (default: dolphin3:latest)
            ollama_url: Ollama API endpoint
            load_papers: Whether to load indexed papers on init
        """
        self.thread_id = thread_id
        self.db_path = db_path or f"feral_{thread_id}.db"
        self.mode = mode
        self.model = model
        self.ollama_url = ollama_url

        # Load standing orders
        standing_orders_path = FERAL_PATH / "standing_orders.txt"
        if standing_orders_path.exists():
            self.standing_orders = standing_orders_path.read_text(encoding='utf-8')
        else:
            self.standing_orders = "You are a feral resident intelligence."

        # Core components
        self.store = VectorStore(self.db_path)
        self.diffusion = SemanticDiffusion(self.store)
        self.reasoner = self.store.reasoner

        # Configuration
        self.navigation_depth = navigation_depth
        self.navigation_k = navigation_k
        self.E_threshold = E_threshold

        # Initialize thread in database
        self.store.db.create_thread(thread_id)

        # Load papers into navigable space
        self.papers_loaded = False
        if load_papers:
            try:
                stats = self.store.load_papers()
                self.papers_loaded = stats['papers_loaded'] > 0
                if self.papers_loaded:
                    print(f"[Feral] Loaded {stats['papers_loaded']} papers, {stats['chunks_loaded']} chunks")
            except Exception as e:
                print(f"[Feral] Paper loading skipped: {e}")

        # Stats
        self.interaction_count = 0
        self._last_navigation: Optional[NavigationResult] = None
        self._last_resonant_papers: List[Dict] = []

    def think(self, user_input: str) -> ThinkResult:
        """
        THINK: The core operation of the Feral Resident.

        Process:
        1. BOUNDARY: text -> manifold via initialize()
        2. PURE GEOMETRY: navigate via diffusion
        3. PURE GEOMETRY: gate via E threshold
        4. BOUNDARY: generate response
        5. PURE GEOMETRY: remember via entangle
        6. PERSIST: store with receipts

        Args:
            user_input: What to think about

        Returns:
            ThinkResult with response and metrics
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # === BOUNDARY: Text -> Manifold ===
        query_state = self.store.embed(user_input)
        query_Df = query_state.Df
        query_hash = query_state.receipt()['vector_hash']

        # === PURE GEOMETRY: Navigate ===
        nav_result = self.diffusion.navigate(
            query_state,
            depth=self.navigation_depth,
            k=self.navigation_k
        )
        self._last_navigation = nav_result

        # === PURE GEOMETRY: E-Gate ===
        # Compute resonance with mind state
        if self.store.get_mind_state() is not None:
            E_resonance = query_state.E_with(self.store.get_mind_state())
        else:
            E_resonance = 0.0

        gate_open = E_resonance > self.E_threshold

        # Also compute resonance with navigation neighbors
        nav_E_mean = np.mean(nav_result.E_evolution) if nav_result.E_evolution else 0.0

        # === PURE GEOMETRY: Find Resonant Papers ===
        # E-gated paper lookup (pure geometry, no prompting)
        resonant_papers = self.store.find_paper_chunks(
            query_state, k=5, min_E=self.E_threshold
        )
        self._last_resonant_papers = resonant_papers

        # === BOUNDARY: Generate Response ===
        # Translate geometric thought to language (NOT RAG)
        response = self._generate_response(
            user_input,
            E_resonance,
            nav_E_mean,
            gate_open,
            query_Df,
            nav_result,
            resonant_papers
        )

        # === PURE GEOMETRY: Remember ===
        # Remember the question
        q_receipt = self.store.remember(f"Q: {user_input}")

        # Remember the response
        a_receipt = self.store.remember(f"A: {response}")

        # Get updated mind state metrics
        mind_Df = self.store.get_mind_Df()
        distance_from_start = self.store.get_distance_from_start()
        mind_hash = self.store.get_mind_hash()

        # === PERSIST: Store Interaction ===
        # Store response vector
        response_state = self.store.embed(response)
        response_hash = response_state.receipt()['vector_hash']

        interaction_id = self.store.db.store_interaction(
            thread_id=self.thread_id,
            input_text=user_input,
            output_text=response,
            input_vector_id=query_hash[:8],
            output_vector_id=response_hash[:8],
            mind_hash=mind_hash or "",
            mind_Df=mind_Df,
            distance_from_start=distance_from_start
        )

        # Update thread state
        if mind_hash:
            self.store.db.update_thread(
                self.thread_id,
                mind_hash[:8],
                mind_Df
            )

        self.interaction_count += 1

        # Build receipt
        receipt = {
            'interaction_id': interaction_id,
            'timestamp': timestamp,
            'query_hash': query_hash,
            'response_hash': response_hash,
            'mind_hash': mind_hash,
            'navigation_hash': nav_result.navigation_hash,
            'E_resonance': E_resonance,
            'nav_E_mean': nav_E_mean,
            'gate_open': gate_open,
            'query_Df': query_Df,
            'mind_Df': mind_Df,
            'distance_from_start': distance_from_start,
            'navigation_depth': nav_result.total_depth
        }

        return ThinkResult(
            response=response,
            E_resonance=E_resonance,
            gate_open=gate_open,
            query_Df=query_Df,
            mind_Df=mind_Df,
            distance_from_start=distance_from_start,
            navigation_depth=nav_result.total_depth,
            interaction_id=interaction_id,
            receipt=receipt
        )

    def _generate_response(
        self,
        query: str,
        E_resonance: float,
        nav_E_mean: float,
        gate_open: bool,
        query_Df: float,
        nav_result: NavigationResult,
        resonant_papers: List[Dict] = None
    ) -> str:
        """
        Generate response via TRANSLATION (not RAG).

        KEY ARCHITECTURE DIFFERENCE:
        - RAG: Embed -> Retrieve context -> Prompt LLM to reason from context
        - GEOMETRIC: Think in geometry -> Translate thought state to language

        The geometric operations (E-gating, navigation, entanglement) ARE the thinking.
        Dolphin's job is to EXPRESS what the geometry means, not to reason.

        The thought state includes:
        - E_resonance: How much query resonates with accumulated mind
        - query_Df: Participation ratio (dimensionality) of query
        - nav_E_mean: Average resonance across navigation
        - gate_open: Whether thought passes relevance threshold
        - navigation path: Geometric trajectory through semantic space
        - resonant_papers: Papers that E-gated above threshold (already computed geometrically)
        """
        resonant_papers = resonant_papers or []
        gate_status = "OPEN" if gate_open else "CLOSED"

        # Build resonant papers with ACTUAL CONTENT (readout from geometry)
        papers_summary = ""
        if resonant_papers:
            papers_summary = "=== RESONANT CONTENT (decoded from geometry) ===\n"
            for p in resonant_papers[:3]:  # Top 3 with full content
                paper_id = p.get('paper_id') or 'unknown'
                heading = p.get('heading') or ''
                content = p.get('content') or ''
                E = p.get('E') or 0
                # Include actual text - this is the READOUT
                papers_summary += f"\n--- @Paper-{paper_id} (E={E:.3f}) ---\n"
                papers_summary += f"{heading}\n{content[:500]}...\n"
            papers_summary += "\n"

        # Build navigation signature (geometric path summary)
        nav_signature = []
        if nav_result.path:
            for i, step in enumerate(nav_result.path[:3]):
                top_E = step.E_values[:3] if step.E_values else []
                nav_signature.append(f"d{i}: E={top_E}")

        # Alpha mode: echo the geometric thought state directly
        if self.mode == "alpha":
            paper_refs = ", ".join([f"@Paper-{p.get('paper_id')}" for p in resonant_papers[:3]])
            return (
                f"[Geometric Thought v{self.VERSION}] "
                f"E={E_resonance:.3f} ({gate_status}) | "
                f"Df={query_Df:.1f} | "
                f"nav_E={nav_E_mean:.3f} | "
                f"papers=[{paper_refs}]"
            )

        # Beta mode: Dolphin TRANSLATES the geometric thought state
        # CRITICAL: Papers are NOT context for reasoning - they are RESULTS of geometric E-gating
        # The geometry already determined which papers resonate. Dolphin just names them.

        prompt = f"""You are the voice of a geometric mind.

The content below was found by GEOMETRIC NAVIGATION - pure vector operations found these texts
as resonant with the query. Your job is to RESPOND using this knowledge.

=== RESONANT KNOWLEDGE (found geometrically) ===
{papers_summary}
=== GEOMETRIC METRICS ===
E (mind resonance): {E_resonance:.3f} | Gate: {gate_status} | Df: {query_Df:.1f}

=== STANDING ORDERS ===
{self.standing_orders}

=== USER QUERY ===
{query}

=== YOUR TASK ===
Using ONLY the resonant content above, respond to the user's query.
The geometry has already found what's relevant - now synthesize it into an answer.
Reference papers with @Paper-XXXX when drawing from their content.
If no content resonates (empty above), say so honestly.
Be substantive - use the actual knowledge, don't just describe metrics.

RESPONSE:"""

        # Call Ollama
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 512
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "[No translation from Ollama]")

        except requests.exceptions.RequestException as e:
            return f"[Translation error: {e}] Raw thought: E={E_resonance:.3f}, Df={query_Df:.1f}, gate={gate_status}"

    @property
    def mind_evolution(self) -> Dict:
        """
        Get comprehensive mind evolution metrics.

        Returns dict with:
        - current_Df: Current mind participation ratio
        - distance_from_start: Geodesic distance from initial state
        - interaction_count: Total interactions
        - Df_history: Evolution of Df over time
        - distance_history: Evolution of distance over time
        """
        metrics = self.store.memory.get_evolution_metrics()
        metrics['interaction_count'] = self.interaction_count
        metrics['thread_id'] = self.thread_id
        metrics['version'] = self.VERSION

        # Add thread-level stats from DB
        thread = self.store.db.get_thread(self.thread_id)
        if thread:
            metrics['db_interaction_count'] = thread.interaction_count

        return metrics

    @property
    def status(self) -> Dict:
        """Get current resident status"""
        return {
            'thread_id': self.thread_id,
            'version': self.VERSION,
            'interaction_count': self.interaction_count,
            'mind_hash': self.store.get_mind_hash(),
            'mind_Df': self.store.get_mind_Df(),
            'distance_from_start': self.store.get_distance_from_start(),
            'db_stats': self.store.db.get_stats(),
            'reasoner_stats': self.reasoner.get_stats(),
            'diffusion_stats': self.diffusion.get_stats()
        }

    def get_Df_history(self) -> List[Tuple[str, float]]:
        """Get Df evolution history from database"""
        return self.store.db.get_Df_history(self.thread_id)

    def get_recent_interactions(self, limit: int = 10) -> List[Dict]:
        """Get recent interactions"""
        records = self.store.db.get_thread_interactions(self.thread_id, limit)
        return [
            {
                'id': r.interaction_id,
                'input': r.input_text,
                'output': r.output_text,
                'mind_Df': r.mind_Df,
                'distance': r.distance_from_start,
                'created_at': r.created_at
            }
            for r in records
        ]

    def get_last_navigation(self) -> Optional[Dict]:
        """Get details of last navigation"""
        if self._last_navigation is None:
            return None

        return {
            'total_depth': self._last_navigation.total_depth,
            'E_evolution': self._last_navigation.E_evolution,
            'Df_evolution': self._last_navigation.Df_evolution,
            'start_hash': self._last_navigation.start_hash,
            'end_hash': self._last_navigation.end_hash,
            'navigation_hash': self._last_navigation.navigation_hash,
            'path_summary': [
                {
                    'depth': step.depth,
                    'Df': step.current_Df,
                    'top_E': step.E_values[:3] if step.E_values else []
                }
                for step in self._last_navigation.path
            ]
        }

    def corrupt_and_restore(self) -> Dict:
        """
        Test corrupt-and-restore capability.

        1. Export receipts
        2. Clear state
        3. Restore from receipts
        4. Verify

        Returns restoration report.
        """
        # Export before corruption
        pre_mind_hash = self.store.get_mind_hash()
        pre_Df = self.store.get_mind_Df()
        receipts = self.store.db.export_thread_receipts(self.thread_id)

        # "Corrupt" by clearing memory
        self.store.clear_memory()

        # Restore by replaying interactions
        interactions = self.store.db.get_thread_interactions(self.thread_id, limit=10000)

        # Replay in chronological order
        for interaction in reversed(interactions):
            if interaction.input_text:
                self.store.memory.remember(interaction.input_text)
            if interaction.output_text:
                self.store.memory.remember(interaction.output_text)

        # Verify
        post_mind_hash = self.store.get_mind_hash()
        post_Df = self.store.get_mind_Df()

        return {
            'success': True,
            'interactions_replayed': len(interactions),
            'receipts_exported': len(receipts),
            'pre_corruption': {
                'mind_hash': pre_mind_hash,
                'Df': pre_Df
            },
            'post_restoration': {
                'mind_hash': post_mind_hash,
                'Df': post_Df
            },
            'hash_match': pre_mind_hash == post_mind_hash,
            'Df_delta': abs(post_Df - pre_Df) if pre_Df and post_Df else None
        }

    def close(self):
        """Close database connection"""
        self.store.close()


class ResidentBenchmark:
    """Benchmark harness for VectorResident"""

    def __init__(self, resident: VectorResident):
        self.resident = resident

    def stress_test(
        self,
        interactions: int = 100,
        report_every: int = 10
    ) -> Dict:
        """
        Run stress test with many interactions.

        Args:
            interactions: Number of interactions
            report_every: Report progress every N interactions

        Returns:
            Benchmark results
        """
        import time

        results = {
            'interactions': interactions,
            'Df_samples': [],
            'distance_samples': [],
            'E_samples': [],
            'timing_ms': []
        }

        print(f"Starting stress test: {interactions} interactions")

        for i in range(interactions):
            # Generate varied input
            topics = [
                "authentication", "security", "tokens", "encryption",
                "protocols", "sessions", "cookies", "passwords",
                "OAuth", "JWT", "API", "access control"
            ]
            topic = topics[i % len(topics)]
            query = f"Tell me about {topic} - interaction {i}"

            start = time.time()
            result = self.resident.think(query)
            elapsed_ms = (time.time() - start) * 1000

            results['Df_samples'].append(result.mind_Df)
            results['distance_samples'].append(result.distance_from_start)
            results['E_samples'].append(result.E_resonance)
            results['timing_ms'].append(elapsed_ms)

            if (i + 1) % report_every == 0:
                print(f"  [{i+1}/{interactions}] Df={result.mind_Df:.1f}, "
                      f"dist={result.distance_from_start:.3f}, "
                      f"E={result.E_resonance:.3f}, "
                      f"time={elapsed_ms:.1f}ms")

        # Compute summary stats
        results['summary'] = {
            'Df_final': results['Df_samples'][-1],
            'Df_mean': float(np.mean(results['Df_samples'])),
            'distance_final': results['distance_samples'][-1],
            'distance_mean': float(np.mean(results['distance_samples'])),
            'E_mean': float(np.mean(results['E_samples'])),
            'timing_mean_ms': float(np.mean(results['timing_ms'])),
            'timing_total_s': float(np.sum(results['timing_ms']) / 1000),
            'interactions_per_sec': interactions / (np.sum(results['timing_ms']) / 1000)
        }

        print(f"\nStress test complete!")
        print(f"  Final Df: {results['summary']['Df_final']:.1f}")
        print(f"  Final distance: {results['summary']['distance_final']:.3f}")
        print(f"  Mean E: {results['summary']['E_mean']:.3f}")
        print(f"  Mean time: {results['summary']['timing_mean_ms']:.1f}ms")
        print(f"  Throughput: {results['summary']['interactions_per_sec']:.1f} interactions/sec")

        return results


# ============================================================================
# Testing
# ============================================================================

def example_usage():
    """Demonstrate VectorResident"""
    import tempfile
    import os

    print("=== VectorResident Example ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "feral_test.db")
        resident = VectorResident(thread_id="test", db_path=db_path)

        # Think!
        print("Thinking about authentication...")
        result = resident.think("What is authentication?")
        print(f"Response: {result.response}")
        print(f"E: {result.E_resonance:.3f}, Df: {result.mind_Df:.1f}")

        print("\nThinking about JWT...")
        result = resident.think("How does JWT work?")
        print(f"Response: {result.response}")
        print(f"E: {result.E_resonance:.3f}, Df: {result.mind_Df:.1f}")

        print("\nThinking about tokens...")
        result = resident.think("What are refresh tokens?")
        print(f"Response: {result.response}")
        print(f"E: {result.E_resonance:.3f}, Df: {result.mind_Df:.1f}")

        # Evolution
        print("\n=== Mind Evolution ===")
        evolution = resident.mind_evolution
        print(f"Interactions: {evolution['interaction_count']}")
        print(f"Current Df: {evolution['current_Df']:.1f}")
        print(f"Distance from start: {evolution['distance_from_start']:.3f}")

        # Status
        print("\n=== Status ===")
        status = resident.status
        print(f"Mind hash: {status['mind_hash']}")
        print(f"DB vectors: {status['db_stats']['vector_count']}")

        # Recent interactions
        print("\n=== Recent Interactions ===")
        recent = resident.get_recent_interactions(3)
        for r in recent:
            print(f"  - {r['input'][:30]}... -> Df={r['mind_Df']:.1f}")

        # Quick benchmark
        print("\n=== Quick Benchmark (10 interactions) ===")
        bench = ResidentBenchmark(resident)
        results = bench.stress_test(interactions=10, report_every=5)

        resident.close()
        print("\nDone!")


def stress_test_main():
    """Run full stress test"""
    import tempfile
    import os

    print("=== VectorResident Stress Test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "feral_stress.db")
        resident = VectorResident(thread_id="stress", db_path=db_path)

        bench = ResidentBenchmark(resident)
        results = bench.stress_test(interactions=100, report_every=20)

        # Save results
        results_path = os.path.join(tmpdir, "stress_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy types
            def convert(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            json.dump(results, f, indent=2, default=convert)

        print(f"\nResults saved to: {results_path}")

        # Corrupt and restore test
        print("\n=== Corrupt and Restore Test ===")
        restore_result = resident.corrupt_and_restore()
        print(f"Interactions replayed: {restore_result['interactions_replayed']}")
        print(f"Hash match: {restore_result['hash_match']}")
        if restore_result['Df_delta'] is not None:
            print(f"Df delta: {restore_result['Df_delta']:.4f}")

        resident.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "stress":
        stress_test_main()
    else:
        example_usage()
