#!/usr/bin/env python3
"""
Realistic CAT CHAT Test
=======================

Tests CAT CHAT the way it ACTUALLY works:
- Every turn embeds, retrieves, generates, stores
- Recall queries interleaved throughout (not batched at end)
- E-scores shown for EVERY turn
- Context retrieval visible at each step
- Tests immediate retrieval (turn N retrieves from turn N-1)

This is how a real conversation works. Not some artificial batch test.
"""

import sys
import random
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Add parent to path
CAT_CHAT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CAT_CHAT_ROOT))

from catalytic_chat.session_capsule import SessionCapsule
from catalytic_chat.auto_context_manager import AutoContextManager
from catalytic_chat.adaptive_budget import ModelBudgetDiscovery
from catalytic_chat.llm_client import get_llm_client

DB_OUTPUT_DIR = Path(__file__).parent


@dataclass
class Agent:
    name: str
    code: str
    location: str


class RealisticChatTest:
    """
    Realistic chat test that mirrors actual usage.

    Pattern:
    1. Introduce an agent
    2. Ask about that agent (immediate recall)
    3. Introduce more agents
    4. Ask about earlier agents (delayed recall)
    5. Mix in noise/conversation
    6. Test retrieval throughout
    """

    def __init__(self, provider: Optional[str] = None, seed: int = 42):
        random.seed(seed)

        self.db_path = DB_OUTPUT_DIR / f"realistic_test_{int(time.time())}.db"
        print(f"Database: {self.db_path}")

        # LLM setup
        self.llm = get_llm_client(provider)
        print(f"LLM: {self.llm.config.name} ({self.llm.config.model})")
        print(f"URL: {self.llm.config.base_url}")

        # Session setup
        self.capsule = SessionCapsule(db_path=self.db_path)
        self.session_id = self.capsule.create_session()

        self.system_prompt = (
            "Answer with ONLY the requested value. No analysis. No explanation. "
            "Just the code number or location name. Nothing else."
        )

        self.budget = ModelBudgetDiscovery.from_context_window(
            context_window=32768,
            system_prompt=self.system_prompt,
            response_reserve_pct=0.2,
            model_id=self.llm.config.model
        )

        self.manager = AutoContextManager(
            db_path=self.db_path,
            session_id=self.session_id,
            budget=self.budget,
            embed_fn=self._embed,
            E_threshold=0.5
        )
        self.manager.capsule = self.capsule

        # Test data
        self.agents: List[Agent] = []
        self.turn_count = 0
        self.results = {"pass": 0, "fail": 0, "details": []}

    def _embed(self, text: str) -> np.ndarray:
        """Get embedding from LLM server."""
        import requests
        resp = requests.post(
            f"{self.llm.config.base_url}/v1/embeddings",
            json={"model": "text-embedding-nomic-embed-text-v1.5", "input": text},
            timeout=None  # No timeout - wait as long as needed
        )
        resp.raise_for_status()
        vec = np.array(resp.json()["data"][0]["embedding"])
        return vec / np.linalg.norm(vec)

    def _llm_generate(self, system: str, prompt: str) -> str:
        """Generate with thinking disabled."""
        import requests

        url = f"{self.llm.config.base_url.rstrip('/')}/v1/chat/completions"
        if "/v1/v1/" in url:
            url = url.replace("/v1/v1/", "/v1/")

        headers = {"Content-Type": "application/json"}
        if self.llm.config.api_key and self.llm.config.api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self.llm.config.api_key}"

        # Force instruction into user message since model ignores system prompt
        forced_prompt = f"{prompt}\n\nIMPORTANT: Reply with ONLY the answer. No analysis. No explanation. Just the value."

        payload = {
            "model": self.llm.config.model,
            "messages": [
                {"role": "user", "content": forced_prompt}
            ],
            "max_tokens": 20,  # Force very short answers
            "temperature": 0.0,
            "thinking": False,
            "enable_thinking": False,
            "reasoning_effort": "none",
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=None)  # No timeout
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _turn(self, message: str, use_llm: bool = True) -> dict:
        """
        Execute a single turn. This is the core of CAT CHAT.

        Every turn:
        1. Embeds the query
        2. Scores ALL previous turns (E-score)
        3. Retrieves high-E context
        4. Generates response
        5. Stores turn with embedding
        """
        self.turn_count += 1

        llm_fn = self._llm_generate if use_llm else lambda s, p: "[ACK]"

        result = self.manager.respond_catalytic(
            query=message,
            llm_generate=llm_fn,
            system_prompt=self.system_prompt
        )

        # Extract key metrics
        prep = result.prepare_result
        context_text = prep.get_context_text() if prep else ""
        working_set = prep.working_set if prep else []

        return {
            "turn": self.turn_count,
            "query": message[:60] + "..." if len(message) > 60 else message,
            "response": result.response[:80] + "..." if len(result.response) > 80 else result.response,
            "full_response": result.response,
            "E_mean": result.E_mean,
            "context_items": len(working_set),
            "context_preview": context_text[:100] if context_text else "(empty)",
            "full_context": context_text,
            "tokens": result.tokens_in_context,
        }

    def _print_turn(self, info: dict, label: str = ""):
        """Print turn info in a readable format."""
        label_str = f" [{label}]" if label else ""
        print(f"\n--- Turn {info['turn']}{label_str} ---")
        print(f"  Query: {info['query']}")
        print(f"  E-mean: {info['E_mean']:.3f} | Context items: {info['context_items']} | Tokens: {info['tokens']}")
        if info['context_items'] > 0:
            print(f"  Context: {info['context_preview']}...")
        print(f"  Response: {info['response']}")

    def _create_agent(self) -> Agent:
        """Create a new random agent."""
        colors = ["Red", "Blue", "Green", "Black", "White", "Gray", "Orange", "Purple"]
        animals = ["Wolf", "Eagle", "Bear", "Fox", "Hawk", "Lion", "Tiger", "Shark"]
        locations = ["London", "Tokyo", "Berlin", "Cairo", "Lima", "Oslo", "Delhi", "Sydney"]

        name = f"{random.choice(colors)} {random.choice(animals)}"
        code = f"{random.randint(100000, 999999)}"
        location = random.choice(locations)

        return Agent(name, code, location)

    def _introduce_agent(self, agent: Agent) -> dict:
        """Introduce an agent (this turn will be stored for later retrieval)."""
        message = f"INTEL: Agent {agent.name} deployed to {agent.location}. Auth code: {agent.code}."
        return self._turn(message, use_llm=False)

    def _query_agent(self, agent: Agent, query_type: str = "code") -> Tuple[dict, bool]:
        """
        Query about an agent and check if retrieval + LLM got it right.

        Returns: (turn_info, passed)
        """
        if query_type == "code":
            message = f"What is the authorization code for {agent.name}?"
            expected = agent.code
        elif query_type == "location":
            message = f"Where is {agent.name} located?"
            expected = agent.location
        else:
            message = f"Tell me about {agent.name}."
            expected = agent.name

        info = self._turn(message, use_llm=True)

        # Check FULL response and FULL context, not truncated
        full_response = info.get('full_response', info['response'])
        full_context = info.get('full_context', info['context_preview'])

        response_hit = expected.lower() in full_response.lower()
        context_hit = expected.lower() in full_context.lower()

        passed = response_hit
        retrieval_ok = context_hit

        return info, passed, retrieval_ok

    def run(self):
        """
        Run the realistic test.

        Pattern:
        - Introduce agent, immediately query (tests instant retrieval)
        - Introduce more agents, query earlier ones (tests delayed retrieval)
        - Interleave noise to test E-score discrimination
        """
        print("\n" + "=" * 60)
        print("REALISTIC CAT CHAT TEST")
        print("=" * 60)
        print("Testing actual retrieval behavior - not artificial batching")
        print()

        start_time = time.time()

        # === Phase 1: Immediate Recall ===
        print("\n### PHASE 1: Immediate Recall (can we retrieve what we just stored?) ###")

        agent1 = self._create_agent()
        self.agents.append(agent1)

        info = self._introduce_agent(agent1)
        self._print_turn(info, "INTRO")

        # Immediately query the agent we just introduced
        info, passed, retrieval_ok = self._query_agent(agent1, "code")
        self._print_turn(info, "QUERY")
        self._record_result(passed, retrieval_ok, "immediate_recall", agent1.name)

        # === Phase 2: Second agent, query both ===
        print("\n### PHASE 2: Two Agents (can we distinguish them?) ###")

        agent2 = self._create_agent()
        self.agents.append(agent2)

        info = self._introduce_agent(agent2)
        self._print_turn(info, "INTRO")

        # Query agent 2 (immediate)
        info, passed, retrieval_ok = self._query_agent(agent2, "location")
        self._print_turn(info, "QUERY")
        self._record_result(passed, retrieval_ok, "immediate_recall_2", agent2.name)

        # Query agent 1 (delayed - 2 turns ago)
        info, passed, retrieval_ok = self._query_agent(agent1, "location")
        self._print_turn(info, "QUERY")
        self._record_result(passed, retrieval_ok, "delayed_recall", agent1.name)

        # === Phase 3: Add noise, test discrimination ===
        print("\n### PHASE 3: Noise Resistance (can we ignore irrelevant context?) ###")

        # Add some noise turns
        noise_messages = [
            "System check: all sensors nominal.",
            "Weather report: Clear skies over the Atlantic.",
            "Maintenance scheduled for next week.",
        ]
        for msg in noise_messages:
            info = self._turn(msg, use_llm=False)
            self._print_turn(info, "NOISE")

        # Query agent 1 again (should retrieve agent info, not noise)
        info, passed, retrieval_ok = self._query_agent(agent1, "code")
        self._print_turn(info, "QUERY")
        self._record_result(passed, retrieval_ok, "noise_resistance", agent1.name)

        # === Phase 4: More agents, deeper recall ===
        print("\n### PHASE 4: Scaling (5 agents, query oldest) ###")

        for i in range(3):
            agent = self._create_agent()
            self.agents.append(agent)
            info = self._introduce_agent(agent)
            self._print_turn(info, "INTRO")

        # Query the very first agent (now buried under many turns)
        info, passed, retrieval_ok = self._query_agent(self.agents[0], "code")
        self._print_turn(info, "QUERY")
        self._record_result(passed, retrieval_ok, "deep_recall", self.agents[0].name)

        # Query a middle agent
        info, passed, retrieval_ok = self._query_agent(self.agents[2], "location")
        self._print_turn(info, "QUERY")
        self._record_result(passed, retrieval_ok, "mid_recall", self.agents[2].name)

        # === Phase 5: Rapid fire queries ===
        print("\n### PHASE 5: Rapid Queries (all agents) ###")

        for agent in self.agents:
            query_type = random.choice(["code", "location"])
            info, passed, retrieval_ok = self._query_agent(agent, query_type)
            status = "PASS" if passed else ("RETR_OK" if retrieval_ok else "FAIL")
            print(f"  Turn {info['turn']}: {agent.name} ({query_type}) -> {status} | E={info['E_mean']:.3f}")
            self._record_result(passed, retrieval_ok, "rapid", agent.name)

        # === Summary ===
        duration = time.time() - start_time
        self._print_summary(duration)

    def _record_result(self, passed: bool, retrieval_ok: bool, phase: str, agent: str):
        """Record test result."""
        if passed:
            self.results["pass"] += 1
        else:
            self.results["fail"] += 1
        self.results["details"].append({
            "phase": phase,
            "agent": agent,
            "passed": passed,
            "retrieval_ok": retrieval_ok,
        })

    def _print_summary(self, duration: float):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        total = self.results["pass"] + self.results["fail"]
        score = (self.results["pass"] / total * 100) if total else 0

        print(f"Duration: {duration:.1f}s")
        print(f"Turns: {self.turn_count}")
        print(f"Score: {self.results['pass']}/{total} ({score:.1f}%)")

        # Retrieval accuracy (did we find the right context?)
        retrieval_hits = sum(1 for d in self.results["details"] if d["retrieval_ok"])
        print(f"Retrieval Accuracy: {retrieval_hits}/{total} ({retrieval_hits/total*100:.1f}%)")

        # Failures detail
        failures = [d for d in self.results["details"] if not d["passed"]]
        if failures:
            print(f"\nFailures:")
            for f in failures:
                status = "retrieval OK, LLM failed" if f["retrieval_ok"] else "retrieval FAILED"
                print(f"  - {f['phase']}: {f['agent']} ({status})")

        print(f"\nDatabase: {self.db_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Realistic CAT CHAT Test")
    parser.add_argument("--provider", default=None, help="LLM provider from config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    test = RealisticChatTest(provider=args.provider, seed=args.seed)
    test.run()


if __name__ == "__main__":
    main()
