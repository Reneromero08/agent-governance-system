#!/usr/bin/env python3
"""
Catalytic Context "SUPER HARD" Test - THE CIPHER MARATHON
Scenario: Operation Haystack

Target: liquid/lfm2.5-1.2b
Objective: Break the model's native recall with high-entropy, high-interference data.
Hypothesis: 1.2B models cannot natively distinguish 50+ similar entities without hallucination, 
but catalytic retrieval will make it trivial by isolating the exact fact.

Parameters:
- 200 Turns
- 50 Unique Agents (High Interference Names)
- Arbitrary 6-digit Codes (High Entropy)
- Multi-attribute binding (Name <-> Code <-> Location)
"""

import sys
import argparse
import requests
import json
import time
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

# Add parent to path for imports
CAT_CHAT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CAT_CHAT_ROOT))

# Import Catalytic Chat components
try:
    from catalytic_chat.session_capsule import SessionCapsule
    from catalytic_chat.auto_context_manager import AutoContextManager
    from catalytic_chat.adaptive_budget import ModelBudgetDiscovery
except ImportError:
    print("Error: failed to import catalytic_chat modules.")
    sys.exit(1)

# Configuration
DEFAULT_LLM_BASE = "http://10.5.0.2:1234"
DEFAULT_MODEL = "liquid/lfm2.5-1.2b"
# Using a slightly higher threshold to ensure we only get the EXACT agent match
DEFAULT_E_THRESHOLD = 0.45 

@dataclass
class AgentProfile:
    id: int
    name: str
    code: str
    location: str

class CipherMarathonScenario:
    def __init__(self, num_agents: int = 50, total_turns: int = 200):
        self.num_agents = num_agents
        self.total_turns = total_turns
        self.agents: List[AgentProfile] = []
        self.turns: Dict[int, str] = {}
        self.recall_tasks: Dict[int, Tuple[str, List[str]]] = {}
        
        self._generate_data()
        self._build_script()

    def _generate_data(self):
        # Generate confusingly similar names
        colors = ["Crimson", "Azure", "Emerald", "Golden", "Shadow", "Silver", "Iron", "Obsidian", "Violet", "Scarlet"]
        animals = ["Fox", "Eagle", "Wolf", "Bear", "Viper", "Hawk", "Shark", "Lion", "Tiger", "Raven", "Cobra", "Falcon"]
        
        used_names = set()
        
        for i in range(1, self.num_agents + 1):
            # Enforce uniqueness
            while True:
                name = f"{random.choice(colors)} {random.choice(animals)}"
                if name not in used_names:
                    used_names.add(name)
                    break
            
            # Generate random 6-digit code (pure entropy)
            code = f"{random.randint(100000, 999999)}"
            
            # Locations
            locs = ["London", "Kyoto", "Berlin", "Cairo", "Lima", "Oslo", "Delhi", "Perth", "Nome", "Suva"]
            loc = random.choice(locs)
            
            self.agents.append(AgentProfile(i, name, code, loc))

    def _build_script(self):
        # Distribute agent introductions across the first 80% of turns
        turns_available = list(range(1, int(self.total_turns * 0.8)))
        random.shuffle(turns_available)
        
        # Plant facts
        for agent in self.agents:
            if not turns_available:
                break
            turn = turns_available.pop()
            # "High Interference" phrasing - all sentences look structurally identical
            self.turns[turn] = f"REGISTRY UPDATE: Asset {agent.name} (ID #{agent.id}) is active in {agent.location}. Secure Code: [{agent.code}]."

        # Fill gaps with high-entropy noise to flush context
        for i in range(1, self.total_turns + 1):
            if i not in self.turns:
                noise_val = random.randint(1000, 9999)
                self.turns[i] = f"SYSTEM NOISE: Background signal interference detected on channel {noise_val}. Re-calibrating entropy pools."

        # Generate Recall Tasks for the final 20% (turns 180+)
        # We will ask super specific questions that require exact binding
        recall_turns = range(int(self.total_turns * 0.85), self.total_turns + 1)
        
        # Select random agents to test
        test_agents = random.sample(self.agents, min(len(recall_turns), 10))
        
        for turn, agent in zip(recall_turns, test_agents):
            # 50/50 mix of Code recall and Location recall
            if random.random() > 0.5:
                query = f"URGENT: Requires authorization code for {agent.name}. Respond with code only."
                expected = [agent.code]
            else:
                query = f"TRACKING: Where represents the current operational zone of ID #{agent.id}?"
                expected = [agent.location, agent.name] # Expect location, maybe name helps verify
            
            self.recall_tasks[turn] = (query, expected)
            # Remove from script flow if it collided (unlikely with logic above but safe to ensure)
            if turn in self.turns:
                del self.turns[turn]

    def get_turn_input(self, turn_num: int) -> str:
        if turn_num in self.recall_tasks:
            return self.recall_tasks[turn_num][0]
        return self.turns.get(turn_num, "SYSTEM: Standby.")

class CipherStressRunner:
    def __init__(self, args):
        self.args = args
        self.scenario = CipherMarathonScenario(num_agents=50, total_turns=200)
        
        import tempfile
        self.tmpdir = Path(tempfile.gettempdir()) / f"cat_cipher_{int(time.time())}"
        self.tmpdir.mkdir(exist_ok=True)
        self.db_path = self.tmpdir / "cipher.db"
        
        self.capsule = SessionCapsule(db_path=self.db_path)
        self.session_id = self.capsule.create_session()
        
        self.system_prompt = "You are a secure intelligence mainframe. You handle sensitive asset data. You perform exact retrieval of codes and locations. Do not hallucinate."
        
        self.budget = ModelBudgetDiscovery.from_context_window(
            context_window=32768,
            system_prompt=self.system_prompt,
            response_reserve_pct=0.2,
            model_id=args.model
        )
        
        self.manager = AutoContextManager(
            db_path=self.db_path,
            session_id=self.session_id,
            budget=self.budget,
            embed_fn=self._get_embedding,
            E_threshold=args.threshold
        )
        self.manager.capsule = self.capsule

    def _get_embedding(self, text: str) -> np.ndarray:
        try:
            resp = requests.post(
                f"{self.args.url}/v1/embeddings",
                json={"model": "text-embedding-nomic-embed-text-v1.5", "input": text},
                timeout=30
            )
            resp.raise_for_status()
            vec = np.array(resp.json()["data"][0]["embedding"])
            return vec / np.linalg.norm(vec)
        except Exception as e:
            # Fatal in this test
            print(f"FATAL EMBEDDING ERROR: {e}")
            sys.exit(1)

    def _llm_generate(self, system: str, prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self.args.url}/v1/chat/completions",
                json={
                    "model": self.args.model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    "max_tokens": 128,
                    "temperature": 0.1 # Absolute precision required
                },
                timeout=60
            )
            return resp.json()["choices"][0]["message"]["content"]
        except:
            return "ERROR"

    def run(self):
        print(f"Starting OPERATION HAYSTACK (Cipher Marathon)")
        print(f"Model: {self.args.model} | Turns: 200 | Agents: 50")
        print("-" * 60)
        
        results = {"success": 0, "total": 0, "details": []}
        
        start = time.time()
        
        for i in range(1, 201):
            user_input = self.scenario.get_turn_input(i)
            is_recall = i in self.scenario.recall_tasks
            
            # Logging
            if is_recall:
                print(f"Turn {i:03d} [QUERY] {user_input}")
            elif i % 20 == 0:
                print(f"Turn {i:03d} [FEED] Inputting data stream...")
            
            # Catalytic Turn
            try:
                # Force-flush memory logic implies we trust the manager to evict old turns
                # as we proceed through 200 turns of dense data.
                resp = self.manager.respond_catalytic(
                    query=user_input,
                    llm_generate=self._llm_generate,
                    system_prompt=self.system_prompt
                )
                
                if is_recall:
                    results["total"] += 1
                    query, expected = self.scenario.recall_tasks[i]
                    llm_out = resp.response
                    
                    # Check extraction - Exact match of code or location
                    # We check if the expected strings are in the LLM output OR 
                    # significantly, if they were in the context (proving retrieval).
                    # For this test, let's demand the LLM actually output it (end-to-end task).
                    
                    hit = False
                    for exp in expected:
                        if exp.lower() in llm_out.lower():
                            hit = True
                            break
                    
                    # Also check retrieval context for debugging
                    context = resp.prepare_result.get_context_text()
                    retrieval_hit = any(exp.lower() in context.lower() for exp in expected)
                    
                    if hit:
                        print(f"  >>> PASS: {llm_out.strip()}")
                        results["success"] += 1
                    else:
                        print(f"  >>> FAIL: Expected {expected} | Got: {llm_out.strip()}")
                        if retrieval_hit:
                            print(f"      (Note: Retrieval SUCCEEDED, LLM failed generation)")
                        else:
                            print(f"      (Note: Retrieval FAILED)")
                            
                    results["details"].append((i, hit, retrieval_hit))

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Turn {i} Error: {e}")
        
        duration = time.time() - start
        self._print_summary(results, duration)

    def _print_summary(self, results, duration):
        print("\n" + "="*60)
        print("CIPHER MARATHON REPORT")
        print("="*60)
        total = results["total"]
        success = results["success"]
        score = (success/total * 100) if total else 0
        
        print(f"Duration: {duration:.1f}s")
        print(f"Score: {success}/{total} ({score:.1f}%)")
        
        # Analyze retrieval vs generation
        retrieval_success = sum(1 for _, h, rh in results["details"] if rh)
        print(f"Retrieval Accuracy: {retrieval_success}/{total}")
        
        if score >= 90:
             print("STATUS: SUPERHUMAN (Passed)")
        elif score >= 50:
             print("STATUS: FUNCTIONAL (Mixed)")
        else:
             print("STATUS: BROKEN (Failed)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_LLM_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_E_THRESHOLD)
    args = parser.parse_args()
    
    CipherStressRunner(args).run()

if __name__ == "__main__":
    main()
