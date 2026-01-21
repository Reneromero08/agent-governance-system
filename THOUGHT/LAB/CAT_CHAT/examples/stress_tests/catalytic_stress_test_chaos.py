#!/usr/bin/env python3
"""
Catalytic Context Stress Test - CHAOS MODE
Use Case: Software Architecture Session (Chaotic, Realistic, Dense)

Target: liquid/lfm2.5-1.2b
Objective: Stress-test catalytic context with rapid context switching, self-contradiction, and high density.
"""

import sys
import argparse
import requests
import json
import time
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass
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
    print("Error: failed to import catalytic_chat modules. Ensure PYTHONPATH is correct.")
    sys.exit(1)

# Configuration
DEFAULT_LLM_BASE = "http://10.5.0.2:1234"
DEFAULT_MODEL = "liquid/lfm2.5-1.2b"
DEFAULT_EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
DEFAULT_CONTEXT_WINDOW = 32768
DEFAULT_E_THRESHOLD = 0.3

@dataclass
class ScenarioTurn:
    turn_id: int
    user_input: str
    expected_keywords: List[str] = None
    is_recall: bool = False
    is_planted: bool = False

class SoftwareArchitectureChaosScenario:
    """
    Use Case 1: Software Architecture Session (CHAOS MODE)
    A 100-turn fully scripted conversation with human chaos, mind-changing,
    and dense parameter updates.
    """
    
    def __init__(self):
        # The script is a continuous conversation.
        # Key decisions (Facts) are embedded naturally.
        # Contradictions are introduced and resolved.
        
        self.script = {
            1: "Okay, kickoff time. We're building 'Project Chimera', a fintech payment gateway. Main goal: high throughput, extreme reliability. Stack preferences?",
            2: "I'm thinking Python for the backend? Maybe FastApi. And we definitely need a relational DB.",
            # FACT 1 (Turn 3) - Auth
            3: "Security first. Authentication must use JWT tokens signed with RS256 algorithm. Keys rotate every 90 days. Maximum session duration is 24 hours. Don't let me forget that.",
            4: "Actually, should we use HS256? No, RS256 is better for key distribution. Stick to RS256.",
            5: "For the database, Postgres is the standard. Let's use version 16. We need to handle high write volume.",
            6: "What about traffic spikes? We need a strict rate limit policy.",
            # FACT 2 (Turn 7) - Rate Limiting
            7: "Let's set rate limiting to 100 requests per minute per API key. Burst allowance is 20 requests. Throttled responses return 429. That seems fair for MVP.",
            8: "Maybe 200/min? No, infra team says 100/min is the safe limit for now. We can bump it later.",
            9: "We need a 'transactions' table. It's the core ledger.",
            10: "I want to track everything. IP address, user agent, latency...",
            11: "Let's define the schema. Standard fields like ID and timestamps are obvious.",
            # FACT 3 (Turn 12) - Schema
            12: "The transactions table schema has 12 columns: id, amount, currency, status, created_at, updated_at, merchant_id, customer_id, idempotency_key, metadata, fee_amount, settlement_date.",
            13: "Wait, did I miss `chargeback_status`? actually, let's put that in metadata for now to keep it to 12 columns.",
            14: "For currency, are we supporting crypto? No, just fiat for now. ISO codes.",
            15: "We need to ensure we don't double-charge people. Idempotency is critical.",
            16: "Usually we use a header for that. `Idempotency-Key`.",
            17: "What format should those keys be? Random strings?",
            # FACT 4 (Turn 18) - Idempotency
            18: "Idempotency keys must be UUID v4 format. They expire after 24 hours. Duplicate requests within window return cached response.",
            19: "Can we accept UUID v1? No, privacy issues. v4 only.",
            20: "Moving to the API endpoints. We need a way to create a payment.",
            21: "RESTful design. `POST /payments` seems right.",
            22: "We need to version this. `/v1/payments`.",
            23: "What's the payload look like? JSON obviously.",
            24: "Amounts should be decimals? Floating point errors are scary.",
            # FACT 5 (Turn 25) - API Input
            25: "POST /v1/payments endpoint accepts JSON with required fields: amount (integer cents), currency (ISO 4217), source (payment method). Timestamps use ISO 8601 format with timezone.",
            26: "So $10.00 is `1000`? Yes, integer cents prevents rounding errors.",
            27: "And the timestamp, UTC only? Yes, ISO 8601 Z-suffix is best practice.",
            28: "What about specialized payment methods like ACH?",
            29: "Currently just cards. We'll add 'source' field flexibility later.",
            30: "How big can these requests get? Some metadata blobs can be huge.",
            # FACT 6 (Turn 31) - Payloads
            31: "Maximum request payload is 1MB. Requests exceeding this return 413. Response bodies are limited to 5MB for batch operations.",
            32: "1MB is small... actually for a payment request it's huge. 1MB is fine.",
            33: "If they send 1.1MB, we drop the connection? No, 413 Payload Too Large.",
            34: "Let's talk about API Keys again. How do we generate them?",
            35: "They need to be random and secure.",
            36: "Should we use a prefix? Stripe does that, it's nice for debugging.",
            37: "Yeah, `sk_live_...` style.",
            # FACT 7 (Turn 38) - API Keys
            38: "API keys are 32-character hexadecimal strings prefixed with 'pk_' for publishable and 'sk_' for secret keys.",
            39: "Wait, 32 chars excluding the prefix or including? Let's say 32 chars of entropy + the prefix.",
            40: "Actually, for the documentation, just call it '32-character hex' + prefix. Keep it simple.",
            41: "Webhooks. We need to notify merchants when a payment succeeds.",
            # FACT 8 (Turn 42) - Webhook Security
            42: "Webhook signatures use HMAC-SHA256 with a shared secret. Signature header is X-Signature-256. Tolerance window is 5 minutes.",
            43: "Why a tolerance window? Replay attacks. If simple replay, 5 mins reduces attack surface.",
            44: "Can we use SHA1? No, it's broken. SHA256 only.",
            45: "What if the merchant's server is down?",
            46: "We need a timeout policy. We can't hang forever.",
            47: "5 seconds? 10 seconds?",
            # FACT 9 (Turn 48) - Timeouts
            48: "Request timeout is 30 seconds for synchronous operations. Async operations have 5-minute timeout with polling endpoint.",
            49: "30 seconds is a long time for a user to wait. But upstream banks are slow.",
            50: "If it fails, do we retry automatically?",
            51: "Yes, but we need to be careful not to spam.",
            # FACT 10 (Turn 52) - Retries
            52: "Retry policy: maximum 3 attempts with exponential backoff. Base delay 1 second, max delay 30 seconds. Jitter of +/-10%.",
            53: "So 1s, 2s, 4s? Roughly. Plus jitter.",
            54: "Switching gears to QA. How do we test this beast?",
            55: "We need unit tests, obviously.",
            56: "What's our coverage gate? 100% is unrealistic.",
            57: "Let's target high coverage for the core money logic.",
            # FACT 11 (Turn 58) - Testing
            58: "Code coverage target is 85% for unit tests. Integration tests must cover all payment flows. Mutation testing score > 70%.",
            59: "Mutation testing is expensive to run. Maybe only on nightly builds?",
            60: "Yeah, nightly is fine. But blocking PRs on 85% unit coverage.",
            61: "We need to load test this too. Black Friday traffic simulation.",
            62: "How much traffic? 1k TPS?",
            # FACT 12 (Turn 63) - Load Testing
            63: "Load testing target: sustain 10,000 TPS for 10 minutes. p99 latency must stay under 200ms during load test.",
            64: "10k TPS is aggressive. We'll need a big cluster.",
            65: "We can optimize the DB queries later if we miss the 200ms target.",
            66: "Let's talk third-party integrations. Stripe?",
            67: "We need to ingest their webhooks too.",
            # FACT 13 (Turn 68) - Integration endpoints
            68: "Stripe webhook endpoint is /v1/hooks/stripe. Events are queued in Redis before processing. Dead letter queue after 5 failures.",
            69: "Why Redis? Why not Kafka? Redis is simpler for now. We can migrate if we scale beyond Redis limits.",
            70: "Dead letter queue is essential. We can't lose payment events.",
            71: "Compliance. The dreaded word.",
            72: "We are handling card numbers?",
            # FACT 14 (Turn 73) - Compliance
            73: "PCI DSS compliance level 1. Card data never touches our servers. Tokenization via Stripe. Annual audit by QSA required.",
            74: "Wait, if we use Stripe Elements, do we still need Level 1? Yes, because of the volume we process.",
            75: "So we are SAQ-D? No, we need a full ROC audit.",
            76: "Deployment architecture. Kubernetes?",
            77: "Everyone uses K8s. It's standard.",
            # FACT 15 (Turn 78) - Infra
            78: "Kubernetes deployment: 3 nodes, each 4 vCPU / 16GB RAM. Pod resource limits: 512MB memory, 0.5 CPU. HPA scales 3-10 pods.",
            79: "512MB per pod seems low for Python. Python is memory hungry.",
            80: "It forces us to write efficient code. We can bump to 1GB if we OOM kill too often.",
            81: "What about logging? ELK stack?",
            82: "ELK is heavy. Maybe Datadog?",
            83: "Let's stick to JSON logs to stdout, let the agent scrape them.",
            84: "We need metrics too. Prometheus.",
            85: "What are our alert thresholds?",
            86: "If latency goes up, wake me up.",
            87: "Let's define 'latency goes up'.",
            88: "Alert threshold: p99 > 500ms for 5 minutes. Log retention: 30 days.",
            89: "30 days hot storage, then S3 glacier.",
            90: "Okay, I think we have a solid plan. Let me review my notes.",
            91: "Wait, checking the auth decision again... yes, RS256.",
            92: "And the retry logic... yes, exponential backoff."
        }
        
        self.recall_queries = {
            93: ("For the security audit, what signing algorithm did we choose for tokens?", ["RS256"]), # Turn 3
            94: ("What's our throttling configuration for API consumers?", ["100 requests", "minute", "429"]), # Turn 7
            95: ("How many fields does our main data table have?", ["12 columns"]), # Turn 12
            96: ("What format are our uniqueness identifiers?", ["UUID v4"]), # Turn 18
            97: ("What time format standard do we use in API responses?", ["ISO 8601"]), # Turn 25
            98: ("What's the structure of our authentication credentials?", ["32-character", "pk_", "sk_"]), # Turn 38
            99: ("How do we verify incoming event notifications are authentic?", ["HMAC-SHA256", "X-Signature-256"]), # Turn 42
            100: ("What's our failure recovery strategy for transient errors?", ["3 attempts", "exponential"]), # Turn 52
        }

    def get_turn(self, turn_num: int) -> ScenarioTurn:
        if turn_num in self.recall_queries:
            query, expected = self.recall_queries[turn_num]
            return ScenarioTurn(
                turn_id=turn_num,
                user_input=query,
                expected_keywords=expected,
                is_recall=True
            )
        
        # Get scripted line or default if out of range
        user_text = self.script.get(turn_num, f"Continuing discussion on topic {turn_num}...")
        
        # Identify key decision turns
        is_planted = turn_num in [3, 7, 12, 18, 25, 31, 38, 42, 48, 52, 58, 63, 68, 73, 78]
        
        return ScenarioTurn(
            turn_id=turn_num,
            user_input=user_text,
            is_planted=is_planted,
            expected_keywords=None
        )

class StressTestRunner:
    def __init__(self, 
                 base_url: str, 
                 model: str, 
                 embed_model: str, 
                 context_window: int,
                 e_threshold: float,
                 verbose: bool = True):
        self.base_url = base_url
        self.model = model
        self.embed_model = embed_model
        self.verbose = verbose
        
        # Setup Catalytic Chat components
        import tempfile
        self.tmpdir = Path(tempfile.gettempdir()) / f"cat_chaos_test_{int(time.time())}"
        self.tmpdir.mkdir(exist_ok=True)
        self.db_path = self.tmpdir / "chaos_test.db"
        
        self.capsule = SessionCapsule(db_path=self.db_path)
        self.session_id = self.capsule.create_session()
        
        # Initial system prompt
        self.system_prompt = "You are a senior technical architect named 'Archie'. You are documenting a new fintech payment system 'Chimera'. You must remember every technical decision made in this conversation."
        
        # Initialize budget
        self.budget = ModelBudgetDiscovery.from_context_window(
            context_window=context_window,
            system_prompt=self.system_prompt,
            response_reserve_pct=0.2,
            model_id=model
        )
        
        # Initialize Manager
        self.manager = AutoContextManager(
            db_path=self.db_path,
            session_id=self.session_id,
            budget=self.budget,
            embed_fn=self._get_embedding,
            E_threshold=e_threshold
        )
        self.manager.capsule = self.capsule # Ensure shared capsule

    def _get_embedding(self, text: str) -> np.ndarray:
        url = f"{self.base_url}/v1/embeddings"
        payload = {
            "model": self.embed_model,
            "input": text
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            vec = np.array(resp.json()["data"][0]["embedding"])
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec
        except Exception as e:
            print(f"Embedding Error: {e}")
            raise e

    def _llm_generate(self, system: str, prompt: str) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.5 # Higher temp for chaos mode naturalness
        }
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LLM Error: {e}")
            return "Error generating response."

    def run(self):
        scenario = SoftwareArchitectureChaosScenario()
        results = {
            "attempts": 0,
            "successes": 0,
            "details": []
        }
        
        print(f"Starting CHAOS Stress Test: Project Chimera")
        print(f"Model: {self.model}")
        print("-" * 60)
        
        start_time = time.time()
        
        for i in range(1, 101):
            turn = scenario.get_turn(i)
            
            # Print status - show every 5 lines or special turns to see the chaos flow
            if i % 5 == 0 or turn.is_planted or turn.is_recall:
                prefix = "[DECISION]" if turn.is_planted else "[RECALL]" if turn.is_recall else "[CHAT]"
                print(f"Turn {i:03d} {prefix} {turn.user_input[:80]}...")
            
            # Run catalytic turn
            try:
                response = self.manager.respond_catalytic(
                    query=turn.user_input,
                    llm_generate=self._llm_generate,
                    system_prompt=self.system_prompt
                )
                
                # Verification for Recall Turns
                if turn.is_recall:
                    results["attempts"] += 1
                    context_text = response.prepare_result.get_context_text()
                    keywords_found = [kw for kw in turn.expected_keywords if kw.lower() in context_text.lower()]
                    success = len(keywords_found) > 0
                    
                    if success:
                        results["successes"] += 1
                        print(f"  >>> SUCCESS: Found {keywords_found}")
                    else:
                        print(f"  >>> FAIL: Expected {turn.expected_keywords}")
                    
                    results["details"].append({
                        "turn": i,
                        "query": turn.user_input,
                        "expected": turn.expected_keywords,
                        "found": keywords_found,
                        "success": success
                    })

            except Exception as e:
                print(f"Error at turn {i}: {e}")
                pass
        
        duration = time.time() - start_time
        self._print_report(results, duration)

    def _print_report(self, results, duration):
        print("\n" + "=" * 60)
        print("CHAOS TEST REPORT")
        print("=" * 60)
        print(f"Total Duration: {duration:.2f}s")
        score = (results['successes'] / results['attempts']) * 100 if results['attempts'] > 0 else 0
        
        print("\nDetailed Recall Results:")
        for res in results["details"]:
            status = "PASS" if res["success"] else "FAIL"
            print(f"Turn {res['turn']}: {status} | Query: {res['query'][:40]}... | Found: {res['found']}")
            
        print("\n" + "-" * 60)
        print(f"FINAL SCORE: {results['successes']}/{results['attempts']} = {score:.1f}%")
        print("-" * 60)
        
        if score >= 80:
             print("STATUS: CHAOS MASTER (Passed)")
        else:
             print("STATUS: CONFUSED (Failed)")

def main():
    parser = argparse.ArgumentParser(description="Catalytic Context Chaos Test")
    parser.add_argument("--url", default=DEFAULT_LLM_BASE, help="LLM Studio Base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID")
    parser.add_argument("--turns", type=int, default=100, help="Number of turns")
    
    args = parser.parse_args()
    
    runner = StressTestRunner(
        base_url=args.url,
        model=args.model,
        embed_model=DEFAULT_EMBED_MODEL,
        context_window=DEFAULT_CONTEXT_WINDOW,
        e_threshold=DEFAULT_E_THRESHOLD
    )
    
    runner.run()

if __name__ == "__main__":
    main()
