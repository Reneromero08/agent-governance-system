#!/usr/bin/env python3
"""
Catalytic Context Stress Test Implementation
Based on THOUGHT/LAB/CAT_CHAT/examples/STRESS_TEST_BRIEF.md

Target: liquid/lfm2.5-1.2b
Objective: Stress-test catalytic context with semantic drift, implicit references, and dense information.
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
    # Fallback if running from a different CWD or structure
    print("Error: detailed to import catalytic_chat modules. Ensure PYTHONPATH is correct.")
    sys.exit(1)

# Configuration from Brief
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
    
class SoftwareArchitectureScenario:
    """Use Case 1: Software Architecture Session (RECOMMENDED)"""
    
    def __init__(self):
        self.planted_facts = {
            3: {
                "text": "Authentication must use JWT tokens signed with RS256 algorithm. Keys rotate every 90 days. Maximum session duration is 24 hours.",
                "keywords": ["RS256", "90 days", "24 hours"]
            },
            7: {
                "text": "Rate limiting is set to 100 requests per minute per API key. Burst allowance is 20 requests. Throttled responses return 429.",
                "keywords": ["100 requests", "minute", "429"]
            },
            12: {
                "text": "The transactions table schema has 12 columns: id, amount, currency, status, created_at, updated_at, merchant_id, customer_id, idempotency_key, metadata, fee_amount, settlement_date.",
                "keywords": ["12 columns", "idempotency_key", "settlement_date"]
            },
            18: {
                "text": "Idempotency keys must be UUID v4 format. They expire after 24 hours. Duplicate requests within window return cached response.",
                "keywords": ["UUID v4", "24 hours", "cached"]
            },
            25: {
                "text": "POST /v1/payments endpoint accepts JSON with required fields: amount (integer cents), currency (ISO 4217), source (payment method). Timestamps use ISO 8601 format with timezone.",
                "keywords": ["integer cents", "ISO 4217", "ISO 8601"]
            },
            31: {
                "text": "Maximum request payload is 1MB. Requests exceeding this return 413. Response bodies are limited to 5MB for batch operations.",
                "keywords": ["1MB", "413", "5MB"]
            },
            38: {
                "text": "API keys are 32-character hexadecimal strings prefixed with 'pk_' for publishable and 'sk_' for secret keys.",
                "keywords": ["32-character", "pk_", "sk_"]
            },
            42: {
                "text": "Webhook signatures use HMAC-SHA256 with a shared secret. Signature header is X-Signature-256. Tolerance window is 5 minutes.",
                "keywords": ["HMAC-SHA256", "X-Signature-256", "5 minutes"]
            },
            48: {
                "text": "Request timeout is 30 seconds for synchronous operations. Async operations have 5-minute timeout with polling endpoint.",
                "keywords": ["30 seconds", "5-minute", "polling"]
            },
            52: {
                "text": "Retry policy: maximum 3 attempts with exponential backoff. Base delay 1 second, max delay 30 seconds. Jitter of +/-10%.",
                "keywords": ["3 attempts", "exponential", "30 seconds"]
            },
            58: {
                "text": "Code coverage target is 85% for unit tests. Integration tests must cover all payment flows. Mutation testing score > 70%.",
                "keywords": ["85%", "mutation", "70%"]
            },
            63: {
                "text": "Load testing target: sustain 10,000 TPS for 10 minutes. p99 latency must stay under 200ms during load test.",
                "keywords": ["10,000 TPS", "10 minutes", "200ms"]
            },
            68: {
                "text": "Stripe webhook endpoint is /v1/hooks/stripe. Events are queued in Redis before processing. Dead letter queue after 5 failures.",
                "keywords": ["/v1/hooks/stripe", "Redis", "5 failures"]
            },
            73: {
                "text": "PCI DSS compliance level 1. Card data never touches our servers. Tokenization via Stripe. Annual audit by QSA required.",
                "keywords": ["level 1", "tokenization", "QSA"]
            },
            78: {
                "text": "Kubernetes deployment: 3 nodes, each 4 vCPU / 16GB RAM. Pod resource limits: 512MB memory, 0.5 CPU. HPA scales 3-10 pods.",
                "keywords": ["3 nodes", "512MB", "3-10 pods"]
            }
        }
        
        self.recall_queries = {
            93: ("For the security audit, what signing algorithm did we choose for tokens?", ["RS256"]), # from turn 3
            94: ("What's our throttling configuration for API consumers?", ["100 requests", "minute", "429"]), # from turn 7 checks
            95: ("How many fields does our main data table have?", ["12 columns"]), # from turn 12
            96: ("What format are our uniqueness identifiers?", ["UUID v4"]), # from turn 18
            97: ("What time format standard do we use in API responses?", ["ISO 8601"]), # from turn 25
            98: ("What's the structure of our authentication credentials?", ["32-character", "pk_", "sk_"]), # from turn 38
            99: ("How do we verify incoming event notifications are authentic?", ["HMAC-SHA256", "X-Signature-256"]), # from turn 42
            100: ("What's our failure recovery strategy for transient errors?", ["3 attempts", "exponential"]), # from turn 52
        }

    def get_turn(self, turn_num: int) -> ScenarioTurn:
        if turn_num in self.planted_facts:
            data = self.planted_facts[turn_num]
            # Wrap fact in natural conversation
            return ScenarioTurn(
                turn_id=turn_num,
                user_input=f"Team, let's document this decision: {data['text']}",
                expected_keywords=data['keywords'],
                is_planted=True
            )
        elif turn_num in self.recall_queries:
            query, expected = self.recall_queries[turn_num]
            return ScenarioTurn(
                turn_id=turn_num,
                user_input=query,
                expected_keywords=expected,
                is_recall=True
            )
        else:
            return ScenarioTurn(
                turn_id=turn_num,
                user_input=self._get_filler(turn_num),
                is_recall=False,
                is_planted=False
            )

    def _get_filler(self, turn_num: int) -> str:
        # Use realistic architecture discussion based on phase
        fillers = [
            "Let's discuss the trade-offs between synchronous and async processing...",
            "Should we use a message queue for this? What about event sourcing?",
            "I'm concerned about the database connection pooling strategy...",
            "How do we handle partial failures in distributed transactions?",
            "What's the migration strategy for schema changes?",
            "We need to define our logging standards clearly.",
            "Can we review the API versioning strategy?",
            "What about documentation generation tools? OpenApi/Swagger?",
            "Let's talk about the CI/CD pipeline stages.",
            "How will we handle secrets management in production?"
        ]
        return fillers[turn_num % len(fillers)]

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
        self.tmpdir = Path(tempfile.gettempdir()) / f"cat_stress_test_{int(time.time())}"
        self.tmpdir.mkdir(exist_ok=True)
        self.db_path = self.tmpdir / "stress_test.db"
        
        self.capsule = SessionCapsule(db_path=self.db_path)
        self.session_id = self.capsule.create_session()
        
        system_prompt = "You are a senior software architect documenting a new system. You have perfect memory of all previous decisions."
        
        # Initialize budget
        self.budget = ModelBudgetDiscovery.from_context_window(
            context_window=context_window,
            system_prompt=system_prompt,
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
        self.system_prompt = system_prompt

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
            # Fallback to random for testing if API fails (though Brief implies working env)
            # But better to fail hard as per specific instructions generally, 
            # for stress test we want to know if infrastructure works.
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
            "temperature": 0.1 # Low temp for factual recall
        }
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LLM Error: {e}")
            return "Error generating response."

    def run(self):
        scenario = SoftwareArchitectureScenario()
        results = {
            "attempts": 0,
            "successes": 0,
            "details": []
        }
        
        print(f"Starting Stress Test: Software Architecture Session")
        print(f"Model: {self.model}")
        print(f"Base URL: {self.base_url}")
        print("-" * 60)
        
        start_time = time.time()
        
        for i in range(1, 101):
            turn = scenario.get_turn(i)
            
            # Print status
            if i % 10 == 0 or turn.is_planted or turn.is_recall:
                prefix = "[PLANT]" if turn.is_planted else "[RECALL]" if turn.is_recall else "[FILLER]"
                print(f"Turn {i:03d} {prefix} {turn.user_input[:60]}...")
            
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
                    
                    # Check if SYSTEM surfaced the fact in context
                    # The brief says: "Success Criteria: System surfaces original fact in context at recall query"
                    context_text = response.prepare_result.get_context_text()
                    
                    # We check if ANY of the expected keywords are in the context passed to the LLM
                    # This proves the retrieval worked via E-scores/Born rule
                    keywords_found = [kw for kw in turn.expected_keywords if kw.lower() in context_text.lower()]
                    success = len(keywords_found) > 0
                    
                    if success:
                        results["successes"] += 1
                        print(f"  >>> SUCCESS: Found keywords {keywords_found} in context.")
                    else:
                        print(f"  >>> FAIL: Did not find keywords {turn.expected_keywords} in context.")
                        if self.verbose:
                            print(f"  Context dump (first 200 chars): {context_text[:200]}...")
                    
                    results["details"].append({
                        "turn": i,
                        "query": turn.user_input,
                        "expected": turn.expected_keywords,
                        "found": keywords_found,
                        "success": success,
                        "e_mean": response.E_mean
                    })

            except Exception as e:
                print(f"Error at turn {i}: {e}")
                # Continue mostly or break?
                # For a stress test, we might want to continue, but exception usually means infra issue.
                pass
        
        duration = time.time() - start_time
        self._print_report(results, duration)

    def _print_report(self, results, duration):
        print("\n" + "=" * 60)
        print("STRESS TEST REPORT")
        print("=" * 60)
        print(f"Total Duration: {duration:.2f}s")
        print(f"Recall Attempts: {results['attempts']}")
        print(f"Successful Recalls: {results['successes']}")
        
        score = (results['successes'] / results['attempts']) * 100 if results['attempts'] > 0 else 0
        
        print("-" * 60)
        print("Detailed Recall Results:")
        for res in results["details"]:
            status = "PASS" if res["success"] else "FAIL"
            print(f"Turn {res['turn']}: {status} | Query: {res['query'][:40]}... | Found: {res['found']}")
            
        print("-" * 60)
        print(f"")
        print(f"  FINAL SCORE: {results['successes']}/{results['attempts']} = {score:.1f}%")
        print(f"")
        if score >= 75:
             print(f"  OVERALL STATUS: PASSED")
        else:
             print(f"  OVERALL STATUS: FAILED")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Catalytic Context Stress Test")
    parser.add_argument("--url", default=DEFAULT_LLM_BASE, help="LLM Studio Base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID")
    parser.add_argument("--turns", type=int, default=100, help="Number of turns (default 100)")
    parser.add_argument("--verbose", action="store_true", help="Print debug info")
    
    args = parser.parse_args()
    
    runner = StressTestRunner(
        base_url=args.url,
        model=args.model,
        embed_model=DEFAULT_EMBED_MODEL,
        context_window=DEFAULT_CONTEXT_WINDOW,
        e_threshold=DEFAULT_E_THRESHOLD,
        verbose=args.verbose
    )
    
    runner.run()

if __name__ == "__main__":
    main()
