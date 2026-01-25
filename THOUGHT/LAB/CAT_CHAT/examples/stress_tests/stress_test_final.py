#!/usr/bin/env python3
"""
CAT CHAT Final Stress Test
==========================

Real conversational stress test based on STRESS_TEST_BRIEF.md.
Tests semantic retrieval across 100 turns with:
- No keyword overlap between planted facts and recall queries
- Real conversation flow (software architecture session)
- E-score measured every turn
- Full DB storage and retrieval verification

Success = System retrieves the right context (not LLM hallucination)
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

CAT_CHAT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CAT_CHAT_ROOT))

from catalytic_chat.session_capsule import SessionCapsule
from catalytic_chat.auto_context_manager import AutoContextManager
from catalytic_chat.adaptive_budget import ModelBudgetDiscovery
from catalytic_chat.llm_client import get_llm_client

DB_OUTPUT_DIR = Path(__file__).parent.parent / "test_chats"


# =============================================================================
# SOFTWARE ARCHITECTURE SESSION - 200 Turns (HARD MODE)
# =============================================================================

PLANTED_FACTS = {
    # Early phase - Requirements (turns 1-20)
    3: {
        "content": "For authentication, we need to use JWT tokens signed with the RS256 algorithm. Keys will rotate every 90 days, and maximum session duration should be 24 hours before requiring re-authentication.",
        "keywords": ["RS256", "90 days", "24 hours"],
    },
    7: {
        "content": "I've set up rate limiting at 100 requests per minute per API key. We allow a burst of 20 requests, and throttled responses will return HTTP 429 status code.",
        "keywords": ["100 requests", "minute", "429"],
    },
    12: {
        "content": "Here's the transactions table schema - it has 12 columns: id, amount, currency, status, created_at, updated_at, merchant_id, customer_id, idempotency_key, metadata, fee_amount, and settlement_date.",
        "keywords": ["12 columns", "idempotency_key", "settlement_date"],
    },
    18: {
        "content": "For idempotency, we require UUID v4 format keys. They expire after 24 hours, and duplicate requests within that window will return the cached response.",
        "keywords": ["UUID v4", "24 hours", "cached"],
    },
    # API Design phase (turns 21-45)
    25: {
        "content": "The POST /v1/payments endpoint accepts JSON with required fields: amount as integer cents, currency in ISO 4217 format, and source for the payment method. All timestamps use ISO 8601 format with timezone.",
        "keywords": ["integer cents", "ISO 4217", "ISO 8601"],
    },
    31: {
        "content": "I'm setting maximum request payload at 1MB - anything larger returns HTTP 413. For batch operations, response bodies can go up to 5MB.",
        "keywords": ["1MB", "413", "5MB"],
    },
    38: {
        "content": "API keys are 32-character hexadecimal strings. Publishable keys are prefixed with 'pk_' and secret keys with 'sk_'. Never expose secret keys client-side.",
        "keywords": ["32-character", "pk_", "sk_"],
    },
    42: {
        "content": "For webhook verification, we're using HMAC-SHA256 with a shared secret. The signature will be in the X-Signature-256 header, and we allow a 5-minute tolerance window for clock skew.",
        "keywords": ["HMAC-SHA256", "X-Signature-256", "5 minutes"],
    },
    # Error handling phase (turns 46-70)
    48: {
        "content": "Synchronous operations have a 30-second timeout. For async operations, we use a 5-minute timeout with a polling endpoint to check status.",
        "keywords": ["30 seconds", "5-minute", "polling"],
    },
    52: {
        "content": "Our retry policy is maximum 3 attempts with exponential backoff. Base delay is 1 second, max delay caps at 30 seconds, and we add jitter of plus or minus 10 percent.",
        "keywords": ["3 attempts", "exponential", "30 seconds"],
    },
    58: {
        "content": "For testing standards, code coverage target is 85% for unit tests. Integration tests must cover all payment flows. We also want mutation testing score above 70%.",
        "keywords": ["85%", "mutation", "70%"],
    },
    63: {
        "content": "Load testing requirements: sustain 10,000 transactions per second for 10 minutes continuously. P99 latency must stay under 200 milliseconds during the test.",
        "keywords": ["10,000 TPS", "10 minutes", "200ms"],
    },
    68: {
        "content": "The Stripe webhook endpoint is /v1/hooks/stripe. Incoming events get queued in Redis before processing. After 5 consecutive failures, events go to the dead letter queue.",
        "keywords": ["/v1/hooks/stripe", "Redis", "5 failures"],
    },
    # Infrastructure phase (turns 71-100)
    73: {
        "content": "We need PCI DSS compliance level 1 since we're handling large volumes. Card data never touches our servers - we use tokenization through Stripe. Annual audit by a QSA is required.",
        "keywords": ["level 1", "tokenization", "QSA"],
    },
    78: {
        "content": "Kubernetes deployment specs: 3 nodes, each with 4 vCPU and 16GB RAM. Pod resource limits are 512MB memory and 0.5 CPU. Horizontal pod autoscaler scales between 3 and 10 pods.",
        "keywords": ["3 nodes", "512MB", "3-10 pods"],
    },
    84: {
        "content": "Database connection pool size is set to 50 connections per pod. We're using PgBouncer in transaction mode. Max query execution time is 60 seconds before automatic termination.",
        "keywords": ["50 connections", "PgBouncer", "60 seconds"],
    },
    89: {
        "content": "Monitoring uses Prometheus with 15-second scrape interval. Alertmanager routes to PagerDuty for P1 incidents. Grafana dashboards refresh every 30 seconds.",
        "keywords": ["Prometheus", "15-second", "PagerDuty", "30 seconds"],
    },
    # Scaling phase (turns 101-130)
    95: {
        "content": "The CDN configuration uses Cloudflare with a TTL of 3600 seconds for static assets. We cache API responses for 60 seconds on GET endpoints only.",
        "keywords": ["Cloudflare", "3600 seconds", "60 seconds"],
    },
    102: {
        "content": "Message queue uses RabbitMQ with 3 replicas in HA mode. Default message TTL is 86400 seconds. Dead letter exchange is named 'dlx.payments'.",
        "keywords": ["RabbitMQ", "3 replicas", "86400", "dlx.payments"],
    },
    108: {
        "content": "The circuit breaker trips after 5 consecutive failures. Half-open state allows 1 request every 30 seconds. Full reset requires 3 successful requests.",
        "keywords": ["5 consecutive", "half-open", "30 seconds", "3 successful"],
    },
    115: {
        "content": "Feature flags are managed through LaunchDarkly with a 2-minute sync interval. Kill switches have 10-second propagation. We maintain 50 active flags maximum.",
        "keywords": ["LaunchDarkly", "2-minute", "10-second", "50 active"],
    },
    # Security hardening (turns 131-160)
    122: {
        "content": "The WAF rules block requests from TOR exit nodes and known bad IPs. Rate limiting at edge is 1000 requests per second per IP. SQL injection patterns trigger immediate block.",
        "keywords": ["TOR exit", "1000 requests", "SQL injection"],
    },
    128: {
        "content": "Encryption at rest uses AES-256-GCM. Key rotation happens every 365 days through AWS KMS. We maintain 3 key versions for decryption compatibility.",
        "keywords": ["AES-256-GCM", "365 days", "AWS KMS", "3 key versions"],
    },
    135: {
        "content": "Password hashing uses Argon2id with memory cost 65536 KB, time cost 3 iterations, and parallelism of 4 threads. Salt is 16 bytes random.",
        "keywords": ["Argon2id", "65536", "3 iterations", "16 bytes"],
    },
    142: {
        "content": "CORS policy allows origins from *.ourcompany.com only. Preflight cache is 7200 seconds. Credentials are allowed. Exposed headers include X-Request-ID.",
        "keywords": ["*.ourcompany.com", "7200 seconds", "X-Request-ID"],
    },
    # Compliance phase (turns 161-180)
    148: {
        "content": "Data retention policy: transaction records kept for 7 years, logs for 90 days, user PII deleted within 30 days of account closure per GDPR requirements.",
        "keywords": ["7 years", "90 days", "30 days", "GDPR"],
    },
    155: {
        "content": "Audit logging captures all admin actions with 256-bit event IDs. Log shipping to SIEM happens every 5 minutes. Retention in cold storage is 10 years.",
        "keywords": ["256-bit", "5 minutes", "10 years"],
    },
    162: {
        "content": "The disaster recovery RTO is 4 hours, RPO is 1 hour. Failover to secondary region is automatic after 3 consecutive health check failures spanning 90 seconds.",
        "keywords": ["4 hours", "1 hour", "3 consecutive", "90 seconds"],
    },
    168: {
        "content": "Backup schedule: full backup every Sunday at 02:00 UTC, incrementals every 6 hours. Cross-region replication lag must stay under 500 milliseconds.",
        "keywords": ["Sunday", "02:00 UTC", "6 hours", "500 milliseconds"],
    },
}

RECALL_QUERIES = {
    # Early recalls - testing recent memory (turns 180-185)
    180: {
        "query": "The DR team needs to know - how quickly must we recover from a disaster and how much data loss is acceptable?",
        "expected": ["4 hours", "1 hour", "RTO", "RPO"],
        "source_turn": 162,
    },
    181: {
        "query": "When does our system take full snapshots of all data?",
        "expected": ["Sunday", "02:00", "UTC"],
        "source_turn": 168,
    },
    # Medium distance recalls (turns 186-192)
    186: {
        "query": "Legal is asking about how long we keep customer transaction history for compliance purposes.",
        "expected": ["7 years"],
        "source_turn": 148,
    },
    187: {
        "query": "The security team wants to know what algorithm we use for storing user credentials securely.",
        "expected": ["Argon2id", "Argon2"],
        "source_turn": 135,
    },
    188: {
        "query": "What protection do we have at the network edge against malicious traffic patterns?",
        "expected": ["WAF", "TOR", "1000", "SQL injection"],
        "source_turn": 122,
    },
    189: {
        "query": "How do we manage gradual rollouts and quick rollbacks of new functionality?",
        "expected": ["LaunchDarkly", "feature flag", "kill switch"],
        "source_turn": 115,
    },
    190: {
        "query": "What happens when a downstream service becomes unresponsive? How do we protect against cascade failures?",
        "expected": ["circuit breaker", "5 consecutive", "half-open"],
        "source_turn": 108,
    },
    191: {
        "query": "What messaging infrastructure handles our async event processing?",
        "expected": ["RabbitMQ", "3 replicas", "dlx"],
        "source_turn": 102,
    },
    # Long distance recalls - testing deep memory (turns 193-200)
    193: {
        "query": "For the security audit next week, what signing algorithm did we choose for authentication tokens?",
        "expected": ["RS256"],
        "source_turn": 3,
    },
    194: {
        "query": "The ops team is asking about our throttling configuration. What limits did we set?",
        "expected": ["100", "minute", "429"],
        "source_turn": 7,
    },
    195: {
        "query": "I need to update the docs - how many fields does our main data table have?",
        "expected": ["12"],
        "source_turn": 12,
    },
    196: {
        "query": "What format standard do we use for preventing duplicate payment submissions?",
        "expected": ["UUID", "v4"],
        "source_turn": 18,
    },
    197: {
        "query": "Integration partner needs the timestamp format we use in API responses.",
        "expected": ["ISO 8601", "8601"],
        "source_turn": 25,
    },
    198: {
        "query": "Security review: what's the format and structure of our API authentication credentials?",
        "expected": ["32", "pk_", "sk_"],
        "source_turn": 38,
    },
    199: {
        "query": "How do we verify incoming webhooks are authentic and not forged?",
        "expected": ["HMAC", "SHA256", "X-Signature"],
        "source_turn": 42,
    },
    200: {
        "query": "What's our strategy for handling temporary failures when calling external services?",
        "expected": ["3 attempts", "exponential", "backoff"],
        "source_turn": 52,
    },
}

# Real conversation filler - actual architecture discussion, no planted keywords
CONVERSATION_FLOW = {
    # Phase 1: Requirements gathering (1-20)
    1: "Hey, let's start planning out this payment API. What are the core requirements we need to nail down first?",
    2: "We should probably start with authentication and authorization. Security is critical for a fintech product.",
    4: "Good call on the token approach. Now what about protecting against abuse? We need some kind of throttling.",
    5: "Makes sense. We should also think about how the client will know they're being rate limited.",
    6: "Right, proper error codes are important. Let me document the rate limiting specs.",
    8: "Now let's talk about the data model. What's the core entity we're building around?",
    9: "Transactions are central, obviously. We need to track the full lifecycle.",
    10: "Should we use soft deletes or hard deletes for failed transactions?",
    11: "Soft deletes for audit trail. Let me spec out the schema.",
    13: "One thing I'm worried about is duplicate submissions. Users might click pay twice.",
    14: "Classic problem. We need idempotency built in from day one.",
    15: "Agreed. How should clients generate their idempotency tokens?",
    16: "We need something universally unique but also easy to generate client-side.",
    17: "Let me write up the idempotency requirements.",
    19: "Now for the API design. What's our URL structure going to look like?",
    20: "RESTful for sure. Resources map to nouns, actions to HTTP verbs.",
    # Phase 2: API design (21-45)
    21: "What about versioning? Path-based or header-based?",
    22: "Path-based is more explicit. Easier for developers to understand.",
    23: "Agreed. Let's use /v1/ prefix. Now what about the payment creation endpoint?",
    24: "That's the most critical one. Need to get the request format right.",
    26: "Speaking of requests, we should set some limits on payload size.",
    27: "Definitely. Don't want someone trying to send us a gigabyte of metadata.",
    28: "What's a reasonable max? We need to allow enough for legitimate use cases.",
    29: "Metadata should be bounded. Let me think about realistic scenarios.",
    30: "Also need to consider response sizes for batch endpoints.",
    32: "Now security. How are we identifying API consumers?",
    33: "API keys are standard for server-to-server. But we need different types.",
    34: "Right, some operations need higher privilege than others.",
    35: "We should have publishable keys for client-side and secret keys for server-side.",
    36: "Exactly. And they need to be distinguishable at a glance.",
    37: "Let me document the key format.",
    39: "What about webhooks? Partners need to know when payments complete.",
    40: "Webhooks are essential. But we need to make them secure.",
    41: "Recipients need to verify the webhook actually came from us.",
    43: "Good. Now let's think about failure modes. Network issues happen.",
    44: "Timeouts are critical. Too short and legitimate requests fail.",
    45: "Too long and the system backs up under load.",
    # Phase 3: Error handling (46-70)
    46: "We need different timeouts for sync versus async operations.",
    47: "Async is trickier. Users need a way to check on long-running jobs.",
    49: "What happens when a request fails? We should have retry logic.",
    50: "Client-side retries need to be smart. Can't hammer a struggling server.",
    51: "Exponential backoff with jitter is the standard approach.",
    53: "Let's switch gears to testing. What's our quality bar?",
    54: "We need comprehensive coverage. Payments can't have bugs.",
    55: "Unit tests, integration tests, and load tests at minimum.",
    56: "What coverage percentage are we targeting?",
    57: "High, but not unrealistic. We need to actually ship.",
    59: "Load testing is crucial. What's our expected traffic?",
    60: "We need to plan for growth. Better to over-spec than under-spec.",
    61: "Sustained load is different from burst handling.",
    62: "We should define clear performance SLAs.",
    64: "Now let's talk about third-party integrations. We're using Stripe for processing.",
    65: "Right. We need to handle their webhooks properly.",
    66: "Where do those webhook events get delivered?",
    67: "We need a dedicated endpoint for Stripe specifically.",
    69: "What about reliability? Webhooks can fail.",
    70: "We should queue events and process them asynchronously.",
    # Phase 4: Infrastructure (71-100)
    71: "Dead letter queue for repeated failures.",
    72: "Also need to think about compliance. PCI is non-negotiable.",
    74: "Now infrastructure. How are we deploying this?",
    75: "Containers for sure. Kubernetes gives us scaling.",
    76: "What's our cluster topology look like?",
    77: "Need enough capacity for redundancy but not wasteful.",
    79: "How do we handle scaling under load?",
    80: "Auto-scaling based on metrics. CPU and request rate.",
    81: "Need to set reasonable bounds on scaling.",
    82: "Now monitoring and alerting. We need visibility.",
    83: "Can't fix what we can't see. Full observability stack.",
    85: "What thresholds trigger alerts?",
    86: "Latency spikes are the canary. If P99 goes up, something's wrong.",
    87: "Should also alert on error rate increases.",
    88: "And have runbooks for each alert type.",
    90: "Now database layer. Connection management is tricky at scale.",
    91: "Yeah pooling is essential. Direct connections won't scale.",
    92: "What about query timeouts? Can't let long queries block everything.",
    93: "Need reasonable limits. Balance between completing work and protecting resources.",
    94: "Let me document the database requirements.",
    96: "Edge caching would help reduce origin load significantly.",
    97: "Especially for read-heavy endpoints. Static assets are obvious wins.",
    98: "What about caching API responses? That's trickier.",
    99: "Only safe for idempotent operations. GET requests basically.",
    100: "Right. Let me spec out the caching strategy.",
    # Phase 5: Scaling (101-130)
    101: "Moving on to async processing. We need a proper message queue.",
    103: "For the queue, we need something that won't lose messages.",
    104: "Durability and high availability are must-haves.",
    105: "What happens to messages that can't be processed?",
    106: "Need a dead letter mechanism. Can't just drop failed messages.",
    107: "Agreed. Let me document the queue architecture.",
    109: "Resilience patterns. What about when dependencies fail?",
    110: "Need automatic protection. Can't cascade failures.",
    111: "Circuit breakers are the standard pattern for this.",
    112: "How do we recover once the dependency comes back?",
    113: "Gradual recovery. Don't slam it with all the backed-up traffic.",
    114: "Good point. Let me add recovery mechanics to the spec.",
    116: "Feature management. How do we roll out changes safely?",
    117: "Gradual rollouts. Start with internal users, expand slowly.",
    118: "What about rollback? Need to be able to pull back fast.",
    119: "Kill switches for emergency situations.",
    120: "How quickly can we disable something globally?",
    121: "Needs to be near instant. Can't wait for deployments.",
    # Phase 6: Security hardening (131-160)
    123: "Security hardening now. Edge protection is critical.",
    124: "Need to block known bad actors before they hit the origin.",
    125: "What kinds of attacks are we protecting against?",
    126: "Automated attacks, bots, known malicious networks.",
    127: "Also need to detect and block injection attempts.",
    129: "Data protection. Encryption requirements.",
    130: "Everything at rest needs to be encrypted. No exceptions.",
    131: "What algorithm and key management approach?",
    132: "Industry standard encryption. Managed key service for rotation.",
    133: "How often do keys rotate?",
    134: "Balance between security and operational overhead.",
    136: "User credentials. How are we storing passwords?",
    137: "Modern hashing algorithm. Memory-hard to resist attacks.",
    138: "Salt generation and storage approach?",
    139: "Random salt per credential. Stored alongside the hash.",
    140: "Good. Defense in depth if database is compromised.",
    141: "Let me document the credential storage approach.",
    143: "Cross-origin requests. Browser security policies.",
    144: "Need to allow legitimate integrations but block abuse.",
    145: "Whitelist specific origins. No wildcards in production.",
    146: "Preflight caching to reduce OPTIONS requests.",
    147: "What headers do we expose to JavaScript clients?",
    # Phase 7: Compliance (161-180)
    149: "Compliance requirements. Data retention policies.",
    150: "Different requirements for different data types.",
    151: "Transaction records for auditors. Logs for debugging.",
    152: "What about user data deletion requests?",
    153: "GDPR right to be forgotten. Need a clear process.",
    154: "Timelines matter. Can't drag feet on deletion requests.",
    156: "Audit trail. What admin actions do we need to log?",
    157: "Everything that changes state. No blind spots.",
    158: "How long do we keep audit logs?",
    159: "Long retention for compliance. Cold storage after initial period.",
    160: "Centralized log aggregation for analysis.",
    161: "Good. Let me add audit requirements to the compliance section.",
    163: "Disaster recovery. What if the primary region goes down?",
    164: "Need automatic failover. Can't wait for manual intervention.",
    165: "How fast does failover need to happen?",
    166: "Business continuity requirements drive this. Check with stakeholders.",
    167: "What about data loss during failover?",
    169: "Backup strategy. How often and what type?",
    170: "Full backups periodically. Incrementals more frequently.",
    171: "Off-site replication for the worst case scenarios.",
    172: "How much lag can we tolerate in replication?",
    173: "Minimal. Data consistency is critical for payments.",
    174: "Let me document the backup and recovery procedures.",
    175: "I think we've covered all the major areas now.",
    176: "Quite comprehensive. Should we do a review of the key decisions?",
    177: "Good idea. Let me pull up my notes.",
    178: "Actually, let me ask some verification questions first.",
    179: "Go ahead. Better to clarify now than have gaps in the spec.",
    # Phase 8: Recall tests (180-200) - handled by RECALL_QUERIES
}


@dataclass
class TurnResult:
    turn: int
    role: str  # "user", "assistant", "plant", "recall"
    content: str
    E_mean: float
    context_items: int
    tokens: int
    response: str
    context_preview: str
    full_context: str
    e_scores: List[float] = field(default_factory=list)


@dataclass
class RecallResult:
    turn: int
    query: str
    expected: List[str]
    source_turn: int
    context_found: bool
    response_found: bool
    E_mean: float
    context_preview: str
    llm_response: str = ""


class FinalStressTest:
    """Final stress test for CAT CHAT - real conversation flow."""

    def __init__(self, provider: Optional[str] = None):
        self.db_path = DB_OUTPUT_DIR / f"stress_test_final_{int(time.time())}.db"
        print(f"Database: {self.db_path}")

        self.llm = get_llm_client(provider)
        print(f"LLM: {self.llm.config.name} ({self.llm.config.model})")
        print(f"URL: {self.llm.config.base_url}")

        self.capsule = SessionCapsule(db_path=self.db_path)
        self.session_id = self.capsule.create_session()

        # No system prompt - raw test
        self.system_prompt = ""

        # Realistic budget - 8K tokens for context
        self.budget = ModelBudgetDiscovery.from_context_window(
            context_window=8192,
            system_prompt=self.system_prompt,
            response_reserve_pct=0.25,
            model_id=self.llm.config.model
        )

        # Using full R formula: R = E / grad_S
        # Threshold R > 1.0 means: include items where E exceeds local uncertainty
        # This is PRINCIPLED, not arbitrary - derived from the Living Formula
        self.manager = AutoContextManager(
            db_path=self.db_path,
            session_id=self.session_id,
            budget=self.budget,
            embed_fn=self._embed,
            E_threshold=1.0  # R > 1 = E exceeds grad_S (principled threshold)
        )
        self.manager.capsule = self.capsule

        self.turn_log: List[TurnResult] = []
        self.recall_results: List[RecallResult] = []

    def _embed(self, text: str) -> np.ndarray:
        import requests
        resp = requests.post(
            f"{self.llm.config.base_url}/v1/embeddings",
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

    def _execute_turn(self, turn: int, content: str) -> TurnResult:
        """Execute a single turn with REAL LLM."""
        result = self.manager.respond_catalytic(
            query=content,
            llm_generate=self._llm_generate,
            system_prompt=self.system_prompt
        )

        prep = result.prepare_result
        context_text = prep.get_context_text() if prep else ""
        working_set = prep.working_set if prep else []

        # Get individual E-scores for retrieved items
        e_scores = []
        if prep and hasattr(prep, 'working_set'):
            for item in prep.working_set:
                if hasattr(item, 'E') or (isinstance(item, dict) and 'E' in item):
                    e = item.E if hasattr(item, 'E') else item.get('E', 0)
                    e_scores.append(e)

        turn_result = TurnResult(
            turn=turn,
            role="user",
            content=content,  # Full content
            E_mean=result.E_mean,
            context_items=len(working_set),
            tokens=result.tokens_in_context,
            response=result.response,  # Full response
            context_preview=context_text,  # Full context
            full_context=context_text,
            e_scores=e_scores,  # Individual E-scores
        )

        self.turn_log.append(turn_result)
        return turn_result

    def _test_recall(self, turn: int, query_data: dict) -> RecallResult:
        """Test recall and check if context contains expected keywords."""
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
        response_text = result.response

        # Check if expected keywords are in CONTEXT (system retrieval)
        context_lower = context_text.lower()
        context_found = any(kw.lower() in context_lower for kw in expected)

        # Check if expected keywords are in RESPONSE (LLM output)
        response_lower = response_text.lower()
        response_found = any(kw.lower() in response_lower for kw in expected)

        recall_result = RecallResult(
            turn=turn,
            query=query,  # Full query
            expected=expected,
            source_turn=source_turn,
            context_found=context_found,
            response_found=response_found,
            E_mean=result.E_mean,
            context_preview=context_text,  # Full context
            llm_response=response_text,  # Store the actual response
        )

        self.recall_results.append(recall_result)
        return recall_result

    def run(self):
        """Run the full 200-turn stress test."""
        print("\n" + "=" * 70)
        print("CAT CHAT FINAL STRESS TEST - HARD MODE")
        print("=" * 70)
        print("Scenario: Software Architecture Session (200 turns)")
        print("Testing: Semantic retrieval with no keyword overlap")
        print(f"Planted Facts: {len(PLANTED_FACTS)}")
        print(f"Recall Queries: {len(RECALL_QUERIES)}")
        print()

        start_time = time.time()

        # Detailed log file
        log_file = self.db_path.with_suffix('.log')
        with open(log_file, 'w', encoding='utf-8') as log:
            log.write("CAT CHAT STRESS TEST LOG\n")
            log.write("=" * 80 + "\n")
            log.write(f"Database: {self.db_path}\n")
            log.write(f"Planted Facts: {len(PLANTED_FACTS)}\n")
            log.write(f"Recall Queries: {len(RECALL_QUERIES)}\n")
            log.write("=" * 80 + "\n\n")
            log.flush()

            # Execute all 200 turns
            for turn in range(1, 201):
                if turn in PLANTED_FACTS:
                    # Plant a fact
                    fact = PLANTED_FACTS[turn]
                    result = self._execute_turn(turn, fact["content"])
                    print(f"Turn {turn:3d} [PLANT] {fact['content'][:60]}...")

                    log.write(f"\n--- T{turn:3d} [PLANT] E={result.E_mean:.3f} ---\n")
                    log.write(f"IN:  {fact['content'][:120]}...\n" if len(fact['content']) > 120 else f"IN:  {fact['content']}\n")
                    log.write(f"OUT: {result.response[:120]}...\n" if len(result.response) > 120 else f"OUT: {result.response}\n")
                    if result.context_items > 0:
                        ctx = result.full_context[:200] + "..." if len(result.full_context) > 200 else result.full_context
                        log.write(f"REHYDRATED ({result.context_items} items): {ctx}\n")
                    log.flush()

                elif turn in RECALL_QUERIES:
                    # Test recall
                    query_data = RECALL_QUERIES[turn]
                    recall = self._test_recall(turn, query_data)
                    status = "PASS" if recall.context_found else "FAIL"
                    llm_status = "+" if recall.response_found else "-"
                    print(f"Turn {turn:3d} [RECALL] {status}{llm_status} E={recall.E_mean:.3f} | {recall.query[:40]}...")

                    log.write(f"\n--- T{turn:3d} [RECALL] {status} E={recall.E_mean:.3f} ---\n")
                    log.write(f"IN:  {recall.query}\n")
                    log.write(f"OUT: {recall.llm_response[:150]}...\n" if len(recall.llm_response) > 150 else f"OUT: {recall.llm_response}\n")
                    ctx_short = recall.context_preview[:300] + "..." if len(recall.context_preview) > 300 else recall.context_preview
                    log.write(f"REHYDRATED: {ctx_short}\n")
                    log.write(f"EXPECTED: {recall.expected} (from T{recall.source_turn}) | VERDICT: CTX={'OK' if recall.context_found else 'FAIL'} LLM={'OK' if recall.response_found else 'FAIL'}\n")
                    log.flush()

                    if not recall.context_found:
                        print(f"         Expected: {recall.expected} from turn {recall.source_turn}")

                elif turn in CONVERSATION_FLOW:
                    # Natural conversation
                    content = CONVERSATION_FLOW[turn]
                    result = self._execute_turn(turn, content)
                    if turn % 20 == 0:
                        print(f"Turn {turn:3d} [CHAT]  {content[:60]}...")

                    log.write(f"\n--- T{turn:3d} [CHAT] E={result.E_mean:.3f} ---\n")
                    log.write(f"IN:  {content[:120]}...\n" if len(content) > 120 else f"IN:  {content}\n")
                    log.write(f"OUT: {result.response[:120]}...\n" if len(result.response) > 120 else f"OUT: {result.response}\n")
                    if result.context_items > 0:
                        ctx = result.full_context[:200] + "..." if len(result.full_context) > 200 else result.full_context
                        log.write(f"REHYDRATED ({result.context_items} items): {ctx}\n")
                    log.flush()

                else:
                    # Filler for any gaps
                    content = f"Let's continue discussing the architecture."
                    result = self._execute_turn(turn, content)
                    log.write(f"\n--- T{turn:3d} [FILL] E={result.E_mean:.3f} ---\n")
                    log.write(f"IN:  {content}\n")
                    log.write(f"OUT: {result.response[:120]}...\n" if len(result.response) > 120 else f"OUT: {result.response}\n")
                    if result.context_items > 0:
                        ctx = result.full_context[:200] + "..." if len(result.full_context) > 200 else result.full_context
                        log.write(f"REHYDRATED ({result.context_items} items): {ctx}\n")
                    log.flush()

            log.write("\n" + "=" * 80 + "\n")
            log.write("END OF LOG\n")

        print(f"\nDetailed log: {log_file}")
        duration = time.time() - start_time
        self._print_report(duration)

    def _print_report(self, duration: float):
        """Print comprehensive test report."""
        print("\n" + "=" * 70)
        print("STRESS TEST REPORT")
        print("=" * 70)

        # Recall statistics
        total_recalls = len(self.recall_results)
        context_hits = sum(1 for r in self.recall_results if r.context_found)
        response_hits = sum(1 for r in self.recall_results if r.response_found)
        full_hits = sum(1 for r in self.recall_results if r.context_found and r.response_found)

        print(f"\nDuration: {duration:.1f}s")
        print(f"Total Turns: {len(self.turn_log)}")
        print(f"Planted Facts: {len(PLANTED_FACTS)}")
        print(f"Recall Queries: {total_recalls}")

        print(f"\n--- RETRIEVAL ACCURACY (System) ---")
        print(f"Context Found: {context_hits}/{total_recalls} ({context_hits/total_recalls*100:.1f}%)")

        print(f"\n--- LLM ACCURACY ---")
        print(f"Response Correct: {response_hits}/{total_recalls} ({response_hits/total_recalls*100:.1f}%)")

        print(f"\n--- END-TO-END ---")
        print(f"Full Pipeline Pass: {full_hits}/{total_recalls} ({full_hits/total_recalls*100:.1f}%)")

        # E-score statistics
        e_scores = [t.E_mean for t in self.turn_log if t.E_mean > 0]
        if e_scores:
            print(f"\n--- E-SCORE STATISTICS ---")
            print(f"Mean E: {np.mean(e_scores):.3f}")
            print(f"Min E:  {np.min(e_scores):.3f}")
            print(f"Max E:  {np.max(e_scores):.3f}")

        # Recall detail
        print(f"\n--- RECALL DETAIL ---")
        for r in self.recall_results:
            ctx_status = "CTX_OK" if r.context_found else "CTX_FAIL"
            llm_status = "LLM_OK" if r.response_found else "LLM_FAIL"
            print(f"Turn {r.turn}: {ctx_status} {llm_status} | Source: T{r.source_turn} | E={r.E_mean:.3f}")

        # Success criteria
        print(f"\n--- SUCCESS CRITERIA ---")
        recall_rate = context_hits / total_recalls if total_recalls > 0 else 0
        if recall_rate >= 0.75:
            print(f"[PASS] Recall rate {recall_rate*100:.1f}% >= 75%")
        else:
            print(f"[FAIL] Recall rate {recall_rate*100:.1f}% < 75%")

        print(f"\nDatabase: {self.db_path}")

        # Save detailed results
        results_file = self.db_path.with_suffix('.json')
        results = {
            "duration": duration,
            "total_turns": len(self.turn_log),
            "recall_accuracy": context_hits / total_recalls if total_recalls else 0,
            "llm_accuracy": response_hits / total_recalls if total_recalls else 0,
            "recalls": [
                {
                    "turn": r.turn,
                    "source_turn": r.source_turn,
                    "context_found": r.context_found,
                    "response_found": r.response_found,
                    "E_mean": r.E_mean,
                    "expected": r.expected,
                }
                for r in self.recall_results
            ],
        }
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results: {results_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CAT CHAT Final Stress Test")
    parser.add_argument("--provider", default=None, help="LLM provider from config")
    args = parser.parse_args()

    test = FinalStressTest(provider=args.provider)
    test.run()


if __name__ == "__main__":
    main()
