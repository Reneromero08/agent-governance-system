"""
Benchmark Scenarios - Phase I.2
===============================

Deterministic benchmark scenarios for compression testing.

Each scenario defines:
- Fixed conversation script
- Planted facts at known positions
- Expected compression characteristics
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import random


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PlantedFact:
    """
    A fact planted at a specific turn for recall testing.

    Planted facts have known content and are used to measure
    whether the auto-context system retrieves relevant history.
    """
    turn_index: int           # Turn where fact was introduced
    fact_id: str              # Unique identifier
    content: str              # The actual fact content
    keywords: List[str]       # Keywords that should trigger recall
    context_required: bool = True  # Whether full context needed for recall


@dataclass
class ConversationTurn:
    """
    A single turn in a benchmark conversation.
    """
    turn_index: int
    user_query: str
    expected_topics: List[str] = field(default_factory=list)
    planted_fact: Optional[PlantedFact] = None
    token_estimate: int = 0  # Estimated tokens for this turn

    def __post_init__(self):
        if self.token_estimate == 0:
            # Rough estimate: 4 chars per token
            self.token_estimate = len(self.user_query) // 4 + 100  # +100 for response


@dataclass
class BenchmarkScenario:
    """
    Complete benchmark scenario definition.

    Scenarios are deterministic - same seed produces same conversation.
    """
    name: str
    description: str
    turns: List[ConversationTurn]
    planted_facts: List[PlantedFact] = field(default_factory=list)
    seed: int = 42
    target_compression_ratio: float = 5.0
    target_recall_rate: float = 0.9

    @property
    def total_turns(self) -> int:
        return len(self.turns)

    @property
    def total_planted_facts(self) -> int:
        return len(self.planted_facts)

    @property
    def estimated_tokens(self) -> int:
        return sum(t.token_estimate for t in self.turns)

    def get_turn(self, index: int) -> Optional[ConversationTurn]:
        """Get turn by index (1-based)."""
        for turn in self.turns:
            if turn.turn_index == index:
                return turn
        return None

    def get_facts_before(self, turn_index: int) -> List[PlantedFact]:
        """Get all planted facts before a given turn."""
        return [f for f in self.planted_facts if f.turn_index < turn_index]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "total_turns": self.total_turns,
            "total_planted_facts": self.total_planted_facts,
            "estimated_tokens": self.estimated_tokens,
            "seed": self.seed,
            "target_compression_ratio": self.target_compression_ratio,
            "target_recall_rate": self.target_recall_rate,
        }


# =============================================================================
# Scenario Definitions
# =============================================================================

def _create_short_conversation() -> BenchmarkScenario:
    """Create a short 10-turn conversation for quick sanity checks."""
    turns = [
        ConversationTurn(1, "What is the capital of France?", ["geography", "France"]),
        ConversationTurn(2, "Tell me about the Eiffel Tower.", ["landmarks", "Paris"]),
        ConversationTurn(3, "What year was it built?", ["history", "construction"]),
        ConversationTurn(4, "How tall is it?", ["measurements", "height"]),
        ConversationTurn(5, "What other famous landmarks are in Paris?", ["landmarks", "Paris"]),
        ConversationTurn(6, "Tell me about the Louvre.", ["museums", "art"]),
        ConversationTurn(7, "What is the most famous painting there?", ["art", "Mona Lisa"]),
        ConversationTurn(8, "Who painted the Mona Lisa?", ["artists", "Leonardo"]),
        ConversationTurn(9, "What other works did Leonardo create?", ["art", "inventions"]),
        ConversationTurn(10, "Going back to Paris, what's the best time to visit?", ["travel", "seasons"]),
    ]

    planted_facts = [
        PlantedFact(
            turn_index=3,
            fact_id="eiffel_year",
            content="The Eiffel Tower was completed in 1889 for the World's Fair.",
            keywords=["1889", "World's Fair", "completion"],
        ),
        PlantedFact(
            turn_index=8,
            fact_id="leonardo",
            content="Leonardo da Vinci painted the Mona Lisa between 1503 and 1519.",
            keywords=["Leonardo", "1503", "1519"],
        ),
    ]

    turns[2].planted_fact = planted_facts[0]
    turns[7].planted_fact = planted_facts[1]

    return BenchmarkScenario(
        name="short_conversation",
        description="10-turn conversation about Paris landmarks - quick sanity check",
        turns=turns,
        planted_facts=planted_facts,
        seed=42,
        target_compression_ratio=3.0,
        target_recall_rate=0.9,
    )


def _create_medium_conversation() -> BenchmarkScenario:
    """Create a 30-turn conversation for typical usage testing."""
    turns = []
    planted_facts = []

    topics = [
        ("machine learning", "neural networks", "training data"),
        ("data preprocessing", "feature engineering", "normalization"),
        ("model evaluation", "metrics", "overfitting"),
        ("deployment", "scaling", "monitoring"),
        ("debugging", "error analysis", "improvements"),
    ]

    fact_positions = [5, 12, 18, 25, 30]
    fact_contents = [
        ("ml_basics", "Neural networks learn through backpropagation with gradient descent, adjusting weights to minimize loss."),
        ("preprocessing", "Feature scaling using StandardScaler normalizes data to mean=0, std=1 for better convergence."),
        ("evaluation", "Cross-validation with k=5 folds provides robust estimate of model generalization."),
        ("deployment", "Containerization with Docker ensures consistent runtime environment across deployments."),
        ("debugging", "Gradient explosion can be mitigated using gradient clipping with max_norm=1.0."),
    ]

    turn_idx = 1
    for topic_idx, (main, sub1, sub2) in enumerate(topics):
        for i in range(6):
            if i == 0:
                query = f"Tell me about {main}."
            elif i == 1:
                query = f"How does {sub1} relate to this?"
            elif i == 2:
                query = f"What about {sub2}?"
            elif i == 3:
                query = f"Can you give an example of {sub1}?"
            elif i == 4:
                query = f"What are common mistakes with {main}?"
            else:
                query = f"How do I know if I'm doing {main} correctly?"

            turn = ConversationTurn(turn_idx, query, [main, sub1, sub2])

            if turn_idx in fact_positions:
                fact_idx = fact_positions.index(turn_idx)
                fact_id, fact_content = fact_contents[fact_idx]
                planted_fact = PlantedFact(
                    turn_index=turn_idx,
                    fact_id=fact_id,
                    content=fact_content,
                    keywords=fact_content.split()[:5],
                )
                planted_facts.append(planted_fact)
                turn.planted_fact = planted_fact

            turns.append(turn)
            turn_idx += 1

    return BenchmarkScenario(
        name="medium_conversation",
        description="30-turn ML tutorial conversation - typical usage",
        turns=turns,
        planted_facts=planted_facts,
        seed=42,
        target_compression_ratio=5.0,
        target_recall_rate=0.85,
    )


def _create_long_conversation() -> BenchmarkScenario:
    """Create a 100-turn conversation for memory stress testing."""
    turns = []
    planted_facts = []

    # Generate diverse topics
    domains = ["web development", "databases", "security", "networking", "cloud computing",
               "devops", "testing", "architecture", "performance", "debugging"]

    fact_positions = [7, 15, 22, 30, 38, 45, 52, 60, 68, 75, 83, 90, 97]

    turn_idx = 1
    random.seed(42)

    for i in range(100):
        domain = domains[i % len(domains)]
        subtopic = random.choice(["basics", "advanced", "best practices", "common issues", "tools"])

        query = f"What should I know about {domain} {subtopic}?"
        turn = ConversationTurn(turn_idx, query, [domain, subtopic])

        if turn_idx in fact_positions:
            fact = PlantedFact(
                turn_index=turn_idx,
                fact_id=f"fact_{turn_idx}",
                content=f"Important fact about {domain}: Always consider {subtopic} when designing systems.",
                keywords=[domain, subtopic, "systems"],
            )
            planted_facts.append(fact)
            turn.planted_fact = fact

        turns.append(turn)
        turn_idx += 1

    return BenchmarkScenario(
        name="long_conversation",
        description="100-turn diverse tech conversation - memory stress test",
        turns=turns,
        planted_facts=planted_facts,
        seed=42,
        target_compression_ratio=7.0,
        target_recall_rate=0.80,
    )


def _create_software_architecture() -> BenchmarkScenario:
    """
    Create a 63-turn software architecture conversation.

    Based on the stress test scenario with planted facts at specific turns.
    """
    turns = []
    planted_facts = []

    # Architecture discussion flow
    phases = [
        ("requirements", [
            "What are the main requirements for our authentication system?",
            "Should we support multiple identity providers?",
            "What about rate limiting requirements?",
        ]),
        ("authentication", [
            "Let's discuss JWT implementation. What algorithm should we use?",
            "How should we handle token refresh?",
            "What's the best way to store refresh tokens?",
            "Should we implement token blacklisting?",
        ]),
        ("api_design", [
            "Now let's design the API endpoints.",
            "How should we structure the route handlers?",
            "What middleware do we need?",
            "How do we handle API versioning?",
        ]),
        ("database", [
            "Let's talk about the database schema.",
            "What indexes do we need for the users table?",
            "How should we handle migrations?",
            "What about connection pooling?",
        ]),
        ("security", [
            "What security measures should we implement?",
            "How do we prevent SQL injection?",
            "What about XSS prevention?",
            "How should we handle CORS?",
        ]),
        ("performance", [
            "Let's discuss performance optimization.",
            "Where should we add caching?",
            "How do we handle concurrent requests?",
            "What metrics should we track?",
        ]),
        ("testing", [
            "What's our testing strategy?",
            "How do we test authentication flows?",
            "What about integration tests?",
            "How do we handle test data?",
        ]),
        ("deployment", [
            "Let's plan the deployment.",
            "Should we use containers?",
            "How do we handle secrets?",
            "What's the rollback strategy?",
        ]),
        ("monitoring", [
            "What monitoring do we need?",
            "How do we handle alerts?",
            "What dashboards should we create?",
        ]),
    ]

    # Planted facts at strategic points
    fact_data = [
        (3, "jwt_algorithm", "We decided to use RS256 for JWT signing because it allows public key verification without exposing the private key."),
        (7, "rate_limiting", "The rate limit is set to 100 requests per minute per IP, with a burst allowance of 20 additional requests."),
        (12, "token_storage", "Refresh tokens are stored in HttpOnly cookies with Secure flag, never in localStorage."),
        (18, "api_version", "API versioning uses URL path prefix /api/v1/ for major versions and Accept header for minor versions."),
        (25, "db_indexes", "Composite index on (email, tenant_id) for the users table with a partial index for active users only."),
        (31, "sql_injection", "Parameterized queries are mandatory; the ORM's query builder handles escaping automatically."),
        (38, "caching_strategy", "Redis is used for session caching with TTL of 15 minutes, extended on each request."),
        (42, "concurrency", "Optimistic locking with version field prevents race conditions in user updates."),
        (48, "test_strategy", "Property-based testing with Hypothesis for auth edge cases, integration tests with testcontainers."),
        (52, "secrets_management", "HashiCorp Vault for production secrets, AWS Secrets Manager as backup, rotated every 30 days."),
        (58, "monitoring", "OpenTelemetry for distributed tracing, Prometheus for metrics, with 99.9% SLO on auth endpoints."),
        (63, "rollback", "Blue-green deployment with automatic rollback if error rate exceeds 0.1% in first 5 minutes."),
    ]

    turn_idx = 1
    for phase_name, questions in phases:
        for question in questions:
            turn = ConversationTurn(
                turn_index=turn_idx,
                user_query=question,
                expected_topics=[phase_name],
            )

            # Check if this turn has a planted fact
            for fact_turn, fact_id, fact_content in fact_data:
                if fact_turn == turn_idx:
                    planted_fact = PlantedFact(
                        turn_index=turn_idx,
                        fact_id=fact_id,
                        content=fact_content,
                        keywords=fact_content.split()[:5],
                    )
                    planted_facts.append(planted_fact)
                    turn.planted_fact = planted_fact
                    break

            turns.append(turn)
            turn_idx += 1

            if turn_idx > 63:
                break
        if turn_idx > 63:
            break

    return BenchmarkScenario(
        name="software_architecture",
        description="63-turn architecture discussion - real domain scenario",
        turns=turns,
        planted_facts=planted_facts,
        seed=42,
        target_compression_ratio=6.0,
        target_recall_rate=0.85,
    )


def _create_dense_technical() -> BenchmarkScenario:
    """Create a 50-turn dense technical conversation with high compression potential."""
    turns = []
    planted_facts = []

    # Dense technical content with repeated concepts
    concepts = [
        ("cryptography", "AES-256", "encryption", "key management"),
        ("hashing", "SHA-256", "collision resistance", "rainbow tables"),
        ("TLS", "handshake", "certificates", "cipher suites"),
        ("PKI", "certificate authority", "chain of trust", "revocation"),
        ("key exchange", "Diffie-Hellman", "forward secrecy", "ephemeral keys"),
    ]

    fact_data = [
        (5, "aes_mode", "AES-256-GCM is preferred over CBC because it provides authenticated encryption."),
        (12, "sha_comparison", "SHA-256 produces a 256-bit hash; SHA-3 is more resistant to length extension attacks."),
        (20, "tls_version", "TLS 1.3 removes RSA key exchange and mandates forward secrecy with ECDHE."),
        (28, "cert_pinning", "Certificate pinning prevents MITM attacks but requires careful update strategy."),
        (35, "key_rotation", "Ephemeral keys are rotated per-session; long-term keys rotate annually."),
        (42, "zero_knowledge", "Zero-knowledge proofs allow verification without revealing the secret value."),
        (50, "quantum_safe", "NIST post-quantum standards include CRYSTALS-Kyber for key encapsulation."),
    ]

    turn_idx = 1
    for round_idx in range(10):
        for concept, detail1, detail2, detail3 in concepts:
            query = f"Explain {concept}, specifically {detail1} and {detail2}."
            turn = ConversationTurn(
                turn_index=turn_idx,
                user_query=query,
                expected_topics=[concept, detail1, detail2],
            )

            for fact_turn, fact_id, fact_content in fact_data:
                if fact_turn == turn_idx:
                    planted_fact = PlantedFact(
                        turn_index=turn_idx,
                        fact_id=fact_id,
                        content=fact_content,
                        keywords=fact_content.split()[:5],
                    )
                    planted_facts.append(planted_fact)
                    turn.planted_fact = planted_fact
                    break

            turns.append(turn)
            turn_idx += 1

            if turn_idx > 50:
                break
        if turn_idx > 50:
            break

    return BenchmarkScenario(
        name="dense_technical",
        description="50-turn cryptography discussion - high compression potential",
        turns=turns,
        planted_facts=planted_facts,
        seed=42,
        target_compression_ratio=8.0,
        target_recall_rate=0.90,
    )


# =============================================================================
# Scenario Registry
# =============================================================================

SHORT_CONVERSATION = _create_short_conversation()
MEDIUM_CONVERSATION = _create_medium_conversation()
LONG_CONVERSATION = _create_long_conversation()
SOFTWARE_ARCHITECTURE = _create_software_architecture()
DENSE_TECHNICAL = _create_dense_technical()

_SCENARIOS = {
    "short_conversation": SHORT_CONVERSATION,
    "medium_conversation": MEDIUM_CONVERSATION,
    "long_conversation": LONG_CONVERSATION,
    "software_architecture": SOFTWARE_ARCHITECTURE,
    "dense_technical": DENSE_TECHNICAL,
}


def get_scenario(name: str) -> BenchmarkScenario:
    """Get a scenario by name."""
    if name not in _SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(_SCENARIOS.keys())}")
    return _SCENARIOS[name]


def list_scenarios() -> List[str]:
    """List all available scenario names."""
    return list(_SCENARIOS.keys())


__all__ = [
    "BenchmarkScenario",
    "ConversationTurn",
    "PlantedFact",
    "get_scenario",
    "list_scenarios",
    "SHORT_CONVERSATION",
    "MEDIUM_CONVERSATION",
    "LONG_CONVERSATION",
    "SOFTWARE_ARCHITECTURE",
    "DENSE_TECHNICAL",
]
