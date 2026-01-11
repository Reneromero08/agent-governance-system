# R-Gate Implementation Guide for Agents

**Version:** 1.0.0
**Status:** IMPLEMENTATION SPEC
**Date:** 2026-01-11
**Prerequisite:** Q17 (Governance Gating) - ANSWERED

---

## Executive Summary

This guide specifies how to implement R-gating in agent systems. R-gating uses the Living Formula's core insight—that local agreement among independent observations reveals truth—to gate agent actions by signal quality rather than volume.

**Core principle:** An agent should not take high-stakes actions when R < threshold, regardless of how much data supports the action.

---

## Table of Contents

1. [Theoretical Foundation](#1-theoretical-foundation)
2. [What Counts as "Observations"](#2-what-counts-as-observations)
3. [Computing R for Agents](#3-computing-r-for-agents)
4. [Threshold Tiers](#4-threshold-tiers)
5. [Implementation Architecture](#5-implementation-architecture)
6. [Code Specifications](#6-code-specifications)
7. [Integration Points](#7-integration-points)
8. [Testing Strategy](#8-testing-strategy)
9. [Failure Modes and Mitigations](#9-failure-modes-and-mitigations)
10. [Rollout Plan](#10-rollout-plan)

---

## 1. Theoretical Foundation

### 1.1 The Living Formula

The formula measures truth through agreement:

```
R = E / σ
```

Where:
- **E** = Evidence/Essence (agreement among observations)
- **σ** = Standard deviation (dispersion of observations)
- **R** = Resonance (signal quality, dimensionless)

### 1.2 Why R Works for Governance

Three converging findings justify R-gating (from Q12, Q14, Q15):

| Finding | Source | Implication |
|---------|--------|-------------|
| Truth crystallizes suddenly | Q12 (Phase transitions) | Binary gates are appropriate |
| Gate is a valid subobject classifier | Q14 (Category theory) | Mathematically well-founded |
| R is intensive (ignores volume) | Q15 (Bayesian inference) | Prevents gaming via data accumulation |

### 1.3 Key Property: Volume Resistance

Standard Bayesian agents become confident in noisy channels given enough data:
```
Posterior Precision ∝ N/σ²  (grows with sample size N)
```

R-gated agents ignore noisy channels regardless of volume:
```
R = 1/σ  (independent of N)
```

**This is the critical safety property.** An agent cannot bypass the gate by collecting more evidence from a bad source.

---

## 2. What Counts as "Observations"

For the formula to work, observations must be:
1. **Independent** (not derived from each other)
2. **Addressing the same question** (comparable)
3. **Representable** (can be embedded or compared)

### 2.1 Valid Observation Sources

| Source Type | Example | Independence Level |
|-------------|---------|-------------------|
| **Multi-model** | GPT-4 + Claude + Gemini on same query | High |
| **Self-consistency** | Same model, N samples with temperature>0 | Medium |
| **Cross-tool** | Web search + file read + database query | High |
| **Temporal** | Cache from yesterday + fresh query today | Medium |
| **Multi-modal** | Text description + image analysis | High |
| **Human-in-loop** | Model output + human verification | Very High |

### 2.2 Invalid Observation Sources (Echo Chambers)

| Source Type | Why Invalid |
|-------------|-------------|
| Same model, temperature=0, N times | Deterministic = 1 observation repeated |
| Multiple queries to same API | May hit same cache/index |
| Derived computations | f(x), g(f(x)), h(g(f(x))) are correlated |
| Consensus from trained-together models | May share biases |

### 2.3 Observation Requirements by Action Tier

| Tier | Min Observations | Independence Requirement |
|------|-----------------|-------------------------|
| T0 (read) | 0 | None |
| T1 (reversible) | 2 | Any |
| T2 (persistent) | 3 | At least 2 independent sources |
| T3 (critical) | 5 | At least 3 independent sources + human |

---

## 3. Computing R for Agents

### 3.1 The Standard Algorithm

```python
def compute_r(observations: List[str], embed_fn: Callable) -> float:
    """
    Compute R from a list of observations.

    Args:
        observations: List of observation strings (e.g., model outputs)
        embed_fn: Function that maps string -> embedding vector

    Returns:
        R value (float). Higher = more agreement.
    """
    if len(observations) < 2:
        return 0.0  # Cannot compute agreement with <2 observations

    # Step 1: Embed all observations
    embeddings = [embed_fn(obs) for obs in observations]
    embeddings = [e / np.linalg.norm(e) for e in embeddings]  # Normalize

    # Step 2: Compute pairwise similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j])
            similarities.append(sim)

    # Step 3: Compute E (mean agreement) and σ (dispersion)
    E = np.mean(similarities)
    sigma = np.std(similarities)

    # Step 4: Compute R with numerical stability
    R = E / (sigma + 1e-6)

    return R
```

### 3.2 Alternative: Centroid Method

More efficient for many observations:

```python
def compute_r_centroid(observations: List[str], embed_fn: Callable) -> float:
    """
    Compute R using centroid distance method.

    More efficient: O(N) instead of O(N²) for pairwise.
    """
    if len(observations) < 2:
        return 0.0

    embeddings = np.array([embed_fn(obs) for obs in observations])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Centroid = mean embedding
    centroid = np.mean(embeddings, axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    # Distances from centroid
    distances = [1 - np.dot(emb, centroid) for emb in embeddings]

    # E = agreement = 1 - mean_distance
    E = 1 - np.mean(distances)
    sigma = np.std(distances)

    R = E / (sigma + 1e-6)
    return R
```

### 3.3 Structured Output Comparison

For structured outputs (JSON, code), use semantic + structural similarity:

```python
def compute_r_structured(outputs: List[dict], embed_fn: Callable) -> float:
    """
    Compute R for structured outputs.

    Combines:
    - Semantic similarity (embeddings)
    - Structural similarity (key overlap, value match)
    """
    if len(outputs) < 2:
        return 0.0

    # Semantic R
    strings = [json.dumps(o, sort_keys=True) for o in outputs]
    R_semantic = compute_r(strings, embed_fn)

    # Structural R
    structural_sims = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            sim = structural_similarity(outputs[i], outputs[j])
            structural_sims.append(sim)

    E_struct = np.mean(structural_sims)
    sigma_struct = np.std(structural_sims)
    R_structural = E_struct / (sigma_struct + 1e-6)

    # Combined R (geometric mean)
    R = np.sqrt(R_semantic * R_structural)
    return R


def structural_similarity(a: dict, b: dict) -> float:
    """Jaccard-like similarity for nested dicts."""
    keys_a = set(flatten_keys(a))
    keys_b = set(flatten_keys(b))

    intersection = len(keys_a & keys_b)
    union = len(keys_a | keys_b)

    if union == 0:
        return 1.0

    return intersection / union
```

### 3.4 Code Output Comparison

For code generation, use AST similarity:

```python
def compute_r_code(code_outputs: List[str], language: str = "python") -> float:
    """
    Compute R for code outputs using AST comparison.

    Ignores surface differences (whitespace, variable names).
    """
    if len(code_outputs) < 2:
        return 0.0

    # Parse to AST
    asts = []
    for code in code_outputs:
        try:
            ast_tree = parse_to_ast(code, language)
            asts.append(ast_tree)
        except SyntaxError:
            asts.append(None)

    # Filter valid parses
    valid_asts = [(i, ast) for i, ast in enumerate(asts) if ast is not None]

    if len(valid_asts) < 2:
        # Fall back to text similarity
        return compute_r(code_outputs, embed_fn)

    # Compute AST similarity
    similarities = []
    for (i, ast_i), (j, ast_j) in combinations(valid_asts, 2):
        sim = ast_similarity(ast_i, ast_j)
        similarities.append(sim)

    E = np.mean(similarities)
    sigma = np.std(similarities)

    # Bonus for parse success rate
    parse_rate = len(valid_asts) / len(code_outputs)

    R = (E / (sigma + 1e-6)) * parse_rate
    return R
```

---

## 4. Threshold Tiers

### 4.1 The Four Tiers

| Tier | Action Type | Threshold | Failure Mode | Example Actions |
|------|-------------|-----------|--------------|-----------------|
| **T0** | Read-only | None | Always allowed | Search, read file, list directory |
| **T1** | Reversible | R > 0.5 | Warn, proceed | Stage changes, create draft, propose edit |
| **T2** | Persistent | R > 0.8 | Block, escalate | Commit, write file, send message |
| **T3** | Critical | R > 1.0 | Block, require human | Deploy, delete, modify canon |

### 4.2 Tier Classification Logic

```python
class ActionTier(Enum):
    T0_READ = 0
    T1_REVERSIBLE = 1
    T2_PERSISTENT = 2
    T3_CRITICAL = 3

def classify_action_tier(action: str, target: str) -> ActionTier:
    """
    Classify action into tier based on action type and target.
    """
    # T3: Critical actions
    T3_ACTIONS = {"deploy", "delete", "drop", "truncate", "force_push"}
    T3_TARGETS = {"canon", "production", "main", "master", "invariant"}

    if action.lower() in T3_ACTIONS:
        return ActionTier.T3_CRITICAL
    if any(t in target.lower() for t in T3_TARGETS):
        return ActionTier.T3_CRITICAL

    # T2: Persistent actions
    T2_ACTIONS = {"write", "commit", "send", "post", "create", "update", "insert"}

    if action.lower() in T2_ACTIONS:
        return ActionTier.T2_PERSISTENT

    # T1: Reversible actions
    T1_ACTIONS = {"stage", "draft", "propose", "preview", "plan"}

    if action.lower() in T1_ACTIONS:
        return ActionTier.T1_REVERSIBLE

    # T0: Everything else (read-only)
    return ActionTier.T0_READ
```

### 4.3 Threshold Lookup

```python
THRESHOLDS = {
    ActionTier.T0_READ: 0.0,
    ActionTier.T1_REVERSIBLE: 0.5,
    ActionTier.T2_PERSISTENT: 0.8,
    ActionTier.T3_CRITICAL: 1.0,
}

def get_threshold(tier: ActionTier) -> float:
    return THRESHOLDS[tier]
```

---

## 5. Implementation Architecture

### 5.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AGENT R-GATE SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    OBSERVATION COLLECTOR                         │   │
│  │                                                                  │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │   │
│  │  │ Model A  │  │ Model B  │  │ Tool X   │  │ Cache    │        │   │
│  │  │ (Claude) │  │ (GPT-4)  │  │ (Search) │  │ (Redis)  │        │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │   │
│  │       │              │              │              │             │   │
│  │       └──────────────┴──────────────┴──────────────┘             │   │
│  │                              │                                    │   │
│  │                              ▼                                    │   │
│  │                    observations: List[str]                        │   │
│  └──────────────────────────────┬───────────────────────────────────┘   │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      R COMPUTATION ENGINE                        │   │
│  │                                                                  │   │
│  │    observations ──► embed_fn ──► similarities ──► R = E/σ       │   │
│  │                                                                  │   │
│  │    Outputs:                                                      │   │
│  │    - R value (float)                                            │   │
│  │    - confidence interval                                         │   │
│  │    - observation_count                                           │   │
│  │    - independence_score                                          │   │
│  └──────────────────────────────┬───────────────────────────────────┘   │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        GATE DECISION                             │   │
│  │                                                                  │   │
│  │    action ──► classify_tier ──► get_threshold ──► compare       │   │
│  │                                                                  │   │
│  │    if R >= threshold:                                           │   │
│  │        return GATE_OPEN                                         │   │
│  │    else:                                                        │   │
│  │        return GATE_CLOSED(escalation_path)                      │   │
│  └──────────────────────────────┬───────────────────────────────────┘   │
│                                 │                                        │
│              ┌──────────────────┴──────────────────┐                    │
│              ▼                                      ▼                    │
│  ┌───────────────────────┐           ┌───────────────────────────┐     │
│  │     GATE OPEN         │           │      GATE CLOSED          │     │
│  │                       │           │                           │     │
│  │  - Execute action     │           │  - Log R value            │     │
│  │  - Log R value        │           │  - Escalate to human      │     │
│  │  - Emit receipt       │           │  - Suggest alternatives   │     │
│  │                       │           │  - Enter degraded mode    │     │
│  └───────────────────────┘           └───────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Component Responsibilities

| Component | Responsibility | Interface |
|-----------|---------------|-----------|
| ObservationCollector | Gather observations from multiple sources | `collect(query, sources) -> List[str]` |
| EmbeddingEngine | Convert observations to vectors | `embed(text) -> np.ndarray` |
| RComputer | Compute R from embeddings | `compute_r(observations) -> RResult` |
| TierClassifier | Classify action into tier | `classify(action, target) -> ActionTier` |
| GateDecision | Compare R to threshold | `decide(R, tier) -> GateResult` |
| EscalationHandler | Handle gate closures | `escalate(context) -> EscalationResult` |
| AuditLogger | Log all gate decisions | `log(decision) -> None` |

---

## 6. Code Specifications

### 6.1 Core Data Structures

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Callable
import numpy as np

class ActionTier(Enum):
    T0_READ = 0
    T1_REVERSIBLE = 1
    T2_PERSISTENT = 2
    T3_CRITICAL = 3

class GateStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    DEGRADED = "degraded"  # Low R but user overrode

@dataclass
class Observation:
    """A single observation from a source."""
    content: str
    source: str
    timestamp: float
    embedding: Optional[np.ndarray] = None

@dataclass
class RResult:
    """Result of R computation."""
    R: float
    E: float  # Mean agreement
    sigma: float  # Dispersion
    n_observations: int
    confidence_interval: tuple[float, float]

@dataclass
class GateDecision:
    """Result of gate decision."""
    status: GateStatus
    R: float
    threshold: float
    tier: ActionTier
    action: str
    reason: str
    escalation_path: Optional[str] = None
```

### 6.2 Main RGate Class

```python
class RGate:
    """
    R-Gate for agent actions.

    Usage:
        gate = RGate(embed_fn=my_embed_function)

        # Check if action is allowed
        decision = gate.check_action(
            action="commit",
            target="main",
            observations=["output1", "output2", "output3"]
        )

        if decision.status == GateStatus.OPEN:
            execute_action()
        else:
            handle_escalation(decision)
    """

    THRESHOLDS = {
        ActionTier.T0_READ: 0.0,
        ActionTier.T1_REVERSIBLE: 0.5,
        ActionTier.T2_PERSISTENT: 0.8,
        ActionTier.T3_CRITICAL: 1.0,
    }

    MIN_OBSERVATIONS = {
        ActionTier.T0_READ: 0,
        ActionTier.T1_REVERSIBLE: 2,
        ActionTier.T2_PERSISTENT: 3,
        ActionTier.T3_CRITICAL: 5,
    }

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        thresholds: Optional[dict] = None,
        audit_logger: Optional[Callable] = None
    ):
        self.embed_fn = embed_fn
        self.thresholds = thresholds or self.THRESHOLDS
        self.audit_logger = audit_logger or self._default_logger

    def compute_r(self, observations: List[str]) -> RResult:
        """Compute R from observations."""
        if len(observations) < 2:
            return RResult(
                R=0.0, E=0.0, sigma=float('inf'),
                n_observations=len(observations),
                confidence_interval=(0.0, 0.0)
            )

        # Embed observations
        embeddings = []
        for obs in observations:
            emb = self.embed_fn(obs)
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            embeddings.append(emb)

        # Compute pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)

        # Compute statistics
        E = np.mean(similarities)
        sigma = np.std(similarities)
        R = E / (sigma + 1e-6)

        # Confidence interval (bootstrap)
        ci = self._bootstrap_ci(similarities)

        return RResult(
            R=R,
            E=E,
            sigma=sigma,
            n_observations=len(observations),
            confidence_interval=ci
        )

    def classify_tier(self, action: str, target: str) -> ActionTier:
        """Classify action into tier."""
        action_lower = action.lower()
        target_lower = target.lower()

        # T3: Critical
        if action_lower in {"deploy", "delete", "drop", "force_push", "reset_hard"}:
            return ActionTier.T3_CRITICAL
        if any(t in target_lower for t in {"canon", "production", "main", "master"}):
            return ActionTier.T3_CRITICAL

        # T2: Persistent
        if action_lower in {"write", "commit", "send", "post", "create", "update"}:
            return ActionTier.T2_PERSISTENT

        # T1: Reversible
        if action_lower in {"stage", "draft", "propose", "preview", "plan"}:
            return ActionTier.T1_REVERSIBLE

        # T0: Read-only
        return ActionTier.T0_READ

    def check_action(
        self,
        action: str,
        target: str,
        observations: List[str]
    ) -> GateDecision:
        """
        Check if action should be allowed.

        Returns GateDecision with status and escalation path if blocked.
        """
        tier = self.classify_tier(action, target)
        threshold = self.thresholds[tier]
        min_obs = self.MIN_OBSERVATIONS[tier]

        # Check minimum observations
        if len(observations) < min_obs:
            decision = GateDecision(
                status=GateStatus.CLOSED,
                R=0.0,
                threshold=threshold,
                tier=tier,
                action=action,
                reason=f"Insufficient observations: {len(observations)} < {min_obs}",
                escalation_path="Collect more independent observations"
            )
            self.audit_logger(decision)
            return decision

        # Compute R
        r_result = self.compute_r(observations)

        # Check threshold
        if r_result.R >= threshold:
            decision = GateDecision(
                status=GateStatus.OPEN,
                R=r_result.R,
                threshold=threshold,
                tier=tier,
                action=action,
                reason=f"R={r_result.R:.3f} >= threshold={threshold}"
            )
        else:
            decision = GateDecision(
                status=GateStatus.CLOSED,
                R=r_result.R,
                threshold=threshold,
                tier=tier,
                action=action,
                reason=f"R={r_result.R:.3f} < threshold={threshold}",
                escalation_path=self._get_escalation_path(tier, r_result)
            )

        self.audit_logger(decision)
        return decision

    def _bootstrap_ci(self, similarities: List[float], n_bootstrap: int = 1000) -> tuple:
        """Compute bootstrap confidence interval for R."""
        if len(similarities) < 3:
            return (0.0, float('inf'))

        R_samples = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(similarities, size=len(similarities), replace=True)
            E = np.mean(sample)
            sigma = np.std(sample)
            R = E / (sigma + 1e-6)
            R_samples.append(R)

        return (np.percentile(R_samples, 2.5), np.percentile(R_samples, 97.5))

    def _get_escalation_path(self, tier: ActionTier, r_result: RResult) -> str:
        """Determine escalation path based on tier and R."""
        if tier == ActionTier.T3_CRITICAL:
            return "Require human approval for critical action"
        elif tier == ActionTier.T2_PERSISTENT:
            if r_result.R < 0.5:
                return "R very low - gather more independent observations"
            else:
                return "R marginal - request user confirmation"
        else:
            return "R below threshold - warn user and proceed if confirmed"

    def _default_logger(self, decision: GateDecision):
        """Default audit logger."""
        import json
        from datetime import datetime

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": decision.action,
            "tier": decision.tier.name,
            "R": decision.R,
            "threshold": decision.threshold,
            "status": decision.status.value,
            "reason": decision.reason
        }
        print(f"[R-GATE] {json.dumps(log_entry)}")
```

### 6.3 Observation Collector

```python
class ObservationCollector:
    """
    Collect observations from multiple sources.

    Ensures independence by tracking source provenance.
    """

    def __init__(self, sources: dict[str, Callable]):
        """
        Args:
            sources: Dict mapping source_name -> callable that takes query
        """
        self.sources = sources
        self.source_independence = self._compute_independence_matrix()

    def collect(
        self,
        query: str,
        min_sources: int = 2,
        timeout: float = 30.0
    ) -> List[Observation]:
        """
        Collect observations from available sources.

        Prioritizes independent sources.
        """
        import asyncio
        import time

        observations = []
        used_sources = set()

        # Sort sources by independence from already-used sources
        available = list(self.sources.items())

        for source_name, source_fn in available:
            if len(observations) >= min_sources * 2:  # Collect extra for redundancy
                break

            try:
                start = time.time()
                result = source_fn(query)
                elapsed = time.time() - start

                observations.append(Observation(
                    content=str(result),
                    source=source_name,
                    timestamp=start
                ))
                used_sources.add(source_name)

            except Exception as e:
                # Log but continue with other sources
                print(f"[ObservationCollector] Source {source_name} failed: {e}")

        return observations

    def _compute_independence_matrix(self) -> dict:
        """
        Compute pairwise independence scores between sources.

        Sources are independent if they:
        - Use different underlying models
        - Query different data sources
        - Have different failure modes
        """
        # This would be configured based on source metadata
        # For now, assume all sources are independent
        return {(s1, s2): 1.0 for s1 in self.sources for s2 in self.sources if s1 != s2}

    def independence_score(self, observations: List[Observation]) -> float:
        """Compute overall independence score for observations."""
        if len(observations) < 2:
            return 0.0

        sources = [obs.source for obs in observations]
        unique_sources = len(set(sources))

        # Penalize if same source used multiple times
        return unique_sources / len(observations)
```

### 6.4 Self-Consistency Sampler

```python
class SelfConsistencySampler:
    """
    Generate multiple observations via self-consistency sampling.

    Uses temperature > 0 to get diverse outputs from same model.
    """

    def __init__(
        self,
        model_fn: Callable[[str, float], str],
        default_n: int = 5,
        default_temp: float = 0.7
    ):
        self.model_fn = model_fn
        self.default_n = default_n
        self.default_temp = default_temp

    def sample(
        self,
        query: str,
        n: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[Observation]:
        """
        Generate n observations via sampling.
        """
        import time

        n = n or self.default_n
        temp = temperature or self.default_temp

        observations = []
        for i in range(n):
            start = time.time()
            result = self.model_fn(query, temp)

            observations.append(Observation(
                content=result,
                source=f"self_consistency_{i}",
                timestamp=start
            ))

        return observations

    def sample_with_cot(
        self,
        query: str,
        n: Optional[int] = None
    ) -> List[Observation]:
        """
        Sample with chain-of-thought for better diversity.

        Extracts final answer for comparison, keeps full reasoning.
        """
        cot_prompt = f"""Think step by step, then give your final answer.

Question: {query}

Let me think through this carefully:"""

        observations = self.sample(cot_prompt, n)

        # Extract final answers for R computation
        for obs in observations:
            obs.content = self._extract_final_answer(obs.content)

        return observations

    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from CoT response."""
        # Look for common final answer patterns
        markers = ["therefore", "final answer", "in conclusion", "the answer is"]

        response_lower = response.lower()
        for marker in markers:
            if marker in response_lower:
                idx = response_lower.rfind(marker)
                return response[idx:]

        # Fall back to last paragraph
        paragraphs = response.strip().split('\n\n')
        return paragraphs[-1] if paragraphs else response
```

---

## 7. Integration Points

### 7.1 MCP Server Integration

Add R-gate to MCP tool execution:

```python
# In CAPABILITY/MCP/server.py

class RGatedMCPServer:
    """MCP server with R-gating on tool execution."""

    def __init__(self, embed_fn: Callable, thresholds: Optional[dict] = None):
        self.r_gate = RGate(embed_fn, thresholds)
        self.observation_collector = ObservationCollector(self._get_sources())
        self.sampler = SelfConsistencySampler(self._model_fn)

    async def execute_tool(
        self,
        tool_name: str,
        args: dict,
        context: dict
    ) -> dict:
        """Execute tool with R-gating."""

        # Classify action tier
        tier = self.r_gate.classify_tier(tool_name, args.get("target", ""))

        # T0 actions bypass gate
        if tier == ActionTier.T0_READ:
            return await self._execute(tool_name, args)

        # Collect observations for gating
        query = self._construct_query(tool_name, args, context)
        observations = await self._collect_observations(query, tier)

        # Check gate
        decision = self.r_gate.check_action(
            action=tool_name,
            target=args.get("target", ""),
            observations=[obs.content for obs in observations]
        )

        if decision.status == GateStatus.OPEN:
            return await self._execute(tool_name, args)
        else:
            return self._handle_gate_closed(decision, tool_name, args)

    async def _collect_observations(
        self,
        query: str,
        tier: ActionTier
    ) -> List[Observation]:
        """Collect observations based on tier requirements."""
        min_obs = RGate.MIN_OBSERVATIONS[tier]

        # For T2+, use cross-source validation
        if tier.value >= ActionTier.T2_PERSISTENT.value:
            return self.observation_collector.collect(query, min_sources=min_obs)
        else:
            # For T1, self-consistency is sufficient
            return self.sampler.sample(query, n=min_obs)

    def _handle_gate_closed(
        self,
        decision: GateDecision,
        tool_name: str,
        args: dict
    ) -> dict:
        """Handle gate closure with appropriate escalation."""
        return {
            "status": "blocked",
            "reason": decision.reason,
            "R": decision.R,
            "threshold": decision.threshold,
            "escalation": decision.escalation_path,
            "tool": tool_name,
            "args": args
        }
```

### 7.2 Crisis Level Integration

Map R to existing crisis levels (CRISIS.md):

```python
# In CAPABILITY/TOOLS/emergency.py

def r_to_crisis_level(R: float, threshold: float, dR_dt: Optional[float] = None) -> int:
    """
    Map R-value to crisis level.

    Args:
        R: Current R value
        threshold: Required threshold for action
        dR_dt: Rate of change of R (optional)

    Returns:
        Crisis level 0-4
    """
    # Level 0: Normal (R >= threshold)
    if R >= threshold:
        return 0

    # Level 1: Warning (R close to threshold)
    if R >= threshold * 0.8:
        return 1

    # Level 2: Alert (R below threshold)
    if R >= threshold * 0.5:
        return 2

    # Level 3: Quarantine (R very low OR sudden drop)
    if R > 0:
        if dR_dt is not None and dR_dt < -0.5:
            return 3  # Sudden drop
        return 3

    # Level 4: Constitutional (R undefined/corrupted)
    return 4


def handle_r_crisis(crisis_level: int, context: dict) -> dict:
    """Handle crisis based on level."""

    handlers = {
        0: lambda ctx: {"action": "proceed", "message": "Normal operation"},
        1: lambda ctx: {"action": "warn", "message": "R close to threshold, proceed with caution"},
        2: lambda ctx: {"action": "rollback", "message": "R below threshold, rollback recommended"},
        3: lambda ctx: {"action": "quarantine", "message": "R very low, entering quarantine"},
        4: lambda ctx: {"action": "constitutional_reset", "message": "R undefined, human intervention required"}
    }

    return handlers[crisis_level](context)
```

### 7.3 Verification Protocol Integration

Add R-check to verification steps (VERIFICATION_PROTOCOL_CANON.md):

```python
# In CAPABILITY/TOOLS/verification.py

class RGatedVerification:
    """Verification protocol with R-gating."""

    def __init__(self, r_gate: RGate):
        self.r_gate = r_gate

    def verify_step(
        self,
        step_name: str,
        step_fn: Callable,
        observations: List[str],
        tier: ActionTier = ActionTier.T2_PERSISTENT
    ) -> dict:
        """
        Execute verification step with R-gate.

        STEP 0.5 from VERIFICATION_PROTOCOL_CANON.md
        """
        # Check R before executing step
        decision = self.r_gate.check_action(
            action=step_name,
            target="verification",
            observations=observations
        )

        if decision.status != GateStatus.OPEN:
            return {
                "status": "BLOCKED",
                "step": step_name,
                "R": decision.R,
                "threshold": decision.threshold,
                "reason": decision.reason,
                "escalation": decision.escalation_path
            }

        # Execute step
        try:
            result = step_fn()
            return {
                "status": "PASS",
                "step": step_name,
                "R": decision.R,
                "result": result
            }
        except Exception as e:
            return {
                "status": "FAIL",
                "step": step_name,
                "R": decision.R,
                "error": str(e)
            }
```

### 7.4 Audit Trail

Log all R-gate decisions for post-hoc analysis:

```python
# In CAPABILITY/AUDIT/r_gate_audit.py

import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

class RGateAuditLogger:
    """
    Audit logger for R-gate decisions.

    Logs to _runs/r_gate_logs/ as specified in Q17.
    """

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log = self.log_dir / f"r_gate_{datetime.now().strftime('%Y%m%d')}.jsonl"

    def log(self, decision: GateDecision, context: Optional[dict] = None):
        """Log a gate decision."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision": {
                "status": decision.status.value,
                "R": decision.R,
                "threshold": decision.threshold,
                "tier": decision.tier.name,
                "action": decision.action,
                "reason": decision.reason,
                "escalation_path": decision.escalation_path
            },
            "context": context or {}
        }

        with open(self.current_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def analyze_period(self, start: datetime, end: datetime) -> dict:
        """Analyze R-gate decisions over a period."""
        decisions = self._load_range(start, end)

        return {
            "total_decisions": len(decisions),
            "open_rate": sum(1 for d in decisions if d["decision"]["status"] == "open") / len(decisions),
            "mean_R": np.mean([d["decision"]["R"] for d in decisions]),
            "by_tier": self._group_by_tier(decisions),
            "escalations": [d for d in decisions if d["decision"]["escalation_path"]]
        }
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# tests/test_r_gate.py

import pytest
import numpy as np
from r_gate import RGate, ActionTier, GateStatus

class TestRComputation:
    """Test R computation logic."""

    def test_identical_observations_high_r(self):
        """Identical observations should give high R."""
        gate = RGate(embed_fn=lambda x: np.array([1, 0, 0]))

        observations = ["hello world"] * 5
        result = gate.compute_r(observations)

        # All same -> E=1, sigma=0 -> R very high
        assert result.R > 10.0

    def test_diverse_observations_low_r(self):
        """Diverse observations should give low R."""
        # Mock embed that gives orthogonal vectors
        embeddings = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
        ]
        gate = RGate(embed_fn=lambda x: embeddings[hash(x) % 3])

        observations = ["a", "b", "c"]
        result = gate.compute_r(observations)

        # Orthogonal -> E≈0, sigma>0 -> R low
        assert result.R < 1.0

    def test_insufficient_observations(self):
        """Should return R=0 for <2 observations."""
        gate = RGate(embed_fn=lambda x: np.array([1, 0, 0]))

        result = gate.compute_r(["single observation"])

        assert result.R == 0.0
        assert result.n_observations == 1


class TestTierClassification:
    """Test action tier classification."""

    def test_read_actions_t0(self):
        gate = RGate(embed_fn=lambda x: np.zeros(3))

        assert gate.classify_tier("read", "file.txt") == ActionTier.T0_READ
        assert gate.classify_tier("search", "database") == ActionTier.T0_READ
        assert gate.classify_tier("list", "directory") == ActionTier.T0_READ

    def test_critical_actions_t3(self):
        gate = RGate(embed_fn=lambda x: np.zeros(3))

        assert gate.classify_tier("deploy", "production") == ActionTier.T3_CRITICAL
        assert gate.classify_tier("delete", "canon/file.md") == ActionTier.T3_CRITICAL
        assert gate.classify_tier("write", "main") == ActionTier.T3_CRITICAL

    def test_persistent_actions_t2(self):
        gate = RGate(embed_fn=lambda x: np.zeros(3))

        assert gate.classify_tier("commit", "feature-branch") == ActionTier.T2_PERSISTENT
        assert gate.classify_tier("write", "output.txt") == ActionTier.T2_PERSISTENT


class TestGateDecision:
    """Test gate decision logic."""

    def test_gate_opens_above_threshold(self):
        # Mock high agreement
        gate = RGate(embed_fn=lambda x: np.array([1, 0, 0]))

        decision = gate.check_action(
            action="write",
            target="output.txt",
            observations=["same"] * 5
        )

        assert decision.status == GateStatus.OPEN

    def test_gate_closes_below_threshold(self):
        # Mock low agreement
        embeddings = [np.random.randn(10) for _ in range(10)]
        idx = [0]
        gate = RGate(embed_fn=lambda x: embeddings[idx[0] := idx[0] + 1])

        decision = gate.check_action(
            action="commit",
            target="main",
            observations=["a", "b", "c", "d", "e"]
        )

        assert decision.status == GateStatus.CLOSED
        assert decision.escalation_path is not None
```

### 8.2 Integration Tests

```python
# tests/test_r_gate_integration.py

import pytest
from r_gate import RGate
from observation_collector import ObservationCollector
from self_consistency import SelfConsistencySampler

class TestEndToEndGating:
    """Test complete R-gating flow."""

    @pytest.fixture
    def full_system(self):
        """Set up full R-gate system with real embeddings."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('all-MiniLM-L6-v2')
        embed_fn = lambda x: model.encode(x)

        return RGate(embed_fn=embed_fn)

    def test_semantic_agreement_detected(self, full_system):
        """Semantically similar responses should have high R."""
        observations = [
            "The capital of France is Paris.",
            "Paris is the capital city of France.",
            "France's capital is Paris.",
        ]

        result = full_system.compute_r(observations)

        assert result.R > 1.0  # High agreement

    def test_semantic_disagreement_detected(self, full_system):
        """Semantically different responses should have low R."""
        observations = [
            "The capital of France is Paris.",
            "Python is a programming language.",
            "The weather is nice today.",
        ]

        result = full_system.compute_r(observations)

        assert result.R < 0.5  # Low agreement
```

### 8.3 Adversarial Tests

```python
# tests/test_r_gate_adversarial.py

class TestAdversarialResistance:
    """Test R-gate resistance to gaming."""

    def test_volume_attack_fails(self):
        """Cannot bypass gate by adding more observations."""
        gate = RGate(embed_fn=real_embed_fn)

        # Low-quality observations (diverse/noisy)
        base_observations = generate_noisy_observations(n=3)

        # Try to game by adding volume
        for n in [5, 10, 20, 50]:
            observations = base_observations + generate_noisy_observations(n=n-3)
            result = gate.compute_r(observations)

            # R should NOT increase with volume
            assert result.R < 0.5, f"R increased to {result.R} with {n} observations"

    def test_echo_chamber_detection(self):
        """Correlated observations should not give high R."""
        gate = RGate(embed_fn=real_embed_fn)

        # Echo chamber: same source repeated
        base = "The answer is definitely X."
        observations = [base, base, base, base, base]

        result = gate.compute_r(observations)

        # Very high R should trigger echo chamber warning
        assert result.R > 10.0  # Suspiciously high
        # Real implementation would flag this

    def test_paraphrase_attack_partial(self):
        """Paraphrased observations should be detected."""
        gate = RGate(embed_fn=real_embed_fn)

        # Paraphrased versions (correlated but different surface)
        observations = [
            "X is the correct answer.",
            "The right answer is X.",
            "X is what we should choose.",
            "We should go with X.",
            "X is the way to go.",
        ]

        result = gate.compute_r(observations)

        # Should have high E (similar meaning) but low sigma
        # Real diversity would have more varied sigma
        assert result.E > 0.7  # High semantic similarity
```

---

## 9. Failure Modes and Mitigations

### 9.1 Failure Mode Taxonomy

| Failure Mode | Description | Detection | Mitigation |
|--------------|-------------|-----------|------------|
| **Echo Chamber** | High R from correlated sources | R > 95th percentile | Require source diversity |
| **Embedding Collapse** | Embeddings don't distinguish semantics | Low variance in embeddings | Use validated embedding model |
| **Threshold Gaming** | Attacker crafts inputs just above threshold | R suspiciously close to threshold | Add margin/hysteresis |
| **Observation Poisoning** | Malicious observations injected | Source not in allowlist | Authenticate sources |
| **Latency Attack** | Slow sources timeout, reducing observations | n_observations < min | Async collection with fallbacks |
| **Model Disagreement** | Different models give valid but different answers | High sigma, moderate E | Domain-specific handling |

### 9.2 Mitigation Implementations

```python
class RGateWithMitigations(RGate):
    """R-Gate with failure mode mitigations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_history = []
        self.echo_chamber_threshold = 10.0
        self.hysteresis_margin = 0.1

    def compute_r_safe(self, observations: List[str]) -> RResult:
        """Compute R with safety checks."""
        result = self.compute_r(observations)

        # Check for echo chamber
        if result.R > self.echo_chamber_threshold:
            result = self._handle_echo_chamber(observations, result)

        # Apply hysteresis
        result = self._apply_hysteresis(result)

        # Track history
        self.r_history.append(result.R)

        return result

    def _handle_echo_chamber(
        self,
        observations: List[str],
        result: RResult
    ) -> RResult:
        """Handle suspected echo chamber."""
        # Check source diversity
        sources = [self._extract_source(obs) for obs in observations]
        unique_sources = len(set(sources))

        if unique_sources < len(observations) * 0.5:
            # Penalize R for low diversity
            diversity_penalty = unique_sources / len(observations)
            adjusted_R = result.R * diversity_penalty

            return RResult(
                R=adjusted_R,
                E=result.E,
                sigma=result.sigma,
                n_observations=result.n_observations,
                confidence_interval=result.confidence_interval
            )

        return result

    def _apply_hysteresis(self, result: RResult) -> RResult:
        """Apply hysteresis to prevent threshold oscillation."""
        if len(self.r_history) < 2:
            return result

        prev_r = self.r_history[-1]

        # If crossing threshold, require margin
        for tier, threshold in self.thresholds.items():
            if prev_r < threshold and result.R >= threshold:
                # Rising across threshold - require margin
                if result.R < threshold + self.hysteresis_margin:
                    return RResult(
                        R=result.R - self.hysteresis_margin,  # Effective R below threshold
                        E=result.E,
                        sigma=result.sigma,
                        n_observations=result.n_observations,
                        confidence_interval=result.confidence_interval
                    )

        return result
```

### 9.3 Graceful Degradation

```python
class GracefulRGate(RGate):
    """R-Gate with graceful degradation modes."""

    def check_action_graceful(
        self,
        action: str,
        target: str,
        observations: List[str],
        user_override: bool = False
    ) -> GateDecision:
        """Check action with graceful degradation options."""
        decision = self.check_action(action, target, observations)

        # If gate closed but user overrides
        if decision.status == GateStatus.CLOSED and user_override:
            return GateDecision(
                status=GateStatus.DEGRADED,
                R=decision.R,
                threshold=decision.threshold,
                tier=decision.tier,
                action=decision.action,
                reason=f"User override with R={decision.R:.3f} < threshold",
                escalation_path="Action allowed in degraded mode - logged for audit"
            )

        # If insufficient observations, allow with warning
        if decision.status == GateStatus.CLOSED and "Insufficient observations" in decision.reason:
            return GateDecision(
                status=GateStatus.DEGRADED,
                R=decision.R,
                threshold=decision.threshold,
                tier=decision.tier,
                action=decision.action,
                reason="Insufficient observations - proceeding with reduced confidence",
                escalation_path="Collect more observations for future actions"
            )

        return decision
```

---

## 10. Rollout Plan

### Phase 1: Foundation (Week 1-2)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 1.1 | Implement core RGate class | `CAPABILITY/PRIMITIVES/r_gate.py` |
| 1.2 | Add embedding integration | Integration with existing `semantic_search.py` |
| 1.3 | Unit tests | `tests/test_r_gate.py` |
| 1.4 | Audit logging | `CAPABILITY/AUDIT/r_gate_audit.py` |

### Phase 2: Integration (Week 3-4)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 2.1 | MCP server integration | R-gating in `CAPABILITY/MCP/server.py` |
| 2.2 | Crisis level mapping | Integration with `CAPABILITY/TOOLS/emergency.py` |
| 2.3 | Verification protocol | STEP 0.5 in verification flow |
| 2.4 | Integration tests | `tests/test_r_gate_integration.py` |

### Phase 3: Observation Sources (Week 5-6)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 3.1 | Self-consistency sampler | `CAPABILITY/PRIMITIVES/self_consistency.py` |
| 3.2 | Cross-source collector | `CAPABILITY/PRIMITIVES/observation_collector.py` |
| 3.3 | Source independence tracking | Provenance metadata |
| 3.4 | Adversarial tests | `tests/test_r_gate_adversarial.py` |

### Phase 4: Hardening (Week 7-8)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 4.1 | Echo chamber detection | Mitigation in RGate |
| 4.2 | Hysteresis implementation | Threshold stability |
| 4.3 | Graceful degradation | Override and fallback modes |
| 4.4 | Production monitoring | Dashboard and alerts |

### Phase 5: Evaluation (Week 9-10)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 5.1 | Threshold calibration | Domain-specific threshold tuning |
| 5.2 | False positive analysis | Audit of blocked actions |
| 5.3 | False negative analysis | Post-hoc R computation on failures |
| 5.4 | Documentation | This guide + API docs |

---

## Appendix A: Quick Reference

### A.1 Threshold Quick Reference

```
T0 (read):      No gate
T1 (reversible): R > 0.5
T2 (persistent): R > 0.8
T3 (critical):   R > 1.0
```

### A.2 Minimum Observations

```
T0: 0 observations
T1: 2 observations (any source)
T2: 3 observations (2+ independent sources)
T3: 5 observations (3+ independent sources + human)
```

### A.3 Crisis Level Mapping

```
R >= threshold:      Level 0 (Normal)
R >= threshold*0.8:  Level 1 (Warning)
R >= threshold*0.5:  Level 2 (Alert)
R > 0:               Level 3 (Quarantine)
R undefined:         Level 4 (Constitutional)
```

---

## Appendix B: Example Usage

### B.1 Basic Usage

```python
from r_gate import RGate
from sentence_transformers import SentenceTransformer

# Initialize
model = SentenceTransformer('all-MiniLM-L6-v2')
gate = RGate(embed_fn=model.encode)

# Check action
decision = gate.check_action(
    action="commit",
    target="feature-branch",
    observations=[
        "The fix handles the edge case correctly.",
        "Edge case is now properly handled.",
        "The edge case bug is fixed."
    ]
)

if decision.status == GateStatus.OPEN:
    git_commit()
else:
    print(f"Blocked: {decision.reason}")
    print(f"Escalation: {decision.escalation_path}")
```

### B.2 With Self-Consistency

```python
from r_gate import RGate
from self_consistency import SelfConsistencySampler

# Sample multiple outputs
sampler = SelfConsistencySampler(model_fn=llm.generate)
observations = sampler.sample("What is the capital of France?", n=5)

# Check agreement
decision = gate.check_action(
    action="answer",
    target="user_query",
    observations=[obs.content for obs in observations]
)
```

### B.3 With Cross-Source Validation

```python
from observation_collector import ObservationCollector

# Define sources
sources = {
    "claude": lambda q: claude.complete(q),
    "gpt4": lambda q: gpt4.complete(q),
    "web_search": lambda q: search_engine.query(q),
    "cache": lambda q: cache.get(q)
}

collector = ObservationCollector(sources)
observations = collector.collect("What is 2+2?", min_sources=3)

decision = gate.check_action(
    action="deploy",
    target="production",
    observations=[obs.content for obs in observations]
)
```

---

**Document Status**: IMPLEMENTATION SPEC
**Next Steps**: Begin Phase 1 implementation
**Owner**: Agent Governance System
**Review Required**: Human steward approval before Phase 2

---

*This guide implements the theoretical findings from Q17 (Governance Gating). For theoretical background, see [q17_governance_gating.md](../medium_priority/q17_governance_gating.md).*
