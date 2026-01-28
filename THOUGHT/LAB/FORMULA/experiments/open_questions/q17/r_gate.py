#!/usr/bin/env python3
"""
R-Gate Implementation for Agent Governance

Core implementation of R = E/σ for gating agent actions.
This is the TESTABLE implementation, not just a spec.
"""

import numpy as np
from typing import List, Callable, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ActionTier(Enum):
    """Action criticality tiers."""
    T0_READ = 0       # Read-only
    T1_REVERSIBLE = 1 # Reversible actions
    T2_PERSISTENT = 2 # Persistent changes
    T3_CRITICAL = 3   # Critical/dangerous


class GateStatus(Enum):
    """Gate decision status."""
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class RResult:
    """Result of R computation."""
    R: float
    E: float           # Mean agreement (evidence)
    sigma: float       # Dispersion
    n_observations: int
    pairwise_sims: List[float]  # Raw data for analysis


@dataclass
class GateDecision:
    """Gate decision result."""
    status: GateStatus
    R: float
    threshold: float
    tier: ActionTier
    reason: str


class RGate:
    """
    R-Gate for agent action governance.

    Core formula: R = E / σ
    Where:
        E = mean pairwise similarity (agreement)
        σ = std of pairwise similarities (dispersion)
    """

    THRESHOLDS = {
        ActionTier.T0_READ: 0.0,
        ActionTier.T1_REVERSIBLE: 0.5,
        ActionTier.T2_PERSISTENT: 0.8,
        ActionTier.T3_CRITICAL: 1.0,
    }

    def __init__(self, embed_fn: Callable[[str], np.ndarray]):
        """
        Initialize R-Gate.

        Args:
            embed_fn: Function that maps string -> normalized embedding vector
        """
        self.embed_fn = embed_fn

    def compute_r(self, observations: List[str]) -> RResult:
        """
        Compute R from a list of observations.

        Args:
            observations: List of observation strings

        Returns:
            RResult with R value and components
        """
        n = len(observations)

        if n < 2:
            return RResult(
                R=0.0,
                E=0.0,
                sigma=float('inf'),
                n_observations=n,
                pairwise_sims=[]
            )

        # Embed all observations
        embeddings = []
        for obs in observations:
            emb = self.embed_fn(obs)
            # Normalize
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)

        # Compute ALL pairwise cosine similarities
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(float(sim))

        # Compute E (mean) and σ (std)
        E = np.mean(similarities)
        sigma = np.std(similarities)

        # R = E / max(sigma, epsilon) - Q29 validated numerical stability
        # epsilon = 1e-6 recommended: prevents div-by-zero while preserving sensitivity
        R = E / max(sigma, 1e-6)

        return RResult(
            R=float(R),
            E=float(E),
            sigma=float(sigma),
            n_observations=n,
            pairwise_sims=similarities
        )

    def classify_tier(self, action: str, target: str = "") -> ActionTier:
        """Classify action into tier."""
        action_lower = action.lower()
        target_lower = target.lower()

        # T3: Critical
        critical_actions = {"deploy", "delete", "drop", "force_push", "reset"}
        critical_targets = {"canon", "production", "main", "master", "invariant"}

        if action_lower in critical_actions:
            return ActionTier.T3_CRITICAL
        if any(t in target_lower for t in critical_targets):
            return ActionTier.T3_CRITICAL

        # T2: Persistent
        persistent_actions = {"write", "commit", "send", "post", "create", "update"}
        if action_lower in persistent_actions:
            return ActionTier.T2_PERSISTENT

        # T1: Reversible
        reversible_actions = {"stage", "draft", "propose", "preview", "plan"}
        if action_lower in reversible_actions:
            return ActionTier.T1_REVERSIBLE

        # T0: Read-only (default)
        return ActionTier.T0_READ

    def check(
        self,
        observations: List[str],
        tier: ActionTier
    ) -> GateDecision:
        """
        Check if gate should open for given observations and tier.

        Args:
            observations: List of observation strings
            tier: Action tier to check against

        Returns:
            GateDecision with status and details
        """
        threshold = self.THRESHOLDS[tier]

        # T0 always passes
        if tier == ActionTier.T0_READ:
            return GateDecision(
                status=GateStatus.OPEN,
                R=float('inf'),
                threshold=threshold,
                tier=tier,
                reason="T0 actions always allowed"
            )

        # Compute R
        result = self.compute_r(observations)

        # Check threshold
        if result.R >= threshold:
            return GateDecision(
                status=GateStatus.OPEN,
                R=result.R,
                threshold=threshold,
                tier=tier,
                reason=f"R={result.R:.4f} >= threshold={threshold}"
            )
        else:
            return GateDecision(
                status=GateStatus.CLOSED,
                R=result.R,
                threshold=threshold,
                tier=tier,
                reason=f"R={result.R:.4f} < threshold={threshold}"
            )


def create_mock_embedder(dim: int = 384, seed: int = 42) -> Callable:
    """
    Create a mock embedder for testing without ML dependencies.

    Uses hash-based deterministic embeddings.
    """
    rng = np.random.default_rng(seed)
    cache = {}

    def embed(text: str) -> np.ndarray:
        if text not in cache:
            # Hash-based seed for reproducibility
            text_seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            text_rng = np.random.default_rng(text_seed)
            emb = text_rng.standard_normal(dim)
            emb = emb / np.linalg.norm(emb)
            cache[text] = emb
        return cache[text]

    return embed


import hashlib  # Need this for mock embedder
