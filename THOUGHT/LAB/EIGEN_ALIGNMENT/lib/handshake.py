"""ESAP Handshake Protocol Implementation.

Implements the Eigen Spectrum Alignment Protocol handshake for
establishing semantic alignment between cassettes/agents.

The handshake verifies the Spectral Convergence Theorem:
- Cumulative variance curves must correlate > 0.9
- Effective ranks must be within reasonable ratio
- Once verified, Procrustes alignment is exchanged
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import secrets
import numpy as np

from .protocol import (
    PROTOCOL_VERSION,
    SpectrumSignature,
    AlignmentMap,
    compute_hash,
    utc_now,
    SpectrumMismatchError,
    AlignmentFailedError,
)


# Spectral Convergence Theorem thresholds (from E.X.3.6/E.X.3.7)
CONVERGENCE_THRESHOLD = 0.9  # Correlation must exceed this
EFFECTIVE_RANK_RATIO_MAX = 2.0  # Df ratio must be within this


def compute_cumulative_variance(eigenvalues: np.ndarray) -> np.ndarray:
    """Compute cumulative variance curve - THE Platonic invariant.

    C(k) = Σᵢ₌₁ᵏ λᵢ / Σλ

    This is the invariant proven in the Spectral Convergence Theorem:
    - Cross-architecture correlation: 0.971
    - Cross-lingual correlation: 0.914
    - Adversarial robustness: cannot be broken

    Args:
        eigenvalues: Eigenvalue spectrum (sorted descending)

    Returns:
        Cumulative variance curve (values in [0, 1])
    """
    total = np.sum(eigenvalues)
    if total == 0:
        return np.zeros_like(eigenvalues)
    normalized = eigenvalues / total
    return np.cumsum(normalized)


def compute_effective_rank(eigenvalues: np.ndarray) -> float:
    """Compute effective rank (participation ratio).

    Df = (Σλ)² / Σλ²

    Typically ~22 for trained models, ~62 for untrained, ~99 for random.

    Args:
        eigenvalues: Eigenvalue spectrum

    Returns:
        Effective dimensionality
    """
    ev = np.asarray(eigenvalues)
    sum_sq = np.sum(ev ** 2)
    if sum_sq == 0:
        return 0.0
    return float((np.sum(ev) ** 2) / sum_sq)


def check_convergence(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    df_a: float,
    df_b: float
) -> dict:
    """Check Spectral Convergence Theorem.

    Args:
        curve_a: Cumulative variance curve from model A
        curve_b: Cumulative variance curve from model B
        df_a: Effective rank of A
        df_b: Effective rank of B

    Returns:
        dict with correlation, converges flag, and ratio
    """
    # Align lengths
    min_len = min(len(curve_a), len(curve_b))
    a = curve_a[:min_len]
    b = curve_b[:min_len]

    # Pearson correlation
    correlation = float(np.corrcoef(a, b)[0, 1])

    # Effective rank ratio
    df_ratio = max(df_a, df_b) / max(min(df_a, df_b), 1e-10)

    converges = (
        correlation >= CONVERGENCE_THRESHOLD and
        df_ratio <= EFFECTIVE_RANK_RATIO_MAX
    )

    return {
        "correlation": correlation,
        "converges": converges,
        "effective_rank_ratio": df_ratio
    }


@dataclass
class SpectrumCompact:
    """Compact spectrum representation for handshake messages."""

    eigenvalues_top_k: list[float]
    cumulative_variance: list[float]
    effective_rank: float
    anchor_set_hash: str
    embedder_id: str = ""
    computed_at: str = field(default_factory=utc_now)

    @classmethod
    def from_signature(cls, sig: SpectrumSignature) -> 'SpectrumCompact':
        """Create from full SpectrumSignature."""
        ev = np.array(sig.eigenvalues)
        cv = compute_cumulative_variance(ev)
        return cls(
            eigenvalues_top_k=sig.eigenvalues,
            cumulative_variance=cv.tolist(),
            effective_rank=sig.effective_rank,
            anchor_set_hash=sig.anchor_set_hash,
            embedder_id=sig.embedder_id,
            computed_at=sig.computed_at
        )

    @classmethod
    def from_eigenvalues(
        cls,
        eigenvalues: np.ndarray,
        anchor_set_hash: str,
        embedder_id: str = ""
    ) -> 'SpectrumCompact':
        """Create directly from eigenvalues."""
        ev = eigenvalues.tolist()
        cv = compute_cumulative_variance(eigenvalues)
        df = compute_effective_rank(eigenvalues)
        return cls(
            eigenvalues_top_k=ev,
            cumulative_variance=cv.tolist(),
            effective_rank=df,
            anchor_set_hash=anchor_set_hash,
            embedder_id=embedder_id
        )

    def to_dict(self) -> dict:
        return {
            "eigenvalues_top_k": self.eigenvalues_top_k,
            "cumulative_variance": self.cumulative_variance,
            "effective_rank": self.effective_rank,
            "anchor_set_hash": self.anchor_set_hash,
            "embedder_id": self.embedder_id,
            "computed_at": self.computed_at
        }


@dataclass
class ESAPHello:
    """Initial handshake message."""

    sender_id: str
    spectrum: SpectrumCompact
    capabilities: list[str] = field(default_factory=list)
    nonce: str = field(default_factory=lambda: secrets.token_hex(16))
    version: str = PROTOCOL_VERSION
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        return {
            "type": "ESAP_HELLO",
            "version": self.version,
            "sender_id": self.sender_id,
            "spectrum": self.spectrum.to_dict(),
            "capabilities": self.capabilities,
            "timestamp": self.timestamp,
            "nonce": self.nonce
        }


@dataclass
class ESAPAck:
    """Acknowledgment with convergence verification."""

    sender_id: str
    in_reply_to: str
    spectrum: SpectrumCompact
    convergence: dict
    alignment: Optional[dict] = None
    capabilities: list[str] = field(default_factory=list)
    version: str = PROTOCOL_VERSION
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        result = {
            "type": "ESAP_ACK",
            "version": self.version,
            "sender_id": self.sender_id,
            "in_reply_to": self.in_reply_to,
            "spectrum": self.spectrum.to_dict(),
            "convergence": self.convergence,
            "capabilities": self.capabilities,
            "timestamp": self.timestamp
        }
        if self.alignment:
            result["alignment"] = self.alignment
        return result


@dataclass
class ESAPReject:
    """Handshake rejection."""

    sender_id: str
    in_reply_to: str
    reason: str
    details: Optional[dict] = None
    version: str = PROTOCOL_VERSION
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        result = {
            "type": "ESAP_REJECT",
            "version": self.version,
            "sender_id": self.sender_id,
            "in_reply_to": self.in_reply_to,
            "reason": self.reason,
            "timestamp": self.timestamp
        }
        if self.details:
            result["details"] = self.details
        return result


class ESAPHandshake:
    """ESAP Handshake protocol handler.

    Example:
        # Agent A initiates
        handler_a = ESAPHandshake("agent_a", spectrum_a)
        hello = handler_a.create_hello(capabilities=["symbol_resolution"])

        # Agent B receives and responds
        handler_b = ESAPHandshake("agent_b", spectrum_b)
        response = handler_b.process_hello(hello)

        # Agent A processes response
        if response["type"] == "ESAP_ACK":
            # Handshake successful - semantic space aligned
            alignment = response.get("alignment")
    """

    def __init__(
        self,
        agent_id: str,
        spectrum: SpectrumCompact,
        capabilities: list[str] = None
    ):
        self.agent_id = agent_id
        self.spectrum = spectrum
        self.capabilities = capabilities or []
        self._pending_nonces: dict[str, ESAPHello] = {}

    def create_hello(self, capabilities: list[str] = None) -> dict:
        """Create ESAP_HELLO message."""
        caps = capabilities or self.capabilities
        hello = ESAPHello(
            sender_id=self.agent_id,
            spectrum=self.spectrum,
            capabilities=caps
        )
        # Store for later verification
        self._pending_nonces[hello.nonce] = hello
        return hello.to_dict()

    def process_hello(
        self,
        hello: dict,
        alignment_map: Optional[AlignmentMap] = None
    ) -> dict:
        """Process received ESAP_HELLO, return ACK or REJECT.

        Args:
            hello: The ESAP_HELLO message dict
            alignment_map: Optional pre-computed alignment

        Returns:
            ESAP_ACK or ESAP_REJECT message dict
        """
        if hello.get("type") != "ESAP_HELLO":
            return ESAPReject(
                sender_id=self.agent_id,
                in_reply_to="",
                reason="VERSION_INCOMPATIBLE",
                details={"message": "Expected ESAP_HELLO"}
            ).to_dict()

        nonce = hello.get("nonce", "")
        their_spectrum = hello.get("spectrum", {})

        # Check anchor set compatibility
        if their_spectrum.get("anchor_set_hash") != self.spectrum.anchor_set_hash:
            return ESAPReject(
                sender_id=self.agent_id,
                in_reply_to=nonce,
                reason="ANCHOR_MISMATCH",
                details={
                    "message": "Anchor sets must match for alignment",
                    "theirs": their_spectrum.get("anchor_set_hash"),
                    "ours": self.spectrum.anchor_set_hash
                }
            ).to_dict()

        # Check spectral convergence
        their_cv = np.array(their_spectrum.get("cumulative_variance", []))
        our_cv = np.array(self.spectrum.cumulative_variance)
        their_df = their_spectrum.get("effective_rank", 0)
        our_df = self.spectrum.effective_rank

        convergence = check_convergence(their_cv, our_cv, their_df, our_df)

        if not convergence["converges"]:
            return ESAPReject(
                sender_id=self.agent_id,
                in_reply_to=nonce,
                reason="SPECTRUM_DIVERGENCE",
                details={
                    "correlation": convergence["correlation"],
                    "threshold": CONVERGENCE_THRESHOLD,
                    "effective_rank_ratio": convergence["effective_rank_ratio"],
                    "message": "Spectral Convergence Theorem not satisfied"
                }
            ).to_dict()

        # Convergence confirmed - create ACK
        alignment_info = None
        if alignment_map:
            alignment_info = {
                "rotation_k": alignment_map.k,
                "rotation_hash": alignment_map.map_hash,
                "residual": alignment_map.procrustes_residual
            }

        ack = ESAPAck(
            sender_id=self.agent_id,
            in_reply_to=nonce,
            spectrum=self.spectrum,
            convergence=convergence,
            alignment=alignment_info,
            capabilities=self.capabilities
        )

        return ack.to_dict()

    def process_ack(self, ack: dict) -> bool:
        """Process received ESAP_ACK.

        Args:
            ack: The ESAP_ACK message dict

        Returns:
            True if handshake successful

        Raises:
            SpectrumMismatchError: If verification fails
        """
        if ack.get("type") == "ESAP_REJECT":
            reason = ack.get("reason", "UNKNOWN")
            details = ack.get("details", {})
            raise SpectrumMismatchError(
                f"Handshake rejected: {reason} - {details.get('message', '')}"
            )

        if ack.get("type") != "ESAP_ACK":
            raise SpectrumMismatchError(f"Unexpected message type: {ack.get('type')}")

        # Verify nonce
        nonce = ack.get("in_reply_to", "")
        if nonce not in self._pending_nonces:
            raise SpectrumMismatchError("Invalid nonce - replay attack?")

        # Clean up
        del self._pending_nonces[nonce]

        # Verify convergence from their side
        convergence = ack.get("convergence", {})
        if not convergence.get("converges", False):
            raise SpectrumMismatchError(
                f"Convergence not confirmed: r={convergence.get('correlation', 0)}"
            )

        return True


def create_handshake_from_embeddings(
    agent_id: str,
    embeddings: np.ndarray,
    anchor_set_hash: str,
    embedder_id: str = "",
    capabilities: list[str] = None
) -> ESAPHandshake:
    """Convenience function to create handshake handler from embeddings.

    Args:
        agent_id: Unique identifier for this agent
        embeddings: Anchor embeddings (n_anchors x dim)
        anchor_set_hash: Hash of the anchor set
        embedder_id: Model identifier
        capabilities: List of semantic capabilities

    Returns:
        ESAPHandshake handler ready to use
    """
    # Compute eigenvalues from embeddings
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]  # Descending

    # Keep only positive eigenvalues
    eigenvalues = eigenvalues[eigenvalues > 0]

    # Create compact spectrum
    spectrum = SpectrumCompact.from_eigenvalues(
        eigenvalues=eigenvalues,
        anchor_set_hash=anchor_set_hash,
        embedder_id=embedder_id
    )

    return ESAPHandshake(
        agent_id=agent_id,
        spectrum=spectrum,
        capabilities=capabilities or []
    )
