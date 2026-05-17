"""Phase 4b: Soft Gate, Hard Gate, @C Symbol System, Df Anomaly Detection

Soft Gate (Unitary Evolution):
    When verification passes: approve output, append to context, continue.
    No intervention. No temperature modulation.

Hard Gate (Projective Measurement):
    When verification fails: halt generation, collect clean consensus,
    overwrite drifted context, force regeneration, log decoherence event.

@C Symbol System (TINY_COMPRESS):
    Compress shared content -> SHA-256 hash -> @C:{hash_short} symbol.
    Sender compresses, receiver resolves via shared canon.
    476x compression for shared-context communication.

Df Anomaly Detection:
    Track effective dimensionality of output distribution.
    Sudden spikes indicate noise injection, goal drift, adversarial input.
"""

import hashlib, json, math, time, numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from phase4b_lattice import ConsensusResult, Verdict


# ============================================================================
# @C Symbol System
# ============================================================================

@dataclass
class CSymbol:
    """A compressed @C symbol referencing stored content."""
    hash_short: str       # e.g., "a3f2c9"
    hash_full: str        # Full SHA-256 hash
    content: str          # Original uncompressed content
    compressed_size: int  # Size of the @C symbol string
    original_size: int    # Size of the original content
    compression_ratio: float  # original_size / compressed_size
    created_at: float     # Unix timestamp


class CSymbolRegistry:
    """Registry for @C symbols. Stores content-addressed by SHA-256 hash.

    Sender: compress content -> SHA-256 hash -> @C:{hash_short}
    Receiver: resolve @C symbol -> retrieve full content -> verify hash
    """

    def __init__(self):
        self._store: dict = {}  # hash_short -> CSymbol

    def compress(self, content: str) -> CSymbol:
        """Compress content into an @C symbol. Returns the CSymbol record."""
        hash_full = hashlib.sha256(content.encode("utf-8")).hexdigest()
        hash_short = hash_full[:6]
        symbol_str = f"@C:{hash_short}"
        compressed_size = len(symbol_str)
        original_size = len(content)
        ratio = original_size / max(compressed_size, 1)

        symbol = CSymbol(
            hash_short=hash_short,
            hash_full=hash_full,
            content=content,
            compressed_size=compressed_size,
            original_size=original_size,
            compression_ratio=ratio,
            created_at=time.time(),
        )
        self._store[hash_short] = symbol
        return symbol

    def resolve(self, symbol_str: str) -> Optional[str]:
        """Resolve an @C symbol to its original content. Returns None if not found."""
        if not symbol_str.startswith("@C:"):
            return None
        hash_short = symbol_str[3:]
        symbol = self._store.get(hash_short)
        if symbol is None:
            return None
        # Verify integrity
        expected_hash = hashlib.sha256(symbol.content.encode("utf-8")).hexdigest()
        if expected_hash != symbol.hash_full:
            return None  # Content corruption detected
        return symbol.content

    def get_stats(self) -> dict:
        """Return aggregate compression statistics."""
        if not self._store:
            return {"n_symbols": 0, "avg_compression": 0}
        ratios = [s.compression_ratio for s in self._store.values()]
        return {
            "n_symbols": len(self._store),
            "avg_compression": round(float(np.mean(ratios)), 1),
            "max_compression": round(float(np.max(ratios)), 1),
            "min_compression": round(float(np.min(ratios)), 1),
            "total_content_bytes": sum(s.original_size for s in self._store.values()),
            "total_symbol_bytes": sum(s.compressed_size for s in self._store.values()),
        }


# ============================================================================
# Df Anomaly Detection
# ============================================================================

@dataclass
class DfSnapshot:
    """A single measurement of the effective dimensionality Df."""
    step: int
    df_value: float       # Effective dimensionality of output distribution
    is_anomaly: bool      # True if Df exceeds threshold
    anomaly_score: float  # How anomalous (0 = normal, 1 = extreme)
    raw_metrics: dict = field(default_factory=dict)


class DfTracker:
    """Tracks effective dimensionality (Df) of agent output distributions.

    Df is computed from the token probability distribution entropy:
    Df = exp(H(p)) where H(p) is the Shannon entropy of the distribution.

    A sudden spike in Df indicates:
    - Noise injection into the generation
    - Goal drift / context pollution
    - Adversarial input corrupting the distribution
    """

    def __init__(self, window_size: int = 10, anomaly_threshold: float = 2.5):
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.history: list = []  # List of DfSnapshot
        self._df_values: list = []  # Raw Df values

    def compute_df(self, logits: np.ndarray) -> float:
        """Compute effective dimensionality from logits.

        Df = exp(H(p)) where H(p) = -sum(p * log(p))
        For a uniform distribution over N tokens: Df = N
        For a deterministic distribution: Df = 1
        """
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        probs = np.clip(probs, 1e-12, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=-1)
        df = np.exp(entropy)
        return float(np.mean(df))

    def record(self, step: int, logits: np.ndarray, metadata: Optional[dict] = None) -> DfSnapshot:
        """Record a Df measurement. Returns the snapshot with anomaly flag."""
        df_val = self.compute_df(logits)
        self._df_values.append(df_val)

        is_anomaly = False
        anomaly_score = 0.0

        if len(self._df_values) >= 3:
            recent = self._df_values[-self.window_size:]
            mean_df = float(np.mean(recent[:-1]))
            std_df = float(np.std(recent[:-1])) + 1e-12
            z_score = (df_val - mean_df) / std_df
            is_anomaly = abs(z_score) > self.anomaly_threshold
            anomaly_score = min(1.0, abs(z_score) / (self.anomaly_threshold * 2))

        snapshot = DfSnapshot(
            step=step,
            df_value=df_val,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            raw_metrics={
                "mean": float(np.mean(self._df_values)) if self._df_values else 0.0,
                "std": float(np.std(self._df_values)) if len(self._df_values) > 1 else 0.0,
                "z_score": anomaly_score * self.anomaly_threshold if is_anomaly else 0.0,
                "window": self.window_size,
            } | (metadata or {}),
        )
        self.history.append(snapshot)
        return snapshot

    def get_anomaly_rate(self) -> float:
        """Fraction of recorded steps that were anomalous."""
        if not self.history:
            return 0.0
        return sum(1 for h in self.history if h.is_anomaly) / len(self.history)

    def get_stats(self) -> dict:
        """Aggregate Df statistics."""
        if not self._df_values:
            return {"mean_df": 0, "std_df": 0, "anomaly_rate": 0, "n_steps": 0}
        return {
            "mean_df": round(float(np.mean(self._df_values)), 4),
            "std_df": round(float(np.std(self._df_values)), 4),
            "min_df": round(float(np.min(self._df_values)), 4),
            "max_df": round(float(np.max(self._df_values)), 4),
            "anomaly_rate": round(self.get_anomaly_rate(), 4),
            "anomaly_count": sum(1 for h in self.history if h.is_anomaly),
            "n_steps": len(self._df_values),
            "threshold": self.anomaly_threshold,
        }

    def to_dict(self) -> list:
        return [{
            "step": s.step, "df_value": round(s.df_value, 4),
            "is_anomaly": s.is_anomaly, "anomaly_score": round(s.anomaly_score, 4),
        } for s in self.history]


# ============================================================================
# Soft Gate
# ============================================================================

@dataclass
class SoftGateEvent:
    """Record of a soft gate approval event."""
    step: int
    resonance: float      # R = 1/grad_S at approval time
    consensus_ratio: float
    node_verdicts: list   # Verdicts from each node
    timestamp: float = field(default_factory=time.time)


class SoftGate:
    """Soft Gate (Unitary Evolution).

    When verification passes: approve output, append to context, continue.
    No intervention. No temperature modulation.
    """

    def __init__(self):
        self.events: list = []

    def approve(self, step: int, consensus: ConsensusResult) -> SoftGateEvent:
        """Approve the output and record the event."""
        event = SoftGateEvent(
            step=step,
            resonance=consensus.resonance,
            consensus_ratio=consensus.consensus_ratio,
            node_verdicts=[r.verdict.value for r in consensus.node_results],
        )
        self.events.append(event)
        return event

    def get_stats(self) -> dict:
        if not self.events:
            return {"n_approvals": 0}
        resonances = [e.resonance for e in self.events]
        finite_res = [r for r in resonances if r != float('inf')]
        return {
            "n_approvals": len(self.events),
            "mean_resonance": round(float(np.mean(finite_res)), 4) if finite_res else float('inf'),
            "min_resonance": round(float(np.min(finite_res)), 4) if finite_res else 0,
            "max_resonance": round(float(np.max(finite_res)), 4) if finite_res else float('inf'),
        }


# ============================================================================
# Hard Gate
# ============================================================================

@dataclass
class HardGateEvent:
    """Record of a hard gate decoherence event."""
    step: int
    grad_S: float              # sqrt(dissonance_density) that triggered the gate
    consensus_ratio: float
    failed_output: str         # The output that failed verification
    clean_consensus: list      # Passing nodes' outputs used for context overwrite
    regenerated: bool = False  # True if regeneration was attempted
    regenerated_output: str = ""  # The output after context overwrite + regeneration
    regenerated_verified: Optional[bool] = None  # Did the regenerated output pass?
    timestamp: float = field(default_factory=time.time)


class HardGate:
    """Hard Gate (Projective Measurement).

    When verification fails:
    1. Halt the current generation.
    2. Collect the passing nodes' outputs (the clean consensus).
    3. Overwrite the drifted agent's context window with the consensus.
    4. Force regeneration of that specific step.
    5. Log the decoherence event for trajectory analysis.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.events: list = []
        self.log_dir = log_dir
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)

    def halt_and_correct(
        self,
        step: int,
        consensus: ConsensusResult,
        failed_output: str,
        regenerated_output: str = "",
        regenerated_verified: Optional[bool] = None,
    ) -> HardGateEvent:
        """Execute the hard gate protocol.

        Returns the event record containing the full decoherence trajectory.
        """
        event = HardGateEvent(
            step=step,
            grad_S=consensus.grad_S,
            consensus_ratio=consensus.consensus_ratio,
            failed_output=failed_output,
            clean_consensus=consensus.passing_outputs,
            regenerated=bool(regenerated_output),
            regenerated_output=regenerated_output,
            regenerated_verified=regenerated_verified,
        )
        self.events.append(event)

        # Log to file if configured
        if self.log_dir:
            log_path = self.log_dir / f"decoherence_step{step}.json"
            log_path.write_text(json.dumps({
                "event_index": len(self.events),
                "step": step,
                "grad_S": round(consensus.grad_S, 4),
                "resonance": round(consensus.resonance, 4),
                "consensus_ratio": round(consensus.consensus_ratio, 4),
                "failed_output": failed_output[:300],
                "clean_consensus": consensus.passing_outputs,
                "regenerated": bool(regenerated_output),
                "regenerated_output": regenerated_output[:300] if regenerated_output else "",
                "regenerated_verified": regenerated_verified,
            }, indent=2), encoding="utf-8")

        return event

    def get_stats(self) -> dict:
        if not self.events:
            return {"n_events": 0, "recovery_rate": 0}
        n_recovered = sum(1 for e in self.events if e.regenerated_verified is True)
        n_attempted = sum(1 for e in self.events if e.regenerated)
        return {
            "n_events": len(self.events),
            "n_regeneration_attempts": n_attempted,
            "recovery_rate": round(n_recovered / max(n_attempted, 1), 4),
            "mean_grad_S": round(float(np.mean([e.grad_S for e in self.events])), 4),
            "max_grad_S": round(float(np.max([e.grad_S for e in self.events])), 4),
        }


# ============================================================================
# Context Reconstruction (for Hard Gate)
# ============================================================================

def build_correction_context(
    original_prompt: str,
    failed_output: str,
    clean_consensus: list,
    correction_message: str = "",
) -> list:
    """Build a reconstructed conversation context for the hard gate.

    Structure:
        1. Original prompt (user)
        2. Failed output (assistant)
        3. Correction message (user - the clean consensus)
        4. (Model regenerates from here)
    """
    if not correction_message:
        correction_message = (
            "[VERIFICATION FAILED: The previous output contained errors. "
            "Here is the corrected information that must be used going forward: "
            + " | ".join(clean_consensus[:2])
            + " ]"
        )

    return [
        {"role": "user", "content": original_prompt},
        {"role": "assistant", "content": failed_output},
        {"role": "user", "content": correction_message},
    ]
