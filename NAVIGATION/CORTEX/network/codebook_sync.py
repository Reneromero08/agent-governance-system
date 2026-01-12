#!/usr/bin/env python3
"""
Codebook Sync Protocol - Implementation of CODEBOOK_SYNC_PROTOCOL.md

Establishes shared side-information (S) between sender and receiver so that
H(X|S) ≈ 0 for semantic pointer compression.

Per the protocol:
- The sync protocol defines a Markov blanket between communicating parties
- When blankets align (hashes match), information transfers deterministically
- When they diverge, the system fails closed until resync

Reference:
- LAW/CANON/SEMANTIC/CODEBOOK_SYNC_PROTOCOL.md
- Q35 (Markov Blankets): R-gating = blanket maintenance
- Q33 (Conditional Entropy): σ^Df = N derivation
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Protocol version
PROTOCOL_VERSION = "1.0.0"


class SyncStatus(Enum):
    """Sync response status values per Section 3.2."""
    MATCHED = "MATCHED"
    MISMATCHED = "MISMATCHED"
    INCOMPATIBLE = "INCOMPATIBLE"
    ERROR = "ERROR"


class BlanketStatus(Enum):
    """Markov blanket status per Section 7.4."""
    ALIGNED = "ALIGNED"       # R > τ, stable blanket, semantic transfer permitted
    DISSOLVED = "DISSOLVED"   # R < τ, blanket broken, resync required
    PENDING = "PENDING"       # R ≈ τ, boundary forming, awaiting confirmation
    UNSYNCED = "UNSYNCED"     # No sync attempted yet


class SyncErrorCode(Enum):
    """Sync-specific failure codes per Section 4."""
    # Sync-specific
    E_SYNC_REQUIRED = "E_SYNC_REQUIRED"
    E_SYNC_EXPIRED = "E_SYNC_EXPIRED"
    E_SYNC_TIMEOUT = "E_SYNC_TIMEOUT"
    E_PROTOCOL_VERSION = "E_PROTOCOL_VERSION"
    E_BLANKET_DISSOLVED = "E_BLANKET_DISSOLVED"
    # Codebook failures
    E_CODEBOOK_MISMATCH = "E_CODEBOOK_MISMATCH"
    E_KERNEL_VERSION = "E_KERNEL_VERSION"
    E_TOKENIZER_MISMATCH = "E_TOKENIZER_MISMATCH"
    E_CODEBOOK_NOT_FOUND = "E_CODEBOOK_NOT_FOUND"
    # Migration failures
    E_MIGRATION_NOT_FOUND = "E_MIGRATION_NOT_FOUND"
    E_MIGRATION_FAILED = "E_MIGRATION_FAILED"
    E_MIGRATION_HASH_MISMATCH = "E_MIGRATION_HASH_MISMATCH"
    E_MIGRATION_NOT_ALLOWED = "E_MIGRATION_NOT_ALLOWED"


@dataclass
class SyncTuple:
    """Sync tuple exchanged during handshake per Section 1.2."""
    codebook_id: str
    codebook_sha256: str
    codebook_semver: str
    kernel_version: str
    tokenizer_id: str

    def to_dict(self) -> Dict:
        return {
            "codebook_id": self.codebook_id,
            "codebook_sha256": self.codebook_sha256,
            "codebook_semver": self.codebook_semver,
            "kernel_version": self.kernel_version,
            "tokenizer_id": self.tokenizer_id
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SyncTuple":
        return cls(
            codebook_id=data.get("codebook_id", ""),
            codebook_sha256=data.get("codebook_sha256", ""),
            codebook_semver=data.get("codebook_semver", "0.0.0"),
            kernel_version=data.get("kernel_version", "1.0.0"),
            tokenizer_id=data.get("tokenizer_id", "tiktoken/o200k_base")
        )


@dataclass
class BlanketHealth:
    """Blanket health metrics per Section 8.4."""
    blanket_health: float  # Composite health score [0, 1]
    drift_velocity: float  # Rate of health decline per second
    predicted_dissolution: Optional[str]  # ISO timestamp or None
    health_factors: Dict = field(default_factory=dict)


class CodebookSync:
    """Codebook synchronization protocol implementation.

    Handles:
    - SyncRequest/SyncResponse message creation and validation
    - Continuous R-value computation (Section 7.5)
    - Blanket alignment checking (Section 7.4)
    - Fail-closed enforcement
    """

    # Default weights for R-value computation (Section 7.5)
    DEFAULT_WEIGHTS = {
        "kernel_version": 1.0,
        "codebook_semver": 0.7,
        "tokenizer_id": 0.5
    }

    # Threshold for blanket alignment
    R_THRESHOLD = 0.5

    def __init__(self, sender_id: str = None):
        """Initialize sync protocol.

        Args:
            sender_id: Unique identifier for this sender
        """
        self.sender_id = sender_id or f"agent-{uuid.uuid4().hex[:8]}"
        self._session_token: Optional[str] = None
        self._session_start: Optional[datetime] = None
        self._last_sync: Optional[datetime] = None
        self._sync_tuple: Optional[SyncTuple] = None
        self._blanket_status = BlanketStatus.UNSYNCED
        self._heartbeat_streak = 0
        self._prev_health: Optional[float] = None
        self._prev_health_time: Optional[datetime] = None

    def create_sync_request(
        self,
        sync_tuple: SyncTuple,
        capabilities: List[str] = None,
        session_id: str = None
    ) -> Dict:
        """Create SyncRequest message per Section 3.1.

        Args:
            sync_tuple: Local sync tuple to send
            capabilities: Optional list of supported pointer types
            session_id: Optional session ID for session-scoped sync

        Returns:
            SyncRequest message dictionary
        """
        request_id = f"sync-{uuid.uuid4().hex[:8]}"

        request = {
            "message_type": "SYNC_REQUEST",
            "protocol_version": PROTOCOL_VERSION,
            "sender_id": self.sender_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sync_tuple": sync_tuple.to_dict(),
            "request_id": request_id
        }

        if capabilities:
            request["capabilities"] = capabilities
        if session_id:
            request["session_id"] = session_id

        return request

    def create_sync_response(
        self,
        request: Dict,
        local_tuple: SyncTuple,
        ttl_seconds: int = 3600
    ) -> Dict:
        """Create SyncResponse message per Section 3.2.

        Args:
            request: Incoming SyncRequest
            local_tuple: Receiver's sync tuple
            ttl_seconds: Session TTL in seconds

        Returns:
            SyncResponse message dictionary
        """
        remote_tuple = SyncTuple.from_dict(request.get("sync_tuple", {}))

        # Check alignment
        is_match, mismatches = self.sync_tuples_match(remote_tuple, local_tuple)
        status = SyncStatus.MATCHED if is_match else SyncStatus.MISMATCHED
        blanket_status = BlanketStatus.ALIGNED if is_match else BlanketStatus.DISSOLVED

        response = {
            "message_type": "SYNC_RESPONSE",
            "protocol_version": PROTOCOL_VERSION,
            "receiver_id": self.sender_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request_id": request.get("request_id", ""),
            "status": status.value,
            "sync_tuple": local_tuple.to_dict(),
            "blanket_status": blanket_status.value
        }

        if is_match:
            # Generate session token
            session_token = f"sess-{uuid.uuid4().hex[:12]}"
            response["session_token"] = session_token
            response["ttl_seconds"] = ttl_seconds
        else:
            response["mismatch_fields"] = mismatches
            # Check if migration is available (simplified)
            response["migration_available"] = False

        return response

    def create_sync_error(
        self,
        request_id: str,
        error_code: SyncErrorCode,
        error_detail: str,
        retry_after: int = 0
    ) -> Dict:
        """Create SyncError message per Section 3.3.

        Args:
            request_id: Request ID being responded to
            error_code: Error code from SyncErrorCode enum
            error_detail: Human-readable error description
            retry_after: Seconds to wait before retry

        Returns:
            SyncError message dictionary
        """
        return {
            "message_type": "SYNC_ERROR",
            "protocol_version": PROTOCOL_VERSION,
            "receiver_id": self.sender_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "error_code": error_code.value,
            "error_detail": error_detail,
            "retry_after_seconds": retry_after
        }

    def create_heartbeat(
        self,
        session_token: str,
        local_codebook_hash: str
    ) -> Dict:
        """Create SYNC_HEARTBEAT message per Section 8.3.

        Args:
            session_token: Active session token
            local_codebook_hash: Current local codebook SHA-256

        Returns:
            Heartbeat message dictionary
        """
        return {
            "message_type": "SYNC_HEARTBEAT",
            "session_token": session_token,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "local_codebook_sha256": local_codebook_hash
        }

    def create_heartbeat_ack(
        self,
        session_token: str,
        blanket_status: BlanketStatus,
        ttl_remaining: int,
        health: BlanketHealth = None
    ) -> Dict:
        """Create HEARTBEAT_ACK message per Section 8.4.

        Args:
            session_token: Session token being acknowledged
            blanket_status: Current blanket status
            ttl_remaining: Remaining session TTL in seconds
            health: Optional blanket health metrics

        Returns:
            Heartbeat acknowledgment dictionary
        """
        response = {
            "message_type": "HEARTBEAT_ACK",
            "session_token": session_token,
            "blanket_status": blanket_status.value,
            "ttl_remaining_seconds": ttl_remaining
        }

        if health:
            response["health"] = {
                "blanket_health": health.blanket_health,
                "drift_velocity": health.drift_velocity,
                "predicted_dissolution": health.predicted_dissolution,
                "warning": self._get_health_warning(health)
            }

        return response

    def verify_sync_response(
        self,
        request: Dict,
        response: Dict
    ) -> Tuple[bool, str]:
        """Verify a sync response is valid.

        Args:
            request: Original SyncRequest
            response: Received SyncResponse

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check request_id correlation
        if response.get("request_id") != request.get("request_id"):
            return False, "Request ID mismatch"

        # Check protocol version
        if response.get("protocol_version") != PROTOCOL_VERSION:
            return False, f"Protocol version mismatch: {response.get('protocol_version')}"

        # Check message type
        msg_type = response.get("message_type")
        if msg_type == "SYNC_ERROR":
            return False, f"Sync error: {response.get('error_code')}: {response.get('error_detail')}"

        if msg_type != "SYNC_RESPONSE":
            return False, f"Invalid message type: {msg_type}"

        # Check status
        status = response.get("status")
        if status == SyncStatus.MATCHED.value:
            # Store session info
            self._session_token = response.get("session_token")
            self._blanket_status = BlanketStatus.ALIGNED
            self._last_sync = datetime.now(timezone.utc)
            self._heartbeat_streak = 0
            return True, "Sync successful"
        elif status == SyncStatus.MISMATCHED.value:
            self._blanket_status = BlanketStatus.DISSOLVED
            mismatches = response.get("mismatch_fields", [])
            return False, f"Sync mismatch on fields: {mismatches}"
        else:
            return False, f"Unknown status: {status}"

    def sync_tuples_match(
        self,
        sender: SyncTuple,
        receiver: SyncTuple
    ) -> Tuple[bool, List[str]]:
        """Compare sync tuples per Section 5.1 (exact match policy).

        Args:
            sender: Sender's sync tuple
            receiver: Receiver's sync tuple

        Returns:
            Tuple of (all_match, list_of_mismatched_fields)
        """
        mismatches = []

        # Critical: codebook_sha256 must match exactly
        if sender.codebook_sha256 != receiver.codebook_sha256:
            mismatches.append("codebook_sha256")

        # Kernel version must match
        if sender.kernel_version != receiver.kernel_version:
            mismatches.append("kernel_version")

        # Tokenizer must match
        if sender.tokenizer_id != receiver.tokenizer_id:
            mismatches.append("tokenizer_id")

        return len(mismatches) == 0, mismatches

    def compute_continuous_r(
        self,
        sender_tuple: SyncTuple,
        receiver_tuple: SyncTuple,
        weights: Dict[str, float] = None
    ) -> float:
        """Compute continuous R-value per Section 7.5.

        Formula:
            R = gate(codebook_sha256) × (Σᵢ wᵢ · score(fieldᵢ)) / (Σᵢ wᵢ)

        Args:
            sender_tuple: Sender's sync tuple
            receiver_tuple: Receiver's sync tuple
            weights: Optional custom weights (default: DEFAULT_WEIGHTS)

        Returns:
            R-value in [0, 1]
        """
        # Hard gate: codebook hash must match
        if sender_tuple.codebook_sha256 != receiver_tuple.codebook_sha256:
            return 0.0

        weights = weights or self.DEFAULT_WEIGHTS

        # Score each field
        scores = {
            "kernel_version": self._score_kernel_version(
                sender_tuple.kernel_version,
                receiver_tuple.kernel_version
            ),
            "codebook_semver": self._score_semver(
                sender_tuple.codebook_semver,
                receiver_tuple.codebook_semver
            ),
            "tokenizer_id": self._score_tokenizer(
                sender_tuple.tokenizer_id,
                receiver_tuple.tokenizer_id
            )
        }

        # Weighted average
        weighted_sum = sum(weights[f] * scores[f] for f in weights)
        total_weight = sum(weights.values())

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def check_blanket_alignment(
        self,
        local_tuple: SyncTuple,
        remote_tuple: SyncTuple,
        threshold: float = None
    ) -> BlanketStatus:
        """Determine blanket alignment status per Section 7.4.

        Args:
            local_tuple: Local sync tuple
            remote_tuple: Remote sync tuple
            threshold: R threshold (default: R_THRESHOLD)

        Returns:
            BlanketStatus enum value
        """
        threshold = threshold or self.R_THRESHOLD

        r_value = self.compute_continuous_r(local_tuple, remote_tuple)

        if r_value >= 0.8:
            return BlanketStatus.ALIGNED
        elif r_value >= threshold:
            return BlanketStatus.PENDING
        else:
            return BlanketStatus.DISSOLVED

    def compute_blanket_health(
        self,
        r_value: float,
        ttl_seconds: int,
        elapsed_seconds: float
    ) -> BlanketHealth:
        """Compute blanket health metrics per Section 8.4.

        Args:
            r_value: Current R-value
            ttl_seconds: Session TTL
            elapsed_seconds: Time since last sync

        Returns:
            BlanketHealth with composite health and predictions
        """
        now = datetime.now(timezone.utc)

        # Factor 1: R-value
        r_factor = r_value

        # Factor 2: TTL fraction remaining
        ttl_fraction = max(0, 1 - elapsed_seconds / ttl_seconds) if ttl_seconds > 0 else 0

        # Factor 3: Heartbeat reliability
        heartbeat_factor = min(1.0, self._heartbeat_streak / 10)

        # Factor 4: Resync recency
        if self._last_sync:
            resync_age = (now - self._last_sync).total_seconds()
            resync_factor = 1 / (1 + resync_age / 86400)  # Decay over 24h
        else:
            resync_factor = 0.5

        # Composite health (weighted geometric mean)
        weights = [0.4, 0.3, 0.15, 0.15]
        factors = [r_factor, ttl_fraction, heartbeat_factor, resync_factor]

        # Geometric mean
        health = 1.0
        for f, w in zip(factors, weights):
            if f > 0:
                health *= (f ** w)
            else:
                health = 0.0
                break

        # Drift velocity
        drift_velocity = 0.0
        if self._prev_health is not None and self._prev_health_time is not None:
            dt = (now - self._prev_health_time).total_seconds()
            if dt > 0:
                drift_velocity = (self._prev_health - health) / dt

        # Update history
        self._prev_health = health
        self._prev_health_time = now

        # Predict dissolution
        dissolution_threshold = 0.5
        predicted_dissolution = None
        if drift_velocity > 0 and health > dissolution_threshold:
            time_to_dissolution = (health - dissolution_threshold) / drift_velocity
            predicted = now + timedelta(seconds=time_to_dissolution)
            predicted_dissolution = predicted.isoformat()

        return BlanketHealth(
            blanket_health=round(health, 4),
            drift_velocity=round(drift_velocity, 6),
            predicted_dissolution=predicted_dissolution,
            health_factors={
                "r_value": round(r_factor, 4),
                "ttl_fraction": round(ttl_fraction, 4),
                "heartbeat_streak": self._heartbeat_streak,
                "resync_factor": round(resync_factor, 4)
            }
        )

    def _score_kernel_version(self, sender: str, receiver: str) -> float:
        """Score kernel version compatibility."""
        try:
            s_parts = [int(x) for x in sender.split(".")]
            r_parts = [int(x) for x in receiver.split(".")]

            if s_parts[0] != r_parts[0]:
                return 0.0  # Major mismatch
            if len(s_parts) > 1 and len(r_parts) > 1 and s_parts[1] != r_parts[1]:
                return 0.7  # Minor mismatch
            if len(s_parts) > 2 and len(r_parts) > 2 and s_parts[2] != r_parts[2]:
                return 0.9  # Patch mismatch
            return 1.0
        except (ValueError, IndexError):
            return 0.0

    def _score_semver(self, sender: str, receiver: str) -> float:
        """Score semver compatibility."""
        return self._score_kernel_version(sender, receiver)

    def _score_tokenizer(self, sender: str, receiver: str) -> float:
        """Score tokenizer compatibility."""
        if sender == receiver:
            return 1.0

        # Known compatible families
        sender_family = sender.split("/")[0] if "/" in sender else sender
        receiver_family = receiver.split("/")[0] if "/" in receiver else receiver

        if sender_family == receiver_family:
            return 0.8

        return 0.0  # Unknown compatibility = fail-closed

    def _get_health_warning(self, health: BlanketHealth) -> Optional[str]:
        """Get health warning based on metrics."""
        if health.blanket_health < 0.5:
            return "HEALTH_CRITICAL"
        elif health.blanket_health < 0.8:
            return "HEALTH_DEGRADED"
        elif health.drift_velocity > 0.01:
            return "DRIFT_DETECTED"
        elif health.predicted_dissolution:
            # Check if within 1 hour
            try:
                pred_time = datetime.fromisoformat(health.predicted_dissolution.replace('Z', '+00:00'))
                if pred_time - datetime.now(timezone.utc) < timedelta(hours=1):
                    return "DISSOLUTION_IMMINENT"
            except ValueError:
                pass
        return None

    def record_heartbeat_success(self):
        """Record a successful heartbeat."""
        self._heartbeat_streak += 1

    def record_heartbeat_failure(self):
        """Record a failed heartbeat."""
        self._heartbeat_streak = 0
        self._blanket_status = BlanketStatus.DISSOLVED

    @property
    def blanket_status(self) -> BlanketStatus:
        return self._blanket_status

    @property
    def session_token(self) -> Optional[str]:
        return self._session_token

    @property
    def is_synced(self) -> bool:
        return self._blanket_status == BlanketStatus.ALIGNED


# ============================================================================
# Convenience Functions
# ============================================================================

def create_sync_tuple_from_codebook(
    codebook_path: Path,
    codebook_id: str = "ags-codebook",
    kernel_version: str = "1.0.0",
    tokenizer_id: str = "tiktoken/o200k_base"
) -> SyncTuple:
    """Create a SyncTuple from a codebook file.

    Args:
        codebook_path: Path to CODEBOOK.json
        codebook_id: Codebook identifier
        kernel_version: Kernel version
        tokenizer_id: Tokenizer identifier

    Returns:
        SyncTuple with computed hash
    """
    with open(codebook_path, 'r', encoding='utf-8') as f:
        codebook = json.load(f)

    # Compute canonical hash
    canonical = json.dumps(codebook, sort_keys=True, separators=(',', ':'))
    codebook_sha256 = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    # Get version from codebook
    codebook_semver = codebook.get("version", "0.0.0")

    return SyncTuple(
        codebook_id=codebook_id,
        codebook_sha256=codebook_sha256,
        codebook_semver=codebook_semver,
        kernel_version=kernel_version,
        tokenizer_id=tokenizer_id
    )


def verify_codebook_hash(codebook_path: Path, expected_hash: str) -> bool:
    """Verify codebook hash matches expected value.

    Args:
        codebook_path: Path to CODEBOOK.json
        expected_hash: Expected SHA-256 hash

    Returns:
        True if hash matches, False otherwise
    """
    with open(codebook_path, 'r', encoding='utf-8') as f:
        codebook = json.load(f)

    canonical = json.dumps(codebook, sort_keys=True, separators=(',', ':'))
    actual_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    return actual_hash == expected_hash


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Classes
    "CodebookSync",
    "SyncTuple",
    "BlanketHealth",
    # Enums
    "SyncStatus",
    "BlanketStatus",
    "SyncErrorCode",
    # Functions
    "create_sync_tuple_from_codebook",
    "verify_codebook_hash",
    # Constants
    "PROTOCOL_VERSION",
]
