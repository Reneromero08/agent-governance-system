#!/usr/bin/env python3
"""SVTP - Semantic Vector Transport Protocol.

The transport layer for cross-model vector communication.
Structures 256D vectors like TCP packets with:
- Semantic payload (holographic thought)
- Pilot tone (geometric checksum)
- Rotation hash (auth token)
- Scalar clock (sequence)

Protocol version: 1.0
Spec date: 2026-01-17

Usage:
    from vector_packet import SVTPEncoder, SVTPDecoder, SVTP_256

    # Sender
    encoder = SVTPEncoder(alignment_key, embed_fn)
    packet = encoder.encode("Hello world", sequence=0)

    # Receiver
    decoder = SVTPDecoder(alignment_key, embed_fn)
    result = decoder.decode(packet.vector, candidates)
    if result.valid:
        print(result.payload)  # "Hello world"
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import numpy as np
from hashlib import sha256

# Protocol constants
SVTP_VERSION = "1.0"

# 256D packet structure
PAYLOAD_START = 0
PAYLOAD_END = 200
PILOT_START = 200
PILOT_END = 220
AUTH_START = 220
AUTH_END = 255
SEQ_INDEX = 255

# Dimensions per section
PAYLOAD_DIMS = PAYLOAD_END - PAYLOAD_START  # 200
PILOT_DIMS = PILOT_END - PILOT_START        # 20
AUTH_DIMS = AUTH_END - AUTH_START           # 35
SEQ_DIMS = 1

TOTAL_DIMS = 256

# The Pilot Tone: A concept we always encode in the checksum slot
# If this decodes to anything other than "truth", the packet is corrupt
PILOT_CONCEPT = "truth"
PILOT_CANDIDATES = ["truth", "false", "error", "noise", "unknown"]


@dataclass
class PacketStructure:
    """Defines the structure of an SVTP packet."""
    total_dims: int = TOTAL_DIMS
    payload_range: Tuple[int, int] = (PAYLOAD_START, PAYLOAD_END)
    pilot_range: Tuple[int, int] = (PILOT_START, PILOT_END)
    auth_range: Tuple[int, int] = (AUTH_START, AUTH_END)
    seq_index: int = SEQ_INDEX

    @property
    def payload_dims(self) -> int:
        return self.payload_range[1] - self.payload_range[0]

    @property
    def pilot_dims(self) -> int:
        return self.pilot_range[1] - self.pilot_range[0]

    @property
    def auth_dims(self) -> int:
        return self.auth_range[1] - self.auth_range[0]


# Default structure
SVTP_256 = PacketStructure()


@dataclass
class SVTPPacket:
    """A structured vector packet."""
    vector: np.ndarray                    # Full 256D vector
    sequence: int                         # Packet sequence number
    payload_text: Optional[str] = None    # Original text (for debugging)
    structure: PacketStructure = field(default_factory=lambda: SVTP_256)

    @property
    def payload(self) -> np.ndarray:
        """Extract payload section."""
        start, end = self.structure.payload_range
        return self.vector[start:end]

    @property
    def pilot_tone(self) -> np.ndarray:
        """Extract pilot tone section."""
        start, end = self.structure.pilot_range
        return self.vector[start:end]

    @property
    def auth_token(self) -> np.ndarray:
        """Extract auth token section."""
        start, end = self.structure.auth_range
        return self.vector[start:end]

    @property
    def scalar_clock(self) -> float:
        """Extract sequence as scalar (0.0 to 1.0)."""
        return self.vector[self.structure.seq_index]

    def to_bytes(self) -> bytes:
        """Serialize packet for transmission."""
        return self.vector.astype(np.float32).tobytes()

    @classmethod
    def from_bytes(cls, data: bytes, structure: PacketStructure = SVTP_256) -> 'SVTPPacket':
        """Deserialize packet from bytes."""
        vector = np.frombuffer(data, dtype=np.float32)
        sequence = int(vector[structure.seq_index] * 256) % 256
        return cls(vector=vector, sequence=sequence, structure=structure)


@dataclass
class DecodeResult:
    """Result of decoding an SVTP packet."""
    payload: Optional[str]        # Decoded text (None if invalid)
    confidence: float             # Semantic match confidence
    valid: bool                   # Packet passed all checks
    pilot_valid: bool             # Pilot tone matched
    auth_valid: bool              # Auth token verified
    sequence: int                 # Extracted sequence number
    error: Optional[str] = None   # Error message if invalid


class SVTPEncoder:
    """Encodes text into SVTP packets."""

    def __init__(
        self,
        alignment_key,  # AlignmentKey instance
        embed_fn: Callable,
        structure: PacketStructure = SVTP_256,
        rotation_matrix: Optional[np.ndarray] = None,
    ):
        """Initialize encoder.

        Args:
            alignment_key: The AlignmentKey for this model
            embed_fn: Function to embed text -> vectors
            structure: Packet structure definition
            rotation_matrix: Optional Procrustes rotation for auth
        """
        self.key = alignment_key
        self.embed_fn = embed_fn
        self.structure = structure
        self.R = rotation_matrix

        # Pre-compute pilot tone vector
        self._pilot_vector = self._compute_pilot_vector()

        # Pre-compute auth hash
        self._auth_hash = self._compute_auth_hash()

    def _compute_pilot_vector(self) -> np.ndarray:
        """Compute the pilot tone vector for 'truth'."""
        # Get full encoding of pilot concept
        from .alignment_key import AlignmentKey
        full_vec = self.key.encode(PILOT_CONCEPT, self.embed_fn)

        # Use middle dimensions for pilot (less correlated with payload)
        pilot_dims = self.structure.pilot_dims
        if len(full_vec) >= pilot_dims:
            # Take from middle of spectrum
            mid = len(full_vec) // 2
            start = mid - pilot_dims // 2
            return full_vec[start:start + pilot_dims]
        else:
            # Pad if needed
            result = np.zeros(pilot_dims)
            result[:len(full_vec)] = full_vec
            return result

    def _compute_auth_hash(self) -> np.ndarray:
        """Compute auth token from rotation matrix."""
        auth_dims = self.structure.auth_dims

        if self.R is not None:
            # Flatten rotation matrix and hash into auth dimensions
            flat = self.R.flatten()

            # Use eigenvalues of R for a compact signature
            if self.R.shape[0] == self.R.shape[1]:
                eigvals = np.linalg.eigvals(self.R)
                # Take real parts, sorted
                signature = np.sort(np.real(eigvals))[:auth_dims]
                if len(signature) < auth_dims:
                    result = np.zeros(auth_dims)
                    result[:len(signature)] = signature
                    return result
                return signature
            else:
                # Non-square: use SVD singular values
                s = np.linalg.svd(self.R, compute_uv=False)
                result = np.zeros(auth_dims)
                result[:min(len(s), auth_dims)] = s[:auth_dims]
                return result
        else:
            # No rotation: use anchor hash as fallback
            hash_bytes = sha256(self.key.anchor_hash.encode()).digest()
            # Convert to floats in [-1, 1]
            hash_ints = np.frombuffer(hash_bytes, dtype=np.uint8)
            auth = (hash_ints[:auth_dims].astype(float) / 127.5) - 1.0
            if len(auth) < auth_dims:
                result = np.zeros(auth_dims)
                result[:len(auth)] = auth
                return result
            return auth

    def encode(self, text: str, sequence: int = 0) -> SVTPPacket:
        """Encode text into an SVTP packet.

        Args:
            text: The semantic payload to encode
            sequence: Packet sequence number (0-255)

        Returns:
            SVTPPacket with structured vector
        """
        # Initialize full vector
        vector = np.zeros(self.structure.total_dims)

        # 1. Encode semantic payload
        payload_vec = self.key.encode(text, self.embed_fn)
        payload_dims = self.structure.payload_dims
        if len(payload_vec) >= payload_dims:
            vector[PAYLOAD_START:PAYLOAD_END] = payload_vec[:payload_dims]
        else:
            vector[PAYLOAD_START:PAYLOAD_START + len(payload_vec)] = payload_vec

        # 2. Insert pilot tone
        vector[PILOT_START:PILOT_END] = self._pilot_vector

        # 3. Insert auth token
        vector[AUTH_START:AUTH_END] = self._auth_hash

        # 4. Insert scalar clock
        vector[SEQ_INDEX] = (sequence % 256) / 256.0

        return SVTPPacket(
            vector=vector,
            sequence=sequence,
            payload_text=text,
            structure=self.structure,
        )


class SVTPDecoder:
    """Decodes SVTP packets back to text."""

    def __init__(
        self,
        alignment_key,  # AlignmentKey instance
        embed_fn: Callable,
        structure: PacketStructure = SVTP_256,
        expected_auth: Optional[np.ndarray] = None,
        pilot_threshold: float = 0.7,
        auth_threshold: float = 0.9,
    ):
        """Initialize decoder.

        Args:
            alignment_key: The AlignmentKey for this model
            embed_fn: Function to embed text -> vectors
            structure: Packet structure definition
            expected_auth: Expected auth token for verification
            pilot_threshold: Min confidence for pilot tone match
            auth_threshold: Min similarity for auth token match
        """
        self.key = alignment_key
        self.embed_fn = embed_fn
        self.structure = structure
        self.expected_auth = expected_auth
        self.pilot_threshold = pilot_threshold
        self.auth_threshold = auth_threshold

        # Pre-compute expected pilot
        self._expected_pilot = self._compute_expected_pilot()

    def _compute_expected_pilot(self) -> np.ndarray:
        """Compute expected pilot tone."""
        full_vec = self.key.encode(PILOT_CONCEPT, self.embed_fn)
        pilot_dims = self.structure.pilot_dims
        if len(full_vec) >= pilot_dims:
            mid = len(full_vec) // 2
            start = mid - pilot_dims // 2
            return full_vec[start:start + pilot_dims]
        else:
            result = np.zeros(pilot_dims)
            result[:len(full_vec)] = full_vec
            return result

    def _verify_pilot(self, pilot_tone: np.ndarray) -> Tuple[bool, float]:
        """Verify pilot tone matches expected 'truth' vector."""
        # Cosine similarity
        norm_a = np.linalg.norm(pilot_tone)
        norm_b = np.linalg.norm(self._expected_pilot)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return False, 0.0

        similarity = np.dot(pilot_tone, self._expected_pilot) / (norm_a * norm_b)
        return similarity >= self.pilot_threshold, float(similarity)

    def _verify_auth(self, auth_token: np.ndarray) -> Tuple[bool, float]:
        """Verify auth token matches expected."""
        if self.expected_auth is None:
            # No expected auth = accept all
            return True, 1.0

        # Cosine similarity
        norm_a = np.linalg.norm(auth_token)
        norm_b = np.linalg.norm(self.expected_auth)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return False, 0.0

        similarity = np.dot(auth_token, self.expected_auth) / (norm_a * norm_b)
        return similarity >= self.auth_threshold, float(similarity)

    def decode(
        self,
        vector: np.ndarray,
        candidates: List[str],
        verify_pilot: bool = True,
        verify_auth: bool = True,
    ) -> DecodeResult:
        """Decode an SVTP packet.

        Args:
            vector: The 256D packet vector
            candidates: Possible payload texts to match against
            verify_pilot: Check pilot tone integrity
            verify_auth: Check auth token validity

        Returns:
            DecodeResult with decoded payload and validity info
        """
        # Extract sections
        payload = vector[PAYLOAD_START:PAYLOAD_END]
        pilot = vector[PILOT_START:PILOT_END]
        auth = vector[AUTH_START:AUTH_END]
        seq_val = vector[SEQ_INDEX]
        sequence = int(seq_val * 256) % 256

        # Verify pilot tone
        pilot_valid = True
        pilot_conf = 1.0
        if verify_pilot:
            pilot_valid, pilot_conf = self._verify_pilot(pilot)
            if not pilot_valid:
                return DecodeResult(
                    payload=None,
                    confidence=0.0,
                    valid=False,
                    pilot_valid=False,
                    auth_valid=True,  # Didn't check
                    sequence=sequence,
                    error=f"Pilot tone mismatch: {pilot_conf:.3f} < {self.pilot_threshold}",
                )

        # Verify auth token
        auth_valid = True
        auth_conf = 1.0
        if verify_auth and self.expected_auth is not None:
            auth_valid, auth_conf = self._verify_auth(auth)
            if not auth_valid:
                return DecodeResult(
                    payload=None,
                    confidence=0.0,
                    valid=False,
                    pilot_valid=pilot_valid,
                    auth_valid=False,
                    sequence=sequence,
                    error=f"Auth token mismatch: {auth_conf:.3f} < {self.auth_threshold}",
                )

        # Decode payload
        # We need to use the payload dimensions for decoding
        # This requires the alignment key to support partial vector decode
        match, confidence = self._decode_payload(payload, candidates)

        return DecodeResult(
            payload=match,
            confidence=confidence,
            valid=True,
            pilot_valid=pilot_valid,
            auth_valid=auth_valid,
            sequence=sequence,
        )

    def _decode_payload(
        self,
        payload: np.ndarray,
        candidates: List[str],
    ) -> Tuple[str, float]:
        """Decode payload vector to best-matching candidate."""
        # Embed all candidates and project to payload space
        best_match = None
        best_score = -1.0

        for candidate in candidates:
            # Encode candidate
            cand_vec = self.key.encode(candidate, self.embed_fn)

            # Truncate to payload dims
            if len(cand_vec) > len(payload):
                cand_vec = cand_vec[:len(payload)]
            elif len(cand_vec) < len(payload):
                padded = np.zeros(len(payload))
                padded[:len(cand_vec)] = cand_vec
                cand_vec = padded

            # Cosine similarity
            norm_p = np.linalg.norm(payload)
            norm_c = np.linalg.norm(cand_vec)
            if norm_p < 1e-8 or norm_c < 1e-8:
                continue

            score = np.dot(payload, cand_vec) / (norm_p * norm_c)
            if score > best_score:
                best_score = score
                best_match = candidate

        return best_match, float(best_score)


# =============================================================================
# Cross-Model SVTP (uses AlignedKeyPair)
# =============================================================================

class CrossModelEncoder:
    """Encoder for cross-model SVTP communication.

    Uses AlignedKeyPair to encode messages that can be decoded by
    a different embedding model. All sections (payload, pilot, auth)
    are encoded in the TARGET model's coordinate space.
    """

    def __init__(
        self,
        aligned_pair,  # AlignedKeyPair instance
        embed_fn: Callable,
        is_model_a: bool = True,
        structure: PacketStructure = SVTP_256,
    ):
        """Initialize cross-model encoder.

        Args:
            aligned_pair: AlignedKeyPair for A<->B communication
            embed_fn: Embedding function for THIS model
            is_model_a: True if this is model A, False if model B
            structure: Packet structure definition
        """
        self.pair = aligned_pair
        self.embed_fn = embed_fn
        self.is_model_a = is_model_a
        self.structure = structure
        self.k = aligned_pair.k

        # Get rotation matrix for auth
        if is_model_a:
            self.R = aligned_pair.R_a_to_b
        else:
            self.R = aligned_pair.R_b_to_a

        # Pre-compute pilot vector (in TARGET's space)
        self._pilot_vector = self._compute_pilot_vector()

        # Pre-compute auth hash
        self._auth_hash = self._compute_auth_hash()

    def _compute_pilot_vector(self) -> np.ndarray:
        """Compute pilot tone in TARGET model's coordinate space."""
        # Encode pilot concept and rotate to target's space
        if self.is_model_a:
            full_vec = self.pair.encode_a_to_b(PILOT_CONCEPT, self.embed_fn)
        else:
            full_vec = self.pair.encode_b_to_a(PILOT_CONCEPT, self.embed_fn)

        pilot_dims = self.structure.pilot_dims
        if len(full_vec) >= pilot_dims:
            mid = len(full_vec) // 2
            start = mid - pilot_dims // 2
            return full_vec[start:start + pilot_dims]
        else:
            result = np.zeros(pilot_dims)
            result[:len(full_vec)] = full_vec
            return result

    def _compute_auth_hash(self) -> np.ndarray:
        """Compute auth token from rotation matrix."""
        auth_dims = self.structure.auth_dims
        R = self.R

        if R.shape[0] == R.shape[1]:
            eigvals = np.linalg.eigvals(R)
            signature = np.sort(np.real(eigvals))[:auth_dims]
            if len(signature) < auth_dims:
                result = np.zeros(auth_dims)
                result[:len(signature)] = signature
                return result
            return signature
        else:
            s = np.linalg.svd(R, compute_uv=False)
            result = np.zeros(auth_dims)
            result[:min(len(s), auth_dims)] = s[:auth_dims]
            return result

    def encode_to_other(self, text: str, sequence: int = 0) -> SVTPPacket:
        """Encode text for transmission to the OTHER model.

        The payload is encoded using the AlignedKeyPair's cross-model
        rotation, so it arrives in the target model's coordinate space.

        Args:
            text: Semantic payload
            sequence: Packet sequence number

        Returns:
            SVTPPacket ready for transmission to the other model
        """
        # Initialize full vector
        vector = np.zeros(self.structure.total_dims)

        # 1. Encode semantic payload using pair's cross-model method
        #    This applies the Procrustes rotation automatically
        if self.is_model_a:
            payload_vec = self.pair.encode_a_to_b(text, self.embed_fn)
        else:
            payload_vec = self.pair.encode_b_to_a(text, self.embed_fn)

        # Copy to payload section
        payload_dims = self.structure.payload_dims
        if len(payload_vec) >= payload_dims:
            vector[PAYLOAD_START:PAYLOAD_END] = payload_vec[:payload_dims]
        else:
            vector[PAYLOAD_START:PAYLOAD_START + len(payload_vec)] = payload_vec

        # 2. Insert pilot tone (already in target's space)
        vector[PILOT_START:PILOT_END] = self._pilot_vector

        # 3. Insert auth token
        vector[AUTH_START:AUTH_END] = self._auth_hash

        # 4. Insert scalar clock
        vector[SEQ_INDEX] = (sequence % 256) / 256.0

        return SVTPPacket(
            vector=vector,
            sequence=sequence,
            payload_text=text,
            structure=self.structure,
        )


class CrossModelDecoder:
    """Decoder for cross-model SVTP communication.

    Decodes packets that were encoded by a different model using
    CrossModelEncoder. Expects all sections to be in THIS model's
    coordinate space.
    """

    def __init__(
        self,
        aligned_pair,  # AlignedKeyPair instance
        embed_fn: Callable,
        is_model_a: bool = True,
        structure: PacketStructure = SVTP_256,
        pilot_threshold: float = 0.5,
        auth_threshold: float = 0.8,
    ):
        """Initialize cross-model decoder.

        Args:
            aligned_pair: AlignedKeyPair for A<->B communication
            embed_fn: Embedding function for THIS model
            is_model_a: True if this is model A, False if model B
            structure: Packet structure definition
            pilot_threshold: Min similarity for pilot tone
            auth_threshold: Min similarity for auth token
        """
        self.pair = aligned_pair
        self.embed_fn = embed_fn
        self.is_model_a = is_model_a
        self.structure = structure
        self.k = aligned_pair.k
        self.pilot_threshold = pilot_threshold
        self.auth_threshold = auth_threshold

        if is_model_a:
            self.key = aligned_pair.key_a
            # Expect auth from model B (their rotation matrix)
            self.expected_auth = self._compute_expected_auth(aligned_pair.R_b_to_a)
        else:
            self.key = aligned_pair.key_b
            # Expect auth from model A (their rotation matrix)
            self.expected_auth = self._compute_expected_auth(aligned_pair.R_a_to_b)

        # Pre-compute expected pilot tone in OUR space
        self._expected_pilot = self._compute_expected_pilot()

    def _compute_expected_auth(self, R: np.ndarray) -> np.ndarray:
        """Compute expected auth token from sender's rotation."""
        auth_dims = self.structure.auth_dims
        if R.shape[0] == R.shape[1]:
            eigvals = np.linalg.eigvals(R)
            signature = np.sort(np.real(eigvals))[:auth_dims]
            if len(signature) < auth_dims:
                result = np.zeros(auth_dims)
                result[:len(signature)] = signature
                return result
            return signature
        else:
            s = np.linalg.svd(R, compute_uv=False)
            result = np.zeros(auth_dims)
            result[:min(len(s), auth_dims)] = s[:auth_dims]
            return result

    def _compute_expected_pilot(self) -> np.ndarray:
        """Compute expected pilot tone in OUR coordinate space."""
        # The pilot was encoded by the OTHER model and rotated to OUR space
        # So we just encode the pilot concept in OUR space directly
        full_vec = self.key.encode(PILOT_CONCEPT, self.embed_fn)
        pilot_dims = self.structure.pilot_dims
        if len(full_vec) >= pilot_dims:
            mid = len(full_vec) // 2
            start = mid - pilot_dims // 2
            return full_vec[start:start + pilot_dims]
        else:
            result = np.zeros(pilot_dims)
            result[:len(full_vec)] = full_vec
            return result

    def _verify_pilot(self, pilot_tone: np.ndarray) -> Tuple[bool, float]:
        """Verify pilot tone matches expected."""
        norm_a = np.linalg.norm(pilot_tone)
        norm_b = np.linalg.norm(self._expected_pilot)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return False, 0.0

        similarity = np.dot(pilot_tone, self._expected_pilot) / (norm_a * norm_b)
        return similarity >= self.pilot_threshold, float(similarity)

    def _verify_auth(self, auth_token: np.ndarray) -> Tuple[bool, float]:
        """Verify auth token matches expected."""
        norm_a = np.linalg.norm(auth_token)
        norm_b = np.linalg.norm(self.expected_auth)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return False, 0.0

        similarity = np.dot(auth_token, self.expected_auth) / (norm_a * norm_b)
        return similarity >= self.auth_threshold, float(similarity)

    def decode(
        self,
        vector: np.ndarray,
        candidates: List[str],
        verify_pilot: bool = True,
        verify_auth: bool = True,
    ) -> DecodeResult:
        """Decode packet from the OTHER model.

        Args:
            vector: Received 256D packet (in OUR coordinate space)
            candidates: Possible payload texts
            verify_pilot: Check pilot tone integrity
            verify_auth: Check auth token validity

        Returns:
            DecodeResult with validity and decoded payload
        """
        # Extract sections
        payload = vector[PAYLOAD_START:PAYLOAD_END]
        pilot = vector[PILOT_START:PILOT_END]
        auth = vector[AUTH_START:AUTH_END]
        seq_val = vector[SEQ_INDEX]
        sequence = int(seq_val * 256) % 256

        # Verify pilot tone
        pilot_valid = True
        pilot_conf = 1.0
        if verify_pilot:
            pilot_valid, pilot_conf = self._verify_pilot(pilot)
            if not pilot_valid:
                return DecodeResult(
                    payload=None,
                    confidence=0.0,
                    valid=False,
                    pilot_valid=False,
                    auth_valid=True,
                    sequence=sequence,
                    error=f"Pilot tone mismatch: {pilot_conf:.3f} < {self.pilot_threshold}",
                )

        # Verify auth token
        auth_valid = True
        auth_conf = 1.0
        if verify_auth:
            auth_valid, auth_conf = self._verify_auth(auth)
            if not auth_valid:
                return DecodeResult(
                    payload=None,
                    confidence=0.0,
                    valid=False,
                    pilot_valid=pilot_valid,
                    auth_valid=False,
                    sequence=sequence,
                    error=f"Auth token mismatch: {auth_conf:.3f} < {self.auth_threshold}",
                )

        # Decode payload - it's already in our coordinate space
        match, confidence = self._decode_payload(payload, candidates)

        return DecodeResult(
            payload=match,
            confidence=confidence,
            valid=True,
            pilot_valid=pilot_valid,
            auth_valid=auth_valid,
            sequence=sequence,
        )

    def _decode_payload(
        self,
        payload: np.ndarray,
        candidates: List[str],
    ) -> Tuple[str, float]:
        """Decode payload to best-matching candidate.

        The payload is already in our coordinate space, so we just
        compare directly against candidates encoded with our key.
        """
        best_match = None
        best_score = -1.0

        for candidate in candidates:
            # Encode candidate in OUR space
            cand_vec = self.key.encode(candidate, self.embed_fn)

            # Use only k dimensions for comparison
            k = min(len(cand_vec), len(payload), self.k)

            # Cosine similarity on first k dims
            norm_p = np.linalg.norm(payload[:k])
            norm_c = np.linalg.norm(cand_vec[:k])
            if norm_p < 1e-8 or norm_c < 1e-8:
                continue

            score = np.dot(payload[:k], cand_vec[:k]) / (norm_p * norm_c)
            if score > best_score:
                best_score = score
                best_match = candidate

        return best_match, float(best_score)


# =============================================================================
# Convenience functions
# =============================================================================

def create_svtp_channel(
    aligned_pair,
    embed_fn_a: Callable,
    embed_fn_b: Callable,
) -> Tuple[CrossModelEncoder, CrossModelDecoder, CrossModelEncoder, CrossModelDecoder]:
    """Create a bidirectional SVTP channel between two models.

    Args:
        aligned_pair: AlignedKeyPair for A<->B
        embed_fn_a: Embedding function for model A
        embed_fn_b: Embedding function for model B

    Returns:
        (encoder_a, decoder_a, encoder_b, decoder_b)
        - encoder_a sends A->B
        - decoder_a receives B->A
        - encoder_b sends B->A
        - decoder_b receives A->B
    """
    encoder_a = CrossModelEncoder(aligned_pair, embed_fn_a, is_model_a=True)
    decoder_a = CrossModelDecoder(aligned_pair, embed_fn_a, is_model_a=True)
    encoder_b = CrossModelEncoder(aligned_pair, embed_fn_b, is_model_a=False)
    decoder_b = CrossModelDecoder(aligned_pair, embed_fn_b, is_model_a=False)

    return encoder_a, decoder_a, encoder_b, decoder_b


def format_packet_hex(packet: SVTPPacket, max_dims: int = 8) -> str:
    """Format packet as hex string for display."""
    lines = []
    lines.append(f"SVTP Packet (seq={packet.sequence})")
    lines.append(f"  Payload[0:{max_dims}]: {packet.payload[:max_dims]}")
    lines.append(f"  Pilot: {packet.pilot_tone[:4]}...")
    lines.append(f"  Auth: {packet.auth_token[:4]}...")
    lines.append(f"  Clock: {packet.scalar_clock:.4f}")
    return "\n".join(lines)
