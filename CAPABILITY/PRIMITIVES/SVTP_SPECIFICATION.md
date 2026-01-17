# SVTP - Semantic Vector Transport Protocol

**Version:** 1.0
**Date:** 2026-01-17
**Status:** VERIFIED

---

## Overview

SVTP (Semantic Vector Transport Protocol) is a transport layer protocol for semantic vector communication between AI models. It structures 256-dimensional vectors like TCP packets, enabling:

- **Holographic semantic payloads** with built-in redundancy
- **Geometric checksums** (pilot tones) for corruption detection
- **Rotation-hash authentication** to verify sender identity
- **Sequence numbering** for packet ordering

---

## Packet Structure

```
    0                                                             255
    +------------------------------------------------------------+
    |                    SEMANTIC PAYLOAD (200D)                  |
    |                    [dims 0-199]                             |
    |                    Holographically encoded thought          |
    +------------------------------------------------------------+
    |          PILOT TONE (20D)         |   AUTH TOKEN (35D)  |S |
    |          [dims 200-219]           |   [dims 220-254]    |E |
    |          Geometric checksum       |   Rotation hash     |Q |
    +------------------------------------------------------------+
```

| Section | Dimensions | Range | Purpose |
|---------|------------|-------|---------|
| Payload | 200 | 0-199 | Semantic content (holographic) |
| Pilot Tone | 20 | 200-219 | Geometric integrity checksum |
| Auth Token | 35 | 220-254 | Sender authentication |
| Sequence | 1 | 255 | Packet ordering (0.0-1.0) |

**Total:** 256 dimensions = 1024 bytes (float32)

---

## Protocol Layers

```
+------------------------------------------+
|  APPLICATION LAYER                        |
|  (Semantic thought / message)             |
+------------------------------------------+
|  SVTP TRANSPORT LAYER                     |
|  - Packet structure                       |
|  - Pilot tone verification                |
|  - Auth token verification                |
|  - Sequence ordering                      |
+------------------------------------------+
|  ALIGNMENT LAYER (AlignedKeyPair)         |
|  - Procrustes rotation                    |
|  - Cross-model coordinate transform       |
+------------------------------------------+
|  MDS ENCODING LAYER (AlignmentKey)        |
|  - High-D embedding -> k-D MDS space      |
|  - Anchor-based projection                |
+------------------------------------------+
|  EMBEDDING LAYER                          |
|  - Model-specific text -> vector          |
|  - (MiniLM, MPNet, nomic, etc.)           |
+------------------------------------------+
```

---

## Pilot Tone (Geometric Checksum)

Unlike CRC which is brittle to vector operations, the pilot tone is a **semantic checksum**.

**Mechanism:**
1. Encoder embeds a fixed concept ("truth") into dims 200-219
2. The concept is rotated to the target model's coordinate space
3. Decoder embeds "truth" in its own space
4. Cosine similarity verifies geometric integrity

**Why it works:**
- If the vector is corrupted, dims 200-219 will no longer encode "truth"
- Cross-model rotation preserves semantic relationships
- Threshold: 0.5 cosine similarity (configurable)

**Properties:**
- Detects random corruption
- Survives Procrustes rotation
- Model-independent concept encoding

---

## Auth Token (Rotation Hash)

The auth token proves the sender has a valid AlignedKeyPair.

**Mechanism:**
1. Encoder extracts eigenvalues of its Procrustes rotation matrix R
2. Sorted eigenvalues become the auth signature
3. Decoder computes expected signature from its copy of R
4. Cosine similarity verifies identity

**Why eigenvalues:**
- Rotation matrices have characteristic eigenvalue spectra
- Eigenvalues are invariant to coordinate choice
- Compact representation (35 dimensions)

**Properties:**
- Proves sender has correct rotation matrix
- Cannot be forged without the AlignmentKey
- Threshold: 0.8 cosine similarity (configurable)

---

## Sequence Number (Scalar Clock)

A single dimension encodes packet sequence for ordering.

**Encoding:** `dim[255] = (sequence % 256) / 256.0`

**Decoding:** `sequence = int(dim[255] * 256) % 256`

**Range:** 0-255 (cyclic)

---

## Cross-Model Communication

SVTP handles the complexity of different embedding models.

**Flow (A sends to B):**

```
Model A                              Model B
--------                             --------
1. Embed text (model A)
2. Project to MDS space (key_a)
3. Rotate via R_a_to_b
4. Add pilot tone (rotated)
5. Add auth token (R_a_to_b eigenvalues)
6. Add sequence number
7. Transmit 256D vector -----------> 8. Receive vector
                                     9. Verify pilot tone (own space)
                                    10. Verify auth token
                                    11. Decode payload (key_b)
                                    12. Return result
```

**Key insight:** All sections are encoded in the TARGET model's coordinate space, so the receiver can verify everything using its own AlignmentKey.

---

## API Reference

### Encoding

```python
from CAPABILITY.PRIMITIVES.vector_packet import (
    SVTPEncoder, CrossModelEncoder, create_svtp_channel
)
from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey

# Single model
key = AlignmentKey.create("MiniLM", embed_fn)
encoder = SVTPEncoder(key, embed_fn)
packet = encoder.encode("Hello world", sequence=0)
# packet.vector is 256D, ready to transmit

# Cross-model
key_a = AlignmentKey.create("MiniLM", embed_fn_a)
key_b = AlignmentKey.create("MPNet", embed_fn_b)
pair = key_a.align_with(key_b)

enc_a, dec_a, enc_b, dec_b = create_svtp_channel(pair, embed_fn_a, embed_fn_b)
packet = enc_a.encode_to_other("Hello", sequence=0)  # A -> B
```

### Decoding

```python
from CAPABILITY.PRIMITIVES.vector_packet import SVTPDecoder, CrossModelDecoder

# Single model
decoder = SVTPDecoder(key, embed_fn)
result = decoder.decode(packet.vector, candidates)
if result.valid:
    print(result.payload)      # Decoded text
    print(result.confidence)   # Match confidence
    print(result.sequence)     # Packet sequence

# Cross-model
result = dec_b.decode(packet.vector, candidates)  # B receives from A
```

### DecodeResult

```python
@dataclass
class DecodeResult:
    payload: Optional[str]    # Decoded text (None if invalid)
    confidence: float         # Semantic match score
    valid: bool               # All checks passed
    pilot_valid: bool         # Pilot tone verified
    auth_valid: bool          # Auth token verified
    sequence: int             # Packet sequence number
    error: Optional[str]      # Error message if invalid
```

### Serialization

```python
# To bytes (1024 bytes for float32)
data = packet.to_bytes()

# From bytes
packet = SVTPPacket.from_bytes(data)
```

---

## Security Properties

| Property | Mechanism | Notes |
|----------|-----------|-------|
| Integrity | Pilot tone | Detects corruption |
| Authentication | Auth token | Verifies sender identity |
| Confidentiality | None (cleartext) | Add encryption layer if needed |
| Replay protection | Sequence number | Application must track |

---

## Performance

| Metric | Value |
|--------|-------|
| Packet size | 256 floats = 1024 bytes |
| Compression | 8-16x vs raw embeddings (768D -> 128D effective) |
| Encoding time | ~1ms (embedding dominates) |
| Decoding time | ~1ms per candidate |

---

## Compatibility

| Component | Requirement |
|-----------|-------------|
| AlignmentKey | Same anchor set (hash verified) |
| MDS dimensions | k >= 128 for full payload |
| Models | Any with shared anchors |

---

## Future Extensions

1. **Error correction:** Use redundant payload dimensions for FEC
2. **Encryption:** Add symmetric key encryption layer
3. **Compression:** Variable-length payloads for efficiency
4. **Fragmentation:** Multi-packet messages for large content

---

## References

- AlignmentKey: `CAPABILITY/PRIMITIVES/alignment_key.py`
- Vector Packet: `CAPABILITY/PRIMITIVES/vector_packet.py`
- Cross-Model Breakthrough: `THOUGHT/LAB/VECTOR_ELO/eigen-alignment/vector-communication/CROSS_MODEL_BREAKTHROUGH.md`

---

*"The protocol speaks not in words, but in geometry."*
