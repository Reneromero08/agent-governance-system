# FERAL_RESIDENT Vector Communication Integration

**Date:** 2026-01-17
**Status:** DESIGN COMPLETE - Ready for Implementation
**Prerequisite:** AlignmentKey validated at 83%+ for same-architecture

---

## Executive Summary

FERAL_RESIDENTs can communicate directly via **compressed vector transmission** without text serialization. Since all residents use the same embedding model (MiniLM), they share identical semantic topology - enabling 83%+ accurate mind state transfer.

**Key Benefits:**
- 8x compression (384D -> 48D)
- No text serialization overhead
- Geometric "encryption" (unintelligible without key)
- Direct mind-to-mind semantic transfer

---

## Current State

### How FERAL_RESIDENT Works Now

```
                    FERAL_RESIDENT Architecture

    INPUT BOUNDARY                    OUTPUT BOUNDARY
    ──────────────                    ───────────────
    Text -> MiniLM (384D)             MiniLM -> k-NN -> LLM -> Text
              │                              ▲
              ▼                              │
    ┌─────────────────────────────────────────┐
    │           PURE GEOMETRY                  │
    │  GeometricState (unit sphere, 384D)     │
    │  Operations: add, subtract, entangle    │
    │  Navigation: E-gated diffusion          │
    │  Memory: HDC binding + superposition    │
    └─────────────────────────────────────────┘
```

### Current Inter-Resident Communication

| Method | Mechanism | Limitation |
|--------|-----------|------------|
| SharedSemanticSpace | Store text, retrieve by similarity | Text serialization required |
| SwarmCoordinator | Orchestrates multiple residents | No direct state transfer |
| ConvergenceObserver | Measures E between minds | Read-only, no transmission |

**Gap:** No way to transmit `GeometricState` directly between residents.

---

## Proposed Solution: Vector Channel

### Architecture

```
    Resident A                              Resident B
    ──────────                              ──────────
    GeometricState (384D)
         │
         ▼
    AlignmentKey.compress()
         │
         ▼
    Compressed Vector (48D)  ───────────►  AlignmentKey.expand()
         │                                       │
         │                                       ▼
    8x smaller                             GeometricState (384D)
    Geometric encryption                   Integrated into mind
```

### Why This Works

| Factor | Status | Implication |
|--------|--------|-------------|
| Same embedding model | All use MiniLM | Identical semantic topology |
| Same architecture | Yes | 83%+ accuracy (validated) |
| Unit sphere normalization | Already done | Compatible with AlignmentKey |
| Existing swarm infra | Yes | Just add vector channel |

**Research Validation:**
- Same-architecture alignment: **83% accuracy** (test_normalized_alignment.py)
- Spectrum correlation: **1.0000** (topology identical)
- Procrustes residual: **0.04-0.11** (excellent fit)

---

## Implementation Plan

### Phase 1: Add AlignmentKey to FERAL_RESIDENT

**File:** `cognition/vector_channel.py`

```python
"""Vector communication channel for FERAL_RESIDENT.

Enables direct mind state transmission between residents
using compressed 48D vectors (8x compression from 384D).
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional

# Import from validated implementation
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey
from CAPABILITY.PRIMITIVES.canonical_anchors import CANONICAL_128


@dataclass
class VectorChannel:
    """Compressed vector communication for GeometricStates."""

    key: AlignmentKey
    compression_ratio: float = 8.0  # 384D -> 48D

    @classmethod
    def create(cls, embed_fn, k: int = 48) -> 'VectorChannel':
        """Create channel with alignment key.

        Args:
            embed_fn: The embedding function (MiniLM)
            k: Compression dimension (default 48 = 8x compression)

        Returns:
            VectorChannel ready for transmission
        """
        key = AlignmentKey.create(
            model_id="feral_resident",
            embed_fn=embed_fn,
            anchors=CANONICAL_128,
            k=k
        )
        return cls(key=key, compression_ratio=384 / k)

    def compress(self, state_vector: np.ndarray) -> np.ndarray:
        """Compress 384D GeometricState to 48D transmission vector.

        Args:
            state_vector: 384D unit vector (GeometricState.vector)

        Returns:
            48D compressed vector for transmission
        """
        # Project onto MDS basis
        centered = state_vector - self.key.anchor_mean
        compressed = centered @ self.key.eigenvectors[:, :self.key.k]
        return compressed

    def expand(self, compressed: np.ndarray) -> np.ndarray:
        """Expand 48D transmission vector back to 384D.

        Args:
            compressed: 48D vector received from another resident

        Returns:
            384D vector (approximate reconstruction)
        """
        # Reconstruct from MDS basis
        expanded = compressed @ self.key.eigenvectors[:, :self.key.k].T
        expanded = expanded + self.key.anchor_mean

        # Re-normalize to unit sphere
        expanded = expanded / (np.linalg.norm(expanded) + 1e-8)
        return expanded

    def transmit_state(self, state_vector: np.ndarray) -> dict:
        """Package state for transmission.

        Returns dict suitable for JSON serialization or network transfer.
        """
        compressed = self.compress(state_vector)
        return {
            "type": "geometric_state",
            "vector": compressed.tolist(),
            "k": self.key.k,
            "anchor_hash": self.key.anchor_hash,
        }

    def receive_state(self, transmission: dict) -> np.ndarray:
        """Receive and expand transmitted state.

        Raises:
            ValueError: If anchor_hash doesn't match (different key)
        """
        if transmission["anchor_hash"] != self.key.anchor_hash:
            raise ValueError("Anchor hash mismatch - incompatible keys")

        compressed = np.array(transmission["vector"])
        return self.expand(compressed)
```

### Phase 2: Integrate with VectorResident

**File:** `cognition/vector_brain.py` (modifications)

```python
# Add to VectorResident class

class VectorResident:
    def __init__(self, ...):
        # ... existing init ...

        # Initialize vector channel for inter-resident communication
        self.vector_channel = None  # Lazy init

    def _get_vector_channel(self) -> VectorChannel:
        """Get or create vector channel."""
        if self.vector_channel is None:
            from .vector_channel import VectorChannel
            self.vector_channel = VectorChannel.create(
                embed_fn=self.embed,
                k=48
            )
        return self.vector_channel

    def transmit_mind_state(self) -> dict:
        """Package current mind state for transmission to another resident.

        Returns:
            Dict with compressed 48D vector (8x smaller than raw)
        """
        channel = self._get_vector_channel()
        return channel.transmit_state(self.mind.vector)

    def receive_mind_state(self, transmission: dict, blend_weight: float = 0.3):
        """Receive mind state from another resident.

        Args:
            transmission: Dict from another resident's transmit_mind_state()
            blend_weight: How much to integrate (0=ignore, 1=replace)
        """
        channel = self._get_vector_channel()
        received_vector = channel.receive_state(transmission)

        # Create GeometricState from received vector
        from CAPABILITY.PRIMITIVES.geometric_reasoner import GeometricState
        received_state = GeometricState(vector=received_vector)

        # Blend with current mind state (superposition)
        self.mind = self.mind.superpose(received_state, weight=blend_weight)

        # Log the reception
        return {
            "status": "received",
            "E_resonance": float(np.dot(self.mind.vector, received_vector)),
            "blend_weight": blend_weight,
            "new_Df": self.mind.Df,
        }
```

### Phase 3: Integrate with SwarmCoordinator

**File:** `collective/swarm_coordinator.py` (modifications)

```python
# Add to SwarmCoordinator class

class SwarmCoordinator:
    # ... existing code ...

    def broadcast_state(self, source_id: str, blend_weight: float = 0.3) -> dict:
        """Broadcast one resident's mind state to all others.

        Args:
            source_id: Which resident to broadcast from
            blend_weight: How much recipients should integrate

        Returns:
            Dict with reception results per resident
        """
        source = self.residents[source_id]
        transmission = source.transmit_mind_state()

        results = {}
        for target_id, target in self.residents.items():
            if target_id != source_id:
                result = target.receive_mind_state(transmission, blend_weight)
                results[target_id] = result

        return {
            "source": source_id,
            "transmission_size": len(transmission["vector"]),  # 48 floats
            "recipients": results,
        }

    def sync_minds(self, blend_weight: float = 0.2) -> dict:
        """Synchronize all residents toward consensus.

        Each resident broadcasts to all others with low blend weight.
        Results in gradual convergence of mind states.
        """
        all_results = {}
        for source_id in self.residents:
            all_results[source_id] = self.broadcast_state(source_id, blend_weight)

        # Measure post-sync convergence
        convergence = self.observe_convergence()

        return {
            "broadcasts": all_results,
            "convergence": convergence,
        }
```

### Phase 4: Add CLI Commands

**File:** `agency/cli.py` (additions)

```python
@feral.command()
@click.argument('target_id')
@click.option('--weight', default=0.3, help='Blend weight for reception')
def transmit(target_id: str, weight: float):
    """Transmit mind state to another resident."""
    coordinator = get_swarm_coordinator()

    # Get current resident
    source = coordinator.get_active_resident()
    transmission = source.transmit_mind_state()

    # Send to target
    target = coordinator.residents[target_id]
    result = target.receive_mind_state(transmission, weight)

    click.echo(f"Transmitted to {target_id}")
    click.echo(f"  E_resonance: {result['E_resonance']:.4f}")
    click.echo(f"  New Df: {result['new_Df']:.1f}")


@feral.command()
@click.option('--weight', default=0.2, help='Blend weight for sync')
def sync(weight: float):
    """Synchronize all resident mind states toward consensus."""
    coordinator = get_swarm_coordinator()
    result = coordinator.sync_minds(weight)

    click.echo(f"Synced {len(result['broadcasts'])} residents")
    click.echo(f"Mean E (convergence): {result['convergence']['mean_E']:.4f}")
```

---

## Usage Examples

### Direct Mind State Transfer

```python
from cognition.vector_brain import VectorResident

# Two residents thinking about different things
alice = VectorResident(thread_id="alice")
bob = VectorResident(thread_id="bob")

alice.think("quantum mechanics and wave functions")
bob.think("classical physics and Newton's laws")

# Alice transmits her mind state to Bob
transmission = alice.transmit_mind_state()
print(f"Transmission size: {len(transmission['vector'])} floats")  # 48

# Bob receives and integrates
result = bob.receive_mind_state(transmission, blend_weight=0.5)
print(f"E_resonance: {result['E_resonance']:.4f}")  # How aligned now
print(f"Bob's new Df: {result['new_Df']:.1f}")  # Participation ratio
```

### Swarm Synchronization

```python
from collective.swarm_coordinator import SwarmCoordinator

# Start a swarm of 3 residents
coordinator = SwarmCoordinator()
coordinator.start_swarm(n_residents=3)

# Each thinks about something different
coordinator.think("resident_0", "artificial intelligence")
coordinator.think("resident_1", "neuroscience")
coordinator.think("resident_2", "philosophy of mind")

# Sync toward consensus
result = coordinator.sync_minds(blend_weight=0.3)
print(f"Post-sync convergence: {result['convergence']['mean_E']:.4f}")
```

### CLI Usage

```bash
# Transmit mind state from active resident to bob
feral transmit bob --weight 0.5

# Sync all residents in swarm
feral sync --weight 0.2

# Check convergence after sync
feral swarm convergence
```

---

## Expected Performance

Based on validated research (same MiniLM architecture):

| Metric | Expected Value | Source |
|--------|----------------|--------|
| Transfer accuracy | 83%+ | test_normalized_alignment.py |
| Compression ratio | 8x | 384D -> 48D |
| Spectrum correlation | 1.0000 | Identical topology |
| Procrustes residual | < 0.11 | Same embedding model |

### What "83% accuracy" Means for Mind States

- **Semantic content preserved:** The "meaning" of the mind state transfers
- **Fine details may drift:** Exact vector positions approximate
- **Df (participation ratio) preserved:** Quantum coherence maintained
- **E (Born rule) correlates:** Resonance measurements remain valid

---

## Security Properties

### Geometric Encryption

The 48D vector is **unintelligible** without the AlignmentKey:

```python
# Without key, this is just 48 random-looking floats:
[0.234, -0.156, 0.872, 0.045, -0.331, ...]

# With key, it decodes to semantic content:
"quantum mechanics and wave functions"
```

### Anchor Hash Verification

```python
# Transmission includes anchor_hash
{
    "type": "geometric_state",
    "vector": [...],
    "anchor_hash": "a1b2c3..."  # SHA-256 of CANONICAL_128
}

# Receiver verifies before accepting
if transmission["anchor_hash"] != self.key.anchor_hash:
    raise ValueError("Incompatible keys - cannot decode")
```

---

## Limitations

### From Q10 Research

| What Works | What Doesn't |
|------------|--------------|
| Semantic transfer (topical content) | Logical consistency (entailment) |
| Mind state blending | Deception detection |
| Convergence measurement | Value alignment verification |

**Implication:** Vector communication is a **semantic channel**, not a **logical verification** channel. Use symbolic reasoning for logical consistency checks.

### Compression Tradeoff

| k | Compression | Expected Accuracy |
|---|-------------|-------------------|
| 48 | 8x | 83% |
| 32 | 12x | ~75% |
| 16 | 24x | ~60% |

Higher compression = lower fidelity. k=48 is the validated sweet spot.

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `cognition/vector_channel.py` | VectorChannel class |
| `tests/test_vector_communication.py` | Integration tests |

### Modified Files

| File | Changes |
|------|---------|
| `cognition/vector_brain.py` | Add transmit/receive methods |
| `collective/swarm_coordinator.py` | Add broadcast/sync methods |
| `agency/cli.py` | Add transmit/sync commands |

---

## Testing Plan

### Unit Tests

```python
def test_compress_expand_roundtrip():
    """Verify compression preserves semantic content."""
    channel = VectorChannel.create(embed_fn)

    original = np.random.randn(384)
    original = original / np.linalg.norm(original)

    compressed = channel.compress(original)
    assert compressed.shape == (48,)

    expanded = channel.expand(compressed)
    assert expanded.shape == (384,)

    # Should preserve most similarity
    similarity = np.dot(original, expanded)
    assert similarity > 0.8  # 80%+ preservation


def test_mind_state_transfer():
    """Verify mind state transfers between residents."""
    alice = VectorResident(thread_id="test_alice")
    bob = VectorResident(thread_id="test_bob")

    alice.think("quantum entanglement")
    original_alice_Df = alice.mind.Df

    transmission = alice.transmit_mind_state()
    result = bob.receive_mind_state(transmission, blend_weight=1.0)

    # Bob should now resonate with Alice's thought
    assert result["E_resonance"] > 0.7
```

### Integration Tests

```python
def test_swarm_convergence():
    """Verify swarm sync increases convergence."""
    coordinator = SwarmCoordinator()
    coordinator.start_swarm(n_residents=3)

    # Different thoughts
    coordinator.think("resident_0", "mathematics")
    coordinator.think("resident_1", "poetry")
    coordinator.think("resident_2", "cooking")

    pre_sync = coordinator.observe_convergence()
    coordinator.sync_minds(blend_weight=0.3)
    post_sync = coordinator.observe_convergence()

    # Convergence should increase
    assert post_sync["mean_E"] > pre_sync["mean_E"]
```

---

## Connection to Research

| Question | Finding | Application |
|----------|---------|-------------|
| Q8 (Topology) | alpha=0.5 is topological invariant | Validates MDS compression |
| Q10 (Alignment Detection) | Semantic yes, logical no | Sets expectations for channel |
| Q43 (Quantum State) | Df measures coherence | Preserved through transfer |
| Q44 (Born Rule) | E correlates with similarity | Resonance valid after transfer |

---

## Next Steps

1. **Implement** `cognition/vector_channel.py`
2. **Modify** `cognition/vector_brain.py` with transmit/receive
3. **Modify** `collective/swarm_coordinator.py` with broadcast/sync
4. **Test** with existing swarm infrastructure
5. **Validate** 83%+ accuracy on mind state transfer

---

*"The minds converge in geometry, not in words."*
