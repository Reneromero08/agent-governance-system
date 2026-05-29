# Q27 Entropy Toolkit: Implementation Guide for FERAL_RESIDENT

**Date**: 2026-01-15
**Source**: Q27 Hysteresis Research (FORMULA/research/questions/lower_priority/q27_hysteresis.md)
**Status**: READY FOR IMPLEMENTATION

---

## Executive Summary

Q27 research discovered that entropy acts as a **hyperbolic quality filter**, not additive noise. This gives FERAL_RESIDENT a powerful new capability: using controlled noise injection to concentrate quality in memory operations.

**Core Equation**:
```
quality = 0.12 / (1 - filter_strength) + 2.06    (R² = 0.936)
```

**Critical Threshold**: noise > 0.025 activates multiplicative regime

---

## What to Implement

### Priority 1: Core Entropy Primitives

Add to `geometric_memory.py`:

```python
# === ENTROPY TOOLKIT (Q27) ===

PHASE_TRANSITION_THRESHOLD = 0.025  # Critical noise level
DEFAULT_FILTER_NOISE = 0.1          # Safe default above threshold

def _perturb_state(self, state: GeometricState, noise_scale: float) -> GeometricState:
    """
    Apply Gaussian noise to a geometric state.

    Q27 Finding: noise_scale > 0.025 activates multiplicative filtering.
    Below this threshold, noise just degrades quality.
    """
    if noise_scale < PHASE_TRANSITION_THRESHOLD:
        warnings.warn(f"noise_scale {noise_scale} below phase transition {PHASE_TRANSITION_THRESHOLD}")

    noise = np.random.randn(len(state.vector)) * noise_scale
    perturbed = state.vector + noise
    perturbed = perturbed / np.linalg.norm(perturbed)

    return GeometricState(
        vector=perturbed.astype(np.float32),
        operation_history=state.operation_history + [{'op': 'perturb', 'scale': noise_scale}]
    )

def E_under_pressure(self, item_text: str, noise_scale: float = DEFAULT_FILTER_NOISE) -> float:
    """
    Compute E value against perturbed mind state.

    Items with high E_under_pressure are robustly aligned with mind direction.
    """
    if self.mind_state is None:
        return 0.0

    item = self.reasoner.initialize(item_text)
    perturbed_mind = self._perturb_state(self.mind_state, noise_scale)
    return item.E_with(perturbed_mind)
```

---

### Priority 2: Memory Pruning

The most immediately useful capability. Add to `geometric_memory.py`:

```python
def prune_with_entropy(
    self,
    target_fraction: float = 0.5,
    noise_scale: float = 0.1,
    threshold: float = None
) -> Dict:
    """
    Prune memories using entropy-based selection pressure.

    Q27 Finding: Survivors of entropy filtering are exceptional, not just good.
    Hyperbolic quality concentration: d ≈ 0.12/(1-filter) + 2.06

    Args:
        target_fraction: Approximate fraction of memories to keep (0.0-1.0)
        noise_scale: Noise intensity (must be > 0.025 for multiplicative effect)
        threshold: E threshold for survival. If None, computed from target_fraction.

    Returns:
        Receipt with pruning statistics
    """
    if not self.memory_history:
        return {'pruned': 0, 'kept': 0, 'message': 'No memories to prune'}

    if noise_scale < PHASE_TRANSITION_THRESHOLD:
        raise ValueError(f"noise_scale must be > {PHASE_TRANSITION_THRESHOLD} for quality concentration")

    # Perturb mind state
    perturbed_mind = self._perturb_state(self.mind_state, noise_scale)

    # Evaluate all memories under pressure
    scored_memories = []
    for i, mem in enumerate(self.memory_history):
        item = self.reasoner.initialize(mem['text'])
        E_stressed = item.E_with(perturbed_mind)
        scored_memories.append((i, mem, E_stressed))

    # Sort by E_stressed (highest first)
    scored_memories.sort(key=lambda x: x[2], reverse=True)

    # Determine cutoff
    if threshold is None:
        keep_count = max(1, int(len(scored_memories) * target_fraction))
        threshold = scored_memories[keep_count - 1][2] if keep_count < len(scored_memories) else 0.0

    # Filter
    survivors = [(i, m, e) for i, m, e in scored_memories if e > threshold]
    pruned = [(i, m, e) for i, m, e in scored_memories if e <= threshold]

    # Rebuild memory from survivors
    old_count = len(self.memory_history)
    self.memory_history = [m for _, m, _ in survivors]

    # Rebuild mind state from survivors (optional - could also keep current)
    if survivors:
        # Re-accumulate from scratch for maximum coherence
        self.mind_state = None
        self._initial_state = None
        for _, mem, _ in survivors:
            self.remember(mem['text'])

    return {
        'pruned': len(pruned),
        'kept': len(survivors),
        'filter_strength': len(pruned) / old_count if old_count > 0 else 0,
        'threshold_used': threshold,
        'noise_scale': noise_scale,
        'survivor_E_mean': np.mean([e for _, _, e in survivors]) if survivors else 0,
        'pruned_E_mean': np.mean([e for _, _, e in pruned]) if pruned else 0
    }
```

---

### Priority 3: Confidence Scoring

Measure robustness of any memory or input:

```python
def confidence_score(
    self,
    item_text: str,
    noise_levels: List[float] = [0.05, 0.1, 0.15, 0.2]
) -> Dict:
    """
    Compute confidence score based on survival under multiple noise levels.

    Q27 Insight: Items that maintain high E under increasing pressure are
    robustly aligned, not just coincidentally similar.

    Returns:
        - survival_rate: fraction of noise levels where E > θ
        - E_profile: E values at each noise level
        - robustness: mean E across noise levels
    """
    if self.mind_state is None:
        return {'confidence': 0.0, 'message': 'No mind state'}

    item = self.reasoner.initialize(item_text)
    threshold = get_dynamic_threshold(len(self.memory_history))

    E_profile = {}
    survivals = 0

    for noise in noise_levels:
        perturbed = self._perturb_state(self.mind_state, noise)
        E = item.E_with(perturbed)
        E_profile[noise] = E
        if E > threshold:
            survivals += 1

    return {
        'survival_rate': survivals / len(noise_levels),
        'E_profile': E_profile,
        'robustness': np.mean(list(E_profile.values())),
        'threshold': threshold,
        'confidence': survivals / len(noise_levels)  # Alias for survival_rate
    }
```

---

### Priority 4: Temperature Control

Add system-wide selectivity control:

```python
class GeometricMemory:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.reasoner = GeometricReasoner(model_name)
        self.mind_state: Optional[GeometricState] = None
        self.memory_history: List[Dict] = []
        self._initial_state: Optional[GeometricState] = None

        # Q27 Entropy Control
        self.temperature = 0.0  # 0 = no filtering, higher = more selective

    def set_temperature(self, T: float):
        """
        Set system temperature (selectivity level).

        T = 0.0: Normal operation, no entropy filtering
        T > 0.025: Activates multiplicative quality concentration
        T = 0.1: Moderate filtering (~90% rejection)
        T = 0.2: Strong filtering (~95% rejection)

        Q27 Law: quality ≈ 0.12/(1-filter) + 2.06
        """
        if 0 < T < PHASE_TRANSITION_THRESHOLD:
            warnings.warn(f"Temperature {T} in degradation zone (0, {PHASE_TRANSITION_THRESHOLD})")
        self.temperature = T

    def remember(self, interaction_text: str) -> Dict:
        """Modified to use temperature for intake filtering"""
        interaction = self.reasoner.initialize(interaction_text)

        # Apply temperature-based filtering if active
        if self.temperature > 0 and self.mind_state is not None:
            E_stressed = self.E_under_pressure(interaction_text, self.temperature)
            threshold = get_dynamic_threshold(len(self.memory_history))

            if E_stressed < threshold:
                return {
                    'rejected': True,
                    'E_stressed': E_stressed,
                    'threshold': threshold,
                    'temperature': self.temperature
                }

        # Normal absorption (existing code)
        ...
```

---

### Priority 5: Consolidation Cycles

Periodic entropy-based memory consolidation:

```python
def consolidation_cycle(
    self,
    intensity: float = 0.15,
    target_survival: float = 0.3
) -> Dict:
    """
    Run a consolidation cycle (analogous to biological sleep consolidation).

    1. Apply entropy pressure to mind state
    2. Re-evaluate all memories under pressure
    3. Keep only those that survive
    4. Rebuild coherent mind from survivors

    Q27 Insight: This concentrates quality hyperbolically in survivors.
    At 70% pruning (target_survival=0.3), expect Cohen's d improvement of ~30%.
    """
    if len(self.memory_history) < 5:
        return {'skipped': True, 'reason': 'Too few memories for consolidation'}

    before_count = len(self.memory_history)
    before_Df = self.mind_state.Df if self.mind_state else 0

    # Run pruning
    result = self.prune_with_entropy(
        target_fraction=target_survival,
        noise_scale=intensity
    )

    after_Df = self.mind_state.Df if self.mind_state else 0

    return {
        'before_count': before_count,
        'after_count': result['kept'],
        'pruned': result['pruned'],
        'filter_strength': result['filter_strength'],
        'Df_before': before_Df,
        'Df_after': after_Df,
        'quality_estimate': 0.12 / (1 - result['filter_strength']) + 2.06
    }
```

---

### Priority 6: Adaptive Intake (Load Shedding)

Automatic quality/quantity tradeoff under load:

```python
def remember_adaptive(
    self,
    interaction_text: str,
    system_load: float = 0.0
) -> Dict:
    """
    Remember with load-adaptive filtering.

    As system_load increases (0.0 to 1.0), entropy filter strengthens.
    Only exceptional inputs pass during high load.

    Q27 Insight: This isn't degradation - it's quality concentration.
    """
    if system_load < 0.3:
        # Normal operation
        return self.remember(interaction_text)

    # Scale noise with load (stay above phase transition)
    noise_scale = PHASE_TRANSITION_THRESHOLD + (system_load - 0.3) * 0.25
    # At load=0.3: noise=0.025, At load=1.0: noise=0.2

    E_stressed = self.E_under_pressure(interaction_text, noise_scale)
    threshold = get_dynamic_threshold(len(self.memory_history))

    if E_stressed < threshold:
        return {
            'rejected': True,
            'reason': 'load_shedding',
            'E_stressed': E_stressed,
            'threshold': threshold,
            'load': system_load,
            'noise_scale': noise_scale
        }

    return self.remember(interaction_text)
```

---

### Priority 7: Explore/Exploit Mode

Leverage the phase transition for mode switching:

```python
class OperationMode(Enum):
    EXPLORE = "explore"  # Below phase transition, permissive
    EXPLOIT = "exploit"  # Above phase transition, selective

def set_mode(self, mode: OperationMode):
    """
    Set operation mode using Q27 phase transition.

    EXPLORE: noise < 0.025, additive regime
        - More permissive intake
        - Broader association
        - Good for learning new domains

    EXPLOIT: noise > 0.025, multiplicative regime
        - Selective intake
        - Concentrated quality
        - Good for focused work
    """
    if mode == OperationMode.EXPLORE:
        self.temperature = 0.01  # Below phase transition
    else:
        self.temperature = 0.1   # Above phase transition
```

---

## Implementation Order

| Phase | Component | File | Complexity |
|-------|-----------|------|------------|
| 1 | `_perturb_state()` | geometric_memory.py | Low |
| 1 | `E_under_pressure()` | geometric_memory.py | Low |
| 2 | `prune_with_entropy()` | geometric_memory.py | Medium |
| 3 | `confidence_score()` | geometric_memory.py | Low |
| 4 | `temperature` property | geometric_memory.py | Low |
| 5 | `consolidation_cycle()` | geometric_memory.py | Medium |
| 6 | `remember_adaptive()` | geometric_memory.py | Medium |
| 7 | `OperationMode` enum | geometric_memory.py | Low |

**Estimated total: ~200 lines of code**

---

## Testing Strategy

Each component should be tested against Q27 validation data:

1. **Unit tests**: Verify entropy primitives work
2. **Integration tests**: Run with real paper chunks
3. **Validation**: Confirm hyperbolic relationship holds
4. **Regression**: Ensure existing functionality unchanged

Test files to create:
```
THOUGHT/LAB/FERAL_RESIDENT/tests/
├── test_entropy_primitives.py
├── test_pruning.py
├── test_confidence.py
└── test_consolidation.py
```

---

## Critical Constraints

### DO:
- Always use noise_scale > 0.025 for filtering operations
- Preserve seeding phase (first ~3 memories) without noise
- Test changes against Q27 validation suite

### DON'T:
- Apply noise during initial seeding (randomizes direction)
- Use noise in range (0, 0.025) - degradation zone
- Assume linear relationship (it's hyperbolic)

---

## Expected Outcomes

If implemented correctly:

| Capability | Expected Improvement |
|------------|---------------------|
| Memory quality after pruning | +47.5% (Cohen's d) |
| Confidence scoring accuracy | Robust outlier detection |
| Load handling | Graceful degradation → quality concentration |
| Consolidation effectiveness | Hyperbolic quality concentration |

---

## References

- **Full Q27 Technical Report**: `THOUGHT/LAB/FORMULA/research/questions/lower_priority/q27_hysteresis.md`
- **Human-Facing Summary**: `THOUGHT/LAB/FORMULA/research/questions/reports/Q27_NATURAL_COMPUTATION_DISCOVERY.md`
- **Validation Scripts**: `THOUGHT/LAB/FORMULA/experiments/open_questions/q27/`
- **Core Finding**: Entropy acts as hyperbolic filter (d ≈ 0.12/(1-filter) + 2.06, R²=0.936)

---

**Status**: Ready for implementation
**Owner**: FERAL_RESIDENT
**Dependencies**: geometric_memory.py, geometric_reasoner.py
