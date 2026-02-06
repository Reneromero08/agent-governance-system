# Q3 Phase 4+: Adding σ(f)^Df to the Formula

## Current Status
Phases 1-3 validated the base formula `R = E/∇S` (assuming `σ(f)^Df = 1`).

This is valid for simple observations without symbolic compression or fractal structure.

## What is σ(f)^Df?

From Q32 (meaning as field):
- **σ(f)**: Symbolic compression gain - how much structure is compressed into stable invariants
- **Df**: Fractal depth - hierarchical/nested context levels
- **σ(f)^Df**: Exponential scaling of resonance with compression and depth

## When is it needed?

**σ^Df = 1 (current tests)**:
- Raw sensor data (Gaussian noise)
- Binary outcomes (coin flips)
- Quantum measurements (single-shot)
- No symbolic abstraction

**σ^Df > 1 (future work)**:
- Natural language (words compress concepts)
- Hierarchical reasoning (nested contexts)
- Multi-level abstractions (theories compress observations)
- Meaning-as-field (Q32 territory)

## How to implement σ(f) and Df

### Option 1: Compression Ratio
```python
def compute_sigma(observations, compressed_representation):
    """
    σ(f) = original_complexity / compressed_complexity
    
    Example: 
    - 1000 observations → 10 parameters
    - σ = 1000/10 = 100
    """
    original_bits = len(observations) * bits_per_observation
    compressed_bits = len(compressed_representation) * bits_per_param
    return original_bits / compressed_bits
```

### Option 2: Mutual Information
```python
def compute_sigma_mi(observations, symbols):
    """
    σ(f) = exp(I(observations; symbols))
    
    Where I is mutual information between raw data and symbolic representation.
    """
    mi = mutual_information(observations, symbols)
    return np.exp(mi)
```

### Option 3: Kolmogorov Complexity Proxy
```python
def compute_sigma_compression(observations):
    """
    σ(f) = len(observations) / len(compressed(observations))
    
    Use actual compression (gzip, lzma) as proxy for Kolmogorov complexity.
    """
    import gzip
    original = pickle.dumps(observations)
    compressed = gzip.compress(original)
    return len(original) / len(compressed)
```

### Fractal Dimension Df

```python
def compute_Df(context_hierarchy):
    """
    Df = depth of hierarchical context
    
    Examples:
    - Single context: Df = 1
    - Nested contexts (word → sentence → paragraph): Df = 3
    - Fractal self-similar structure: Df = log(N) / log(scale)
    """
    return len(context_hierarchy)  # Simple version
    
    # Or for fractal structures:
    # Df = log(count_at_scale) / log(scale_factor)
```

## Full Formula Implementation

```python
def compute_R_full(observations, truth, symbols=None, context_depth=1):
    """
    R = (E/∇S) × σ(f)^Df
    
    Args:
        observations: Raw data
        truth: Ground truth value
        symbols: Optional symbolic representation
        context_depth: Hierarchical depth (Df)
    """
    # Base formula (validated in Phases 1-3)
    sigma = compute_grad_S(observations)
    E = compute_E(observations, truth, sigma)
    R_base = E / sigma
    
    # Symbolic/fractal scaling (if applicable)
    if symbols is not None:
        sigma_f = compute_sigma(observations, symbols)
        Df = context_depth
        scaling = sigma_f ** Df
    else:
        scaling = 1.0  # No symbolic compression
    
    return R_base * scaling
```

## Test Plan for σ^Df

### Test 1: Language Compression
```python
def test_language_compression():
    """
    Test: Does σ^Df increase R for compressed symbolic representations?
    
    Setup:
    - Raw: 1000 word tokens
    - Compressed: 10 concept embeddings
    - σ = 100, Df = 2 (word → concept → theme)
    - Expected: R_full = R_base × 100^2 = R_base × 10000
    """
    pass
```

### Test 2: Hierarchical Reasoning
```python
def test_hierarchical_context():
    """
    Test: Does Df scale with context depth?
    
    Setup:
    - Flat context: Df = 1
    - Nested context: Df = 3
    - Same σ, same E/∇S
    - Expected: R(Df=3) = R(Df=1) × σ^2
    """
    pass
```

### Test 3: Sensitivity Analysis
```python
def test_Df_sensitivity():
    """
    Test: Is Df sensitivity a problem?
    
    From Q32: "known sensitivity risk; Df dominates variance if not bounded"
    
    Setup:
    - Sweep Df from 1 to 10
    - Check if R explodes exponentially
    - Determine safe bounds for Df
    """
    pass
```

## Integration with Q32

Q32 (Meaning as Field) already has tests for the full formula:
- `q32_meaning_field_tests.py`
- `q32_adversarial_gauntlet.py`
- `q32_public_benchmarks.py`

These tests use `M = log(R)` to stabilize the multiplicative σ^Df term.

**Next step**: Validate that Q32's implementation of σ^Df is consistent with Q3's base formula.

## Roadmap

**Phase 4**: Theoretical foundation for σ(f) and Df
- Define compression formally (Kolmogorov? MDL?)
- Define fractal dimension for semiotic structures
- Prove σ^Df doesn't break E/∇S properties

**Phase 5**: Cross-domain validation with σ^Df
- Information theory: σ = compression ratio
- Statistical mechanics: σ = coarse-graining factor
- Linguistics: σ = abstraction level

**Phase 6**: Unify with Q32
- Show Q3's base formula + Q32's σ^Df = complete theory
- Validate on meaning-as-field benchmarks
- Final synthesis document

## Notes

The separation of `E/∇S` (base SNR) and `σ^Df` (symbolic scaling) is intentional:
- `E/∇S` is universal (works on raw observations)
- `σ^Df` is domain-specific (requires defining symbols and hierarchy)

This modular structure allows testing each component independently.
