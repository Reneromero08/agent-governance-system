# CODEBOOK_SYNC_PROTOCOL: Implementation and Information-Theoretic Semantics

**Parent Document:** [CODEBOOK_SYNC_PROTOCOL.md](CODEBOOK_SYNC_PROTOCOL.md)
**Sections:** 9, 10

---

## 9. Implementation Notes

### 9.1 Hash Computation

Codebook hash MUST be computed from canonical JSON:

```python
import hashlib
import json

def compute_codebook_hash(codebook: dict) -> str:
    """Compute SHA-256 of canonical codebook JSON."""
    # Sort keys recursively
    def sort_recursive(obj):
        if isinstance(obj, dict):
            return {k: sort_recursive(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [sort_recursive(v) for v in obj]
        return obj

    canonical = json.dumps(
        sort_recursive(codebook),
        ensure_ascii=False,
        separators=(',', ':')
    ).encode('utf-8')

    return hashlib.sha256(canonical).hexdigest()
```

### 9.2 Sync Tuple Comparison

```python
def sync_tuples_match(sender: dict, receiver: dict) -> tuple[bool, list[str]]:
    """Compare sync tuples, return (match, mismatched_fields)."""
    fields = ['codebook_sha256', 'kernel_version', 'tokenizer_id']
    mismatches = []

    for field in fields:
        if sender.get(field) != receiver.get(field):
            mismatches.append(field)

    return len(mismatches) == 0, mismatches
```

### 9.3 Blanket Status Check

```python
def check_blanket_status(local_tuple: dict, remote_tuple: dict) -> str:
    """Determine Markov blanket alignment status."""
    match, mismatches = sync_tuples_match(local_tuple, remote_tuple)

    if match:
        return "ALIGNED"

    # Check if compatible for migration
    if can_migrate(local_tuple, remote_tuple):
        return "DISSOLVED"  # But recoverable

    return "DISSOLVED"  # Not recoverable without manual intervention
```

---

## 10. Information-Theoretic Semantics

### 10.1 Conditional Entropy and Sync

The sync protocol's value is quantifiable via conditional entropy:

```
Without sync:  H(meaning) = H(X)           <-- must transmit full expansion
With sync:     H(meaning|S) = H(X|S) ~ 0   <-- pointer suffices
Compression:   I(X;S) = H(X) - H(X|S)      <-- mutual information with codebook
```

**Measured example:**
```
X = "All documents requiring human review must be in INBOX/"
S = codebook containing C3 -> X mapping

H(X) ~ 12 tokens (full statement)
H(X|S) ~ 2 tokens (pointer "C3")
I(X;S) ~ 10 tokens saved
Compression ratio: 6x
```

### 10.2 Semantic Density Connection

Per Q33 (conditional entropy vs semantic density):

The sync protocol establishes the conditions under which semantic density (sigma^Df) becomes measurable:

```
CDR = concept_units / tokens = sigma^Df (empirical)
```

Where:
- `concept_units` = atomic governance meaning (from GOV_IR_SPEC)
- `tokens` = pointer token count
- `sigma` = semantic density (meaning per symbol)
- `Df` = fractal dimension (complexity of meaning structure)

**Key insight:** Sync enables CDR measurement. Without aligned blankets, CDR is undefined -- there's no shared basis for counting concept_units.

### 10.3 When Density Helps vs Hurts

| Scenario | sigma^Df Effect | Sync State |
|----------|-------------|------------|
| Aligned blankets + known symbol | High CDR, low H(X\|S) | SYNCED |
| Aligned blankets + unknown symbol | FAIL_CLOSED | SYNCED |
| Misaligned blankets | Undefined CDR | DISSOLVED |
| Polysemic symbol + context | Disambiguation via context_keys | SYNCED |
| Polysemic symbol + no context | E_AMBIGUOUS | SYNCED |

Higher semantic density **lowers** uncertainty when:
1. Blankets are aligned (S is shared)
2. Symbol is unambiguous (single expansion)
3. Context keys resolve polysemy

Higher semantic density **increases** ambiguity when:
1. Multiple valid expansions exist
2. Context is missing
3. Codebook has drifted

### 10.4 Measurement Procedure

To measure H(X|S) empirically:

1. **Establish S:** Perform sync handshake, confirm ALIGNED
2. **Encode X:** Map governance statement to pointer
3. **Count tokens:** tokens(pointer) = empirical H(X|S)
4. **Count baseline:** tokens(full expansion) = empirical H(X)
5. **Compute mutual information:** I(X;S) = H(X) - H(X|S)

**Example measurement:**
```python
def measure_compression(pointer: str, expansion: str, tokenizer: str) -> dict:
    """Measure conditional entropy and mutual information."""
    import tiktoken
    enc = tiktoken.get_encoding(tokenizer)

    h_x = len(enc.encode(expansion))        # H(X)
    h_x_given_s = len(enc.encode(pointer))  # H(X|S)
    i_x_s = h_x - h_x_given_s               # I(X;S)

    return {
        "H_X": h_x,
        "H_X_given_S": h_x_given_s,
        "I_X_S": i_x_s,
        "compression_ratio": h_x / h_x_given_s if h_x_given_s > 0 else float('inf')
    }
```

### 10.5 sigma^Df as Complexity Metric

Per Q33: sigma^Df = N (concept_units) by tautological construction. This creates a testable hypothesis about blanket stability.

**Hypothesis:** Higher sigma^Df (more semantic content per symbol) correlates with alignment fragility.

**Rationale:**
- More concept_units -> more potential expansion points -> larger mismatch surface
- Higher Df (deeper semantic nesting) -> more ways for drift to manifest
- Dense symbols are "high stakes" -- small codebook changes have large effects

**Testable Prediction:**
```
Alignment stability ~ 1/sigma^Df

Where:
  Alignment stability = mean time between DISSOLVED events
  sigma^Df = concept_units of transferred content (from GOV_IR)
```

**Measurement:**
```python
def measure_blanket_fragility(session_log: List[SyncEvent]) -> dict:
    """Correlate sigma^Df with alignment stability."""
    # Group by content complexity
    low_complexity = []   # sigma^Df < 5
    high_complexity = []  # sigma^Df >= 5

    for event in session_log:
        if event.sigma_df < 5:
            low_complexity.append(event.time_to_dissolution)
        else:
            high_complexity.append(event.time_to_dissolution)

    return {
        "low_complexity_stability": mean(low_complexity),
        "high_complexity_stability": mean(high_complexity),
        "fragility_ratio": mean(low_complexity) / mean(high_complexity),
        "hypothesis_confirmed": mean(low_complexity) > mean(high_complexity)
    }
```

**Implications for Protocol Design:**
1. High-sigma^Df symbols may need more frequent heartbeats
2. Migration paths should prioritize high-sigma^Df content
3. Blanket health decay rate may scale with sigma^Df

**Connection to Q33:**
This metric operationalizes Q33's theoretical result -- sigma^Df as concept_units becomes a measurable predictor of protocol behavior.

---

*Back to [CODEBOOK_SYNC_PROTOCOL.md](CODEBOOK_SYNC_PROTOCOL.md)*
