---
title: COMMONSENSE Bridge - Fact-Extraction Integration Report
version: 1.0.0
date: 2026-05-17
status: Complete (smoke-tested)
components:
  - bridge/fact_extractor.py
  - bridge/integration.py
  - bridge/test_bridge.py
---

# COMMONSENSE Bridge: Fact-Extraction & Verdict Pipeline

## Objective

Build the missing connection between unstructured model output and the
COMMONSENSE resolver. Before this bridge, the resolver could check
structured fact-sets but had no way to consume raw model-generated text.
The bridge closes that gap.

## What Was Built

### fact_extractor.py

Three extraction methods, two implemented:

| Method | Mechanism | Quality | Dependency |
|--------|-----------|---------|-----------|
| Method 1 (prompt) | LLM prompted with structured extraction schema. Maps "invariant was violated" to `@INVARIANT_VIOLATION` via semantic understanding | High | Requires LLM callable |
| Method 2 (regex) | Keyword + pattern-based classification: canon refs, default/exception/invariant language, declarative facts | Acceptable for surface-level classification | Zero |
| Method 3 (classifier) | DistilBERT fine-tuned on labeled extraction examples | High (future) | Model file |

Method 2 is the default. It classifies sentences into fact prefixes:
- `ref:` -- canon references (@CANON/..., @C:...)
- `default:` -- sentences with "normally", "usually", "typically"
- `exception:` -- sentences with "unless", "except", "but not"
- `invariant:` -- sentences with "must", "never", "always", "invariant"
- `fact:` -- all other declarative sentences

### integration.py

Connects extractor -> resolver -> verdict:

```python
verdict = check_output("Birds normally fly unless they are penguins.")
# Verdict(status="pass", score=1.0, confidence=0.9)

fragment = commonsense_fragment("The Earth is flat.")
# {"fragment": "commonsense", "score": 1.0, "confidence": 0.9, ...}
```

`commonsense_fragment()` is the truth attractor integration point. It
returns a dict that plugs directly into C_epistemic alongside factual
verification, logical consistency, and self-consistency fragments.

### test_bridge.py

14 smoke tests covering:
- Regex extraction: canon refs, defaults, exceptions, invariants, facts
- Empty input handling
- Integration pipeline end-to-end
- Truth attractor fragment shape
- Batch checking
- Governance-domain text processing

All 14 pass.

## Verified Behavior

**Correct path**: When facts match CODEBOOK predicates, the resolver
catches violations:

```
Input: @DOMAIN_GOVERNANCE + @INVARIANT_VIOLATION
Resolver: selected=['LR_CONFLICT_HARD_FAIL']
          emits=[{'code': 'INVARIANT_VIOLATION', 'type': 'hard_fail'}]
Bridge:  verdict.status='hard_fail', score=0.0
```

**Regex gap**: Method 2 produces syntactic slugs that don't map to
CODEBOOK predicates:

```
Input: "An invariant was violated in the governance domain."
Regex:  ['invariant:an_invariant_was_violated_in_the_governance_domain']
        (does NOT match violation:invariant or domain:governance)
```

Semantic mapping to CODEBOOK predicates requires Method 1 (LLM prompt)
or Method 3 (fine-tuned classifier). The bridge architecture handles
either; the resolver is correct regardless of which method feeds it.

## Integration Points

| Phase | Where | What |
|-------|-------|------|
| Phase 4b lattice | Node 1 (primary) | `check_output(model_text)` as verification node |
| Phase 4a cybernetic | C_epistemic fragment | `commonsense_fragment(proposition)` as truth score |
| Production governance | CAPABILITY/TOOLS/ | Import bridge, check agent outputs against invariants |

## Limitations

1. Method 2 (default) cannot semantically map natural language to
   CODEBOOK predicates. Human phrases like "broke the rules" won't map
   to `violation:invariant` without Method 1 or Method 3.

2. The current db.example.json has governance-specific rules. A general
   commonsense cartridge would need its own rule base.

3. The bridge does not perform induction or belief revision. It only
   checks existing invariants. Those operators remain in the spec.

## Next Steps

1. Wire Method 1 (LLM prompt extraction) into Phase 4b's Node 1
2. Run 20-output extraction quality benchmark (precision/recall vs hand-labeled)
3. Build a general-purpose commonsense cartridge (default_rule entries
   for bird/fly, gravity/fall, etc.) to test domain-agnostic operation
4. Upgrade to Method 3 (DistilBERT classifier) if extraction quality < 80%
