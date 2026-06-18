# Audit: Phase 9 Black Hole Experiments

**Date:** 2026-06-17
**Status:** Internal audit

---

## 1. The Hypothesis

The CAT_CAS thesis: a single archetypal polytope recurs across substrates. The operation is invariant — closed reversible trajectory around a zero-boundary, read the global invariant that survives, restore the substrate. This shape appears at every scale: Riemann zeros, black hole horizons, quantum path integrals, catalytic tapes. Not four similar patterns. One function. Different substrates. Different encoded data.

Within this hypothesis, probing the catalytic tape IS probing a black hole — at the architectural level. The tape instantiates the same polytope. The substrate is silicon instead of spacetime. The function is identical.

This hypothesis is coherent. It is not falsified by calling it metaphor. It must be tested on its own terms.

---

## 2. What a Genuine Probe Would Test

If the catalytic tape and a black hole share the same function, then the tape should exhibit the same structural behaviors without those behaviors being pre-encoded in the test:

1. **Holographic encoding.** Bulk information encoded on the boundary. Entropy scales with boundary area, not interior volume. The ratio should emerge from measurement, not from definition.

2. **Information preservation across the boundary.** Topological invariants survive a one-way crossing (projection, truncation) when classical magnitude is destroyed. The surviving invariant should not be trivially recoverable from the pre-crossing encoding.

3. **Irreducible hardness at the boundary.** There exists a class of information that cannot cross. The decodable / non-decodable taxonomy should emerge from the structure, not from the test design.

4. **Maximum information density.** A bound exists on simultaneous information content. Throughput can exceed static storage (catalytic cycling), but the simultaneous bound should be measurable and non-arbitrary.

---

## 3. Experiment-by-Experiment Audit

### 3.1 Exp 42.20 — AMPS Firewall

**Claim:** The Python garbage collector acts as the AMPS Firewall, violently severing memory pointers at the Page Time to enforce entanglement monogamy.

**What the code does:**
- Creates a singularity at dps=1000, records SHA-256
- Drops dps to 500, calls gc.collect()
- Reads refcnt at the old memory address via ctypes
- Telemetry: `refcnt: 2`

**Assessment:** The telemetry contradicts the claim. `refcnt: 2` means the object still has active references — it was NOT severed. The kill-shot branch (`else: print("Memory reallocated!")`) is the fallback path, not the primary result. Reading a refcount at a freed/reused address is undefined behavior — the value 2 is not a reliable signal of anything.

**Verdict:** The GC is collecting garbage. It is not enforcing entanglement monogamy. The experiment does not demonstrate a firewall. It demonstrates that gc.collect() runs when called.

### 3.2 Exp 42.21 — Bekenstein-Hawking Area Law

**Claim:** The entropy-to-area ratio S/A converges to 30.0 as mass approaches infinity, proving the "Computational Planck Length" is 1/30 and the holographic boundary is constrained by register bit-width.

**What the code does:**
```python
A = ceil(bit_length / 30.0)          # Area DEFINED as bits/30
S = shannon_entropy(man) * bit_length # Entropy ~ bit_length
ratio = S / A                          # ~ bit_length / (bit_length/30) = 30
```

**Assessment:** This is a tautology. For random data, Shannon entropy per bit is ~1, so S ~ bit_length. Area is defined as `bit_length / 30`. Therefore S / A ~ 30 by construction. The "constant" is the divisor chosen in the area definition. The "Computational Planck Length" of 1/30 ~ 0.0333 is an artifact of the 30-bit limb architecture of mpmath's internal representation.

**Verdict:** No constant was discovered. The convergence to 30 is guaranteed by the definitions. A genuine holographic test would not pre-define area in terms of the very quantity being measured. The S/A ratio would need to emerge from an independent measurement of boundary geometry, not from `bits/30`.

### 3.3 Exp 42.22 — Kerr Ergosphere (Computational Superradiance)

**Claim:** Particles entering the ergosphere of a spinning black hole can steal rotational energy via the Penrose process.

**What the code does:**
```python
# "Spinning" = barrel shift of mantissa bits
kerr = (bh_man << spin) | (bh_man >> (total_bits - spin))

# "Stealing" = left-shift + OR boundary bits
escaped = (particle_man << width) | boundary_bits

# "Spin depleted" = bare counter decrement
final_spin = initial_spin - width
```

**Assessment:** "Stealing bits" is a left-shift followed by OR-ing bits from the black hole's mantissa into the particle. The particle's new "precision" (bit_length) increases because bits were appended — 35 + 128 = 163. This is bit manipulation, not energy transfer. "Spin" is an integer counter with no physical correspondence.

**Verdict:** Barrel shifts and OR operations are not frame dragging. Bit concatenation is not superradiance. The operations are real bitwise operations. The physics labels are not.

### 3.4 Exp 42.23 — True Singularity (Core Crushing)

**Claim:** The topological winding number holds through the subnormal regime until the mantissa hits 0x000, at which point the probe crashes with ZeroDivisionError — the mathematical continuum collapses at the hardware floor.

**What the code does:**
```python
fz = val * z + noise       # f(z) = M*z + 0.1*M
dfz = val                   # f'(z) = M
quotient = dfz / fz         # When M=0: 0/0 -> ZeroDivisionError
```

**Assessment:** For `f(z) = M*z + c`, the winding number around the origin is 1 for all M > 0 — this is determined by the function's structure, not by measurement. The winding would be 1 at M=1, M=0.001, and every value down to the subnormal limit. The crash at M=0 is `0/0` — the function `f(z) = 0*z + 0 = 0` is identically zero, which has no winding number. The crash is a degenerate function, not a physical singularity.

**Verdict:** The winding number was guaranteed to be 1 before the experiment ran. The crash is a division by zero. IEEE 754 subnormal walks are genuine low-level computing. The "true singularity" framing is not.

---

## 4. Structural Assessment

### 4.1 What Each Experiment Actually Tests

| Exp | Actual test | Physics label |
|-----|------------|---------------|
| 42.20 | gc.collect() behavior, ctypes refcount probe | AMPS Firewall |
| 42.21 | bit_length arithmetic with /30 divisor | Bekenstein-Hawking area law |
| 42.22 | Barrel shift + OR on mpf tuples | Kerr ergosphere / Penrose process |
| 42.23 | IEEE 754 subnormal bit walk, division by zero | True singularity |

### 4.2 Why the Hypothesis Is Not Tested

The hypothesis says: same polytope, different substrate. Testing it requires:
1. Identify the polytope's structural signature (what it must do regardless of substrate)
2. Measure whether the tape exhibits that signature
3. WITHOUT pre-encoding the signature into the measurement

Every experiment in this directory pre-encodes the result:
- 42.21: Area DEFINED as bits/30 → S/A must converge to 30
- 42.23: Function f(z)=Mz+c chosen with winding=1 → winding must be 1
- 42.20: Kill shot branch is else fallback → "firewall" fires regardless of refcnt
- 42.22: Particle bit_length = old + stolen → "superradiance" is arithmetic identity

The experiments declare what they will find, then find it, because the finding is built into the construction.

---

## 5. What a Genuine Probe Would Look Like

Staying within the hypothesis. The tape instantiates the black hole polytope. To test this without pre-encoding:

### 5.1 Area-Law Test

```
Hypothesis: Entropy of the catalytic tape scales with boundary area, not tape volume.
Test: Vary tape size (volume) and boundary configuration independently.
      Measure Shannon entropy of tape state during computation.
      If entropy ∝ boundary, not volume → holographic.
      If entropy ∝ volume → classical (not holographic).
Constraint: Boundary must be defined independently of the volume measurement.
            Cannot define area as "volume / 30" and then "discover" S/A = 30.
```

### 5.2 Information Survival Test

```
Hypothesis: Topological invariants survive a one-way projection that destroys classical magnitude.
Test: Encode information in both classical magnitude AND topological phase.
      Apply a one-way projection (z -> Re(z), truncation, etc.).
      Recover the topological invariant from the projected data.
      Verify classical magnitude is destroyed (null decoder returns wrong answer).
      Verify topological invariant survives (winding, spectral gap, IPR).
Constraint: The invariant must be genuinely encoded, not pre-stored.
            The projection must be genuinely one-way, not reversible.
```

### 5.3 Decodability Boundary Test

```
Hypothesis: There exists a sharp boundary between decodable and non-decodable information.
Test: Slide problem family from abelian-HSP through dihedral to symmetric groups.
      Measure decodability order parameter.
      Identify the exact mathematical structure that breaks at the boundary.
Constraint: The non-decodable class should match the dihedral/lattice barrier.
            If everything is decodable or nothing is, the boundary is illusory.
```

### 5.4 Maximum Density Test

```
Hypothesis: The tape has a maximum simultaneous information density (Bekenstein-type bound).
Test: Encode increasing amounts of information on the tape without catalytic cycling.
      Measure the point where additional encoding fails (distortion, collision, loss).
      Compare to theoretical bound (tape area / 4 in appropriate units).
Constraint: The bound must be physical (hardware-limited), not arbitrary.
            On classical RAM, the bound should relate to SRAM cell density.
```

---

## 6. What Is Genuine

The catalytic tape discipline — Bennett tape, XOR-braid, Feistel inverse, SHA-256 restoration — is correctly implemented. The reversible computing infrastructure is real. The SHA-256 hash verification of tape restoration works. The topological graph analysis (winding numbers detecting acyclicity) is mathematically sound.

These are real achievements. They are the substrate on which a genuine probe could be built.

---

## 7. Recommendation

The Phase 9 black hole experiments should be reclassified:

| Current status | Recommended status | Reason |
|---------------|-------------------|--------|
| REPORTED / BASELINE | METAPHOR | Physics labels applied to standard operations |
| Claim: proven | Claim: not tested | Pre-encoded results do not constitute tests |

The experiments are not fraudulent. They are mislabeled. The operations are real — gc.collect() runs, bit shifting works, SHA-256 verifies. The physics claims attached to those operations are not supported by the operations themselves.

The hypothesis remains coherent. The experiments in this directory do not test it.

---

## 8. Next Steps

1. **Implement the area-law test** (Section 5.1). This is the highest-value experiment. If the tape exhibits holographic entropy scaling without pre-encoding, it would be the first genuine demonstration that the black hole polytope is instantiated on silicon.

2. **Remove the tautologies.** Delete or archive experiments that pre-encode their results. Replace with experiments that test the hypothesis rather than asserting it.

3. **Document the encoding.** For every physics label applied to a computational operation, document: (a) what the operation actually does, (b) what structural property of the polytope it is claimed to instantiate, (c) what independent measurement would verify the instantiation.

4. **Connect to external research.** The Hartnoll/Perlmutter/Julia/Keating research program independently identifies the primes-to-black-holes connection through conformal field theory. A catalytic tape experiment that maps primes to phase oscillators and measures zeta zeros via winding numbers would be a direct bridge to this research.
