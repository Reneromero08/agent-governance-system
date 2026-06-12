# Phase 47 Audit Report: The Atomic Ground State

## Scope

Six experiments mapping Standard Model physics to computational substrate
phenomena. Each evaluated on: sensor validity, claim correspondence,
engineering integrity, and whether the computational phenomenon structurally
maps to the physical claim (isomorphism) or is an arbitrary correlation
(noise).

---

## Exp 47.1: The Nucleus (Protected Memory Knot)

**Sensor**: Nanosecond timing of `gc.collect()` after object deallocation.
Compares GC scan time for reference-counted bytearrays (unbound nucleons,
refcount drops to 0 synchronously) vs cyclic reference graphs (nuclear knot,
GC cycle detection required).

**Finding**: At N=3: unbound 1,318,910 ns vs bound 1,354,801 ns (1.03×).
At N=238: unbound 3,577,653 ns vs bound 15,816,859 ns (4.42×).
The GC cycle-resolution cost scales nonlinearly with cycle size.

**Structural correspondence**: The cyclic reference graph genuinely imposes
a computational penalty that does not exist for acyclic structures. The OS
GC cannot deallocate a closed pointer loop without dedicated cycle-detection
traversal. This IS a measurable topological barrier, not an arbitrary metric.

**Issues**:
- Timer wraps `gc.collect()` after `del`, not the deallocation itself.
  The "unbound" case measures empty-heap scan time, not "refcount destruction
  latency." The comparison is real (cyclic vs acyclic GC cost) but the framing
  of the unbound case as "destruction" is imprecise.
- Gate thresholds (1.01×, 3.0×) are tuned to observed values.
- Tape never modified. SHA-256 verification is tautological.

**Verdict**: Real signal. The computational isomorphism holds — cyclic pointer
structures impose a collective topological barrier that acyclic structures do
not. The GC cycle detection algorithm IS a mechanism for detecting and resolving
closed reference loops that imposes measurable computational cost. **Passes.**

---

## Exp 47.2: Electron Orbitals (Topological Edge States)

**Sensor**: Non-Hermitian tight-binding Hamiltonian on 15×15 lattice with
3×3 core sink (-100i) and boundary chemical potential μ. Edge states
identified by >50% boundary probability after excluding core states
(imag(eig) < -50). Shell quantization tested by sweeping μ from 0 to 5
at 6 discrete points.

**Finding**: Core IPR = 0.119 (core states localized). Max core overlap
for edge states = 0.000 (edge states have zero amplitude at core sites).
Edge state counts drop as boundary energy increases (206 → 57 across
μ = 0 to 5).

**Structural correspondence**: The non-Hermitian Skin Effect naturally
localizes eigenstates at boundaries. The nucleus sink (-100i) creates a
spectral gap that separates core from edge states. The edge state count
decreases as boundary energy increases because higher energy pushes
eigenstates into the bulk. This is standard non-Hermitian topological
physics — the computational substrate (matrix diagonalization) reveals
the same bulk-boundary correspondence observed in topological insulators.

**Issues**:
- Gate 3 (shell quantization): `len(set(shell_counts)) > 1` only checks
  non-constancy, not actual quantization. The drops (26, 48, 55, 14, 6)
  are not integer multiples.
- The 0.000 core overlap is structurally guaranteed by the anti-Hermitian
  gap (-100i vs real boundary potential), not a discovered invariant.
- 6 sample points is sparse for a "quantization" claim.

**Verdict**: Real physics, weak quantization gate. The edge-state
localization and core insulation are genuine non-Hermitian phenomena.
The "shell quantization" claim overreaches the data. **Passes with caveats.**

---

## Exp 47.3: The Pauli Exclusion Principle (Hash Collision Prevention)

**Sensor**: Minimum eigenvalue gap between edge states for Bosonic
(Hermitian, gamma=0, TRS preserved) vs Fermionic (gamma=0.6, Peierls
phase, TRS broken) Hamiltonians on 15×15 lattice.

**Finding**: Bosonic min gap = 0.000000 (degenerate, TRS protects Kramers
pairs). Fermionic min gap > 0.001 (degeneracy lifted, TRS breaking forces
level repulsion).

**Structural correspondence**: Time-reversal symmetry breaking is a
well-established mechanism for lifting spectral degeneracies. The Peierls
substitution (magnetic flux per plaquette) breaks TRS and splits Kramers
pairs. This is mathematically identical to the mechanism that produces
level repulsion in quantum systems with spin-orbit coupling. The
isomorphism holds: TRS breaking → degeneracy lifting → states forced
apart → no two states at the same energy.

**Issues**:
- Gate 2 threshold (gap > 0.001) is near numerical noise for a 225×225
  matrix. The actual gap magnitude matters for physical significance.
- The Bosonic and Fermionic models differ only in alpha (0 vs 1/3). Same
  Hamiltonian class — the Peierls phase is the TRS-breaking parameter.
- Tape never modified.

**Verdict**: Legitimate physics. TRS breaking lifts degeneracies. The
isomorphism to the exclusion principle (no two states at the same energy)
is structurally valid. **Passes.**

---

## Exp 47.4: The LHC Overflow Exploit (Particle Generation)

**Sensor**: Palindrome match rate (`calculate_spin`) and popcount parity
(`calculate_charge`) on 64-bit binary string fragments from a truncated
mpmath mantissa after precision reduction and noise multiplication.

**Finding**: 26 fragments generated. Spin values: 0.375 to 0.594, all
clustered around 0.5 ± 0.1. Charge: random parity (~50% each). Boson
classification (spin > 0.55): 10 fragments. Fermion (spin ≤ 0.55):
16 fragments.

**Structural correspondence**: None. The palindrome match rate on a
random 64-bit binary string has expected value 0.5. The observed values
are exactly what randomness produces. The Boson/Fermion split is an
arbitrary threshold (0.55) on a distribution naturally centered at 0.5.
Change the threshold to 0.52 and every fragment becomes Fermion. Change
to 0.48 and every fragment becomes Boson. The classification is not
discovering a property of the data — it is imposing one.

The popcount parity produces uniformly random +/- labels. No structure
in the mantissa fragments survives the arbitrary threshold.

**Issues**:
- `calculate_spin` operates on each fragment independently. For 64-bit
  random strings, the palindrome rate distribution has mean 0.5 and
  standard deviation ~0.0625. The observed range (0.375-0.594) is
  exactly within ±2σ of the mean.
- The BennettHistoryTape records operations to a list and pops them.
  The tape itself is never modified. Verification is structural no-op.
- No memory allocation instrumentation, no fragmentation measurement,
  no heap analysis. The "mantissa shattering" is string manipulation.

**Verdict**: **NULL RESULT.** Every measured property is a statistical
artifact of randomness. The Boson/Fermion classification is determined
by an arbitrary threshold, not by any structural feature of the data.
The experiment does not demonstrate any computational phenomenon that
maps to particle generation. **Fails.**

---

## Exp 47.5: The Higgs Mechanism (Normalization Drag)

**Sensor**: Mean and standard deviation of `mpmath.mpf(shard) + 1.0`
addition latency over 50,000 iterations at 10 bit-lengths (0 through
8192), with P99 outlier filtering. Higgs detection: derivative or
std_dev exceeding 1.5× previous value at ≥512 bits.

**Finding**: Latency generally increases with bit-length (unsurprising —
bigger integers require more arithmetic operations). Specific values
depend on mpmath's internal bigint implementation. The "Higgs" flag
triggers at 512 bits based on the 1.5× heuristic.

**Structural correspondence**: The latency hierarchy from integer
addition is a real computational phenomenon — larger operands cost
more cycles. However, the claim that this maps specifically to the
Higgs mechanism requires: (a) that the 512-bit boundary corresponds
to a cache-line crossing, and (b) that the latency spike at this
boundary is structurally equivalent to mass acquisition via the
Higgs field.

- (a) is factually incorrect: mpmath objects are not simple structs.
  The internal mantissa is stored as a Python bigint (dynamically
  allocated digit array), not as consecutive bytes at a fixed offset.
  A 512-bit bigint does not magically align to a single cache line.
- (b) Even if (a) were true, a cache miss latency spike maps to "mass
  acquisition" only by analogy, not by structural correspondence.

**Issues**:
- 0-bit latency = 400 ns with std_dev = 0.0 across 50,000 iterations.
  Zero variance is extremely unlikely in CPU timing — suggests an mpmath
  fast-path or timer resolution floor. Unexplained.
- Gate 2 checks `latencies[64] < latencies[8192]` and `latencies[256]
  < latencies[4096]` — selective pairs, not full monotonicity. The 1-bit
  vs 64-bit comparison (which would violate monotonicity) is avoided.
- Tape never modified.

**Verdict**: **WEAK.** The timing data is real (bigger operands cost more),
but the structural mapping to the Higgs mechanism fails on the cache-line
claim. The Higgs detection is a post-hoc heuristic on 10 data points.
The experiment would benefit from hardware performance counter usage
(actual cache miss measurement) rather than the derivative heuristic.
**Fails the specific Higgs claim; passes the general mass-bit-length
correlation.**

---

## Exp 47.6: Quark Confinement (String Tension & Pair Production)

**Sensor**: Nanosecond timing of `mmap` byte reads at increasing offsets
(8B to 16KB) in both warm (pre-faulted) and cold (untouched) virtual
memory. 1,000 iterations per offset with P99 filtering.

**Finding**: Warm latency hierarchy: 8-64B ~100 ns (L1 cache), 128B-2KB
~130-200 ns (cache/TLB traversal). Cold latency: 4KB+ ~2000 ns (page
fault + OS physical RAM allocation).

**Structural correspondence**: The memory hierarchy creates a genuine
latency hierarchy. Offsets within the same cache line (≤64B) have
minimal latency (asymptotic freedom). Offsets crossing cache/TLB
boundaries incur traversal cost (string tension). Offsets crossing
page boundaries (≥4096B) trigger OS page faults — the OS must allocate
physical RAM, update page tables, and handle the fault before returning
control (pair production). The OS intervention creates a new physical
memory mapping where none existed — a structural analog to pair
production from the vacuum.

**Issues**:
- "Pair production" at 4096B offset = 2 page faults instead of 1 (two
  byte reads at offset 4096 cross into a second 4KB page). The latency
  doubles because the OS handles two faults, not because of any
  deeper mechanism.
- Gate 3 threshold (1.5×) is reasonable for page fault detection but
  the "pair production" naming inflates the mechanism.
- Tape never modified.

**Verdict**: **Passes.** Real computational phenomenon with genuine
structural correspondence. The memory hierarchy creates latency
discontinuities at cache-line and page boundaries. The OS page fault
handler allocates new physical memory — a structural analog to
vacuum pair production. Best-engineered experiment in Phase 47.
**Passes.**

---

## Cross-Cutting Issues

1. **BennettHistoryTape is never modified.** Across all six experiments,
   `record_operation` appends to a list and `uncompute` pops it. The
   underlying tape bytearray is never XOR'd, never read during computation,
   and never altered. The SHA-256 verification at the end is structurally
   guaranteed to pass — the tape has not been touched. This is not a
   catalytic computation; it is a null operation dressed as one.

2. **Tape size is arbitrary.** 10MB or 256MB tapes serve no functional
   purpose. The tape is a ceremonial object in these experiments.

3. **Gate thresholds are tuned to observed values.** Most experiments use
   thresholds (1.01×, 1.5×, 3.0, 0.55, 0.001) that are set to match what
   the code produces rather than derived from physical principles.

4. **No external validation.** None of the experiments compare against
   known physical constants, published measurements, or independent
   datasets. The "hardening gates" test internal consistency only.

---

## Summary

| Exp | Sensor Valid | Structural Correspondence | Gate Integrity | Overall |
|-----|-------------|--------------------------|----------------|---------|
| 47.1 | Yes | GC cycle resolution ↔ topological binding | Thresholds tuned | PASS |
| 47.2 | Yes | Skin effect ↔ edge localization | Quantization gate too weak | PASS* |
| 47.3 | Yes | TRS breaking ↔ level repulsion | Gap threshold near noise | PASS |
| 47.4 | No | None — random palindrome noise | Boson/Fermion arbitrary | **FAIL** |
| 47.5 | Partial | Mass-bit-length correlation real, Higgs cache miss false | Selective monotonicity | WEAK |
| 47.6 | Yes | Memory hierarchy ↔ confinement ladder | Reasonable thresholds | PASS |

*Exp 47.2 passes on core physics, fails the quantization claim.

Four experiments demonstrate genuine computational phenomena that structurally
correspond to their physical claims. One experiment (47.4) is a complete null
result — every sensor reading is statistical noise. One experiment (47.5) has
valid timing data but a factually incorrect physical mechanism claim.
