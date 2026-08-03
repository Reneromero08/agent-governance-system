# CAT_CAS Phase Frontier Lab

This directory is intentionally a mutable research surface. It is not a frozen
package, promotion packet, or new stopping point.

## Canonical claim reconciliation

Claim authority is reconciled through source head
`466557668e8ea4497f58794591291990048d3964`. The canonical per-milestone
verification levels, restoration classifications, source-audit authority,
and strict ceilings are recorded in
`../CLAIM_AUTHORITY_REGISTRY.json`.

The original CATVM open-intermediate atomic transaction is
`REJECTED_SOURCE_DEFECT_ATOMICITY`: its `PROJECT Z` response precedes a later,
independent `RESTORE` command. Its bounded F-to-G algebra, hidden resident
`Y`, direct consumption of the actual `Y`, later inverse restoration,
same-carrier reuse, and same-UID process controls remain valid separately.
The exact 15-node rank-two scheduler remains valid only at its recorded
fixture ceiling. The homogeneous Boolean suffix quotient remains valid only
for its neighbor-AND/OR family; mixed/nonperiodic depth-6 ranks reach 14
against the homogeneous ceiling 7.

Every result after audit head `16b3db783f86d966fdb52830d591a0aa8d27cc5d`
is `SOURCE_AUDITED_PACKAGE_LOCAL` unless the registry records stronger
evidence. Tolerance-defined inverse-plus-canonical closure is
`INVERSE_PLUS_CANONICAL_NUMERICAL_QUOTIENT`, not unqualified exact
restoration. Snapshot reload is a recovery baseline, not catalytic
restoration.

No distinct phase resource, computational advantage, Small Wall crossing,
physical waveform execution, or replacement of physical bits with pi has
been established. The explicit verification/resume goal superseded the
earlier development stop. The completed public-observation package was not
restarted; the required shared-latent successor and its distinct owner-binding
repair are recorded below.

## Active construction

The first bottleneck repair replaces the prior `O(n*k*M)` time-layer carrier
with one fixed `O(k*M)` resident torus.

For odd prime cyclic width `M` and nonzero public shift `w`, the map

```text
y[s] = x[s] * x[s-w]
```

is a bijection on each odd `p`-root phase group. The inverse follows from the
same public shift:

```text
2*x[0] = y[0] + y[w] - y[2w] + ... - y[(M-1)w]
x[r]   = y[r] - x[r-1]
```

All arithmetic in the engine remains complex phase multiplication,
conjugation, fixed phase powers, and fixed p-fold locking. It does not decode a
residue during native evolution.

The construction removes the carrier-growth defect and removes any hardcoded
program-length bound. It does not by itself remove the equivalent compact
classical dynamic program. C5 therefore remains the next live bottleneck.

## Fixed-resident development result

The twin-rail form removes the carrier-specific seed backup used by the earlier
layered engine. Both rails borrow the same dirty common-mode phase. Computation
occupies only their relative phase, the seed is public, and the inverse returns
the rails to their original relation.

Development evidence:

```text
resident carrier                 310 complex cells
Python exact cases               10 / 10
Python maximum steps             4,096
direct-metal maximum steps       1,048,576
million-step result              exact
million-step restoration         1.77722398948e-16
retained inverse factors         0
carrier-specific seed backup     none
```

The engine accepts any positive step count representable by the host runtime;
there is no package-scale `MAX_STEPS`.

## Streaming nonlinear phase VM

The same twin-rail law now supports a streaming ternary phase machine:

```text
ROT(t,c)       z[t] <- z[t] * omega^c
ADD(s,t)       z[t] <- z[t] * z[s]
MULADD(a,b,t)  z[t] <- z[t] * omega^(a*b)
SWAP(a,b)      exchange both physical rails
```

`MULADD` does not decode `a` or `b`. Its factor is the fixed roots-of-unity
polynomial

```text
F(x,y) = (
    1 + x + x^2 + y + y^2
    + omega^2 * (xy + x^2y^2)
    + omega   * (xy^2 + x^2y)
) / 3.
```

On ternary roots, `F(omega^a, omega^b) = omega^(a*b)`. `ROT`, `ADD`, and
`MULADD` provide constants, field addition, and field multiplication. Since
every finite function over `F_3` has a finite polynomial representation, the
gate algebra can evaluate arbitrary finite ternary functions with ancillas;
compute-copy-uncompute makes the evaluation catalytic. This is a functional
completeness construction for scalable finite phase circuits, not a claim that
one fixed finite carrier stores infinite information.

Direct-metal development evidence on isolated Core 3:

```text
program 1                         10,000,000 instructions
program 2                         10,000,017 instructions
resident carrier                 48 complex cells / 24 registers
retained inverse factors         0
both independent boundaries      exact
maximum root distance            2.58641936433e-13
maximum restoration error        8.52841676469e-13
wrong inverse error              1.73205080757
omitted inverse error            1.85405371962
actual restored reuse            PASS
```

The optimized strict C forward path is about `2.00-2.02x` the independent
compact ternary evaluator. The `-Ofast` development probe is
`1.66-1.69x`. Full forward-plus-restoration is `3.34-4.29x`, depending on
compiler mode.

That is not C5 advantage. It does show that the earlier hundreds-fold penalty
was implementation friction rather than intrinsic phase-work growth. The
remaining gap is now the cost of digitally emulating complex phase algebra on
the same conventional processor that runs the compact evaluator.

## Public programs, scalable time, and scalable memory

`streaming_phase_vm.c` now accepts a strict public `.holo` program. The native
engine reads only boundary loads and the public `ROT`, `ADD`, `MULADD`,
`SWAP`, `CSWAP`, and `PCSWAP` stream. It does not contain or link the scalar
adjudicator.

An optional `PASSES n` field repeats the complete instruction body without
expanding the program. The inverse indexes the same public body in reverse:

```text
public program storage             O(instructions)
resident carrier                   O(registers)
per-step inverse history           0
forward / inverse time             O(passes * instructions)
```

Direct-metal development evidence:

```text
unrelated public programs exact    2 / 2
large phase memory                 65,536 registers / 131,072 complex cells
large-memory boundary digest       exact against independent reference
compact repeated program           8 stored instructions
repeated native gates              8,000,000
repeated-program restoration       8.66678137878e-13 maximum on reuse
wrong inverse error                1.73205080757
omitted inverse error              1.77145811692
paired syntax parser negatives     8 / 8
embedded-NUL parser negatives      3 / 3
ASan + UBSan                       PASS
```

Large carriers emit a deterministic digest and nonzero count instead of
serializing every zero-valued register. The independent reference emits the
same digest.

Boolean values fit as the `0/1` boundary subset of `F3`. The phase polynomials

```text
NOT(a)     = 1 + 2a
AND(a,b)   = ab
XOR(a,b)   = a + b + ab
OR(a,b)    = a + b + 2ab
NAND(a,b)  = 1 + 2ab
```

matched the independent reference on all four input pairs. Since NAND is
functionally complete, ordinary finite Boolean circuits can be compiled into
this phase algebra. That does not mean the current digital carrier is faster
than Boolean hardware; it establishes syntax and semantic reach.

## Phase-native conditional routing and stored program state

`CSWAP(c,a,b)` adds a Fredkin operator without decoding `c`. The engine forms
the phase indicator

```text
g(c) = c^2 * F(c,c)^2 = omega^(2c + 2c^2)
```

which selects only phase symbol `1`. The two target relations are transformed
simultaneously from their original values. The operator is identity for
control symbols `0` and `2`, swap for symbol `1`, and self-inverse.

Development evidence:

```text
exhaustive F3 CSWAP table         27 / 27 exact
repeated conditional gates        3,000,000
repeated conditional reuse error  2.15739188326e-13
decoded control feedback          none
```

A two-slot fixed Fredkin fabric then placed a one-hot program counter, program
enable bits, gate workspace, and data in phase relations. The host executed
the same fixed 18-gate schedule in both cases:

```text
program bits [1,0]  -> data [2,1]
program bits [1,1]  -> data [1,2]
```

Each slot computed `enable = pc * program`, conditionally routed data, and
uncomputed `enable` back to zero before advancing the phase-resident program
counter. Both boundaries matched the separate C reference and the carrier
restored below `8.5e-14`.

This closes the narrow decoded-branch gap for predicated reversible circuits.
It is the construction pattern used by the compiler below; it is not an
unbounded stored-program fabric or a physical parallel Fredkin array.

## Compact Fredkin compiler and exact-byte custody

`fredkin_phase_compiler.c` lowers a public multi-gate Fredkin network to a
phase-resident fabric mechanically. Its first construction used a one-hot
program counter and scanned all slots for every counter position. That was
correct but paid `O(N^2)` native gates for `N` public Fredkin slots.

The current compiler replaces that avoidable scan with the native
`PCSWAP(program, control, left, right)` relation. `PCSWAP` composes the
roots-of-unity product polynomial and the Fredkin selector without decoding
either phase. The compiler now emits one phase program enable per gate, the
data relations, and exactly one native instruction per public gate per cycle.
It never evaluates the circuit or chooses its result.

The fused instruction is not counted as free: its C kernel evaluates three
product polynomials directly, and the phase selector evaluates one additional
product polynomial internally, for four total plus two relation writes. That
is constant work per gate, so the repaired law is linear rather than a hidden
quadratic scan.

The same three-gate routed-network fixture now compiles to three native
instructions and executes three native gates:

```text
circuit wires                       5
Fredkin slots                       3
compiled registers                  8
compiled public instructions        3
native gates                        3
boundary symbols                    [1,1,1,1,1,0,2,1]
boundary digest                     df444ca2a82c721d
nominal restoration                 1.01391585001e-13
actual-restored reuse restoration   1.03720799810e-13
wrong inverse restoration           1.73205080757
omitted inverse restoration         1.73205080757
compiler byte reproduction          exact
```

The fused operator matched the separate scalar reference on all 81 total
`F3` combinations of program enable, control, left, and right. A deterministic
100,000-gate circuit then compiled to exactly 100,000 native steps and
100,003 phase registers. Its boundary matched the reference and restoration
remained `1.89494289067e-13`. The previous scan would have required
`79,999,900,000` native instructions for the same gate count; the new resource
law removes that quadratic artifact.

On the same isolated CPU core, one observed matched median for the
complex-phase forward kernel was `15.976x` slower than the compact byte-valued
C reference at 100,000 gates; an independent replay measured `15.602x`.
The target's DVFS was left unchanged and produced other slower runs, so these
ratios are descriptive rather than fastest-case or clock-normalized
benchmarks. Every observation remains far from C5 advantage. The result is an
architecture and resource-law improvement; the next necessary change is
parallel or physical phase coupling, not another interpreter constant-factor
pass.

The compiler, native engine, and separate reference now parse length-aware
raw bytes and reject embedded NUL bytes. The complete nine-program suite was
retransferred without text normalization and reproduced 9/9 native/reference
boundaries. This repaired an evidence-identity defect found by focused review;
it did not change the phase mechanism or its numerical results.

This is a compact compiler for finite Fredkin networks, not a claim of C5
advantage, infinite storage, physical parallelism, or completed phase
computing.

## Spatial pthread phase fabric

`spatial_phase_fredkin.c` removes the sequential gate-stream schedule from
the next experiment. A layer contains `width` disjoint complex-phase
`PCSWAP` interactions over a shifted partition of `3*width` data relations.
The partition shifts by one relation between layers, so information propagates
through the spatial carrier. A persistent pthread pool executes each layer
with explicit synchronization that is visible to ThreadSanitizer.

The native kernel still operates on twin complex rails. It does not decode
the program, control, or data while evolving. Its `F3` product polynomial now
uses the unit-circle identity `z^2 = conjugate(z)`, reducing each product
polynomial from six complex multiplications to three without converting the
phase relation to a scalar symbol.

Six parameter sets matched the independently compiled scalar reference,
including identical boundaries at one and six threads and a case with more
threads than gates. The largest current probe had:

```text
spatial width                       20,000 gates
layers / logical depth             32
total phase gates                   640,000
program phase registers             640,000
data phase registers                60,000
resident complex cells              1,400,000
full boundary digest                9453c5a6d7c6f665
data boundary digest                9bc8f4d1201e5665
program-variant data digest         c65025a0431ff9e5
nominal restoration                 5.00138098873e-13
actual-restored reuse restoration   9.792812436e-13
cross-program reuse restoration     9.89511047202e-13
wrong inverse                       1.73205080757
omitted inverse                     1.73205080757
```

ASan, UBSan, leak detection, and ThreadSanitizer pass for both executables.
Strict decimal parsing also rejects negative, signed, overflowing, zero, and
trailing-garbage parameters before execution. The program-sensitivity control
reports rather than rejects legitimate identity computations: even repetition
of a self-inverse layer can correctly erase the distinction between two
programs. A nondegenerate canonical case must and does change its data result.
Focused reviewer `SOL-XHIGH-SPATIAL-PTHREAD-FREDKIN-01` independently extended
the checks to all 81 `F3` PCSWAP cases, 25,957 layer partitions, a 771-layer
ThreadSanitizer stress, and a 1,088,000-gate recurrence. Verdict: `PASS`, with
no remaining findings.

The SSH daemon had inherited affinity `0-1`, but its cgroup permits all six
online physical cores. Applying `taskset -c 0-5` only to each experiment
process produced a six-core phase median of `66,593,768 ns` versus
`114,427,905 ns` on one core: a real `1.718x` spatial wall-time reduction.
The strongest one-core compact scalar median remained `12,372,273 ns`, so the
six-core phase execution was still `5.383x` slower. DVFS was unchanged and
uncontrolled. No system configuration changed. The construction therefore
establishes finite spatial logical parallelism and a race-clean C runtime, not
C5 or physical phase computation.

## Dependency-layered public phase VM

`parallel_phase_vm.c` now connects the existing `.holo` language and Fredkin
compiler directly to the pthread phase runtime. The scheduler assigns each
instruction to the maximum ready layer over every register it accesses. It
never reads phase data, boundary data, or expected results. Instructions share
a layer only when their complete accessed-register sets are disjoint, so they
commute. A mechanical verifier rejects any lost instruction or within-layer
register collision.

Narrow layers execute directly; layers with at least 256 independent
instructions use the persistent pthread pool. Forward execution traverses
passes and layers; restoration reverses both. The exact public instruction
stream remains the source of the inverse, with no stored inverse history.

A deterministic C generator emitted an 8,192-wide program repeated for 127
passes:

```text
public program bytes                527,301
public program SHA-256              50137707408e1d0a529baed6f9820f0e09fc9b20d765552ff2259f8d43f88d04
stored instructions                 8,192
total phase gates                   1,040,384
dependency layers per pass          1
logical depth                       127
phase registers                     32,768
resident complex cells              65,536
boundary digest                     3b39182758a1e325
nominal restoration                 4.99022921114e-13
actual-restored reuse restoration   5.23070005001e-13
wrong inverse                       1.73205080757
omitted inverse                     1.73205080757
```

The parallel, sequential phase, and independent scalar executables produced
the same boundary. Nine committed programs and 20 deterministic mixed-opcode
programs also matched across one-thread parallel, six-thread parallel,
sequential phase, and scalar execution. ASan, UBSan, leak detection, and
ThreadSanitizer pass.

On the wide public program, the six-core phase median was `90,459,486 ns`,
`1.786x` faster than the one-core layered phase median and `1.980x` faster
than the sequential phase VM. The compact scalar evaluator remained
`17.741x` faster than the six-core phase VM. This is a real reduction in
phase-program wall time and logical depth, not an asymptotic or C5 advantage.

## Next active work

The phase VM and spatial scheduler are now support substrate. The primary
frontier is no longer further instruction-stream scaling. It is the
relational lift:

```text
open typed many-to-many phase relations
-> composition through shared interfaces
-> unresolved internal-port closure
-> idempotent relational boundary
-> inverse restoration
-> restored-carrier reuse
```

Any useful successor must keep the relation unresolved rather than enumerate
tuples, internal assignments, witnesses, or one ordinary circuit per case.

## Typed open quotient-relation calibration

`open_relation_phase.c` is the first relational lift that does not lower the
full `Z_N` relation into a gate stream or tuple table. A
`CYCLIC_PARITY(N)` port is represented by its quotient geometry, not by `N`
values. Each local open relation is the complete two-slot characteristic
vector of a Boolean-lattice subset of the two parity-difference cosets:

```text
EMPTY
SAME
OPPOSITE
BOTH
```

Two relations sharing one typed internal port close through idempotent Boolean
convolution over `Z2`. `AND` and `OR` are roots-of-unity phase polynomials.
The native composition function contains no loop over `N`, boundary pairs,
internal assignments, or witnesses.

At `N=64`, the strongest calibration is:

```text
left relation                       BOTH
right relation                      BOTH
boundary relation                   BOTH
boundary pairs                      4,096
derivations                         262,144
witnesses per valid boundary pair   64
native complex carrier cells        8
native witness slots                0
retained inverse factors            0
nominal restoration                 1.11022302463e-16
actual-restored reuse restoration   1.57009245868e-16
wrong inverse                       1.73205080757
reordered inverse                   1.73205080757
omitted inverse                     1.73205080757
```

The independent bounded extensional oracle agrees on all eight calibrations,
including the neutral `EMPTY o EMPTY` edge.
Duplicate presentation and witness multiplicity do not change the lawful
boundary. Port/relation presentation permutation preserves it. Empty relation
has an intact carrier and differs mechanically from an injected carrier
failure. A cut shared port is rejected, so no local cached answer survives a
disconnected diagram. The same restored carrier executes a different second
process successfully.

The native carrier remains eight complex cells from `N=4` through a
non-enumerated `N=1,000,000,000` run. Bounded exhaustive references at
`N=4, 8, 16, 32, 64, 128` all agree with the phase boundary. This is exact for
the declared quotient algebra; it is not evidence that an arbitrary
billion-state relation was exhaustively computed.

The mechanism is a genuine structured open relation and is non-enumerative
over `Z_N`, but it is quotient-extensional: the fixed `Z2` relation is fully
stored in two characteristic slots and composed by a fixed four-AND/two-OR
Boolean-convolution circuit. It is not yet a general holographic relational
computer. It cannot represent arbitrary relations such as Boolean `LEQ`,
arbitrary arity, branching diagrams, or general relational trace. A compact
classical Boolean-convolution equivalent exists. The next frontier is a richer
typed relation signature with native algebraic composition and elimination,
without full-domain truth-table or internal-assignment expansion.

Focused reviewer `SOL-XHIGH-OPEN-RELATION-PHASE-01` independently found and
closed four bounded defects: neutral `EMPTY o EMPTY` rejection, fixed-`Z2`
intensional overclaim, unenforced canonical decimal syntax plus imprecise
inverse-source wording, and the `NONE` sentinel/name ambiguity. The exact
repaired candidate passed strict GCC and static analysis, sanitizers, all
eight fixtures, all 16 quotient-relation pairs, duplicate and permutation
controls, scale and max-domain controls, 24 inverse controls, 64
cross-process restored-carrier reuse trials, deterministic replay, 17 parser
adversaries, no-smuggle inspection, and oracle non-linkage. Verdict: `PASS`;
remaining findings: none.

## Algebraic relation lift

`algebraic_relation_phase.c` moves beyond the fixed two-slot `Z2`
characteristic vector. A typed `BOOLEAN_F3` binary relation is now the zero set
of a public multiaffine polynomial:

```text
c00 + c10*x + c01*y + c11*x*y = 0 mod 3
```

The four coefficients live as relative complex phases. Given
`f(x,y)=A(x)y+B(x)` and `g(y,z)=C(z)y+D(z)`, the shared internal port closes
through the phase-native linear resultant

```text
R(x,z) = B(x)C(z) - A(x)D(z).
```

The primary calibration is the non-functional order relation
`LEQ(x,y)=x(1-y)=0`. Direct metal execution produced:

```text
LEQ o LEQ boundary coefficients    [0,1,0,2]
lawful boundary pairs              3
extensional derivations            4
native carrier cells               12
tuple / witness slots              0 / 0
retained inverse factors           0
native displacement                4.24264068712
nominal restoration                1.66533453694e-16
actual-restored reuse              1.57009245868e-16
wrong / reordered / omitted        1.73205080757 each
```

Eight heterogeneous or presentation-varied fixtures match the independent
bounded Boolean oracle. Strict GCC, `-fanalyzer`, ASan, UBSan, leak detection,
20 deterministic replays, cross-process restored reuse, cut-geometry
rejection, and native/oracle source separation all pass.

An exhaustive independent C survey found the raw resultant exact on only
`3,217 / 6,561` coefficient-signature pairs. The result is therefore not
promoted to unrestricted Boolean elimination.

The implemented repair is a prospective algebraic admission law: the left
relation must be total toward its second/internal port for each first-port
value, and the right relation must be total toward its first/internal port for
each second-port value. Each affine internal fiber is then `{0}`, `{1}`, or
`{0,1}`. The determinant is zero exactly when the two nonempty Boolean root
sets intersect. There are 25 admissible coefficient signatures on each side;
all `625 / 625` admitted pairs match exact Boolean existential composition.

The retained `EMPTY o ANY` counterexample is now rejected by both native and
reference parsers before phase evolution because its left internal fiber is
empty. This admission test uses only public input coefficients and never
computes or chooses a boundary.

The next correction is repeatable multi-internal composition using an
algebraic class closed under its own boundary outputs, then branching
relational trace.

Focused reviewer `SOL-XHIGH-ALGEBRAIC-RELATION-PHASE-01` found and closed two
bounded defects: reference/native identifier-uniqueness drift and a
counterfactual degenerate resultant mislabeled as native execution. On the
exact repaired bytes, the reviewer independently executed all 625 admitted
signature pairs: native/reference boundaries, nominal restoration, and reuse
passed 625/625; 1,875 applicability-gated inverse controls passed; cross-
process reuse passed 64/64; and strict compilation, static analysis,
sanitizers, determinism, parser adversaries, no-smuggle inspection, and oracle
non-linkage passed. Verdict: `PASS`; remaining findings: none.

## Repeatable algebraic relation chain

The stronger bi-total `BOOLEAN_F3` subclass closes the single-port mechanism
under its own outputs. Exhaustive independent adjudication found:

```text
bi-total coefficient signatures       17
distinct extensional relations         7
ordered admitted pairs               289 / 289 exact
resultant outputs remaining bi-total 289 / 289
ordered admitted triples           4,913 / 4,913 exact
left/right grouping extensional     4,913 / 4,913
```

`algebraic_relation_chain_phase.c` therefore applies the same phase-native
resultant repeatedly, one unresolved internal port at a time. It retains one
four-phase derived relation per layer, not assignments, witnesses, or
branch histories:

```text
public relation cells        4*n
derived history cells        4*(n-1)
complete carrier cells       8*n-4
native evolution             O(n)
tuple / witness slots        0 / 0
retained inverse factors     0
```

The parser initially hid an `O(n^2)` graph-normalization defect. That was
repaired with streaming byte custody and indexed endpoint normalization. No
fixed relation-count cap remains in the format; address space and available
memory are the practical bounds.

On the authorized userspace C target, two different deterministic
100,000-relation processes executed against the same actual restored carrier:

```text
internal ports closed                    99,999
carrier cells                           799,996
input bytes per process               4,566,792
native cross-process lifecycle time       1.888 s
maximum phase-root error             5.70825378262e-12
maximum correct restoration          3.51083346858e-16
carrier displacement                 1095.44237639
wrong inverse restoration error      1.73205080757
forward-order inverse error          1.73205080757
omitted inverse error                1.73205080757
native/reference boundary agreement  exact
```

Strict C11 compilation, GCC static analysis, ASan, UBSan, leak detection,
fresh-process determinism, presentation invariance, four generated relation
families, committed parser adversaries, and closure surveys pass.

The first focused review found one qualification defect at the exact minimum
size. A two-relation chain has only one resultant, so forward-order inverse
and reverse-order inverse are identical. The engine incorrectly called that
control applicable and falsely failed otherwise exact minimum chains. The
control is now honestly inapplicable only for that one-closure geometry.
All 289 ordered bi-total two-relation pairs now pass native/reference
agreement and restoration with zero false order applicability; the existing
dependency-sensitive order control remains effective for larger chains.
Renewed focused reviewer
`SOL-XHIGH-ALGEBRAIC-RELATION-CHAIN-01-R2` independently reproduced all
289 minimum pairs, all 4,913 triples, all 83,521 four-relation order-control
cases, the 100,000-relation execution, strict/static/sanitized builds,
determinism, parser adversaries, no-smuggle separation, restoration, and
reuse. Verdict: `PASS`; remaining findings: none.

This establishes repeatable native internal closure for a finite linear
bi-total relation chain with linear carrier history, actual inverse traversal,
restoration, and reuse. It does not establish branching geometry, arbitrary
relation classes, advantage, physical execution, or unlimited computation.
The active correction is now a branching typed relational trace whose public
geometry materially determines the boundary without host tuple expansion.

## Branching algebraic relation star

The first branching trace closes one typed Boolean hub shared by three or
more public binary relations. After canonical orientation, branch `i` has the
form

```text
f_i(x_i,h) = A_i(x_i)*h + B_i(x_i) = 0
```

and every external-port pair receives the native resultant

```text
R_ij(x_i,x_j) = B_i(x_i)*A_j(x_j) - A_i(x_i)*B_j(x_j).
```

Because every admitted hub fiber is a nonempty subset of the two-point
Boolean domain, all fibers have a common hub value exactly when every pair
intersects. The pairwise resultant boundary is therefore an exact
factorization of the existential star projection for this admitted class.
The native engine constructs those factors through products of complex
cube-root phases and conjugate phase subtraction. It does not enumerate
external assignments, select a hub witness, or materialize a truth table.

Independent scalar adjudication covers all `1,375,640` assignment rows across
all 4,913 admitted three-branch signature tuples and all 83,521 four-branch
tuples. The complete native three-branch sweep then executed all 4,913
ordered signature triples. Native and independent-reference boundary hashes,
nominal restoration, reuse, and applicability-gated controls passed
`4,913 / 4,913`.

The first exhaustive native sweep exposed one real control-accounting defect.
For signature triple `1:0:1`, an earlier pair factor was nontrivial but the
canonical final pair factor was zero. The engine marked omitted-inverse
applicability from any nontrivial pair even though the control omits only the
final pair. Applicability is now derived exclusively from the exact omitted
factor. The counterexample correctly restores to `1.11022302463e-16` with
that control marked inapplicable; the renewed 4,913-case sweep has zero
false or missed failures.

The authorized userspace C scale execution records:

```text
public branches                         1,000
closed internal hubs                        1
boundary relation factors             499,500
boundary coefficient phases         1,998,000
complete carrier cells              2,002,000
tuple / witness / truth-table slots    0 / 0 / 0
retained inverse factors                    0
maximum correct restoration      3.51083346858e-16
maximum phase-root error          9.93013661299e-16
wrong inverse error                  1.73205080757
geometry-scrambled inverse error     1.73205080757
omitted inverse error                1.73205080757
fresh-process native output          byte-identical
```

Two different 1,000-branch stars produced different factorized boundaries
and reused the same actual restored carrier. Strict C11 compilation, GCC
static analysis, ASan, UBSan, leak detection, deterministic generation,
presentation invariance, committed geometry/parser adversaries, and source
no-smuggle inspection pass.

This removes the single-linear-chain limitation and establishes one finite
branching typed relational trace. It does not yet establish multiple
connected internal hubs, cycles, arbitrary typed process graphs,
computational advantage, physical phase computation, or unlimited catalytic
phase computation. The next correction is composable relation-valued phase
memory across multiple internal hubs so a boundary relation can feed another
native closure without host expansion.

## Two-hub relation-valued phase memory

The next bounded process connects two internal hubs:

```text
A -- U -- V -- C
B --'    '-- D
```

The native engine first closes `U` between each left relation and the `U--V`
bridge. Each result remains in four carrier-relative complex phases as an
unresolved binary relation from `A` or `B` to `V`. Those phase cells feed the
second closure with the `C--V` and `D--V` relations directly. No intermediate
coefficient is decoded.

The final boundary is a six-factor relation over all external-port pairs.
Independent exhaustive adjudication records:

```text
bi-total coefficient signatures            17
distinct extensional relations               7
first-stage relation-message pairs         289
messages remaining bi-total            289 / 289
five-relation coefficient tuples     1,419,857
boundary assignment rows            22,717,712
factorized exact rows                22,717,712
multi-witness rows                      938,000
```

The primary direct userspace C execution records:

```text
public relations                              5
internal hubs                                 2
phase-resident relation messages              2
final boundary factors                        6
complex carrier cells                        52
tuple / witness / truth-table slots      0 / 0 / 0
decoded intermediate coefficients             0
retained inverse factors                      0
borrowed-carrier displacement     9.16515138991
correct restoration              1.66533453694e-16
wrong boundary inverse error         1.73205080757
geometry-scrambled inverse error      1.73205080757
omitted message inverse error         1.73205080757
```

The correct primary boundary hash is `762241e0138b94e8`. Forward bypass of the
resident relation message, which treats `U` and `V` as though they were one
hub, produces `1634d56063838ddd`. The bypass path reverses and restores
cleanly, so this is a causal result discriminator rather than generic
corruption. A different two-hub process then consumes the same actual
restored carrier and produces boundary `b981be2d7776c0e2`.

Strict C11 compilation, GCC static analysis, ASan, UBSan, leak detection,
fresh-process byte determinism, presentation invariance, all-ANY
multi-witness behavior, exact independent reference comparison, and committed
parser/geometry negatives pass.

This establishes the first finite relation-valued phase memory in the lane:
one native closure result remains unresolved in phase and is consumed by a
second closure. It does not establish an arbitrary hub tree, cycle, unbounded
recursive memory, advantage, physical phase computation, or unlimited
catalytic phase computation. The next correction is a streaming public tree
whose same four-phase relation message can propagate across arbitrary finite
internal depth without host decoding.

## Topology-generic bounded relation-tree phase closure

The fixed-depth limitation is now removed for public typed trees. The generic
C engine validates a tree with up to 64 nodes and 32 external leaves, finds
every unique external path, and recursively composes each path through
four-phase resident relation messages. No intermediate coefficient is
decoded; only the final external-pair factors cross the boundary.

The three-hub/five-leaf discriminator uses seven public relations, ten
external-pair paths, twelve resident message relations, and 116 complex
carrier cells. It produces boundary `e6bb33ad7c0cbbe0`, restores within
`1.57009245868e-16`, and then reuses the same actual carrier for a different
process producing `da5a04939bd2d482`. A four-hub/six-leaf process grows
without source specialization to 25 resident messages and 196 carrier cells,
matches exact scalar closure, and restores within `2.00148302124e-16` on
reuse.

The strengthened causal controls discriminate both result and reversal:

```text
wrong boundary inverse                 restoration error 1.73205080757
scrambled final-edge geometry          boundary 17bfcd6650455e2a, clean inverse
omitted resident-message inverse       restoration error 1.73205080757
interior-message bypass                boundary 76ce9cefd10b119e, clean inverse
```

The complete extensional survey ranges all seven bi-total Boolean relations
over all seven edges of the three-hub topology:

```text
relation tuples                         823,543
external assignment rows             26,353,376
exactly extendable rows               13,486,304
multi-witness rows                     4,538,592
mismatches                                     0
```

Strict C11 compilation, `-Werror -pedantic`, GCC `-fanalyzer`, ASan, UBSan,
leak detection, fresh-process byte determinism, scalar-reference separation,
all-`ANY` witness multiplicity, and committed tree/parser negatives pass.

Two independent Sol/xhigh reviews pass with zero remaining findings:

```text
SOL-XHIGH-ALGEBRAIC-RELATION-TREE-01-R2
SOL-XHIGH-ALGEBRAIC-RELATION-TREE-MECHANISM-02-R2
```

Review repaired a blank-record safety defect in the separate reference parser
and separated its 20-internal-node exhaustive-enumeration limit from the
native engine's 64-node topology capacity. A reported missing post-`END` gate
was retracted after exact-source inspection showed that gate already existed;
a committed negative now binds it.

This establishes topology-generic bounded tree closure and recursive
relation-valued phase memory. It does not establish cyclic relational trace,
unrestricted domains, advantage over compact classical tree algorithms,
physical phase computation, or unlimited catalytic phase computation. The
next mechanism is a phase-native cycle invariant that detects and closes loop
consistency without enumerating internal assignments.

## Exact cyclic relational phase closure

The first relational cycle now closes without forcing the earlier bi-total
tree law into a loop. Four-phase polynomial algebra in
`F3[x,y]/(x^2-x,y^2-y)` supplies two exact native operators:

```text
intersection(f,g) = f^2 + g^2
compose(f,g)      = product over u in {0,1}
                    of (f(x,u)^2 + g(u,y)^2)
```

The native implementation executes these as fixed roots-of-unity phase
polynomials. The `u` values are algebraic factors in the operator definition;
there is no runtime witness loop, coefficient decode, or scalar feedback.

The fixed public diamond `U-W-V-Z-U` holds four phase-resident relation
messages in a 44-cell carrier. Its primary exact boundary accepts only
`{00}`. Bypassing one cycle path accepts `{00,01}`; replacing intersection
with ordinary coefficient addition accepts `{00,11}`. Both altered forward
paths reverse their actual histories and restore cleanly, proving that the
loop operator changes the result rather than merely damaging the carrier.

Wrong boundary inversion and omission of the cycle-intersection inverse each
leave error `1.73205080757`. Correct primary restoration and actual-restored
cross-process reuse are both `1.57009245868e-16`.

The separate complete survey covers all 6,561 ordered pairs of all 81
multiaffine F3 polynomials:

```text
exact composition rows       26,244 / 26,244
exact intersection rows      26,244 / 26,244
```

The separate scalar cycle reference enumerates all 64 assignments of the six
public/internal Boolean variables and matches both committed processes.
Strict compilation, `-fanalyzer`, ASan, UBSan, leaks, deterministic outputs,
and committed parser negatives pass.

Two independent Sol/xhigh reviews pass with zero findings:

```text
SOL-XHIGH-ALGEBRAIC-CYCLE-01
SOL-XHIGH-ALGEBRAIC-CYCLE-MECHANISM-02
```

The mechanism review additionally swept the actual native phase operators
over all 6,561 composition pairs and all 6,561 intersection pairs. All
13,122 outputs were exact and all inverse traversals restored, with maximum
root error `2.24803028762e-15` and restoration error
`2.48253415325e-16`.

This establishes one fixed cyclic relational phase closure and removes the
bi-total restriction from the exact binary operators. It does not establish
a generic cyclic graph language, compact bounded-treewidth elimination,
advantage, physical phase computation, or unlimited catalytic phase
computation. The next mechanism is generic graph parsing plus a public,
phase-native elimination order whose carrier/provenance growth is explicit.

## Public series/parallel relational phase closure

The fixed-cycle limitation is now removed for public two-terminal
series/parallel graphs. A strict source declares arbitrary Boolean/F3 edge
relations and a public internal-node elimination order. Compilation reads
topology and message addresses only. Degree-two internal interfaces close by
exact phase-native composition; parallel paths merge by exact phase-native
intersection. Every intermediate relation remains in four carrier-relative
complex phases until the final two-port boundary.

The nested two-cycle discriminator compiles 10 public relations into seven
compositions and two intersections on an 80-cell carrier. Exact closure
accepts `{00}`. Bypassing the first parallel merge admits `{00,10}`, while
ordinary coefficient addition admits `{00,11}`. Both wrong forward paths
reverse their actual histories and restore cleanly. Rotating the boundary
inverse or omitting one resident-message inverse leaves error
`1.73205080757`; correct restoration and cross-process reuse are
`1.57009245868e-16`.

The independent scalar executable checks every complete assignment for 120
deterministic arbitrary-coefficient graphs. Native reduction, scalar
reduction, and existential projection agree in all 120 cases. Fifteen scale
graphs reach 46 nodes, 60 relations, 59 native operations, and 480 carrier
cells. The 15-diamond carrier restores within `1.66533453694e-16`.

That capacity probe exposed and repaired amplifying floating phase drift. A
continuous three-well phase lock, `unit(2z + conj(z)^2)`, reduces maximum
root error from `0.0740742833362` to `1.57009245868e-16` without phase-label
decode. Independent review then found the UBSan wrapper was fail-open; the
qualifier now uses nonrecovering undefined-behavior sanitization and fail-fast
runtime options. The complete suite and two independent Sol/xhigh reviews
pass with no remaining findings:

```text
SOL-XHIGH-ALGEBRAIC-SERIES-PARALLEL-01
SOL-XHIGH-SERIES-PARALLEL-MECHANISM-03
```

This establishes bounded public series/parallel relational phase closure,
not arbitrary treewidth, automatic ordering, broader arity/domain,
computational advantage, physical phase computation, or unlimited catalytic
phase computation. The next mechanism is recursive typed relational programs
whose module boundaries remain open phase relations and compose without host
expansion.

## Original CATVM open-intermediate package: atomic claim rejected

The branch-native Boolean/F3 phase engine now runs behind its first enforced
machine boundary. A carrier-owning Linux `SOCK_SEQPACKET` service is
non-dumpable before allocation, locks a private anonymous mapping, rejects
forked mappings and core dumps, unlinks its single-client socket, and installs
a post-accept seccomp allowlist. The IPC-only controller is not linked to the
phase core.

The source sequence is:

```text
seal A, B, C
-> F: exact native composition leaves Y in four resident phase cells
-> G: exact native intersection consumes those actual Y cells
-> project only final Z = [0,2,1,1]
-> actual G^-1
-> actual F^-1
-> exact discrete-state and tolerant complex-state restoration
-> unrelated program on the same restored carrier, Z = [1,2,0,2]
-> 1000 further alternating same-carrier transactions
```

There are zero intermediate decodes and no serialized Y, intermediate hash,
witness list, candidate set, truth table, decoded relation, or retained
inverse factor. While Y is resident, the controller's attempts through
`/proc/<pid>/mem`, `/proc/<pid>/maps`, `/proc/<pid>/fd/0`,
`process_vm_readv`, `ptrace`, and `pidfd_getfd` are all denied; the service
then completes the transaction. `PROJECT Y`, malformed protocol requests,
debug/detail requests, dumps, reads, snapshot commands, embedded NULs, and
oversize packets receive fixed errors.

One carrier allocation completes 1002 restoration generations. Maximum
restoration error is `4.99600361081e-16` against a predeclared `2e-12`
tolerance. Wrong-G, missing-G, and prospectively noncommuting reordered
inverse controls each leave error `1.73205080757`. The final result is emitted
before restoration and therefore survives outside inverse history, but that
ordering also defeats the claimed atomic machine law: the controller can
receive `Z` before the later independent `RESTORE` command.

The separately measured snapshot baseline reloads a saved copy, performs no
actual inverse, locks 8192 bytes rather than the in-place path's 4096 bytes,
and establishes only:

```text
CATVM_SNAPSHOT_BACKED_TRANSACTIONAL_REUSE_ESTABLISHED
```

Warmed average transaction times in the repaired qualification are:

```text
direct process phase          14.853 us
isolated inert boundary       58.019 us
snapshot CATVM                72.610 us
in-place inverse CATVM        76.615 us
```

GCC strict builds, `-fanalyzer`, ASan, UBSan, deterministic replay, native
series-parallel semantic comparison, resource gates, no-smuggle source and
runtime gates, and all controls pass. The focused independent review reports
no claim blocker. Its two accounting/evidence findings were repaired by
counting both snapshot mappings and replacing response constants with measured
carrier-creation, receive-buffer, and socket-queue state.

The former atomic claim:

```text
CATVM_OPEN_INTERMEDIATE_COMPOSITION_ESTABLISHED_ON_PHASE_BACKEND
```

is `REJECTED_SOURCE_DEFECT_ATOMICITY`. Clean-room adversarial verification
preserves only the bounded algebra, hidden-intermediate custody, later actual
inverse restoration, reuse, and same-UID controls. Later CATVM packages that
place restoration before response are separate claims. This package does not
establish root/kernel or microarchitectural secrecy, arbitrary topology,
compact wide-interface relations, general holographic relational
computation, computational advantage, physical waveform or silicon
computation, Small Wall crossing, or unlimited catalytic computation.

## Bounded recursive typed relational modules

The first typed module compiler now treats independently compiled
series-parallel closures as nominally typed open relations. A composite
descriptor is admitted only when the left child's right-domain identity
exactly equals the right child's left-domain identity. A control with two
distinct names backed by the same `BOOLEAN_F3` representation rejects,
showing nominal type enforcement rather than a width check.

Each module export is only:

```text
left domain
right domain
resident final_start phase address
orientation
```

The parent native composition reads the actual child addresses. There is no
child decode, serialization, hash, persistent export copy, witness list,
candidate set, or truth table. Only the root relation is copied to the
four-cell public boundary and decoded. Reverse execution visits parent
operations before child operations, recomputes the actual factors, restores
the carrier, and then runs a different module program on that same carrier.

The shallow three-descriptor split of the reviewed nested graph matches the
flattened series-parallel engine exactly:

```text
primary [0,2,1,2]
reuse   [0,1,1,1]
```

The stronger balanced tree instantiates one cached leaf definition four times:

```text
nominal domains                    5
module descriptors                 7
tree depth                         3
unique leaf sources                1
native operation descriptors       19
resident relation messages         19
carrier cells                      160
live carrier bytes                 5120
primary root                       [0,2,2,0]
reuse root                         [0,1,1,1]
maximum correct restoration        2.00148302124e-16
```

Wrong-boundary and omitted-resident-module inverses each leave error
`1.73205080757`. Nominal mismatch, non-root projection, leaf-intersection
bypass, ordinary-sum intersection, deterministic replay, flattened parity,
strict compilation, static analysis, ASan, UBSan, and leak detection pass.
Focused independent review finds no remaining issue.

This establishes:

```text
BOUNDED_RECURSIVE_TYPED_RELATIONAL_MODULE_COMPOSITION
```

The resource disclosure also identifies the next blocker. Leaf definition
parsing is cached, but each instantiation still expands native operation
descriptors and resident message cells:

```text
compact_definition_reuse              false
expanded_native_operation_descriptors 19
native operation descriptor bytes     912
```

`module_export_copy_cells = 0` excludes persistent cross-module export
artifacts; native operators still use bounded transient operand arrays. This
does not establish compact recursive closure, wide-interface relations,
arbitrary topology or arity, advantage, physical phase execution, Small Wall
crossing, or unlimited catalytic computation. The next mechanism must execute
compiled module bodies without per-instance operation inlining while
preserving phase-resident locals, actual inverse restoration, and reuse; the
following lift is an unresolved interface wider than two Boolean ports.

## Compact compiled-body typed module execution

The per-instance operation-inlining limitation is now removed for the bounded
tested module trees. Eight typed leaf instances share one parsed four-operation
leaf body. Each instance retains only typed module metadata and phase-message
addresses. Forward and inverse execution reconstruct one transient native
operation descriptor at a time and relocates the shared body's addresses into
the instance carrier region.

The depth-four balanced tree reports:

```text
leaf instances                                      8
composite modules                                   7
unique leaf sources                                 1
persistent per-instance native operation records    0
persistent shared leaf operation records             4
persistent composite module descriptors              7
simultaneous transient operation records              1
executed native operations                          39
resident operation messages                         39
carrier cells                                      320
primary root                                  [0,2,2,0]
reuse root                                    [0,1,1,1]
maximum correct restoration           2.00148302124e-16
```

The expanded typed backend retains 39 native-operation records for the same
tree and produces the same boundaries. The compact executor retains the
actual child `final_start` addresses, inverses parent operations before
children, and runs the unrelated reuse program on the same restored carrier
allocation. Wrong-boundary and omitted-parent inverses each leave error
`1.73205080757`.

Independent review found and caused repair of two evidence defects. The
resource ledger now counts the four-value retained root boundary factor,
four-value wrong-inverse rotation, 52-value maximum operator workspace,
carrier plus comparison snapshot, both concurrently loaded typed sources,
both leaf definitions, and both layouts. The no-smuggle qualifier now enforces
an exact runtime output-key allowlist and rejects output sinks before the
declared root projection. The shallow case also proves that definition-reuse
reporting is dynamic rather than hardcoded.

This establishes:

```text
COMPACT_COMPILED_BODY_TYPED_RELATIONAL_MODULE_EXECUTION
```

only for shared compiled leaf bodies in bounded software module trees.
Composite descriptors and all 39 phase-history messages remain
instance-specific, so this is not compact carrier/history or constant-space
recursion. It does not establish interfaces wider than two Boolean ports,
arbitrary topology or arity, advantage, physical phase execution, Small Wall
crossing, or unlimited catalytic computation.

The next experiment is a native width-two separator: 16-cell four-port
Boolean/F3 relations composed by one reusable `CONTRACT2` body, with the
16-cell intermediate retained unresolved and consumed directly before actual
inverse restoration and cross-program reuse. Its exponential
separator-width resource law must remain explicit.

## Native width-two relational phase contraction

The branch now closes two shared Boolean ports in one native phase operator.
A four-port relation is represented by 16 multiaffine F3 coefficients in 16
relative phase cells. For resident `F(X,Y)` and `G(Y,Z)`, the reusable body
computes:

```text
J = F^2 + G^2
N_t(P) = P|t=0 * P|t=1
CONTRACT2(F,G) = N_Y1(N_Y0(J))
```

Boolean-quotient multiplication is OR-index phase convolution. F3
intersection and the two Boolean norm products are exact without decoding a
coefficient or iterating shared assignments.

The demonstrated chain is:

```text
F          cells  0..15
G          cells 16..31
K          cells 32..47
resident H cells 48..63 = CONTRACT2(F,G)
resident Z cells 64..79 = CONTRACT2(actual H,K)
public     cells 80..95
```

Only the public boundary is decoded. The boundary result survives while
actual inverse execution removes the parent contraction before the child,
restores the carrier within `1.66533453694e-16`, and runs a different source
on the same restored allocation.

Primary and reuse boundaries are:

```text
[1,0,2,1,2,1,2,1,0,0,1,1,1,1,1,1]
[1,0,2,1,0,0,1,1,2,1,2,1,1,1,1,1]
```

Wrong boundary inversion, omitted parent inversion, and prospectively
noncommuting child-before-parent reversal each leave error
`1.73205080757`. A bypassed Boolean norm, ordinary coefficient sum in place
of its product, and swapped shared-port order each change the final relation
and reverse their actual altered histories cleanly. Projection of the
resident intermediate and null-carrier execution reject before state output.

Strict compilation, `-fanalyzer`, ASan, UBSan, leaks, exact output allowlists,
preprojection sink gates, parser negatives, replay hashes, and focused
independent review pass. The final vectors are predeclared fixture assertions
backed by algebra and implementation inspection; there is intentionally no
second controller-side coefficient oracle that materializes the resident
intermediate.

This establishes:

```text
NATIVE_WIDTH2_TYPED_RELATIONAL_PHASE_CONTRACTION
```

with claim ceiling:

```text
SOFTWARE_COMPACT_WIDTH2_TYPED_RELATIONAL_PHASE_CONTRACTION_AND_RESTORATION_REFERENCE_ONLY
```

The compactness is the single reusable body rather than four expanded shared
assignments. Dense storage remains exponential:

```text
relation/message cells       2^(2w)
transient union coefficients 2^(3w)
```

It does not establish compact separator storage, arbitrary width, arity,
topology or treewidth, CATVM enforcement for the wider operator, advantage,
physical execution, Small Wall crossing, or unlimited catalytic computation.
The next blocker is a factorized wider-interface relation representation whose
native closure preserves structure without dense separator expansion.

## Bounded width-three rank-two lazy tensor-train composition

The first factorized wider-interface experiment now represents each
six-variable Boolean/F3 relation in genuinely coupled rank-two tensor-train
cores:

```text
rank shape                      1 -> 2 -> 2 -> 1
phase cells per relation        32
dense coefficients avoided      64
nonzero public rank minors       F=1, G=1
```

The accepted descriptor retains two custody references, eight public
selectors, and six topology-only operator nodes:

```text
SQUARE(F)
SQUARE(G)
ADD(LIFT_XY(SQUARE(F)), LIFT_YZ(SQUARE(G)))
NORM_Y0
NORM_Y1
NORM_Y2
```

It contracts each selected coefficient directly from the actual resident
cores through branch-native F3 symbol products and Boolean OR convolution.
The descriptor contains no coefficient values. A sparse phase cache is
cleared between selectors, and no dense 64-coefficient `H`, assignment
expansion, truth table, decoded intermediate, or serialized intermediate is
created.

The resident layout is:

```text
F rank-two cores       cells  0..31
G rank-two cores       cells 32..63
sparse H messages      cells 64..71
public boundary        cells 72..79
```

Only the eight public cells are decoded. The actual resident messages drive
the boundary copy. Their inverse factors are recomputed while the actual F/G
cores remain resident, then G and F are inversed. An unrelated fixture and 32
more alternating programs execute on the same restored allocation:

```text
primary boundary                    [0,0,1,2,2,2,1,2]
reuse boundary                      [1,1,2,2,1,0,2,0]
single-transaction restoration max  1.57009245868e-16
34-transaction reuse max            2.48253415325e-16
```

Wrong, missing, and noncommuting reordered inverses each leave error
`1.73205080757`. Cutting both TT bonds to channel zero changes the boundary to
`[1,0,0,0,0,0,0,0]`; mismatching one G bond changes it to
`[0,0,2,1,2,1,0,2]`. Both altered algebra histories reverse cleanly.
Intermediate projection, null carrier, malformed input, and eager output-TT
materialization reject. Snapshot reload is separately labeled and does not
support the in-place claim.

The width-one degeneration matches the reviewed native `OP_COMPOSE` to
`1.11022302463e-16`. Independent review also reimplemented the discrete F3
sparse recurrence and reproduced both boundaries.

The review found one resource-label defect: the original `38,688`-byte
quantity counted selected buffers but not nested call frames. The repaired
evidence labels it a material-buffer subtotal, gates a `37,088`-byte GCC
`-O2 -fstack-usage` active call chain, and reports a deliberately conservative
combined ceiling of `75,776` bytes. No space or performance advantage is
claimed.

Exact eager structural ranks disclose the unresolved closure barrier:

```text
input -> square -> intersection -> N_Y0 -> N_Y1 -> N_Y2
   2       4            8            64    4096   16777216
```

This establishes:

```text
BOUNDED_WIDTH3_RANK2_PHASE_RESIDENT_LAZY_TT_RELATION_COMPOSITION_WITH_SPARSE_BOUNDARY_INVARIANT
```

with ceiling:

```text
BOUNDED_SPARSE_BOUNDARY_LAZY_TT_RELATION_COMPOSITION_REFERENCE_ONLY
```

It establishes compact coupled input relations and exact sparse lazy
contraction, not a compact full output relation, bounded-rank closure,
polynomial scaling, arbitrary tensor-network topology, computational
advantage, CATVM enforcement for the wider operators, physical execution,
Small Wall crossing, or unlimited catalytic computation.

The selected next experiment is the minimal width-two CATVM enclosure: place
the already qualified two-`CONTRACT2` chain and its actual 16-cell unresolved
intermediate behind the proven Linux custody boundary. Factor-preserving
compact full-relation closure remains the next representation blocker after
that enforced transaction.

## Enforced CATVM width-two open-intermediate contraction

The dense width-two relation engine now runs behind the protected CATVM
controller/service boundary. The controller sends 48 public F3 coefficients
and fixed typed commands, but it neither links the phase kernel nor contains
an expected boundary. Inside the carrier-owning service:

```text
H = CONTRACT2(F,G)          resident cells 48..63
Z = CONTRACT2(actual H,K)   resident cells 64..79
PROJECT Z                   final 16 coefficients only
```

The actual 16-cell `H` is never decoded or serialized. The service's private,
locked, non-dumpable machine mapping owns the 96-cell carrier, canonical
machine metadata, and all 240 complex `CONTRACT2` workspace values. Every
contraction requires an exactly zero workspace on entry and securely clears
it on return.

The accepted inverse removes the public boundary, recomputes the parent factor
from actual `H` and `K`, removes the parent, removes `K`, recomputes the child
factor from actual `F` and `G`, then removes the child, `G`, and `F`.
Canonical equality includes carrier cells, workspace, topology and morphism
digests, lease, carrier-creation count, morphism stack, open-port and pending
state, restoration generation, receive buffer, and backend queue.

The accepted run reports:

```text
primary boundary             [1,0,2,1,2,1,2,1,0,0,1,1,1,1,1,1]
reuse boundary               [1,0,2,1,0,0,1,1,2,1,2,1,1,1,1,1]
carrier creations            1
accepted generations         258
maximum restoration error    8.61764809305e-16
predeclared tolerance        2e-12
native CONTRACT2 calls       1,032
locked in-place mapping      8,192 bytes
```

Wrong, missing, and applicable child-before-parent inverses each leave
`1.73205080757` carrier error. Projection of the intermediate, null carrier,
strict-protocol attacks, and same-UID `/proc`, `process_vm_readv`, `ptrace`,
and `pidfd_getfd` inspection all fail. Service stdout and stderr remain empty,
the sole decoder projects only `Z`, and the controller binary excludes the
phase symbols. Strict compilation, static analysis, ASan, UBSan, leak checks,
deterministic replay, direct branch-native parity, exact output allowlists,
and the original four-cell CATVM regression pass.

Snapshot reload remains a separate weaker baseline:

```text
CATVM_WIDTH2_SNAPSHOT_BACKED_TRANSACTIONAL_REUSE_ESTABLISHED
```

It maps another 4,096 locked bytes, writes and reloads 1,536 carrier bytes per
transaction, performs no actual inverse, and is not credited toward the
stronger claim. Warm direct-process, inert-boundary, snapshot, and in-place
paths are all measured; no performance advantage is claimed.

Focused independent review found no substantive scientific, security, or code
defect and bound its verdict to the final source/result hashes plus a passing
regression of the original CATVM proof.

This establishes:

```text
CATVM_WIDTH2_OPEN_INTERMEDIATE_CONTRACT2_COMPOSITION_ESTABLISHED_ON_PHASE_BACKEND
```

with ceiling:

```text
SOFTWARE_ENFORCED_FIXED_WIDTH2_CATVM_CONTRACT2_CHAIN_REFERENCE_ONLY
```

It does not establish compact separator storage, factor-preserving full
relation closure, width-three CATVM custody, arbitrary width or topology,
arbitrary CATVM programs, root/kernel or microarchitectural secrecy,
computational advantage, physical execution, Small Wall crossing, or
unlimited catalytic computation.

The selected frontier returns to the representation barrier:
`FACTOR_PRESERVING_COMPACT_FULL_RELATION_CLOSURE`. The next experiment must
seek a relation family or exact representation whose complete surviving
boundary remains factorized after native closure; another dense-width record
or sparse-only query does not resolve that blocker.

## Projected-affine compact full relation closure

The first exact factor-preserving full-boundary family is now resident on the
phase carrier. A typed port has `q` quotient bits and `f` free fibre bits, and
the relation:

```text
R(A,a) = {(x,y) : pi(y) = A pi(x) XOR a}
```

is retained as `q*q` binary matrix coefficients plus `q` offset coefficients.
For invertible `A`, every source has `2^f` targets and every target has `2^f`
sources. The free fibres therefore make the complete relation genuinely
many-to-many in both directions without materializing a fibre value or
witness.

Composition closes exactly:

```text
R(B,b) o R(A,a) = R(B*A, B*a XOR b)
```

Each Boolean coefficient remains an F3 phase symbol. Native
`symbol_product` supplies coefficient AND, and:

```text
lock(x * y * symbol_product(x,y))
```

supplies Boolean XOR without decoding. The implemented chain writes the first
complete descriptor to actual resident `H`, consumes that same `H` in the
second composition, projects every final descriptor coefficient, then
recomputes and reverses parent before child. No sparse selector replaces the
full boundary.

The accepted `q=8, f=4` transaction reports:

```text
port bits                              12
degree in each direction               16
factor coefficients                    72
dense pair/coefficient exponent        24
carrier cells                          432
primary full-boundary hash  33f962d84bbdd8db
reuse full-boundary hash    1b6d1f9d91bbe788
same-carrier transactions              130
maximum restoration error   2.48253415325e-16
```

Both 72-coefficient vectors match a separately compiled compact GF2
reference. Wrong, missing, and applicable reordered inverses fail;
ordinary-F3-sum and reversed-composition controls change the complete
boundary while reversing their own altered histories. Projection of `H` and
null-carrier execution reject. Snapshot reload remains a separate weaker
path.

The minimum nonabelian `q=2, f=1` family contains 24 compact affine relations.
The phase engine passes:

```text
ordered pair closures          576 / 576
associativity triples       13,824 / 13,824
retained relation lookups         0
tuple/assignment/witness          0 / 0 / 0
```

The two parenthesizations use different resident intermediates and agree on
the complete boundary. This is not an enumeration of fibre assignments or
relation tuples.

Scaling builds at quotient ranks `2,4,8,16,32,64` match the independent
reference. At `q=64, f=20`:

```text
port bits                              84
factor coefficients                 4,160
carrier cells                      24,960
live carrier bytes                798,720
dense pair/coefficient exponent       168
symbol products per transaction  2,129,920
restoration error        2.22044604925e-16
```

The exact resource law is quadratic descriptor/carrier storage and cubic
native composition. No advantage over ordinary compact binary matrix
arithmetic is claimed. Accepted `q=8` accounting records 27,648 carrier-plus-
comparison heap bytes, a 5,776-byte compiler-measured nested stack chain, and
a conservative current-ABI total of 33,424 bytes.

Focused review independently reproduced both `q=8` boundaries, verified
noncommutation and bi-many fibre semantics, and closed two evidence findings:
the qualifier now hashes itself, and the documentation limits heap/stack
accounting to the accepted `q=8` ABI while scaling claims remain tied to
explicitly measured laws.

This establishes:

```text
BOUNDED_PROJECTED_AFFINE_PHASE_RELATION_COMPOSITION_WITH_COMPACT_FULL_BOUNDARY
```

with ceiling:

```text
BOUNDED_PROJECTED_AFFINE_F2_RELATION_SUBCATEGORY_REFERENCE_ONLY
```

Compact closure holds because the family ignores free kernel coordinates and
acts invertibly on the quotient. It does not establish kernel-sensitive or
general affine-system projection, nonlinear compact closure, arbitrary
relation topology, advantage, CATVM custody for this new family, physical
execution, Small Wall crossing, or unlimited catalytic computation.

The selected next experiment is
`OBLIVIOUS_PHASE_NATIVE_GENERAL_AFFINE_RELATION_ELIMINATION`: represent general
open GF2 affine equation systems in phase cells and eliminate a shared
interface with a fixed coefficient-oblivious schedule, phase-resident pivot
controls, no host-selected pivots, and a complete canonical affine boundary.

## Coefficient-oblivious general affine relation closure

The fixed-width successor removes the projected-affine quotient restriction.
Any affine subset of two two-bit ports is stored as at most four equation
rows plus an empty flag. Composition embeds two such relations in a resident
eight-row augmented phase system and eliminates the two shared bits before
canonicalizing the complete external relation.

The schedule is coefficient-oblivious. Six public column stages always
execute the same selection, swap, and row-add topology. Pivot selection,
row-active flags, elimination controls, input emptiness, and inconsistency
remain Boolean-subset F3 phase symbols. Rank, pivot positions, coefficients,
and contradictions never select a host pivot, loop bound, or memory address.

The primary child composition is:

```text
x0 XOR y0 XOR y1 = 0
y0 XOR y1 XOR z0 = 1
--------------------------------
x0 XOR z0 = 1
```

Each admissible `(X,Z)` pair retains two unresolved `Y` witnesses; no witness
is selected or materialized. The actual 25-cell `H` is consumed by a second
native composition, which yields the complete final relation:

```text
x0 XOR w0 XOR w1 = 1
```

The only decoder latches the final 25 coefficients. The boundary survives
while the parent and child are recomputed and reversed from the actual
resident operands. The same restored carrier then runs unrelated,
rank-deficient, universal, empty, and full-rank programs.

The accepted semantic suite reports:

```text
phase/reference exact boundary matches       9 / 9
primary boundary hash           a96f5625e4054ee6
unrelated reuse hash             0183d6882c23ee46
full-rank four-row hash          2a656a080ba3a76b
same-carrier actual-inverse transactions                  73
maximum restoration error                    2.00148302124e-16
predeclared tolerance                                      2e-12
```

Row permutation and duplicate equations preserve the exact canonical
boundary. Rank-zero and rank-one shared systems both project to canonical
universal. Explicit input empty propagates, while two simultaneous
contradiction rows correctly produce canonical empty; the latter
discriminates the required OR accumulation from XOR. A conventional
rank-pointer GF2 eliminator, compiled as a separate binary and structurally
different from the phase pivot schedule, reproduces all nine boundaries.

Wrong parent inverse, missing parent inverse, and applicable child-before-
parent inversion each leave `1.73205080757` error. Intermediate projection,
control projection, null carrier, and unknown command paths reject. Snapshot
reload remains separately labeled and is not credited as actual inverse.
Strict GCC, `-fanalyzer`, ASan, UBSan, leak checks, deterministic replay,
exact JSON allowlists, source/result hashes, syscall write tracing, and the
sole-final-decode static gate pass.

Focused review closed four evidence defects: it required a genuine rank-four
boundary, narrowed the obliviousness wording to the demonstrated fixed host
pivot/loop/address schedule, made expanded syscall tracing fail closed, and
corrected snapshot inverse-recomputation accounting to zero. The repaired
source and result hashes passed closure review with no remaining finding.

The accepted carrier has:

```text
relation blocks                         6 * 25 cells
joint augmented matrix                         56 cells
resident phase controls                       112 cells
total reusable workspace                      168 cells
total carrier                                 318 cells
live carrier bytes                         10,176
carrier plus comparison heap bytes         20,352
compiler-measured nested stack chain        6,000
conservative current-ABI accounted bytes   26,352
phase AND/XOR/OR/NOT per transaction
                              6,372 / 10,416 / 232 / 440
```

At width two the 25-cell canonical descriptor is larger than the 16-entry
dense membership table, so no concrete storage or performance advantage is
claimed. Its importance is semantic generality: singular, kernel-sensitive,
rank-deficient, empty, and universal affine relations now share one fixed
phase-native closure law.

This establishes:

```text
BOUNDED_WIDTH2_COEFFICIENT_OBLIVIOUS_GENERAL_AFFINE_PHASE_RELATION_COMPOSITION
```

with ceiling:

```text
FIXED_WIDTH2_SOFTWARE_GF2_AFFINE_RELATION_SYSTEM_REFERENCE_ONLY
```

It does not establish wider interfaces, arbitrary equation capacity,
nonlinear Boolean elimination, arbitrary topology or treewidth, automatic
elimination-order discovery, CATVM custody for this kernel, computational
advantage, physical execution, Small Wall crossing, or unlimited catalytic
computation.

The selected next experiment is
`WIDTH_PARAMETRIC_OBLIVIOUS_AFFINE_PHASE_ELIMINATION`: retain the same machine
law while increasing interface width and public equation capacity, emit a
complete polynomial affine boundary, and measure the actual phase/resource
law against conventional compact Gaussian elimination.

## Width-parametric general affine relation closure

The width-two calibration now compiles unchanged across:

```text
w = 2, 3, 4, 8, 12, 16
```

At width `w`, a complete affine relation between two `w`-bit ports is stored
as `2w` equation slots plus an explicit empty flag:

```text
B(w) = 4w^2 + 4w + 1 phase cells
```

Composition embeds `F(X,Y)` and `G(Y,Z)` into `4w` resident augmented rows
over public column order `Y,X,Z`. All candidate selection, conditional swaps,
row activity, pivot state, elimination, emptiness, and contradiction logic
remain Boolean-subset F3 phase symbols. The same public `3w`-stage host
schedule executes for every coefficient pattern; rank and pivot positions do
not select host pivots, loop bounds, addresses, or native-operation counts.

The exact resident resource laws are:

```text
relation cells       4w^2 + 4w + 1
workspace cells     36w^2 + 11w + 2
carrier cells       60w^2 + 35w + 8
phase ANDs         576w^3 + 450w^2 - 18w
phase XORs       1,152w^3 + 384w^2 - 168w
```

Nine complete semantic boundaries match a separately compiled conventional
rank-pointer GF2 reference at every accepted width. The suite includes
rank-deficient many-to-many composition, unrelated restored-carrier reuse,
row permutation, duplicate equations, rank-zero and rank-one universal
projection, explicit input empty, multiple simultaneous contradictions, and
maximum-rank `2w` output. It materializes no assignment, relation tuple,
candidate, or witness expansion.

At width sixteen:

```text
equation capacity                           32
complete relation cells                  1,089
dense membership entries         4,294,967,296
workspace cells                          9,394
carrier cells                           15,928
live carrier bytes                     509,696
phase ANDs                            2,474,208
phase XORs                            4,814,208
phase-cell updates                    3,692,636
native kernel reads                   4,929,526
logical carrier cell inspections      5,047,017
maximum repeated restoration error  2.00148302124e-16
predeclared tolerance                        2e-12
```

The width-sixteen compiler-measured nested call chain is 105,840 bytes.
Carrier plus comparison heap is 1,019,392 bytes, giving a conservative
current-ABI accounted total of 1,125,232 bytes. The complete affine
descriptor becomes smaller than dense membership at width three; this is a
polynomial closure law for affine relations, not a compact representation of
arbitrary Boolean relations or a speed advantage over ordinary GF2
elimination.

The actual resident child relation is consumed by its parent. Only the final
complete boundary is decoded; parent and child are then recomputed and
reversed from the actual resident operands, restoration is checked exactly
within tolerance, and the actual restored carrier runs unrelated programs.
Wrong, missing, and applicable reordered inverses fail. Intermediate and
control projection reject. No pre-inverse whole-carrier scalar is calculated
or emitted. Snapshot reload remains separately labeled and performs zero
inverse-factor recomputations. Mandatory file, network, write, vector-write,
positional-write, and send tracing finds no extra output channel.

Focused review independently swept every integer width from two through
sixteen. All nine phase boundaries matched the conventional reference at all
fifteen widths. The review closed four findings: it removed a whole-carrier
displacement scalar that leaked the hidden intermediate's Hamming weight,
made carrier inspection and snapshot-copy accounting complete, counted all
nine simultaneously resident semantic programs, and included the
`ga_apply_compose` frame in the measured stack chain. Closure review found no
remaining finding within the bounded software claim.

This establishes:

```text
BOUNDED_WIDTH_PARAMETRIC_COEFFICIENT_OBLIVIOUS_GENERAL_AFFINE_PHASE_RELATION_COMPOSITION
```

with ceiling:

```text
BOUNDED_WIDTH16_SOFTWARE_GF2_AFFINE_RELATION_SYSTEM_REFERENCE_ONLY
```

It does not establish runtime-unbounded width, arbitrary equation capacity,
nonlinear Boolean closure, arbitrary graph topology or treewidth, automatic
elimination-order discovery, CATVM custody for this wider kernel,
computational advantage, physical execution, Small Wall crossing, or
unlimited catalytic computation.

The selected next experiment is
`WIDTH_PARAMETRIC_AFFINE_COMPOSITION_AND_INTERSECTION_MODULE_CLOSURE`: reuse
the polynomial canonical boundary and fixed phase reduction kernel for both
wide-interface composition and native relation intersection, then execute a
recursive public series/parallel affine module with actual resident
child-to-parent messages, boundary-only projection, actual inverse
restoration, and restored-carrier reuse.

## Width-parametric mixed affine module closure

The accepted affine backend now executes one typed public mixed topology:

```text
R    = COMPOSE(F : X->U, G : U->B)
I    = INTERSECT(actual R : X->B, P : X->B)
ROOT = COMPOSE(actual I : X->B, K : B->C)
```

Nominal signature validation is bound to the actual carrier-block addresses.
A width-compatible descriptor redirected to a block with the wrong nominal
signature rejects before carrier execution.

Native intersection concatenates both input equation systems in the existing
`4w` resident row workspace. The public first `w` shared-interface columns
are zero and the later `2w` columns represent the common external boundary.
The unchanged fixed pivot/elimination schedule canonicalizes the
intersection without decoding coefficients, rank, pivots, emptiness, or
contradictions. It requires no new workspace/control cells and has the same
native schedule as one composition.

At width three, the primary module produces the complete four-row relation:

```text
x0 = 1
c0 = 1
c1 = x2
c2 = x1 XOR x2 XOR 1
```

with hash `594d12aa095dab79`. The same actual restored carrier then runs an
unrelated program whose final relation is:

```text
x2 = 0
c0 = 1
c1 = x0
c2 = x1 XOR 1
```

with hash `4057ecbc8bf19d29`.

Six complete phase boundaries match a separately compiled coefficient-aware
GF2 reference at widths `3,4,8,12,16`. Empty propagation, universal
intersection identity, cross-branch contradiction, and duplicate
intersection idempotence pass. The idempotence fixture also places a
right-operand-only equation at `X[w-1],B[w-1]`, so scaled executions cover
the highest external coefficient columns on the intersection's right input.

Wrong root inverse, missing root inverse, and the applicable order `I^-1`
before `ROOT^-1` each leave `1.73205080757` error. Bypassing intersection
with either input or substituting composition changes the complete boundary,
while each altered path reverses its own history. Snapshot reload remains a
separate weaker path with zero inverse-factor recomputations.

The accepted transaction uses:

```text
B(w)        = 4w^2 + 4w + 1 relation cells
W(w)        = 36w^2 + 11w + 2 workspace cells
Carrier(w)  = 68w^2 + 43w + 10 phase cells
```

At width sixteen:

```text
complete relation cells                  1,089
dense membership entries         4,294,967,296
workspace cells                          9,394
carrier cells                           18,106
live carrier bytes                     579,392
phase ANDs                            3,711,312
phase XORs                            7,221,312
phase-cell updates                    5,536,776
native kernel reads                   7,393,200
logical carrier cell inspections      7,552,623
compiler-measured nested stack chain     98,320
conservative current-ABI bytes        1,257,104
```

Only `ROOT` is copied to the surviving boundary and decoded. Resident `R`
and `I` are never decoded, serialized, hashed, committed, aggregated, or
exported. Strict compilation, analyzer, sanitizers, deterministic replay,
exact output allowlists, source/result hashes, reference separation, and
expanded syscall no-smuggle tracing pass.

Focused review closed three evidence defects: the stack chain now includes
the separately emitted reduction frame, scaled fixtures cover the right
intersection operand's highest coefficient columns, and matching high-index
equations through `R` and `P` restore genuine duplicate idempotence. Closure
review reproduced all 30 phase/reference boundaries and found no remaining
finding.

This establishes:

```text
BOUNDED_WIDTH_PARAMETRIC_AFFINE_COMPOSITION_AND_INTERSECTION_MODULE_CLOSURE
```

with ceiling:

```text
BOUNDED_WIDTH16_SOFTWARE_GF2_AFFINE_MIXED_MODULE_REFERENCE_ONLY
```

It does not establish an arbitrary mixed-module compiler, unbounded module
depth or topology, arbitrary graphs or treewidth, nonlinear Boolean closure,
CATVM custody for this module, computational advantage, physical execution,
Small Wall crossing, or unlimited catalytic computation.

The selected next experiment is
`RECURSIVE_WIDTH_PARAMETRIC_AFFINE_MODULE_COMPILER`: compile arbitrary
bounded public series/intersection trees to resident canonical message
blocks, validate nominal signatures against addresses, schedule dependency-
ordered forward execution and exact reverse-dependency restoration, and
measure carrier growth against live separator count.

## Recursive width-parametric affine tree compiler

The accepted compiler removes the hardcoded mixed-module topology for a
bounded public tree language:

```text
root ID
leaf ID DOMAIN CODOMAIN LEAF_BODY_ID
compose ID LEFT RIGHT
intersect ID LEFT RIGHT
```

The qualified scrambled manifest contains fifteen nodes, eight uniquely owned
leaf bodies, five composition nodes, two intersection nodes, fourteen edges,
and depth three. The compiler validates unique identities, child resolution,
acyclicity, exactly one parent per non-root node, reachability, mixed
operators, and bottom-up nominal signatures before assigning any carrier
address.

Runtime custody is carried by live relation leases bound to slot, owner,
generation serial, nominal signature, and reserved address. An output cannot
alias either operand. The forward evaluator computes the actual two children,
produces the parent, then applies the actual inverse operations needed to
uncompute both children. Restoration reconstructs the actual child
dependencies, inverses the actual parent, and recursively clears those
reconstructions. All 85 allocated relation blocks are released only after
their complete `B(w)` cells restore, and the accepted transaction ends with
zero outstanding leases.

The reversible pebble schedule needs seven working relation slots plus one
dedicated final boundary slot. A retain-all comparison needs fifteen working
slots plus the same boundary. Their carrier laws are:

```text
B(w)                 = 4w^2 + 4w + 1
W(w)                 = 36w^2 + 11w + 2
pebbled carrier       = W(w) + 8B(w)  = 68w^2 + 43w + 10
retain-all carrier    = W(w) + 16B(w) = 100w^2 + 75w + 18
```

At width sixteen, the accepted path uses 18,106 phase cells and 579,392 live
carrier bytes versus 26,818 cells and 858,176 bytes for retain-all. It
executes 21 forward and 21 inverse native operations, including dependency
recomputation, with 25,979,184 phase ANDs, 50,549,184 phase XORs, and
52,667,985 logical carrier-cell inspections. The conservative current-ABI
accounting, including carrier verification copy, topology, programs,
boundary, stack, and manifest, is 1,287,096 bytes.

Five default semantic boundaries and scaled primary/reuse boundaries at
widths `3,4,8,12,16` match a separately compiled coefficient-aware compact
GF2 reference exactly. The width-three primary boundary is:

```text
x0 = 1
x1 + e1 = 0
x2 + e1 = 0
e0 + e1 = 1
e2 = 0
```

Only `ROOT` is copied and decoded. No intermediate coefficient, rank, pivot,
hash, commitment, witness, tuple, or assignment expansion is exported. The
expanded no-smuggle gate traces all file and network syscalls plus ordinary
and positioned writes; it permits only loader/manifest reads and stdout
writes.

Wrong and missing root inverses each leave `1.73205080757` error. A
producer-before-consumer inverse reorder rejects through the stale-or-missing
lease law. Snapshot reload is a separately labelled weaker path. Seventeen
same-carrier transactions, including an unrelated identity reuse program,
restore below `8.7e-16`. Strict compilation, analyzer, sanitizers,
deterministic replay, malformed-manifest controls, and retain-all comparison
pass.

Focused review found one evidence gap in the initial no-smuggle trace. The
expanded file/network trace closes it, and closure review found no remaining
finding.

This establishes:

```text
BOUNDED_PUBLIC_RECURSIVE_AFFINE_SERIES_INTERSECTION_TREE_COMPILATION
```

with ceiling:

```text
BOUNDED_WIDTH16_SOFTWARE_GF2_AFFINE_TREE_COMPILER_REFERENCE_ONLY
```

It does not establish arbitrary DAG fanout, arbitrary graph topology or
treewidth, runtime-unbounded width or topology, nonlinear Boolean closure,
CATVM custody for this compiler, computational advantage, physical
execution, Small Wall crossing, or unlimited catalytic computation.

The selected next experiment is
`BOUNDED_AFFINE_DAG_FANOUT_WITH_SHARED_RESIDENT_MESSAGES`: extend the public
compiler with one shared affine producer consumed by multiple typed parents
while keeping one actual resident message, reversible shared custody,
boundary-only projection, actual inverse restoration, and carrier accounting
against duplicate-tree expansion.

## Bounded affine DAG shared-message fanout

The tree-only occurrence restriction is now removed for one native-produced
degree-two fanout:

```text
S    = COMPOSE(F : A->B, G : B->C)
L    = COMPOSE(actual S : A->C, H : C->D)
R    = COMPOSE(actual same S : A->C, K : C->D)
ROOT = INTERSECT(actual L : A->D, actual R : A->D)
```

The public manifest contains eight nodes, eight edges, four leaves, three
composition nodes, one intersection node, and depth three. It is declared in
scrambled order. The compiler resolves the DAG once, derives all nominal
signatures before assigning addresses, and emits every unique node exactly
once.

Each public operand edge transitions:

```text
DECLARED -> FORWARD_CONSUMED -> INVERSE_CONSUMED
```

The actual shared relation `S` has one observed owner allocation, one forward
materialization, one peak live instance, and one actual inverse/release. Both
forward consumers and both inverse consumers bind the same observed slot,
owner, serial, signature, and carrier allocation. The live lease is validated
again at final projection and after `ROOT^-1`. `S^-1` rejects while either
consumer edge remains live.

At width three the primary final relation is:

```text
x0 + z2 = 0
x1 + z0 = 0
x2      = 0
z1      = 0
```

Its complete canonical boundary hash is `630132cdcd942021`. Left-neutral and
right-neutral variants change the complete boundary; an opposed branch makes
the intersection empty.

The matched occurrence-expanded tree instantiates `F`, `G`, and
`S=COMPOSE(F,G)` twice using the same immutable compiled body table. It
reaches the exact same final boundary and restores by actual inverse, but it
has eleven nodes, ten native calls, twelve leaf toggles, and two distinct
producer owners/serials. It therefore fails the sharing predicate. The
accepted DAG has eight nodes, eight native calls, eight leaf toggles, and one
shared resident block.

Carrier laws are:

```text
B(w)                  = 4w^2 + 4w + 1
W(w)                  = 36w^2 + 11w + 2
accepted DAG carrier  = W(w) + 9B(w)  = 72w^2 + 47w + 11
duplicate tree        = W(w) + 12B(w) = 84w^2 + 59w + 14
```

At width sixteen, the accepted path uses 19,195 carrier cells and 614,240
live carrier bytes. Eight forward/inverse native calls execute 4,948,416
phase ANDs, 9,628,416 phase XORs, and 10,064,805 logical carrier-cell
inspections. Conservative current-ABI accounting, including the carrier
verification copy, topology, program bodies, boundary, measured stack, and
both manifests, is 1,375,336 bytes.

Twenty-five complete phase boundaries match the separately compiled compact
GF2 reference at widths `3,4,8,12,16`. Seventeen transactions on one carrier
allocation, alternating with an unrelated identity program, restore below
`2.3e-16`. The accepted state resets all live leases, edge tokens, pending
counts, scheduler state, and the shared custody receipt; serial advances by
exactly eight allocations and restoration generation by exactly one.

Wrong and missing root inverses each leave `1.73205080757` error. Premature
producer inverse, stale serial, skipped consumer, duplicate consumer,
projection, null-carrier, and copy controls reject. The copy control
constructs an equal-content clone in a new typed lease and proves that
coefficient equality cannot substitute for exact custody identity. All
successor relation copies pass through a counted wrapper: the accepted path
records two boundary-copy calls and zero intermediate-copy calls.

The expanded no-smuggle trace covers file, network, IPC, memory-mapping,
ordinary and positioned writes, memory-file creation, splice/sendfile/copy
routes, cross-process writes, ptrace, and ioctl. It observes only read-only
loader/manifest access, private mappings, and stdout final-boundary JSON.

Focused review found and closed two evidence defects: sharing fields are now
runtime observations rather than topology assertions, the copy control now
attempts a real clone substitution, and resource reporting separates active
edge descriptors from mutable custody state without double counting. Closure
review found no remaining finding.

This establishes:

```text
BOUNDED_NATIVE_PRODUCED_AFFINE_DAG_SHARED_RESIDENT_MESSAGE_FANOUT
```

with ceiling:

```text
BOUNDED_WIDTH16_SOFTWARE_GF2_SINGLE_FANOUT_AFFINE_DAG_REFERENCE_ONLY
```

It does not establish multiple shared nodes, arbitrary DAGs or graphs,
runtime-unbounded topology or width, automatic live-range optimization,
nonlinear Boolean closure, CATVM custody for this compiler, computational
advantage, physical execution, Small Wall crossing, or unlimited catalytic
computation.

The selected next experiment is
`MULTI_FANOUT_AFFINE_DAG_CUSTODY_AND_LIVE_RANGE_COMPILATION`: compile a
bounded public DAG with multiple native-produced shared nodes and overlapping
consumer lifetimes, retain per-node observed custody, release only at proven
inverse readiness, and compare carrier/native work against its
occurrence-expanded tree.

## Nested affine DAG multi-message custody

The scalar shared-owner receipt has been replaced with node-indexed owner
receipts and edge-indexed generation observations. The successor proves the
smallest nested two-fanout case:

```text
S    = COMPOSE(F : A->B, G : B->A)
T    = COMPOSE(actual S : A->A, H : A->A)
I    = INTERSECT(actual S : A->A, actual T : A->A)
ROOT = COMPOSE(actual I : A->A, actual same T : A->A)
```

`S` and `T` are same-typed native-produced messages. Each is allocated and
materialized once, observed by exactly two forward and two inverse consumer
edges through one slot/serial generation, inversed once, and released once.
The two distinct owners are observed simultaneously live through projection
and `ROOT^-1`. The runtime lifecycle is:

```text
birth(S) < birth(T) < projection < inverse(ROOT)
         < inverse(T) < release(T) < inverse(S) < release(S)
```

Forward edge custody now follows validate, native operation, then commit; a
failed parent cannot leave a false consumed-edge token. Producer inverse
readiness checks both pending counts and every outgoing edge receipt.
Same-typed `S`-for-`T` and `T`-for-`S` controls reject by exact owner
generation, not by type. Equal-content clones, stale serials, skipped and
duplicate edges, and premature inverse are targeted independently at both
owners.

At width three the primary complete boundary is:

```text
x0 + z0 = 0
x1 + z1 = 0
x2      = 0
z2      = 0
```

Its canonical hash is `59f0245207f2a0f1`. Five semantic variants match the
separate conventional GF2 reference at each width `3,4,8,12,16`, for 25
complete-boundary matches.

The accepted unique-node schedule retains seven working relations. The exact
occurrence expansion retains fifteen:

```text
B(w)                 = 4w^2 + 4w + 1
W(w)                 = 36w^2 + 11w + 2
accepted carrier     = W(w) + 8B(w)  = 68w^2 + 43w + 10
expanded tree        = W(w) + 16B(w) = 100w^2 + 75w + 18
```

At width sixteen the accepted path uses 18,106 carrier cells and 579,392 live
carrier bytes. It performs eight native forward/inverse calls and six leaf
toggles, versus fourteen and sixteen for the matched tree. The tree produces
the same complete boundary and restores by actual inverse but has no shared
owner and fails the custody predicate. This is a bounded resource comparison,
not an advantage claim.

Seventeen alternating transactions on one carrier allocation restore below
`2.8e-16`. All seven leases, eight edge tokens, owner and edge receipts,
pending counts, and scheduler state reset. Serial advances by seven
allocations and restoration generation by one. Snapshot reload remains a
separate weaker baseline.

The expanded no-smuggle trace covers file, network, IPC, memory mappings,
ordinary and positioned writes, memory files, splice/sendfile/copy routes,
cross-process writes, ptrace, and ioctl. The accepted path records two
boundary block copies and zero intermediate copies. It emits no intermediate
coefficient, hash, rank, pivot, content equality bit, tuple, witness,
candidate, or assignment expansion.

Focused review found one evidence-provenance mismatch after a compatibility-
only backend edit. The complete qualifier was rerun against the current
source hash, the prior single-fanout qualifier also passed against that
backend, and closure review found no remaining scientific finding.

This establishes:

```text
BOUNDED_NESTED_AFFINE_DAG_MULTI_SHARED_RESIDENT_MESSAGE_CUSTODY
```

with ceiling:

```text
BOUNDED_WIDTH16_SOFTWARE_GF2_EXACTLY_TWO_NESTED_FANOUTS_REFERENCE_ONLY
```

It does not establish a general DAG compiler, more than two fanout nodes,
unbounded fanout, compact live-range release or rematerialization, arbitrary
graphs or treewidth, CATVM enforcement for the affine compiler,
computational advantage, physical execution, Small Wall crossing, or
unlimited catalytic computation.

The selected next experiment is
`BOUNDED_GENERAL_AFFINE_DAG_CUSTODY_COMPILATION_AND_COMPACT_PEBBLING`:
derive the shared-owner set and fanout counts from arbitrary bounded public
DAGs, retain per-edge generation custody for every shared producer, and add a
lawful compact release/rematerialization schedule compared against both
retain-all unique-node and occurrence-expanded baselines.

## Compact leaf-pebbled execution of the nested affine DAG

The exact seven-node nested graph now has a lawful four-logical-lease
execution. The native-produced shared messages `S` and `T` remain pinned in
their original slot/serial generations. `I` and `ROOT` also remain their
original resident relations. Only the three immutable public degree-one
leaves are reversibly unencoded after their forward edge commits and
re-encoded in fresh leases immediately before the matching inverse edge.

The schedule performs the same four native forward and four native inverse
operations as retain-all execution. It performs twelve leaf encodes instead
of six and ten allocations instead of seven, but reduces the working
relation lease cap from seven to four:

```text
                         compact     retain-all     occurrence tree
working relation slots         4              7                  15
native calls                    8              8                  14
leaf encode calls              12              6                  16
lease allocations              10              7                  15
```

Under the exact declared law, forward `ROOT` requires the still-pinned
original `S`, its actual inputs `T` and `I`, and a distinct `ROOT` output.
Thus four is a lower bound for allocator-visible working leases on this
fixture. A three-slot build fails causally at that allocation. This is not a
physical-block or general scheduling lower bound: the substrate has a fixed
floor of six physical relation blocks including the copied boundary.

With `B(w)=4w^2+4w+1` and `W(w)=36w^2+11w+2`:

```text
compact carrier       = W(w) + 6B(w)  = 60w^2 + 35w + 8
retain unique carrier = W(w) + 8B(w)  = 68w^2 + 43w + 10
occurrence tree       = W(w) + 16B(w) = 100w^2 + 75w + 18
```

At width sixteen the compact carrier is 15,928 complex cells and 509,696
live bytes, versus 18,106 cells and 579,392 bytes for retain-all. This is
phase-carrier compactness only. Compact-specific stack, metadata, binary, and
output resources are not combined into a total-memory comparison, and no
performance or computational advantage is claimed.

Structural reconstruction obligations bind each public leaf body, nominal
signature, public-program epoch, and exact pending edge without retaining
relation contents. Wrong-body, missing, double, and incomplete
reconstruction controls reject. Shared/internal eviction and premature `S`
or `T` inverse controls also reject. No internal relation is rematerialized,
no operator is recomputed, only `ROOT` reaches the copied boundary, and the
expanded no-smuggle trace remains clean.

Five semantic variants match the separate GF2 reference at widths
`3,4,8,12,16`, for 25 complete boundaries. Seventeen alternating
transactions on one carrier restore below `6.7e-16`; wrong and missing root
inverses each leave `1.73205080757` error. The retain-all nested and earlier
single-fanout qualifiers pass unchanged against the generalized backend.
Focused adversarial review found no remaining finding after narrowing the
claim to its exact logical-lease and carrier-cell scope.

This establishes:

```text
BOUNDED_NESTED_AFFINE_DAG_PINNED_SHARED_COMPACT_LEAF_PEBBLING
```

with ceiling:

```text
BOUNDED_WIDTH16_SOFTWARE_GF2_EXACT_SEVEN_NODE_DAG_FOUR_LOGICAL_LEASE_CAP_SIX_PHYSICAL_RELATION_BLOCKS_DISTINCT_COPIED_BOUNDARY_PINNED_S_T_ORIGINAL_I_ROOT_PUBLIC_IMMUTABLE_LEAF_ONLY_RECONSTRUCTION_CARRIER_COMPACTNESS_ONLY
```

It does not establish generic or compiler-generated pebbling, a global
four-slot optimum, internal or operator rematerialization, exact generation
preservation for reconstructed leaf edges, more than two shared nodes,
arbitrary DAGs or graphs, CATVM enforcement for this compiler, total-memory
or performance advantage, physical execution, Small Wall crossing, or
unlimited catalytic computation.

The selected next experiment is now split deliberately. First compile and
execute a larger retain-all public DAG with compiler-derived custody for at
least four shared owners and fanout up to four. Only after that semantic
generalization is independently established should automatic schedule
generation attempt compact release or rematerialization.

## Four-owner heterogeneous-fanout retain-all DAG custody

The exact-two shared-owner fixture is now generalized once to a scrambled
fifteen-node public DAG. The topology compiler derives four native-produced
shared owners `S,T,U,V` with fanouts `4,3,3,2`, twelve shared edges, and
twenty-two total edges. All four actual owners are simultaneously live at
projection and after `ROOT^-1`; their six slot pairs are all nonaliased.

Every shared outgoing edge independently records forward and inverse slot and
serial observations. Per-owner exactness is recomputed from all actual edge
receipts and exact allocation, materialization, consumption, inverse, and
release counts. It is not trusted from an initialized flag. The observed
lifecycle is:

```text
birth(S) < birth(T) < birth(U) < birth(V) < projection < ROOT^-1
         < V^-1 < release(V) < U^-1 < release(U)
         < T^-1 < release(T) < S^-1 < release(S)
```

Only `ROOT` is copied to the boundary. There are zero intermediate relation
copies, hashes, coefficient decodes, tuple slots, assignment slots, witness
slots, or content-dependent receipts.

Five nontrivial semantic variants match the separate conventional GF2
evaluator at widths `3,4,8,12,16`, giving twenty-five exact complete-boundary
matches. The primary width-three boundary hash is `2e2200557469163d`;
width sixteen is `a5d0a65b5a670ae6`.

Each of the twelve shared edges also has an independent semantic necessity
control. A distinct upstream `A->A` operand replaces that edge in the public
identifier graph before reference compilation. Every modified graph passes
alias, cycle, connectivity, and fanout validation and produces a nontrivial
final boundary different from the primary result.

The exact 51-node occurrence expansion has 26 leaf occurrences, 50 edges,
and no fanout. It reaches the same complete boundary at widths three and
sixteen and restores by actual inverse, but is a separate reference-only
path:

```text
                                 retain-all DAG     occurrence expansion
working relation blocks                     15                       51
native calls                                22                       50
leaf encode calls                            8                       52
lease allocations                           15                       51
```

With `B(w)=4w^2+4w+1` and `W(w)=36w^2+11w+2`:

```text
retain-all carrier = W(w) + 16B(w) = 100w^2 + 75w + 18
occurrence carrier = W(w) + 52B(w) = 244w^2 + 219w + 54
```

At width sixteen these paths use respectively 26,818 and 66,022 complex
carrier cells, or 858,176 and 2,112,704 live bytes. These are bounded
phase-carrier counts relative to this exact occurrence expansion, not a
performance or computational-advantage claim.

Seventeen alternating transactions use one actual carrier allocation and
restore below `2.1e-16`. Wrong and missing root inverses each leave
`1.73205080757` error. Exact discrete state resets all fifteen leases,
twenty-two edge tokens, twelve shared-edge receipts, owner receipts, pending
counts, and scheduler state. Serial advances by fifteen allocations and
restoration generation by one.

Controls reject all twelve ordered same-typed cross-owner substitutions,
equal-content clones, stale/skip/double/reordered operations for every owner,
missing and stale receipt for every shared edge, swapped cross-owner
receipts, intermediate projection, custody projection, null carrier, and a
degree-four graph compiled with a fanout cap of three. The clone control uses
a separate bounded sixteen-slot build; the accepted executable remains
fifteen-slot. The expanded no-smuggle trace is clean.

Focused review initially rejected two controls. The semantic controls had
mutated resolved indices after compilation into aliased operands, and the
missing-receipt control had failed on edge state before receipt validation.
Both were repaired: public identifiers are now changed before compilation,
and missing receipt clears only `forward_seen` while preserving the forward
edge state. All twelve repaired missing controls fail through the intended
generation-receipt check with empty stdout. A direct width-sixteen occurrence
boundary comparison was also added. Fresh closure review and all three
predecessor qualifiers pass.

This establishes:

```text
BOUNDED_FOUR_OWNER_HETEROGENEOUS_FANOUT_RETAIN_ALL_AFFINE_DAG_CUSTODY_ESTABLISHED
```

with ceiling:

```text
BOUNDED_WIDTH16_SOFTWARE_GF2_EXACT_15_NODE_RETAIN_ALL_DAG_FOUR_NATIVE_SHARED_OWNERS_FANOUT_HISTOGRAM_4_3_3_2_REFERENCE_ONLY
```

It does not establish a general DAG compiler, automatic compact scheduling,
unbounded fanout, arbitrary graph topology, CATVM enforcement for this
compiler, performance or computational advantage, physical execution, Small
Wall crossing, or unlimited catalytic computation.

The selected next experiment is automatic compact scheduling for this exact
fifteen-node custody graph. It must derive legal release or rematerialization
decisions from the compiled topology, preserve exact generation custody for
all pending shared edges, and compare against both the fifteen-block
retain-all and 51-block occurrence baselines.

## Compiler-emitted public-leaf pebbling on the four-owner DAG

The established fifteen-node custody graph now has a topology-compiled
reversible compact tape. The compiler derives all four public degree-one
leaves from the validated graph, emits nineteen forward steps, and records
the exact reverse tape. It inverse-encodes each leaf after its sole forward
edge and reconstructs it from its public body only when the reverse tape
requires that pending inverse edge.

The plan binds topology `41d917d4a3308fbe`, schedule
`aa5719d149bc55e0`, every step's public node identity, and the complete
opcode/node/edge/live-count sequence into plan `627f298bb1d2c4e8`.
Predicted and observed peak residency are both eleven working relation
blocks; a ten-slot build fails causally at clean-pool exhaustion.

All four native shared owners `805,806,807,808` remain pinned and pairwise
nonaliased. The edge audit distinguishes eighteen exact resident-generation
edges from the four explicitly lawful reconstructed public-leaf edges
`801->805`, `802->805`, `803->806`, and `804->806`. All twelve shared edges
retain exact slot and serial generation. There are zero internal-node
rematerializations and zero operator recomputations.

The automatic path has:

```text
working relation blocks                 11
physical blocks including boundary      12
native operator calls                   22
leaf encode calls                       16
lease allocations                       19
```

For `B(w)=4w^2+4w+1` and `W(w)=36w^2+11w+2`:

```text
automatic leaf pebbling = W(w) + 12B(w) = 84w^2 + 59w + 14
retain-all              = W(w) + 16B(w) = 100w^2 + 75w + 18
occurrence expansion    = W(w) + 52B(w) = 244w^2 + 219w + 54
```

At width sixteen the automatic path uses 22,462 complex carrier cells and
718,784 live bytes. This is a bounded carrier-cell comparison, not a
total-memory or performance claim. The final evidence also counts automatic
scheduler and temporary inverse resources explicitly: 4,092 current-ABI
bytes for the plan and live scheduler arrays, a 4,416-byte transaction
summary, and a compiler-measured 61,600-byte concurrent `main + ac_execute`
stack floor. A 273,224-byte all-function-frame sum is retained as a
conservative upper bound.

Five semantic variants match the separate GF2 reference at widths
`3,4,8,12,16`, giving twenty-five complete-boundary matches. Seventeen
transactions use one actual carrier allocation and restore below `7.1e-16`.
Only the final root reaches the boundary; actual inverse and actual restored
carrier reuse remain intact. Snapshot is a separately marked weaker branch.

Controls reject illegal shared/internal eviction, wrong body, stale epoch,
missing/double/skipped reconstruction, shared inverse reordering, stale
internal generation, tape tampering, capacity ten, all intermediate
projection attempts, null carrier, and wrong/missing root inverse. The
expanded no-smuggle trace is clean. Focused review closed topology/schedule
provenance and resource-accounting findings and reports no remaining
substantive defect.

This establishes:

```text
BOUNDED_15_NODE_FOUR_OWNER_AUTOMATIC_PUBLIC_LEAF_PEBBLING_ESTABLISHED
```

with ceiling:

```text
BOUNDED_WIDTH16_SOFTWARE_GF2_EXACT_15_NODE_DAG_COMPILER_EMITTED_19_STEP_REVERSIBLE_PUBLIC_DEGREE_ONE_LEAF_ONLY_PEBBLING_11_WORKING_SLOTS_12_PHYSICAL_RELATION_BLOCKS_FOUR_PINNED_SHARED_OWNERS_REFERENCE_ONLY
```

It does not establish automatic general DAG pebbling, internal operator
rematerialization, a global optimum beyond the declared leaf-only planner,
arbitrary graph topology, CATVM enforcement for this compiler, total-memory
or performance advantage, physical execution, Small Wall crossing, or
unlimited catalytic computation.

The selected next experiment is compiler-planned internal operator
rematerialization with multi-epoch shared-edge custody. It must release and
later recompute eligible internal values while making every edge receipt
identify the correct activation generation, without saving decoded
intermediate content or relaxing final-boundary-only projection.

## Compiler-planned one-layer internal operator rematerialization

The compact compiler now derives four eligible public internal operators
`809,810,811,812` from the same validated fifteen-node topology. Each is a
nonshared degree-one operator whose two operands are pinned native shared
owners. The compiler emits a coefficient-oblivious 23-step reversible tape,
clears each operator after its sole forward consumer, and records only a
structural reconstruction obligation. No relation coefficients, decoded
intermediate, saved carrier copy, or answer-bearing table enters an
obligation.

During actual inverse execution, the pending consumer inverse causes the
operator to be recomputed natively from the original resident shared-owner
generations. The actual recomputed relation is consumed by the pending
inverse edge, then its own native inverse clears and releases it. The four
shared owners remain pinned and pairwise nonaliased throughout.

Logical edge custody and physical activation custody are now distinct. The
graph still has 22 public edges, each with one logical forward/inverse
transition. Rematerialization produces 30 physical activation receipts:
eight input edges activate twice, while six exact edges, four public-leaf
reconstruction edges, and four internal-operator reconstruction edges
activate once. All 22 producer-generation activation pairs close exactly.
A causal control changes only a consumer activation generation while keeping
the actual slot, serial, and producer activation unchanged; it is rejected.

The bound plan hashes topology `41d917d4a3308fbe`, schedule
`aa5719d149bc55e0`, and all 23 public steps into
`f0345b7ae7bfe27d`. Predicted and observed peak residency are eight working
blocks. An eight-slot build passes and a seven-slot build fails causally at
clean relation-pool exhaustion.

The one-layer path has:

```text
working relation blocks                  8
physical blocks including boundary       9
native operator calls                    30
leaf encode calls                        16
lease allocations                        23
logical edge transitions                 22
physical activation receipts             30
```

For `B(w)=4w^2+4w+1` and `W(w)=36w^2+11w+2`:

```text
internal rematerialization = W(w) +  9B(w) =  72w^2 +  47w + 11
automatic leaf pebbling    = W(w) + 12B(w) =  84w^2 +  59w + 14
retain-all                 = W(w) + 16B(w) = 100w^2 +  75w + 18
occurrence expansion       = W(w) + 52B(w) = 244w^2 + 219w + 54
```

At width sixteen the accepted path uses 19,195 complex carrier cells and
614,240 live carrier bytes. The evidence separately accounts for logical
custody, activation receipts, the reversible plan, obligations, execution
state, compiler-measured concurrent stack, the executable, and stdout.
These are bounded software resource counts, not a total-memory or
performance-advantage claim.

Five semantic variants match the independent GF2 reference and the leaf-only
predecessor at widths `3,4,8,12,16`, for twenty-five complete boundaries.
Seventeen transactions use one actual carrier allocation and restore below
`6.7e-16`; unrelated reuse consumes that actual restored carrier. Strict,
analyzer, sanitizer, deterministic-replay, expanded no-smuggle, and
predecessor-regression checks pass. Controls reject stale or missing
activation closure, stale producer activation, missing/double/stale operator
reconstruction, noneligible eviction, tape tampering, every forbidden
projection, null carrier, wrong or missing root inverse, and the weaker
snapshot branch. Focused independent review reports no blocking defect.

This establishes:

```text
BOUNDED_15_NODE_FOUR_OWNER_INTERNAL_OPERATOR_REMATERIALIZATION_ESTABLISHED
```

with ceiling:

```text
BOUNDED_WIDTH16_SOFTWARE_GF2_EXACT_15_NODE_ONE_LAYER_FOUR_INTERNAL_OPERATOR_REMATERIALIZATION_23_STEP_REVERSIBLE_TAPE_8_WORKING_SLOTS_9_PHYSICAL_BLOCKS_MULTI_ACTIVATION_EDGE_CUSTODY_REFERENCE_ONLY
```

It does not establish recursive internal rematerialization, automatic general
DAG pebbling, arbitrary graph topology, a global pebbling optimum, CATVM
enforcement for this compiler, total-memory or performance advantage,
physical execution, Small Wall crossing, or unlimited catalytic computation.

The selected next experiment is multi-layer recursive internal operator
rematerialization. It must evict an operator whose inverse reconstruction
depends on at least one operator that has itself been evicted, derive nested
obligations and activation generations from topology, and close exact custody
at activation depth greater than two without retaining decoded relation
content.

## Rank-2 recursive custody finishes the current affine scheduler

The exact public fifteen-node graph already contains a rank-two dependency:
`813` consumes rank-one operators `809,810`, and root `815` consumes `813`.
The compiler derives ranks from public topology and selects the maximum rank,
then lowest public ID. This chooses `813`; symmetric candidate `814` remains
resident as an anchor rather than enlarging the calibration.

The compiler emits a 28-action forward tape and mechanically derives a
28-action literal reverse. After the established one-layer prefix constructs
root `815`, it reconstructs dormant `809,810`, actually inverses and releases
`813`, then actually inverses and releases those temporary children.
Projection retains only `805,806,807,808,814,815`.

Literal reversal reconstructs `809,810` as generation two, then reconstructs
`813` from those actual resident values. It suspends the temporary children,
later reconstructs them as generation three, and uses those actual values in
the native inverse of rematerialized `813`. The nested frame depth is two;
including pinned owners, activation-chain depth is three.

Custody is now receipt-specific rather than one policy per public edge. Each
receipt binds exact edge, ordinal, consumer activation, forward and inverse
producer activations, and both action IDs. Exact receipts require the same
slot, serial, and generation. Rebind receipts require the exact unconsumed
compiler authorization, a structural obligation, the declared replacement
generation, and a fresh serial. A changed generation or serial alone is
insufficient.

There are still 22 logical public edges with one forward/inverse transition
each. The physical ledger closes:

```text
activation receipts                         40
exact receipts                              29
public-leaf rebind receipts                  4
internal-operator rebind receipts            7
multi-activation edges                      10
second-or-later activation receipts         18
maximum activation ordinal                   3
```

The four shared owners remain their original activation-zero generations.
Their physical forward/inverse totals are `805=8/8`, `806=6/6`,
`807=10/10`, and `808=4/4`.

Nine working relation slots are both predicted and observed; an eight-slot
build fails causally at clean pool exhaustion. Projection has six live
working blocks. With `B(w)=4w^2+4w+1` and
`W(w)=36w^2+11w+2`:

```text
rank-2 recursive scheduler = W(w) + 10B(w) = 76w^2 + 51w + 12
one-layer scheduler        = W(w) +  9B(w) = 72w^2 + 47w + 11
leaf-only scheduler        = W(w) + 12B(w) = 84w^2 + 59w + 14
retain-all                 = W(w) + 16B(w) = 100w^2 + 75w + 18
occurrence expansion       = W(w) + 52B(w) = 244w^2 + 219w + 54
```

The extra block relative to the one-layer calibration is the bounded cost of
holding the sibling anchor and actual recursive operands simultaneously. At
width sixteen the path uses 20,284 complex carrier cells and 649,088 live
carrier bytes. Each transaction performs 20 native forward and 20 native
inverse calls, 16 leaf encode calls, and 28 allocations/releases.

Five semantic variants match the independent GF2 evaluator and one-layer
predecessor at widths `3,4,8,12,16`, for twenty-five complete boundaries.
Seventeen same-carrier transactions restore below `5.0e-16`. Strict,
analyzer, sanitizer, deterministic replay, expanded no-smuggle tracing, and
the complete predecessor qualifier pass.

Controls reject a deep stale consumer with the same shared lease, deep stale
producer, cross-edge rebind authorization swap, generation reuse, missing
nested close, missing nested child, tape tamper, every intermediate/debug
projection, null carrier, capacity eight, and wrong or missing root inverse.
Snapshot remains a separately labelled weaker branch. Focused independent
review reports no blocking finding.

This establishes:

```text
BOUNDED_15_NODE_RANK2_RECURSIVE_OPERATOR_REMATERIALIZATION_ESTABLISHED
```

with ceiling:

```text
BOUNDED_WIDTH16_SOFTWARE_GF2_EXACT_15_NODE_SINGLE_RANK2_BORROWER_28_FORWARD_28_REVERSE_ACTIONS_9_WORKING_SLOTS_10_PHYSICAL_BLOCKS_MAX_ACTIVATION_ORDINAL3_REFERENCE_ONLY
```

It does not establish both rank-two branches, automatic general-DAG
pebbling, arbitrary topology, unbounded depth, CATVM enforcement for this
scheduler, non-affine relations, advantage, physical execution, Small Wall
crossing, or unlimited catalytic computation.

Per the durable main-goal guardrail, this finishes the current affine
scheduler calibration. The selected next experiment is CATVM enforcement of
the automatically scheduled shared relational DAG behind a carrier-owning
Unix-domain service. The controller may submit only the public manifest and
program selection and receive only the final boundary plus content-oblivious
custody receipts; it must never access action-time carrier state.

## CATVM now enforces the automatic rank-two shared-DAG carrier law

The reviewed scheduler is now privately embedded in a separate
carrier-owning Linux service. A default-preserving `RR_PUBLIC_MAIN` hook lets
the service link the actual scheduler while garbage-collecting its standalone
entry point and detailed reporting functions. The original standalone
semantics were requalified through the complete width and predecessor chain.

The production service accepts only:

```text
HELLO
EXECUTE <public variant 0..4>
SHUTDOWN
```

The controller is not linked to the phase core and contains no evaluator,
expected boundary, or answer table. Every internal projection, tape,
obligation, generation, receipt, carrier-read, dump, and debug request
returns one fixed denial. Embedded NUL, oversized, and unknown packets return
one fixed protocol error.

Each `EXECUTE` is atomic from the controller's perspective. Inside the
non-dumpable service, the actual reviewed path runs all 28 forward actions,
latches only root `815`, removes that actual boundary factor, runs the literal
28-action reverse, closes receipts/nodes/obligations/allocator, checks the
workspace and carrier, and advances restoration generation by exactly one.
Only then does it construct and send the final 49-coefficient boundary.

The service context and carrier are private, locked, `DONTDUMP`, and
`DONTFORK`. After accepting one same-UID client it unlinks the Unix socket and
installs a seccomp default-kill allowlist with no file-open, new socket,
process creation, execution, inspection, or shared-memory syscall. Six
same-UID access paths are denied. Accepted service stdout and stderr are
byte-zero, and a dedicated traced build confirms the post-custody syscall
surface.

The accepted run executes:

```text
primary program transactions                 1
unrelated reuse program transactions         1
alternating restored-carrier transactions  256
carrier creations                            1
request / response packets             273 / 273
request / response bytes          2,703 / 39,339
```

The primary and reuse boundaries match both the direct reviewed engine and
the independent GF2 reference. Every accepted transaction hardcodes
`RC_RESTORE_CORRECT` and `RR_FAULT_NONE`; the production protocol contains no
restore or fault selector. The second and all later programs consume the
actual carrier restored by the preceding inverse.

Separate test-only services establish wrong-root and missing-root restoration
failure, the weaker snapshot path, an inert boundary baseline, and a
dependency-reordered reverse tape that is killed fail-closed by seccomp
before a payload escapes. Strict, analyzer, sanitizer, deterministic replay,
direct/reference parity, output-key allowlist, symbol separation, and the
complete scheduler regression chain pass.

At width three the accepted machine accounts for:

```text
service context                         17,888 bytes
compiled topology                       2,664 bytes
activation plan                        13,192 bytes
program table                           1,960 bytes
actual carrier                         27,168 bytes
verification baseline                  27,168 bytes
production service binary              86,248 bytes
controller binary                      21,576 bytes
```

One uncontrolled 1,000-transaction comparison measured roughly equal direct
and in-place time (`37.01 s` versus `36.96 s`), snapshot at `18.29 s`, and
the inert boundary at `0.015 s`. This establishes no performance or
total-memory advantage.

This establishes:

```text
CATVM_ENFORCED_15_NODE_RANK2_AUTOMATIC_SCHEDULED_SHARED_RELATIONAL_DAG_ESTABLISHED_ON_PHASE_BACKEND
```

within:

```text
BOUNDED_LINUX_SAME_UID_SOFTWARE_GF2_WIDTH3_EXACT_15_NODE_ATOMIC_RUN_28_FORWARD_28_REVERSE_9_WORKING_SLOTS_REFERENCE_ONLY
```

The public fixture has root `815`, shared owners `805,806,807,808`, and
fanouts `4,3,3,2`. Clean-room adversarial verification confirms this exact
schedule exists but rejects generic transfer: source action identities,
obligations, and tape remain tied to this fixture. Its status is
`FIXTURE_SCOPED_VALID`; generic scheduler and arbitrary-DAG wording are
forbidden.

It does not establish a generic CATVM DAG service, non-affine relations,
general holographic relational computation, advantage, Small Wall crossing,
physical waveform or silicon computation, or unlimited catalytic
computation.

This closes the selected machine-boundary obstruction. Per the durable
main-goal guardrail, the next experiment is not another affine fixture. The
active successor is the smallest compact non-affine relation signature and
composition law that the existing phase carrier can close without truth
tables, assignment expansion, or decoded intermediate relations. If that
mechanism cannot be made compact, the fallback is a controlled
baseline/sham/augmented CATVM Small Wall experiment using the new enforced
service as the machine substrate.

## Fixed-schema quadratic ANF closes two hidden ports beyond affine GF(2)

The first post-affine successor uses monic Boolean-ANF relation factors:

```text
F(a,b;u) = u + alpha + beta*a*b
G(u,c;v) = v + gamma + delta*u*c
J(v,e;d) = d + eta + theta*v*e
```

The port/support schema is fixed before coefficients. No fixture supplies
selectors, expected results, membership masks, candidates, or witnesses.
Phase-resident Boolean-subset F3 products compile the two substitutions:

```text
H = v + gamma + delta*alpha*c + delta*beta*a*b*c
Z = d + eta
      + theta*gamma*e
      + theta*delta*alpha*c*e
      + theta*delta*beta*a*b*c*e
```

`H` occupies four resident phase cells and is consumed directly by the five
resident `Z` cells. `H` is never decoded or copied. Only the complete fixed
five-coefficient `Z` signature crosses the declared boundary.

The primary path closes:

```text
u+a*b = 0
v+u*c = 0
d+v*e = 0
```

to:

```text
d+a*b*c*e = 0
boundary coefficients [1,0,0,0,1]
```

Its nonzero fourth Boolean derivative proves the functional graph is not an
affine GF(2) relation. The unrelated reuse path closes to
`d+1+e+c*e+a*b*c*e=0`, also non-affine. A degree-two counterexample closes to
`d+c*e=0`; it has zero degree-four coefficient but a nonzero `c*e`
coefficient and remains non-affine. An affine sham closes to `d+e=0`.
Primary, reuse, degree-two, and sham paths share plan
`f8198cf1e338bbb5` and exactly the same native operation counts.

One accepted transaction uses:

```text
input signature cells                 9
resident H cells                      4
resident Z cells                      5
final-boundary cells                  5
total carrier cells                  23
baseline-plus-working carrier       736 bytes
comparison snapshot                 736 bytes
phase products                       18
carrier reads                        97
phase-cell updates                   46
intermediate decodes / copies       0 / 0
final boundary decodes               5
```

The primary and reuse boundaries match a separately linked reference. It
streams the bounded 32 external rows and 128 hidden probes without retaining
an extensional relation. The native path contains neither those loops nor the
reference machinery.

Wrong, missing, and applicable reordered inverses each leave restoration
error `1.73205080757`. Altering the resident-H source and cutting the
quadratic input term change the final boundary while their actual altered
histories reverse cleanly. Intermediate projection, null carrier, nonmonic
definitions, and malformed coefficients reject. Snapshot reload is a
separate weaker path with no inverse. One actual carrier restores through the
primary, unrelated reuse, and 256 alternating sentinels.

Strict compilation, analyzer, sanitizer, deterministic replay, reference
parity, native/reference linkage separation, fixed output allowlist, and a
one-write no-smuggle trace pass. Focused review found and repaired two
certificate defects: the first classifier ignored a legal degree-two
non-affine boundary, and its `c*e` coefficient was initially mislabeled as a
complete derivative. Fresh evidence `/tmp/qanf-full-eighth` binds 24
provenance entries and passes closure review with no remaining finding.

This establishes:

```text
ALGEBRAIC_FIXED_SCHEMA_QUADRATIC_ANF_TWO_HIDDEN_PORT_PHASE_COMPOSITION_ESTABLISHED
```

within:

```text
BOUNDED_BOOLEAN_GF2_MONIC_QAND_CHAIN_DEGREE4_FIVE_COEFFICIENT_BOUNDARY_SOFTWARE_REFERENCE_ONLY
```

It does not establish arbitrary non-affine relation closure, general Boolean
ANF elimination, bounded degree or term growth, a many-to-many non-affine
boundary, CATVM enforcement for this backend, advantage, Small Wall crossing,
physical execution, or unlimited catalytic computation.

The active successor is not a longer QAND fixture. It is the smallest CATVM
enclosure of this exact nonlinear path: seal F/G/J, keep H and Z service-local,
return only the final five coefficients after actual Z/H/input reversal, and
reuse the actual carrier for an unrelated nonlinear program. Once enforced,
the next alternative is a controlled baseline/sham/augmented CATVM Small Wall
experiment.

## CATVM now enforces the nonlinear QANF hidden-intermediate law

The reviewed quadratic-ANF engine is now privately embedded in a separate
carrier-owning Linux service. The ordinary controller has no phase code,
fixture parser, Boolean evaluator, expected result, or answer table. It can
request only:

```text
HELLO
EXECUTE <public variant 0..3>
SHUTDOWN
```

All attempts to project U, V, H, Z, F, G, J, carrier state, dumps, or state
detail receive one fixed denial. Embedded-NUL, oversized, and unknown packets
receive one fixed protocol error. The production protocol exposes no inverse,
snapshot, fault, carrier-disabled, or coefficient selector.

The service seals four public programs and one actual 23-cell carrier in a
locked, non-dumpable, non-forking context. After accepting one same-UID
client, it unlinks the Unix socket and installs a seccomp default-kill
allowlist. The six same-UID `/proc`, `process_vm_readv`, `ptrace`, and
`pidfd_getfd` inspection attempts fail.

Each accepted request atomically executes:

```text
encode F/G/J
-> construct unresolved resident H
-> construct Z directly from that actual H
-> copy and decode only final Z
-> boundary^-1
-> Z^-1
-> H^-1
-> (J,G,F)^-1
-> verify sealed carrier state
-> advance restoration generation
-> send final boundary
```

The four H cells are never decoded, copied, serialized, or committed. The
five Z coefficients are latched before inverse execution but no response is
constructed or sent until the actual inverse, exact discrete custody, and
complex restoration law pass.

The accepted service executes primary, unrelated nonlinear reuse, and 256
alternating transactions on one actual carrier. Exact per-transaction counts
remain:

```text
phase products                       18
carrier reads                        97
phase-cell updates                   46
final boundary decodes                5
intermediate decodes / copies       0 / 0
boundary copies                       2
snapshot loads                        0
```

Primary `[1,0,0,0,1]` and reuse `[1,1,1,1,1]` match both the direct phase
engine and separately linked reference. Wrong, missing, and applicable
reordered inverses are detected. Snapshot remains a distinct weaker
test-only path whose restoration generation stays zero; inert and null-carrier
controls cannot masquerade as the accepted path.

Strict compilation, static analysis, ASan/UBSan, deterministic replay,
predecessor regression, symbol separation, fixed output allowlisting, and
nonvacuous post-custody tracing pass. The traced suffix contains 27 actual
receives and 27 sends, zero forbidden channels, and service stdout/stderr are
byte-zero. Focused review first required the distinct trace build and every
control artifact to be provenance-bound, then required exact sealed-copy
equality for the discrete program table instead of relying on a 64-bit
custody hash. Fresh evidence `/tmp/catvm-qanf-full-fifth` binds 37 top-level
artifacts plus the nested predecessor bundle and passes with no remaining
finding.

This establishes:

```text
CATVM_ENFORCED_FIXED_SCHEMA_QUADRATIC_ANF_TWO_HIDDEN_PORT_COMPOSITION_ESTABLISHED_ON_PHASE_BACKEND
```

within:

```text
BOUNDED_LINUX_SAME_UID_SOFTWARE_BOOLEAN_GF2_MONIC_QAND_CHAIN_DEGREE4_FIVE_COEFFICIENT_ATOMIC_TRANSACTION_REFERENCE_ONLY
```

It does not establish general ANF elimination, arbitrary non-affine closure,
advantage, Small Wall crossing, physical waveform execution, or unlimited
catalytic computation.

The exact fixed schema has only six free public coefficient bits and admits
the compact conventional boundary law:

```text
[1, eta, theta*gamma, theta*delta*alpha, theta*delta*beta]
```

so a longer fixture would not remove the live obstruction. The selected next
experiment is a matched persistent-service triad: four-AND compact baseline,
snapshot sham, and in-place CATVM, with identical public requests and
boundaries. It will adjudicate the fixed-schema obstruction honestly; batch
repetition is throughput, not problem-size growth or mission Gamma.

## The fixed-schema CATVM triad confirms the compact-baseline obstruction

The same fixed QANF program family now runs behind three separately linked
persistent services with the same `AF_UNIX/SOCK_SEQPACKET` request, response,
fixture order, and final-boundary format:

```text
compact baseline -> direct four-AND coefficient law, no carrier
snapshot sham    -> phase forward path, then 368-byte working reload
in-place CATVM   -> phase forward, actual inverse, restoration, reuse
```

The compact arm contains no phase symbols and the phase arms contain no
compact fallback. All four public fixtures match the independent reference.
Projection of `H` or `Z`, embedded NULs, oversized packets, unknown commands,
and null-carrier requests are rejected in every arm. Post-custody traces are
nonempty and contain no forbidden channel; all service stdout and stderr are
byte-zero.

The exact per-transaction laws are:

```text
compact baseline  4 Boolean ANDs, no cache, table, carrier, or snapshot
snapshot sham     9 phase products, 51 reads, 23 updates, 5 final decodes,
                  1 snapshot load, 368 working bytes reloaded, no inverse
in-place CATVM   18 phase products, 97 reads, 46 updates, 5 final decodes,
                  no snapshot load, actual inverse and restored reuse
```

All 18 warm, CPU-pinned raw runs preserve identical boundaries and timed
traffic. Timing remains descriptive: the fixed symmetric arm order is not a
statistically counterbalanced performance study. Across batches 1024, 4096,
and 16384, compact/in-place wall-time ratios were approximately `0.793`,
`0.700`, and `0.737`; no performance or total-memory advantage appears.

The decisive obstruction is analytic rather than timing-based. The schema has
six free public coefficient bits, only 64 public programs, five final bits per
program, and the constant formula:

```text
[1, eta, theta*gamma, theta*delta*alpha, theta*delta*beta]
```

Thus even the complete unmaterialized answer vector is bounded above by 320
bits. Repeating transactions cannot create a growing problem family,
capacity separation, mission Gamma, or a Small Wall crossing.

Strict compilation, analyzer, ASan/UBSan, exact operation and traffic laws,
symbol separation, predecessor regressions, deterministic replay, resource
accounting, no-smuggle tracing, and 82 top-level provenance bindings pass.
Focused review `QANF-SMALL-WALL-OBSTRUCTION-01` found no blocking defect.
Evidence `/tmp/qanf-small-wall-fourth` establishes:

```text
FIXED_SCHEMA_QANF_COMPACT_BASELINE_OBSTRUCTION_CONFIRMED_UNDER_MATCHED_CATVM_BOUNDARY
```

within:

```text
BOUNDED_LINUX_WARM_FIXED_SCHEMA_QANF_BASELINE_SNAPSHOT_IN_PLACE_RESOURCE_COMPARISON_REFERENCE_ONLY
```

It does not establish advantage, capacity separation, Small Wall crossing,
arbitrary non-affine closure, physical execution, or unlimited catalytic
computation.

This retires further fixed-schema QANF enlargement. The selected successor is
a width-parametric compact non-affine many-to-many factor algebra whose
interface width or separator structure actually grows while native
composition preserves compact unresolved state. Dense truth-table
materialization, witness expansion, and longer fixed QAND fixtures are not
acceptable substitutes.

## Boolean relation tensor trains establish width-growing non-affine closure

The fixed QANF obstruction is now bypassed by a relation family whose public
word width grows. A Boolean-semiring tensor train represents
`R:X<->Y` with local binary cores and unresolved internal bond states:

```text
chi_R(X,Y) = OR_bonds AND_i A_i[left,x_i,y_i,right]
```

For `A:X<->Y` and `B:Y<->Z`, native composition writes each output core cell
from only two local shared-bit cofactors:

```text
C_i[x,z,(a,c),(b,d)]
  = OR_y A_i[x,y,a,b] AND B_i[y,z,c,d]
```

Boolean distributivity makes this the exact existential composition over the
complete word Y. There is no width-wide shared-assignment loop, witness list,
truth table, or dense `4^w` relation buffer. Ranks multiply deterministically.

The tested primary family is a rank-two, nonfunctional neighbor relation:

```text
y_i = x_i AND x_(i+1) for i < w
y_w is free
```

Three actual resident leaves execute:

```text
H = F ; G  at rank 4
Z = H ; J  at rank 8
```

Z consumes the actual phase-resident H. H is never decoded, serialized,
hashed, or materialized into a second block; its ordinary operand reads are
included in the carrier-read count. Only the final rank-eight core block is
copied and decoded; it survives while the machine removes the boundary copy, applies
`Z^-1`, applies `H^-1`, reverses the three leaf encodings, verifies
restoration, and reuses the actual carrier for the unrelated neighbor-NAND
program.

For each `i<=w-3`, the primary root has:

```text
z_i = x_i*x_(i+1)*x_(i+2)*x_(i+3)
```

Its fourth Boolean derivative is one. The projection onto those four inputs
and z is the degree-four AND graph, so the full relation is not affine GF(2).
Width four embeds the previous fixed QANF `d=abce`; widths 5, 8, 12, and 16
produce 2, 5, 9, and 13 overlapping degree-four windows. Exact leaf
multiplicities and conservative root lower bounds are kept distinct after
focused review corrected the initial labels.

The compact core laws are:

```text
N_2 =  16w - 16
N_4 =  64w - 96
N_8 = 256w - 448
carrier = 3N_2 + N_4 + 2N_8 = 624w - 1040 phase cells
```

At width four the carrier has 1,456 cells; at width sixteen it has 8,944.
Rank-eight output storage first becomes smaller than dense `4^w` storage at
width five. The best matched generic classical TT evaluator stores
`3N_2+N_4+N_8 = 368w-592` bits and performs linear local contraction. The
stronger fixture-specialized baseline directly emits the public
neighbor-AND-cubed rank-eight cores in `O(N_8)`, so no advantage follows.

Independent compact-reference parity matches every final core for primary and
reuse at all five widths. Strict compilation, GCC analyzer, ASan/UBSan,
deterministic replay, symbol separation, one-write output tracing,
wrong/missing/reordered inverse controls, snapshot separation, null-carrier
rejection, hidden-H projection denial, dense-request denial, and rank-cap
preflight pass. Thirty-four actual transactions reuse one carrier per width;
maximum repeated restoration error is `2.98936698014e-16` against tolerance
`2e-12`.

Focused review repaired the multiplicity labels and required the qualifier
source to join the provenance closure. Fresh evidence
`/tmp/boolean-tt-phase-sixth` passes with no remaining finding and establishes:

```text
BOUNDED_WIDTH_PARAMETRIC_BOOLEAN_TT_MANY_TO_MANY_RELATION_COMPOSITION_WITH_PRODUCT_RANK_NATIVE_PHASE_CLOSURE_AND_RESIDENT_INTERMEDIATE
```

within:

```text
BOUNDED_LINUX_SOFTWARE_WIDTHS4_5_8_12_16_BOOLEAN_SEMIRING_TT_RANK2_TO_RANK4_TO_RANK8_REFERENCE_ONLY
```

The union of finite TT ranks is closed under the product-rank construction,
but no fixed rank cap is closed under unbounded depth. This does not establish
native rank minimization, arbitrary QANF compactness, arbitrary graph
topology, CATVM enforcement, advantage, Small Wall crossing, physical
execution, or unlimited catalytic computation.

The selected successor is a minimal CATVM enclosure for this width-parametric
transaction. It must enforce hidden H custody and actual `Z^-1,H^-1`
restoration behind the process boundary. After that, the scientific fork is
exact relation-preserving rank reduction or a growing-instance Small Wall
triad, not additional widths at the same fixed ranks.

## CATVM enforces the width-growing Boolean-TT handoff

The width-parametric non-affine transaction now runs inside one separately
linked same-UID service per tested width. Each service creates one carrier,
privately embeds the reviewed phase-TT engine, accepts one
`AF_UNIX/SOCK_SEQPACKET` client, unlinks the endpoint, and installs a
default-kill seccomp allowlist. The controller contains no phase core,
relation generator, reference evaluator, expected boundary hash, witness
list, or answer table.

The production command surface is limited to `HELLO`, `EXECUTE 0/1`, and
`SHUTDOWN`. Inside each atomic execution, actual rank-two F/G/J leaves produce
resident rank-four H; Z reads the actual H cells as counted carrier operands.
H has zero decoded cells, zero serialized cells, and zero second-block
materializations. Only the final rank-eight core block is decoded and reduced
to a custody receipt. The service then applies the actual boundary, Z, H, and
leaf inverses, verifies canonical restoration, advances generation, and sends
the already-latched receipt.

At widths `4,5,8,12,16`, one service and one carrier complete primary,
unrelated neighbor-NAND reuse, and 32 alternating transactions. All 34
transactions preserve exact direct-backend boundary parity. The carrier
creation count remains one, generation reaches 34, and the second program
consumes the actual carrier restored by the first.

Wrong, missing, and prospectively noncommuting reordered inverse builds fail
restoration. Snapshot reload remains a separate generation-zero path. An
inert test build provides a carrier-disabled transport control with no final
result. The earlier no-argument check is labeled only as malformed startup,
not as a null-carrier experiment. Intermediate projection, embedded-NUL,
oversized, and unknown requests are rejected.

All six same-UID `/proc`, `process_vm_readv`, `ptrace`, and `pidfd_getfd`
inspection attempts are denied. A 3,701-byte post-custody trace contains 28
receive and 28 send calls and no file-open, file-create, rename, connect, or
stdout/stderr write channel. Service stdout and stderr are byte-zero.

Resource evidence counts payload arrays, sealed verification state,
transaction comparison state, protocol buffers and traffic, binaries, and the
exact phase-operation laws. Direct, inert, snapshot, and in-place paths are
compared only at operation-law scope; this is not comprehensive RSS or a
performance study. The best generic classical TT evaluator and the stronger
fixture-specialized O(N8) generator remain available, so no advantage follows.

Focused review repaired the malformed-startup and overbroad zero-copy labels.
Fresh evidence `/tmp/catvm-boolean-tt-third` passes both manifests,
analyzers, ASan/UBSan, replay, direct regression, no-smuggle tracing, and
review with no remaining finding. It establishes:

```text
CATVM_ENFORCED_WIDTH_PARAMETRIC_BOOLEAN_TT_RESIDENT_H_COMPOSITION_ESTABLISHED_ON_PHASE_BACKEND
```

within:

```text
BOUNDED_LINUX_SAME_UID_SOFTWARE_WIDTHS4_5_8_12_16_BOOLEAN_TT_RANK2_TO_RANK4_TO_RANK8_ATOMIC_TRANSACTION_REFERENCE_ONLY
```

The machine boundary is no longer the immediate obstruction for this family.
Product ranks still grow with depth, and neither exact rank reduction nor a
fixed-rank closure law is established. The selected successor is the smallest
exact relation-preserving Boolean-TT quotient experiment that can decide
whether this public nonlinear family admits native fixed-rank recursive
closure. If it only exploits fixture-specific structure or fails to reduce
general products, record that obstruction and move to the growing-instance
Small Wall triad rather than adding more widths.

## Suffix bisimulation reduces this family from exponential to linear rank

The selected exact quotient experiment is now established for homogeneous
chains of the neighbor-AND relation and, separately, homogeneous chains of
the neighbor-OR relation. At composition depth `d`, an unquotiented internal
bond state is a `d`-bit column and has raw product rank `2^d`. For these two
declared families the live columns are monotone threshold strings, so public
family geometry admits an exact suffix-bisimulation quotient.

Let `L` be the number of word sites remaining to the right of a bond. The
exact quotient ranks are:

```text
outer boundaries                         1
final internal bond, L=1                 2
other internal bonds, L>=2     min(d+1,L+2)
```

For `2<=L<d`, threshold heights below `L` remain singleton classes, heights
`L..d-1` merge into one suffix-indistinguishable middle class, and height
`d` remains a distinct all-leading top class. An early prototype incorrectly
merged that top state; focused review caught the error because width
four/depth three then used 96 rather than 120 final cells and failed raw
product semantics. The accepted implementation and an explicit bad-horizon
control preserve this repair.

The quotient plan depends only on public family, width, depth, site, and
remaining suffix. Every output cell is written directly from the actual
resident depth-`d-1` quotient stage and a resident depth-one leaf. It never
materializes the raw product core, a width-wide shared assignment, a truth
table, witness list, candidate set, or dense `4^w` relation. Only the final
stage is copied and decoded. The machine then removes that boundary copy,
reverses all quotient stages and the leaf, verifies restoration, and reuses
the actual carrier for the other homogeneous family.

Fifteen cases cover widths `4,5,8,12,16`, depths two and three at every
width, and `(4,4),(5,5),(8,8),(12,8),(16,8)`. At width sixteen/depth eight:

```text
raw product rank                         256
maximum quotient rank                      9
raw final TT cells                  3,672,064
quotient final TT cells                 3,548
final representation reduction      1,034.967x
retained resident stages               13,740 cells
boundary copy                           3,548 cells
total carrier                          17,288 phase cells
```

When width is at least depth, maximum rank is `d+1`, not `2^d`; at fixed
width it saturates no higher than `w+1`. This is a family-scoped linear-rank
law, not fixed-rank unbounded-depth closure or general Boolean-TT
minimization.

The independent verifier deliberately materializes raw product-rank TT cores
to certify the bounded result. Its largest final stage is 3,672,064 one-byte
cells. It verifies prefix reachability, suffix coaccessibility, class-uniform
suffix signatures, exact quotient edges, and final core parity for both
families. This exponential-in-depth verifier cost is disclosed and is not
part of the accepted phase path.

Wrong, missing, and noncommuting reordered inverses detect restoration
failure. Snapshot reload remains generation zero. Bad-horizon overmerge fails
semantic parity, the wrong OR phase law leaves the Boolean alphabet,
intermediate projection and null carrier are rejected, and deterministic,
analyzer, ASan/UBSan, and predecessor CATVM checks pass. Thirty-four
transactions reuse each actual carrier; maximum observed restoration error is
`2.48253415325e-16` against a predeclared `2e-12` tolerance.

Focused evidence `/tmp/boolean-tt-quotient-fourth` establishes:

```text
BOUNDED_BOOLEAN_TT_SUFFIX_BISIMULATION_QUOTIENT_REDUCES_PRODUCT_RANK_GROWTH_FROM_EXPONENTIAL_TO_LINEAR_WITH_PHASE_RESIDENT_CLOSURE
```

within:

```text
BOUNDED_LINUX_SOFTWARE_WIDTHS4_5_8_12_16_DEPTHS2_3_4_5_8_NEIGHBOR_AND_OR_FAMILY_SCOPED_QUOTIENT_REFERENCE_ONLY
```

Clean-room adversarial verification matches all 112 homogeneous formula
cases. Alternating, nonperiodic, and reverse-nonperiodic depth-six controls
instead produce ranks `2,7,14,14,14,14`, exceeding the homogeneous maximum
rank 7. The result is therefore `FAMILY_SCOPED_VALID`, not a general
Boolean-TT quotient law. Frozen evidence is bound to phase-source SHA-256
`1a8678d11df0be641a5c035d2e117eccb7086d23a4046362818a64ef4cb3a655`;
the current SHA-256
`281e1d1eb455cfdd0abaa77943459d7dbb0df340025023e6ee1cb6d91f3e2520`
differs only by the `QTT_EMBEDDED_MAIN` preprocessor wrapper added at
`c0cee6a9`, not by the quotient body.

The strongest fixture-specialized conventional baseline directly emits the
public quotient cores in `O(final quotient cells)`. The phase path retains
every stage for inverse execution, so neither compute nor memory advantage
follows. The selected successor is the matched growing-instance compact
baseline, snapshot-sham, and in-place CATVM triad. Its purpose is to expose
whether direct quotient generation and retained inverse history are the next
machine-law obstruction, not to enlarge the same family with more depths.

## Growing matched triad identifies the compact recurrence obstruction

The selected triad is now complete for `(4,4)`, `(5,5)`, `(8,8)`,
`(12,8)`, and `(16,8)`. Three separately linked persistent services receive
the same public width, depth, homogeneous AND/OR schedule, final receipt, and
protocol traffic:

```text
direct baseline   public threshold-quotient generator, O(final cells)
snapshot sham     actual phase forward path, then working-state reload
in-place CATVM    actual phase forward path, actual inverse, restored reuse
```

The direct service contains no phase engine, carrier, cache, answer table, or
raw product. The phase services contain no direct-generator fallback, and the
controller contains neither computation engine nor expected receipt. The
independent verifier still materializes raw product TT cores, but only outside
the timed arms.

At width sixteen/depth eight, every arm returns the same AND receipt
`4da4cbe210c58b26` and OR receipt `1582e559b5414ed6` after projecting all
3,548 final quotient cells. The retain-all phase carrier contains 17,288
logical cells: 13,740 retained stages, including 10,192 predecessor-history
cells, plus the 3,548-cell boundary. Live carrier, sealed verification state,
and per-transaction comparison snapshot each occupy 553,216 bytes. Snapshot
reload copies 276,608 working bytes per transaction.

Across two observations per arm after 32 warm transactions, the bounded
width-sixteen service-CPU means were:

```text
direct baseline                      3,788,813 ns
snapshot phase                     506,511,099 ns   133.686x baseline
in-place phase                     962,470,024 ns   254.029x baseline
```

These warm Linux software timings are descriptive, not a general performance
law. The stronger adjudication is structural: homogeneous neighbor-AND/OR
quotient cores have a public threshold recurrence that emits each final cell
directly. The current Boolean root-locking law therefore has no useful phase
resource distinct from the best matched compact classical representation.
Avoidance of the exponentially large raw-product verifier is not a Small
Wall crossing.

Projection denial, null-carrier request rejection, byte-identical arm
traffic, snapshot generation zero, actual in-place restoration/reuse,
analyzers, ASan/UBSan, and post-custody no-smuggle tracing pass. Carrier
creation CPU is exposed separately from warm transaction CPU; at width
sixteen the sampled baseline, snapshot, and in-place seal costs were 2,484,
936,436, and 898,906 ns. Evidence
`/tmp/boolean-tt-small-wall.TNqwzG/evidence` establishes:

```text
GROWING_NONAFFINE_BOOLEAN_TT_SUFFIX_QUOTIENT_SMALL_WALL_TRIAD_CONFIRMS_COMPACT_RECURRENCE_AND_RETAINED_HISTORY_OBSTRUCTION
```

within:

```text
BOUNDED_LINUX_WARM_WIDTHS4_5_8_12_16_DEPTHS4_5_8_HOMOGENEOUS_NEIGHBOR_AND_OR_REFERENCE_ONLY
```

The compact recurrence is the primary route obstruction. Retained inverse
history is a separate phase-machine defect because `sum(H1..Hd)` adds a depth
factor to carrier storage. The selected immediate repair is public-topology
reversible pebbling with actual slot reuse; further benchmark variants or
larger homogeneous fixtures are not selected.

## Reversible quotient-stage pebbling reduces the actual phase carrier

The phase-owned repair now compiles a reversible path-pebble tape for
`H1 -> ... -> Hd` directly from public depth. A stage toggle is legal only
while the actual predecessor and permanent H1 leaf are resident. Every move
binds exact node, clean slot, activation generation, and predecessor
generation into the schedule hash. An inverse toggle must restore the
complete phase region below `2e-12` before the region can be rebound.

At width sixteen/depth eight, a 13-move forward tape reaches H8 with three
slots of 3,548, 2,968, and 1,848 cells. H5, H7, and H8 are resident at the
final projection. Only H8 is copied and decoded; the boundary copy is removed
and the exact slot-tagged tape is then reversed before H1 is removed.

```text
retain-all carrier                    17,288 cells / 553,216 bytes
pebbled carrier                       12,152 cells / 388,864 bytes
actual carrier reduction               5,136 cells / 29.708%

retain-all phase updates              34,576
pebbled phase updates                 46,896
recomputation multiplier               1.356x
weighted stage-move cells             39,320
reconstruction additions / cells       6 / 6,160
```

The comparison snapshot also falls to 388,864 bytes, but remains verification
state rather than the restoration path. Exact final hashes and one-counts
match the retain-all phase machine and independent raw-product verifier in all
five cases. Wrong boundary inverse, missing inverse move, and applicable
noncommuting reordered inverse controls fail restoration; dirty-slot
injection, live predecessor-generation tampering, and schedule-hash tampering
are rejected by custody. Snapshot reload is separate, and 18 transactions
reuse each actual restored allocation. Maximum observed repeated restoration
is `4.99600361081e-16`.

Evidence `/tmp/boolean-tt-pebble.gKMVMI/evidence` establishes:

```text
TOPOLOGY_DERIVED_REVERSIBLE_BOOLEAN_TT_QUOTIENT_STAGE_PEBBLING_REDUCES_RETAINED_PHASE_HISTORY
```

within:

```text
BOUNDED_LINUX_SOFTWARE_WIDTHS4_5_8_12_16_DEPTHS4_5_8_HOMOGENEOUS_NEIGHBOR_AND_OR_PHASE_PEBBLING_REFERENCE_ONLY
```

This changes the phase machine rather than its wrapper: actual phase regions
are restored early and rebound. It does not change the family-specific
Boolean recurrence, improve compute cost against that recurrence, establish
fixed-rank unbounded-depth closure, or cross the Small Wall.

The homogeneous suffix family is now exhausted as an advantage route. The
next selected work is the smallest exact non-Boolean phase relation signature
whose composition preserves unresolved interference information that is not
immediately homomorphic to Boolean threshold or GF(2) affine recurrence. It
must retain boundary-only projection, actual inverse restoration, restored
carrier reuse, and a matched compact classical signature.

## Nonlinear unit-phase shear establishes a native non-Boolean law

The first coherent U(2) prototype was rejected because it decoded the whole
resident matrix into the matched classical recurrence. A direct additive-wave
repair was also rejected because it bypassed unit-phase carrier semantics.
Neither rejected prototype entered the accepted evidence.

The accepted repair operates on two baseline-relative unit phasors. Each
public morphism rotates one actual resident target by
`exp(i*k*Im(source))`, using only `relative` and `multiply_cell`. The source
is unchanged by its shear, so reverse topology order with conjugate factors
is the actual inverse. Alternating targets make adjacent morphisms nonlinear
and noncommuting.

Across depths `3,32,128,512,2048,4096`, exact quantized final hashes match an
independent two-angle recurrence. Only the copied final two-phase boundary is
decoded as an interference probability. Sixteen additional transactions
reuse the same restored carrier. At depth 4096:

```text
resident plus boundary phase cells           4
native forward-plus-inverse updates       8,200
maximum unit-modulus error          3.33066907388e-16
maximum repeated restoration       7.58532257521e-14
predeclared restoration tolerance             2e-12
```

Wrong boundary, missing shear, wrong shear, and noncommuting reordered
inverses leave raw restoration residuals. Coupling-disabled and
baseline-neutralized executions change the boundary, actual null-carrier
execution is rejected, snapshot reload remains generation zero, and analyzer,
ASan/UBSan, replay, and source-level no-bypass checks pass.

Evidence `/tmp/nonlinear-phase-shear-final.uC5B2f/evidence` establishes:

```text
BOUNDED_NONLINEAR_UNIT_PHASE_TORUS_SHEAR_COMPOSITION_WITH_ACTUAL_RESTORATION_AND_INTERFERENCE_PROJECTION
```

within:

```text
BOUNDED_LINUX_SOFTWARE_NONLINEAR_TORUS_SHEAR_DEPTHS3_32_128_512_2048_4096_DOUBLE_COMPLEX_PHASE_REFERENCE_ONLY
```

This is phase-owned progress: phase is the primitive state, the update is
nonlinear and noncommuting, and the carrier is actually restored and reused.
It is not a distinct phase resource or a Small Wall crossing. The best
matched recurrence stores two double angles in 16 bytes and has the same
`O(depth)` cost. The next selected mechanism is a public-topology compiled
shared nonlinear phase-shear graph behind CATVM. It must grow relational
geometry beyond fixed two-angle state without hiding an equivalent compact
classical solver or materializing phase paths.

## Shared nonlinear phase graphs now execute behind CATVM

The successor replaces the implicit alternating two-cell schedule with a
public dependency graph. Six scrambled shear declarations compile to a
deterministic topological tape over four resident phasors. Two source epochs
feed multiple consumers. The compiler rejects cycles and every noncommuting
hazard pair not ordered by reachability; declaration order does not affect
the compiled topology or boundary.

Each morphism consumes the actual baseline-relative source and mutates only
the actual target through `multiply_cell`. No source phase or forward factor
is retained. Reverse execution traverses exact compiled custody and
recomputes the conjugate factor from the then-restored resident source. Only
two final latch cells are decoded.

The direct backend passes rounds `1,3,128,512,2048,4096` with exact quantized
parity against an independently compiled four-angle recurrence. The CATVM
service is non-dumpable, locked, single-peer `AF_UNIX/SOCK_SEQPACKET`. At 128
rounds:

```text
transactions on one carrier                         256
forward / inverse shears per transaction        768 / 768
native phase updates                                  1,548
resident phase reads                                   1,540
unit-modulus complex-cell checks                        9,240
live carrier / comparison snapshot bytes            192 / 192
snapshot creation / reload bytes                    192 / 96
page-rounded locked context bytes                       8,192
best matched classical state bytes                         32
```

All six same-UID process inspection paths and every intermediate projection
request are denied. Wrong, missing, and applicable reordered inverses fail
restoration. Snapshot recovery remains generation zero. The unrelated second
program and 254 following alternations consume the same actual restored
carrier.

A matched warm diagnostic uses three observations per arm, each after 32
warmups with 1,024 timed transactions. Boundary arms have identical request
and response byte counts:

```text
compact four-angle direct process             5,696 ns/transaction
direct phase process                        110,397 ns/transaction
isolated inert boundary                      10,922 ns/transaction
snapshot CATVM                               63,525 ns/transaction
in-place CATVM                              127,496 ns/transaction
```

These Linux software timings are descriptive and show overhead, not
leverage. Evidence `/tmp/catvm-nonlinear-phase-resource2.VKjf3e/evidence` and
focused independent review establish:

```text
BOUNDED_CATVM_TOPOLOGY_COMPILED_SHARED_NONLINEAR_UNIT_PHASE_GRAPH_WITH_ACTUAL_RESTORATION_AND_REUSE
```

within:

```text
LINUX_USERSPACE_AF_UNIX_WIDTH4_EDGES6_ROUNDS128_DOUBLE_COMPLEX_PHASE_REFERENCE_ONLY
```

This generalizes phase topology, shared custody, and the machine boundary. It
does not establish a distinct phase resource, computational advantage,
arbitrary relational geometry, physical execution, or a Small Wall crossing.
The exact four-double recurrence remains the primary obstruction.

The selected next mechanism is not a larger point-phase graph. It is the
smallest compact non-affine open phase-relation signature with native
shared-port closure that preserves unresolved relational interference beyond
an immediately equivalent point-angle recurrence.

## Full-F5 conics now close through two resident hidden ports at fixed rank

The selected repair replaces the Boolean/GF(2) affine calibration with a
full-F5, six-coefficient conic signature. A total-degree-two conic
`Q(u,v)` is bracketed by monic affine graph relations `u=A(x)` and `v=C(z)`.
Native root-of-unity product interpolation computes `H(x,v)=Q(A(x),v)` and
then consumes the actual resident `H` to compute `K(x,z)=H(x,C(z))`. Neither
hidden coefficient vector is decoded or serialized.

The primary fixture closes to `K=xz`, whose F5 zero set has nine points and
therefore cannot be an affine F5 subset. The carrier preserves six signature
cells after each closure. Correct reverse dependency order restores all 22
borrowed cells, and the actual restored carrier runs an unrelated dense
conic program plus sixteen alternating reuse transactions.

The qualified component accounting is:

```text
resident phase storage                         352 bytes
hidden baseline storage                        352 bytes
borrowed verification copy                     704 bytes
explicit h/k phase-factor temporaries          192 bytes
projected boundary structure                    32 bytes
compact classical live coefficient state       64 bytes
retain-all classical coefficient state          88 bytes
largest individual compiler stack frame       1648 bytes
```

These are separately measured components, not a summed peak-process-memory
claim. Analyzer, ASan/UBSan, deterministic replay, wrong/missing/applicable
reordered inverse, snapshot, bypass, null-carrier, and intermediate-projection
controls pass. Focused review found and repaired one evidence leak: the
classical reference had serialized `H`; corrected evidence hard-gates its
absence.

Evidence `/tmp/algebraic-f5-conic-recorded.GK4aG4/evidence` establishes:

```text
BOUNDED_F5_MONIC_AFFINE_CONIC_TWO_HIDDEN_PORT_FIXED_RANK_PHASE_CLOSURE_WITH_RESTORATION_AND_REUSE
```

within:

```text
BOUNDED_FULL_F5_MONIC_AFFINE_PORT_MAPS_TOTAL_DEGREE2_CONIC_TWO_HIDDEN_PORTS_FIXED_FIXTURES_SOFTWARE_REFERENCE_ONLY
```

This is a broader non-affine phase-relation algebra, but not general
conic-conic composition. Monic substitution also has an exact 64-byte compact
modular recurrence, so this result is not a distinct phase resource,
computational advantage, or Small Wall crossing.

The next selected mechanism changes the native operation from coefficient
substitution to coherent hidden-port summation: reversible nondegenerate F5
quadratic phase-kernel Gauss contraction with fixed two-block custody. Exact
chirps will be compared against one-entry off-manifold perturbations and
generic unit-phase kernels. If only the chirp manifold preserves unit-modulus
fixed-rank closure, the compact seven-integer Weil recurrence is the
adjudicated obstruction and further depth variants will not be selected.

## Coherent Gauss contraction closes at fixed rank, but only on the tested chirp manifold

The phase machine now contracts actual resident 5-by-5 unit-phase kernels
through a coherent five-path shared-port sum. Two 25-cell blocks alternate:
the new block is formed from the live block and public quadratic kernel, then
the new actual resident block and public adjoint reconstruct and clear the
old block. Reverse execution repeats those contractions in exact reverse
order. No inverse kernels are retained.

Depths `2,4,8,32,128,512,2048` preserve 50 hidden phase cells. At depth
2,048, the accepted lifecycle performs 204,850 native updates and 1,024,000
coherent path terms. Final hashes and probabilities match an independent
seven-integer F5 Weil recurrence. Restoration remains below
`1.997e-14` against a predeclared `2e-10` tolerance, and the actual restored
carrier runs an unrelated kernel plus eight alternating transactions.

The decisive raw-modulus control does not normalize a failed fixture into the
claim:

```text
exact chirp maximum raw modulus error       2.465e-14
one-entry 2^-20 radian perturbation          4.056e-7
deterministic generic unit-phase fixture       0.6265
```

Thus the two tested off-manifold fixtures do not close in a phase-only
unit-modulus block. This is not a theorem about every off-manifold kernel.
The 28-byte classical number is the semantic seven-integer recurrence state,
not complete peak memory with temporaries and final expansion.

Evidence `/tmp/algebraic-f5-gauss-recorded.RnbSLQ/evidence` and focused review
establish:

```text
BOUNDED_F5_QUADRATIC_PHASE_KERNEL_COHERENT_SHARED_PORT_GAUSS_CLOSURE_WITH_FIXED_TWO_BLOCK_CUSTODY_RESTORATION_AND_REUSE
```

within:

```text
BOUNDED_F5_NONDEGENERATE_QUADRATIC_PHASE_KERNEL_DEPTHS2_4_8_32_128_512_2048_DOUBLE_COMPLEX_SOFTWARE_REFERENCE_ONLY
```

This changes the native operation from coefficient substitution to coherent
hidden-port interference and removes depth-growing inverse history for the
tested class. Direct two-block role custody is not CATVM enforcement. The
exact Weil recurrence and failed tested off-manifold closure mean there is
still no distinct phase resource, advantage, or Small Wall crossing.

The next repair must represent lawful contraction magnitude without moving
it into hidden classical state. The selected experiment uses a canonical
pair of unit phasors per complex kernel entry and tests generic off-chirp
unitary composition. Its matched baseline is the full 25-complex matrix
recurrence; success can establish a broader waveform carrier law, not
advantage.

## Paired phasors admit generic off-chirp unitaries but expose amplitude and gauge costs

Each complex matrix entry is represented as the coherent average of two unit
phasors. A 5-by-5 product entry is computed from 20 actual phasor path
products and canonically split back into a phase pair. Two 50-cell blocks
alternate through produce, adjoint reconstruction, source release, reverse
execution, and final unsealing.

Structured generic dense unitaries outside the F5 chirp manifold pass at
depths `1,2,4,8,16,32`. Final hashes and probabilities match an independent
25-complex dense recurrence. The depth-32 carrier uses 100 hidden phase cells,
retains no inverse matrices, performs 64,000 phasor path products, restores
within `2.369e-9` against a predeclared `1e-8` phase-cell tolerance, and
reuses the actual restored carrier for an unrelated depth-8 program plus
eight following transactions.

The accepted class declares a `1e-6` nonzero-entry floor; its observed
minimum is `0.004346`. Canonical pair gauge becomes ill-conditioned near
zero. The shared production validator rejects a nonunitary candidate, and
the production encoder rejects an entry outside the unit disk. Missing,
wrong, reordered inverse, snapshot, null-carrier, and
intermediate-projection controls also pass.

Evidence `/tmp/algebraic-paired-phase-recorded.EAtbac/evidence` and focused
review establish:

```text
BOUNDED_GENERIC_5X5_UNITARY_PAIRED_UNIT_PHASOR_COHERENT_COMPOSITION_WITH_TWO_BLOCK_CUSTODY_RESTORATION_AND_REUSE
```

within:

```text
BOUNDED_DOUBLE_COMPLEX_5X5_UNITARY_PUBLIC_CONSTANT_STATE_GENERATOR_PAIRED_PHASE_COORDINATE_EMBEDDING_DEPTHS1_2_4_8_16_32_ENTRY_FLOOR1E_6_SOFTWARE_REFERENCE_ONLY
```

This is broader waveform custody, not a distinct phase resource. The split
materializes a 25-complex result plus 50 phase factors in 1,200 bytes of
explicit host temporaries and performs magnitude-sensitive arithmetic.
Resident paired-phase storage is 1,600 bytes versus a 400-byte dense complex
recurrence; U(5) has a 25-double, 200-byte coordinate lower bound before
chart metadata and workspace. CATVM enforcement, global arbitrary-unitary
stability, phase-only computation without amplitude arithmetic, advantage,
and Small Wall crossing remain unestablished.

Three materially different phase mechanisms now localize the same
obstruction: discrete F5 conic coefficients collapse to modular
substitution; one-phasor coherent kernels close only on the tested compact
chirp manifold; paired phasors admit generic unitary magnitude only through
an equivalent complex recurrence and gauge-sensitive amplitude split. The
next phase-owned advance must supply a native coupling resource that does
not merely move this complex recurrence into host temporaries or extra gauge
state.

## Finite phasor means have a global gauge-section obstruction

The ordered arithmetic mean of two unit phasors is rank-one at its antipodal
zero. Three equilateral phasors repair that local defect: the full Jacobian
has rank two, and fixing the third phase leaves determinant `sqrt(3)/18`.
Thus a smooth local amplitude section genuinely exists near zero.

That repair cannot be global over the closed unit disk. On the disk boundary,
triangle equality forces every unit summand to equal the boundary value.
Each of the three component phases would therefore have winding one, while a
circle-valued map extending continuously over a disk must have boundary
winding zero. The fixed-third-phase chart independently fails at `z=-r`,
where it demands a two-phasor sum of modulus four although the maximum is
two.

Evidence `/tmp/phase-coupler-gauge-recorded.tUAnYE/evidence` and focused
review establish:

```text
ORDERED_THREE_UNIT_PHASOR_MEAN_HAS_SMOOTH_ZERO_LOCAL_SECTION_BUT_NO_GLOBAL_CONTINUOUS_GAUGE_FREE_DISK_SECTION
```

within:

```text
FINITE_ORDERED_UNIT_PHASOR_ARITHMETIC_MEAN_ENCODING_TOPOLOGICAL_OBSTRUCTION_SOFTWARE_REFERENCE_ONLY
```

This retires only memoryless finite ordered phasor arithmetic-mean encoding
as the amplitude repair. It does not rule out weighted or nonmean encodings,
stateful gauge transport, conserved phase geometries, or general phase
couplers.

## Kerr/interference waves keep amplitude resident without re-encoding

The successor changes the carrier law instead of adding phasor coordinates.
Four normalized complex wave cells receive an intensity-dependent Kerr phase
kick followed by alternating `SU(2)` interference couplers. Interference
changes amplitude while amplitude remains resident in the actual wave cells;
there is no phasor split, magnitude chart, or gauge canonicalization.

Public topology rematerializes the inverse without retained matrices or
intermediate wave states. Depths `1,4,32,128,512,2048` pass with a fixed
four-complex-cell carrier. At depth 2048, maximum norm error is `5.773e-15`
and restoration error is `1.207e-11` against a predeclared `2e-10`
tolerance. Sixteen varying programs consume the same restored carrier and
remain below `2.935e-14` restoration error. Missing, wrong, and reordered
inverse controls leave raw residuals; disabled-Kerr and disabled-coupler
paths change the boundary and lawfully restore.

Evidence `/tmp/nonlinear-kerr-wave-recorded.dCFXDX/evidence` and focused
review establish:

```text
BOUNDED_FIXED_RANK_NONLINEAR_KERR_INTERFERENCE_WAVE_CARRIER_WITH_ACTUAL_RESTORATION_AND_REUSE
```

within:

```text
BOUNDED_FOUR_COMPLEX_CELL_NORMALIZED_KERR_SU2_WAVE_MESH_DEPTHS1_4_32_128_512_2048_DOUBLE_COMPLEX_SOFTWARE_REFERENCE_ONLY
```

Here fixed rank means fixed carrier dimension only. An independent
eight-double recurrence has the same 64-byte semantic state and matches
quantized boundaries through depth 512. Cross-implementation parity at depth
2048 is not claimed. Direct-process CLI denial is not machine enforcement.
The next selected experiment puts this wave law behind CATVM and compares it
against the exactly matched compact complex recurrence; no distinct phase
resource, advantage, Small Wall crossing, physical execution, or unlimited
computation is established.

## CATVM now enforces nonlinear wave custody, restoration, and reuse

The four-cell Kerr/`SU(2)` carrier now runs inside a non-dumpable, locked,
single-peer `AF_UNIX` service with a default-kill seccomp filter. The public
controller contains no complex carrier or wave update. It can request only a
complete primary or reuse transaction and receives three final real
observables after the service has reversed the actual resident wave.

One carrier completes 256 alternating depth-128 transactions. Restoration
generation reaches 256, maximum restoration error is `5.016e-13`, and
maximum repeated final-boundary drift is `1.358e-13` under the predeclared
`2e-10` continuous tolerance. Six same-UID inspection paths deny access.
Intermediate projection, null-carrier, missing inverse, wrong Kerr inverse,
and applicable reordered inverse controls fail; snapshot remains generation
zero and is not actual restored-carrier reuse.

Evidence `/tmp/catvm-kerr-wave-recorded.QUkVYO/evidence` and focused review
establish:

```text
CATVM_ENFORCED_NONLINEAR_KERR_INTERFERENCE_WAVE_COMPOSITION_WITH_ACTUAL_RESTORATION_AND_REUSE
```

within:

```text
LINUX_USERSPACE_AF_UNIX_FOUR_COMPLEX_CELL_DEPTH128_DOUBLE_COMPLEX_KERR_SU2_SOFTWARE_REFERENCE_ONLY
```

The raw response recorder and exact parsers close the no-smuggle response
surface. Resource evidence separates the 208-byte logical context and
4096-byte context mapping from observed warm process `VmRSS=4120 KiB` and
`VmLck=4112 KiB`.

Matched CATVM inert, snapshot, and in-place arms have identical traffic.
Descriptive warm medians are approximately `10.7 us`, `25.1 us`, and
`40.4 us` per transaction. The compact scalar forward recurrence is
approximately `13.2 us`; direct wave full lifecycle is approximately
`29.2 us`. Those direct timings do not have identical lifecycle semantics
and establish no leverage. The decisive obstruction remains an exactly
matched 64-byte, four-complex-value recurrence.

The next selected phase-owned experiment is not another CATVM wrapper or
larger Kerr fixture. It tests whether the nonlinear symplectic wave law
admits a compact open relation signature with native shared-port closure,
or whether exact elimination forces relation-rank or harmonic growth. That
directly reconnects the waveform machine to unresolved relational geometry.

## Nonlinear Kerr Lie signatures expose bounded polynomial rank growth

Two rational Kerr/`SU(2)` Hamiltonians now undergo native Poisson
canonical-index contraction with every coefficient resident as both an
`F17` and `F19` root phase. No coefficient is decoded before the complete
five-grade boundary. Exact rational and phase hashes agree at degrees
`4,6,8,10,12`.

The exact rational support grows:

```text
degree                 4    6    8    10    12
nonzero terms          6   32   85   126   231
full basis cells      35   84  165   286   455
```

Correct reverse contractions restore within `7.772e-16`; the same carrier
runs a different mixer and seven further alternating transactions within
`2.887e-15`. Missing, wrong, and dependency-reordered inverses leave both
large continuous residuals and nonidentity modular cells. Failed controls
receive no successful restoration receipt.

Evidence `/tmp/symplectic-lie-final.slaEJp/evidence` and focused review
establish:

```text
BOUNDED_DUAL_PRIME_PHASE_RESIDENT_NONLINEAR_SYMPLECTIC_LIE_SIGNATURE_GROWTH_WITH_ACTUAL_RESTORATION_AND_REUSE
```

within:

```text
BOUNDED_TWO_MODE_RATIONAL_SU2_KERR_HAMILTONIAN_DUAL_PRIME_LIE_GRADES4_6_8_10_12_SOFTWARE_REFERENCE_ONLY
```

The 2,050 active dual-prime cells are 32,800 logical packed payload bytes,
but the allocated C carrier is 154,800 bytes before compiler and stack
temporaries. Poisson contraction contracts canonical derivative indices; it
does not eliminate the coordinate interface of a general open relation.
This is bounded growth in the tested polynomial Lie-signature class, not a
full BCH materialization or an unbounded-growth theorem.

## The exact Stokes quotient removes redundant phase and norm state

The first phase-owned repair maps normalized two-mode waves to the Stokes
sphere and reduces every Lie-Poisson product by
`x^2+y^2+z^2=1`. This removes global optical phase and the conserved norm
Casimir exactly. Canonical monomials have `z` exponent zero or one.

Across degree limits `2,3,4,5,6`, exact rational support is
`3,4,9,9,16` and quotient basis size is `9,16,25,36,49`. Thus the five-grade
basis falls from 1,025 four-coordinate cells to 135 Stokes cells. The
dual-prime carrier falls from 2,050 to 270 phase cells, or from 32,800 to
4,320 logical packed payload bytes. Actual CPython object allocation is not
claimed.

Phase hashes match an independent exact-rational Groebner oracle at every
grade. Correct restoration is `2.220e-16`; eight same-carrier reuse
transactions remain within `1.110e-15`. Wrong inverse now leaves residual
`1.993` and 31 nonidentity modular cells, closing the focused review finding.

Evidence `/tmp/stokes-lie-final.ji9Ao2/evidence` and focused review
establish:

```text
BOUNDED_STOKES_SPHERE_REDUCED_DUAL_PRIME_PHASE_RESIDENT_NONLINEAR_SYMPLECTIC_LIE_SIGNATURE_WITH_RESTORATION_AND_REUSE
```

within:

```text
BOUNDED_NORMALIZED_TWO_MODE_STOKES_SPHERE_RATIONAL_SU2_KERR_DUAL_PRIME_LIE_GRADES2_3_4_5_6_SOFTWARE_REFERENCE_ONLY
```

This is an exact relation-preserving rank reduction, but not fixed-rank
closure. Spherical-harmonic degree and rank still grow, and the matched
point-evaluation recurrence remains 64 bytes. The next selected experiment
must use irreducible Stokes harmonic sectors or a lawful nontrivial invariant
quotient; adding more grades to the same monomial fixture would not remove
the obstruction.

## Topology-derived parity closure removes impossible harmonic cells

The Stokes Lie-Poisson carrier now compiles only canonical monomials whose
total-degree parity matches the public grade. Quadratic generation changes
parity once per bracket and exact sphere reduction changes degree by two, so
this 80-cell sector is closed by construction. It replaces the prior
135-cell all-degrees-through-limit allocation without inspecting answers.

Dual-prime custody falls from 270 to 160 resident phase cells, or from 4,320
to 2,560 logical packed payload bytes. Exact rational hashes match at every
grade. Correct restoration is `2.220e-16`; eight same-carrier transactions
remain within `1.110e-15`. Missing, wrong, and dependency-reordered inverses
leave nonidentity modular cells, while snapshot receives no restoration
receipt.

The independent homogeneous-sphere quotient is nonzero at highest degrees
`2,3,4,5,6`, with canonical term counts `2,3,5,5,7`. This certifies bounded
survival of the highest harmonic shell, not an explicit irreducible
decomposition or an unbounded-growth theorem.

Evidence `/tmp/stokes-harmonic-final.ZwLLlN/evidence` and focused independent
review establish:

```text
BOUNDED_PARITY_ADMISSIBLE_STOKES_HARMONIC_SECTOR_DUAL_PRIME_PHASE_SIGNATURE_REDUCTION_WITH_RESTORATION_AND_REUSE
```

within:

```text
BOUNDED_NORMALIZED_TWO_MODE_PARITY_ADMISSIBLE_STOKES_HARMONIC_DUAL_PRIME_LIE_GRADES2_3_4_5_6_SOFTWARE_REFERENCE_ONLY
```

The remaining obstruction is whether those nonzero highest shells obey a
compact exact phase recurrence or require increasing independent rank.

## The highest Stokes shell has an exact fixed-rank factorization

For the repeated quadratic generator `Q=L^2/625`, the identity
`ad_Q(f)=(2/625)L ad_L(f)` and `ad_L(L)=0` give
`ad_Q^n(z^2)=L^n q_n`. In a rational frame aligned with `L`, every `q_n`
after the first bracket remains in the four-coordinate space
`ac,ab,bc,b^2-c^2`.

The phase machine stores each modular coefficient as its complete `F17` or
`F19` unit-phase character orbit. Public scalar multiplication is a reversible
index permutation, eliminating the deep floating-power drift without a
contractive phase lock or decoded residue shadow. The carrier has 144
unit-phase cells, 2,304 logical packed bytes, and zero retained inverse
history independent of depth.

At depths `1,2,4,8,32,128,512,2048`, phase boundaries match the exact
rational recurrence. At maximum depth the unexpanded highest shell has
degree 2,050 and expanded dimension 4,101. Root error and restoration
residual are both zero; 16 transactions consume the same restored carrier.
Missing and wrong inverses leave 136 and 52 nonidentity character cells.

Evidence `/tmp/stokes-factorized-shell-final.YPrr4F/evidence` and focused
independent review establish:

```text
FIXED_RANK_FACTORIZED_HIGHEST_STOKES_HARMONIC_SHELL_PHASE_RECURRENCE_WITH_RESTORATION_AND_REUSE
```

within:

```text
EXACT_REPEATED_SINGLE_AXIS_QUADRATIC_STOKES_KERR_HIGHEST_HOMOGENEOUS_SHELL_FACTOR_L_POWER_N_TIMES_Q4_DUAL_PRIME_SOFTWARE_DEPTHS1_TO_2048
```

The algebraic factorization law holds at every positive depth, but executed
software evidence is bounded and covers only the highest shell. The lower
shells and full Stokes signature remain open. A matched classical dual-prime
machine stores the same Q4 recurrence in eight residue bytes, so this is not
a distinct phase resource, advantage, Small Wall crossing, or unbounded
catalytic computation.

## Character-phase addition exposes noncommuting harmonic rank growth

The complete-character phase representation now generalizes beyond the
monomial Q4 recurrence. For each modular coefficient `v`, native
componentwise phase multiplication performs coefficient addition. Every
executed nonzero public scalar action is a phase-index permutation; modular
zero contributions are skipped. Sparse Lie-Poisson contraction consumes
actual resident coefficients without decoding, residue shadows, phase
locking, or retained inverse history.

With tilted and axial quadratic generators alternating through degree 14,
the parity-admissible carrier holds 676 logical coefficient cells or 24,336
unit phases. Its logical packed payload is 389,376 bytes. A non-emitting
verifier compares all 2,704 primary/reuse dual-prime boundary cells exactly
against the rational oracle. Correct restoration is `5.551e-16`; eight
same-carrier transactions stay within `1.832e-15`. Missing, wrong, and
noncommuting dependency-reordered inverses all leave modular mismatches.

All thirteen projected grades are the declared final coefficient-jet
boundary. After unique harmonic projection, exact middle catalecticant ranks
are:

```text
3,3,5,5,7,7,9,9,11,11,13,13,15
```

This establishes bounded separable/Waring-rank growth through degree 14. It
is not a lower bound against every compact representation and not an
unbounded theorem.

Evidence `/tmp/stokes-alternating-axis.HZY9E7/evidence` and focused
independent review establish:

```text
BOUNDED_NONCOMMUTING_ALTERNATING_AXIS_STOKES_CHARACTER_PHASE_HARMONIC_CATALECTICANT_RANK_GROWTH_WITH_RESTORATION_AND_REUSE
```

within:

```text
BOUNDED_NORMALIZED_TWO_MODE_RATIONAL_ALTERNATING_NONCOMMUTING_QUADRATIC_STOKES_AXES_DUAL_PRIME_CHARACTER_PHASE_GRADES2_TO14_SOFTWARE_REFERENCE_ONLY
```

The same-output dual-prime classical jet needs 1,352 logical one-byte
residues; actual classical allocation is unmeasured. The 64-byte point
evaluator has different boundary semantics. The next phase repair must find
compact nonseparable closure or a resource not immediately equivalent to
classical point evolution.

## Reversible BCH rematerialization removes Lie-word history

Two noncommuting Stokes modules are now composed through the exact bounded
signature `log(exp(A) exp(B))`. Public tensor-log coefficients are
Dynkin-Specht-Wever projected into right-nested Poisson brackets. The phase
backend rematerializes one Lie word at a time, adds its complete character
orbit to the declared final grade, and immediately applies the actual inverse
to its reusable scratch chain.

Through word grade six, 72 nonzero words require at most six live scratch
blocks. The carrier holds 116 final and 116 reusable scratch coefficient
cells, or 8,352 unit phases and 133,632 logical packed bytes. Retained
Lie-word history is zero. A non-emitting verifier checks all 464
primary/reuse dual-prime cells. Correct restoration is `4.554e-13`; reuse
restores within `6.054e-13`. Missing and wrong inverses leave residual
`1.993`, and swapped noncommuting module order changes the boundary.
The matched grade-six classical signature is 232 logical residue bytes.
An executed snapshot sham transfers 133,632 logical bytes at both creation
and reload, mints no restoration receipt, and successfully runs the reuse
program. The null-carrier path fails closed.

Evidence `/tmp/stokes-bch-rematerialized-final.3Xj1ap/evidence` establishes:

```text
BOUNDED_TOPOLOGY_REMATERIALIZED_NONCOMMUTING_STOKES_BCH_CHARACTER_PHASE_CLOSURE_WITH_RESTORATION_AND_REUSE
```

within:

```text
BOUNDED_NORMALIZED_TWO_MODULE_NONCOMMUTING_STOKES_BCH_GRADES1_TO6_PHASE_AND_EXACT_RATIONAL_DIAGNOSTIC_THROUGH_GRADE10_SOFTWARE_ONLY
```

The independent exact diagnostic through word grade ten has harmonic
catalecticant ranks `3,3,5,5,7,7,9,9,11,11`. Rematerialization therefore
removes retained inverse history but not growth of the declared final
signature. The same-output grade-ten dual-prime signature needs 720 logical
residue bytes; the 64-byte point evaluator has different boundary semantics.
This is not fixed-rank closure, an arbitrary representation lower bound,
advantage, Small Wall crossing, or physical execution.

## Reflection grading halves the BCH carrier but preserves rank growth

Both noncommuting quadratic generators are even under `y -> -y`. The Stokes
Lie-Poisson bracket maps parities `(p,q)` to `p+q+1 mod 2`, so public topology
fixes the `y` parity of every length-`n` Lie word to `n-1 mod 2`.

Compiling this grading removes 58 of 116 final and 58 of 116 reusable scratch
coefficient cells. The character carrier falls from 8,352 to 4,176 unit
phases, or from 133,632 to 66,816 logical packed bytes. A non-emitting exact
verifier compares 232 retained primary/reuse dual-prime cells and proves all
116 excluded primary/reuse cells exactly zero. Actual restoration remains
`4.554e-13`; reuse remains within `6.054e-13`.

Evidence `/tmp/stokes-bch-reflection-final.pzfxDb/evidence` establishes:

```text
BOUNDED_REFLECTION_GRADED_TOPOLOGY_REMATERIALIZED_NONCOMMUTING_STOKES_BCH_CHARACTER_PHASE_QUOTIENT_WITH_RESTORATION_AND_REUSE
```

within bounded grade-six phase execution and a grade-ten exact diagnostic.
The matched classical signature is also halved to 116 logical residue bytes.
Harmonic catalecticant ranks remain `3,3,5,5,7,7,9,9,11,11`; the exact
quotient changes the constant but not the rank-growth obstruction. It is not
fixed-rank closure, a distinct phase resource, advantage, Small Wall
crossing, or physical execution.

## Cubic phase/Fourier waves expose exact sequential rank growth

The next phase machine leaves the integrable Stokes point recurrence. It
stores an unresolved exact wave tensor train over `Q(zeta_5)` and interleaves
normalized local Fourier interference with the nonseparable two-site phase
law `zeta^(gamma*(x^2*y+x*y^2))`. Arbitrary cyclotomic amplitudes remain
resident; roots of unity are not being used merely as encoded digits.

Across widths `2,4,6` and one, two, and three central crossings, exact central
bond ranks are `4,14,64`. Separate `F11` and `F31` implementations reproduce
all bond vectors and projected boundary residues. No global `5^width` wave,
assignment expansion, statevector, or truth table is materialized.

At width six, the carrier reaches 9,710 TT cells and 129,678 logical
coefficient bytes. Exact factorization scratch reaches 1,642,143 logical
bytes, with 128/129-bit numerator/denominator height. Actual inverse
restoration is exact, and an unrelated circuit consumes the same restored
carrier. Missing, wrong, reordered, Fourier-disabled, forced-rank,
separable, Clifford, snapshot, projection, and null-carrier controls pass.

Evidence `/tmp/cyclotomic-f5-cubic-tt-final.7jbmXC/evidence` establishes:

```text
BOUNDED_EXACT_CYCLOTOMIC_CUBIC_PHASE_FOURIER_TENSOR_TRAIN_SEQUENTIAL_RANK_GROWTH_WITH_ACTUAL_RESTORATION_AND_REUSE
```

within widths `2,4,6` and bounded software exact arithmetic. The matched
exact classical TT is identical to the accepted representation, so the same
rank, coefficient, state, and scratch growth defeats any distinct-resource,
advantage, or Small Wall claim. This is bounded phase-native rank growth, not
fixed-rank or unbounded computation.

## CATVM enforces hidden cyclotomic bond composition

A non-dumpable Linux Unix-domain service now owns one persistent width-four
exact cubic/Fourier TT carrier. The controller imports only a protocol
framing module; current-source AST inspection and runtime module inspection
prove that it loads neither the service nor the phase engine. Fixed
1,024-byte requests select only public programs; fixed 4,096-byte responses
contain one final cyclotomic amplitude, restoration generation, flags, and a
one-way custody receipt. Tensor entries, bonds, ranks, pivots, and Fourier
intermediates have no protocol path.

Primary and unrelated reuse amplitudes match direct exact reference.
Restoration generations advance `1,2`, proving that the second circuit
consumes the actual restored carrier. The mode-`0600` service checks peer
credentials, emits no process output, denies `/proc/<pid>/mem` inspection,
and rejects intermediate, null, and cross-mode commands.

The accepted in-place service contains no snapshot image. A separate sham
service matches the primary boundary and charges all three actual logical
payload copies: 160 bytes at image creation, 160 at execution load, and 160
at restoration reload. Python object-graph resident sizes are reported
separately. The sham sets snapshot-loaded and reports restoration generation
zero.

Evidence `/tmp/catvm-cyclotomic-f5-tt-repaired.MNFtp2/evidence` establishes:

```text
CATVM_ENFORCED_CYCLOTOMIC_CUBIC_TT_HIDDEN_BOND_COMPOSITION_WITH_ACTUAL_RESTORATION_AND_REUSE
```

within Linux userspace width four. This strengthens machine custody but does
not change the phase rank, matched classical TT, advantage, Small Wall, or
physical-execution boundaries.

## Continuous phase coherence yields a bounded bandwidth plateau

The next carrier leaves finite cyclotomic arithmetic for a continuous
sampled wave. Each fixed Floquet step applies the phase kick
`exp(-i sqrt(2) cos(theta))`, a Fourier transform, the irrational quadratic
free phase `exp(-i sqrt(3) n^2/2)`, and the inverse Fourier transform.

At tail-energy tolerance `1e-12`, periodic radii at depths
`64,128,256,512,1024,2048` are `26,26,26,26,26,24`. An identical-strength
deterministic 17-step phase schedule reaches
`50,77,131,239,452,882`. Grid doubling, an epsilon sweep, and a
63-bit-mantissa replay preserve the contrast. The depth-2048 actual inverse
restores within `1.412e-14`; an unrelated 31-step program and eight further
cycles consume the same restored carrier.

Evidence `/tmp/continuous-kicked-phase-repaired.I8PlYF/evidence` supports the bounded
numerical claim:

```text
BOUNDED_NUMERICAL_CONTINUOUS_IRRATIONAL_KICKED_PHASE_COHERENT_FOURIER_LOCALIZATION_CONTRAST_WITH_EFFECTIVE_BANDWIDTH_PLATEAU_ACTUAL_RESTORATION_AND_REUSE
```

The matched adaptive Bessel recurrence needs only 97 complex coefficients
and matches the 2,048-grid phase state within `7.298e-13`. It is smaller and
faster than the dense FFT implementation. The plateau is therefore
phase-coherent behavior but not a distinct computational resource or
advantage. Exact compact support, asymptotic localization, control
delocalization, Small Wall crossing, unbounded computation, CATVM
enforcement for this carrier, and physical execution remain unestablished.

The fixed one-rotor vector is the new obstruction. The selected repair lifts
the periodic law to four nonseparably coupled rotors in a Fourier TT/MPS and
tests the central `2|2` interface rank without materializing the dense
four-dimensional wave.

## Four-rotor phase relations move the obstruction to interface rank

The one-rotor recurrence is lifted to four continuous rotors with
nearest-neighbor nonseparable Bessel-factorized phase coupling. Canonical TT
compression measures the physical central `2|2` Schmidt cut.

At mode radius 14 and discarded-L2 tolerance `1e-11`, central ranks grow
`13,100,246` over three rounds while maximum local Fourier radius remains
12. The separable control stays rank one. Guards 14 and 16 agree in the
declared boundary within `8.895e-13`.

The exact depth-three rank is guard-dependent (`242,246,247`); only
qualitative monotone growth is claimed. Ranks are tolerance-truncated values
at central closure. Restoration is physical-state tolerant rather than
canonical-rank restoration, and high-rank numerical residue prevents any
stable compact-reuse claim.

The actual inverse restores within `5.236e-8`; an unrelated two-round
program consumes the actual restored carrier and restores within
`1.251e-8`. Retained inverse history is zero and generations advance `1,2`.

Evidence `/tmp/four-rotor-kicked-phase-tt-final.MhLb0S/evidence` supports:

```text
BOUNDED_FOUR_ROTOR_NONSEPARABLE_CONTINUOUS_KICKED_PHASE_FOURIER_TT_CENTRAL_INTERFACE_RANK_GROWTH_WITH_ACTUAL_RESTORATION_AND_REUSE
```

This is an obstruction result. The nominal unmaterialized dense wave has
707,281 cells (11.316 MB), but inverse closure peaks at 5,380,718 TT/MPO
cells (86.091 MB) and a 707,281-cell interface core. The matched classical
TT is identical. No fixed rank, distinct resource, advantage, Small Wall
crossing, unbounded computation, CATVM enforcement for this carrier, or
physical execution is established.

The selected repair is matrix-free streamed coupling Schmidt closure. It
must rematerialize public Bessel terms into matvecs, retain only the needed
interface subspace with exact Frobenius-tail accounting, and eliminate both
the expanded live MPO bond and dense interface core.

## Matrix-free closure removes the expanded inverse structures

The public Bessel coupling is now rematerialized term by term into
deterministic matrix-free Schmidt matvecs. Streamed full-column Frobenius
accounting certifies the retained subspace without constructing the expanded
MPO bond or 707,281-cell dense interface core.

At discarded-L2 tolerance `1e-6`, central ranks are `11,41,97`; the
same-tolerance dense reference gives `11,40,96`. Boundary disagreement is
`9.459e-10`. Actual inverse restoration is `1.922e-8`; unrelated reuse is
`4.087e-9`, with generations `1,2`.

Evidence `/tmp/four-rotor-matrix-free-accounted.sTo4ay/evidence` supports:

```text
BOUNDED_MATRIX_FREE_STREAMED_BESSEL_SCHMIDT_CLOSURE_WITHOUT_EXPANDED_MPO_OR_DENSE_INTERFACE_CORE_WITH_ACTUAL_RESTORATION_AND_REUSE
```

A conservative simultaneous-array upper bound falls from 86.091 MB to
43.532 MB, including QR/SVD factors, output arrays, and nested contraction
temporaries. The largest workspace array is smaller than the eliminated
dense core. Total matrix-free state still exceeds the 11.316 MB
dense-equivalent because inverse cancellation requires probe rank 492. The
identical classical matrix-free TT defeats any advantage claim.

The next repair is post-inverse canonical phase closure with
fresh-versus-restored reuse rank/resource parity. Numerical high-rank residue
must not move into later transactions.

## Post-inverse canonical closure stops reuse-rank accumulation

One standard TT-rounding sweep now acts directly on the actual strict
inverse-restored carrier. The sweep uses a global `1e-7` L2 budget divided
over the three cuts and never consults a baseline state.

Residual bond ranks `29,166,29` reduce to `1,1,1`; carrier cells fall from
280,894 to 116. The inverse error before closure is `5.235e-8`, the closure
delta is `5.108e-8`, and postclosure restoration error is `1.147e-8`.
This is a tolerance-defined numerical quotient after actual inverse
execution, not exact inverse restoration. Its canonical classification is
`INVERSE_PLUS_CANONICAL_NUMERICAL_QUOTIENT`; it must never be reported as
unqualified exact restoration.

The unrelated matrix-free reuse program has rank history `9,27` on both the
actual restored carrier and a separately created fresh diagnostic carrier.
Probe rank, probe columns, matvec counts, retained rank, largest array,
workspace, and total live cells match exactly. Boundary disagreement is
`6.681e-12`; both carriers close to 116 cells and ranks `1,1,1`. The accepted
carrier advances to generation two with no snapshot or retained inverse
history.

Evidence `/tmp/four-rotor-canonical-final.dQzIFv` supports:

```text
BOUNDED_ACTUAL_POST_INVERSE_TT_CANONICAL_QUOTIENT_CLOSURE_WITH_FRESH_RESTORED_MATRIX_FREE_REUSE_RANK_AND_RESOURCE_PARITY
```

The matrix-free counter was also repaired to use the selected live workspace
rather than recursively adding a historical maximum. Compact factors,
squared singular values, and retained NumPy backing allocations are counted.
Its primary peak is 43.532 MB, not 46.522 MB.

The primary inverse still needs probe rank 492 and exceeds the 11.316 MB
dense-equivalent wave. Canonical reuse is repaired; inverse cancellation
without probe-space expansion is now the active obstruction.

## Incremental Schmidt algebra removes the probe-space expansion

The 13 public Bessel factors are now added directly to a compact Schmidt
factorization. Each update reorthogonalizes the combined old/new bases with
explicit in-place ZGEQRF/ZUNGQR and applies a small-core ZGESVD. The certified
finite-plus-analytic Bessel tail is subtracted from the `1e-6` per-coupling
budget before the remainder is divided over 13 updates. Factor lifetimes,
retained NumPy backing allocations, queried QR workspaces, simultaneous
ZGESVD input/output/workspace, and explicit BLAS outputs are counted.
Every combined basis passes a `1e-10` ZHERK orthogonality gate.

At radius 14 and depth three, the accepted path uses zero probes and peaks at
677,010 complex-cell equivalents (10.832 MB), below the 707,281-cell
(11.316 MB) nominal dense wave. Central ranks are `11,47,117`; boundary
disagreement against the same-tolerance dense reference is `7.380e-8`.
No expanded MPO bond or dense interface core is constructed.

The actual inverse restores within `2.687e-7` and canonical closure within
`7.495e-8`. Unrelated generation-two reuse restores within `1.499e-7`.
Fresh and restored reuse have rank history `11,32`, exact deterministic
resource signatures, and boundary disagreement `6.003e-8`. Missing, wrong,
and reordered inverse controls separate.

Evidence `/tmp/four-rotor-incremental-schmidt-certified.DHPL2i` supports:

```text
BOUNDED_PROBE_FREE_INCREMENTAL_BESSEL_SCHMIDT_PHASE_CLOSURE_BELOW_DENSE_EQUIVALENT_MEMORY_WITH_ACTUAL_RESTORATION_AND_REUSE
```

This is a dense-representation memory-threshold crossing, not a Small Wall
crossing. The identical best classical incremental TT has the same resource
law, so no distinct phase resource or advantage is established.

The next diagnostic places this phase machine behind CATVM custody and
compares the in-place path with an identical compact direct baseline and a
snapshot sham. Dense expansion is not the baseline.

## CATVM custody exposes actual inverse coupling cost

The probe-free incremental carrier now runs behind a Linux same-UID
`AF_UNIX/SOCK_SEQPACKET` service. The protocol-only controller cannot import
the phase backend, intermediate projection and null-carrier paths are denied,
ordinary outputs pass the no-smuggle scan, and the accepted carrier is
actually restored and consumed by a generation-two unrelated program.

Matched direct, isolated-sham, snapshot-sham, and in-place arms use identical
public programs, boundary schemas, fixed packet shapes, and 64,512 logical
protocol bytes. All primary boundaries agree exactly; restored reuse agrees
within `6.003e-8`. The in-place primary uses 18 native coupling applications
versus 9 for each forward-only baseline. This identifies:

```text
ACTUAL_INVERSE_REQUIRES_2X_NATIVE_COUPLING_APPLICATIONS_IN_THIS_BOUNDED_IMPLEMENTATION
```

Evidence `/tmp/catvm-four-rotor-incremental-repaired.KpxZnT` supports the
bounded CATVM claim. It does not establish cross-UID secrecy, seccomp
confinement, leverage, a distinct phase resource, or Small Wall crossing.

## Sector inversion reduces closures but moves cost into rematerialization

Public total-pair-momentum sector LU solves replace 117 inverse incremental
updates with 9 inverse sector closures. Primary restoration is `6.423e-8`,
canonical closure is `6.424e-8`, and generation-two reuse agrees with fresh
execution within `7.381e-8`.

The repair is a negative tradeoff. Exact Grams require 73,167 sector RHS
rematerializations, and wrapper peak payload rises from 10.834 MB to
21.974 MB. Evidence `/tmp/four-rotor-sector-inverse-final.drDP8n` identifies:

```text
EXACT_GRAM_AND_SECTOR_RHS_REMATERIALIZATION_COST
```

No memory, warm-time, advantage, or distinct-resource claim follows.

## An exact cyclic phase law fixes state size across depth

The successor removes sector and Gram reconstruction. A `17^4` resident
finite-torus wave receives onsite and nearest-neighbor phase multipliers
directly in angle coordinates and separable free phases through an
orthonormal Fourier transform. Its inverse conjugates the same public phase
law in reverse step order without retained history.

Across depths `1,2,4,8,16,32,64`, carrier payload remains 1,336,336 bytes,
accounted explicit engine arrays remain 2,009,672 bytes, and wrapper payload
including phase construction, projection arrays, and the non-reloaded
verification baseline remains 3,346,008 bytes. Depth-64 restoration error is
`1.698e-14`. The depth-32 primary restores at `7.671e-15`; unrelated
generation-two reuse restores at `2.244e-15` and agrees with fresh execution
within `6.661e-15`.

Evidence `/tmp/four-rotor-cyclic-phase-law-repaired.HgR1j4` supports:

```text
BOUNDED_DENSE_FINITE_TORUS_CYCLIC_PHASE_UPDATE_LAW_WITH_DEPTH_INDEPENDENT_EXPLICIT_NUMPY_ARRAY_PAYLOAD_ACTUAL_RESTORATION_AND_REUSE
```

This is fixed-depth-growth explicit array payload for a fixed dense
four-rotor grid, not compact growth in width; PocketFFT internal workspace is
outside the bound. The matched direct classical cyclic FFT is identical.
This path bypasses rather than solves the compact TT sector obstruction. The
active obstruction is compact factorization of this phase law across rotor
count or grid width without restoring Bessel/Gram costs or reducing
immediately to an equivalent compact classical recurrence.

## Generic cyclic phase TT saturates the dense central interface

The exact cyclic law was applied through native two-site phase multiplication
and SVD closure in a four-site TT. Central boundary ranks at depths
`1,2,3,4,5` are `17,92,201,280,289`; the final value saturates the exact
`17^2` central cut. Logical TT storage reaches 167,620 cells, `2.007x` the
83,521-cell dense wave, and an 83,521-cell dense-equivalent two-site core is
materialized from depth two.

Shared boundaries agree with dense execution within `5.792e-15`. The actual
inverse is hard-gated before closure at `1.041e-9`; baseline-free closure
restores ranks `1,1,1` within `1.141e-9`. The actual restored carrier
executes an unrelated generation-two program, with fresh/restored boundary
error `4.338e-15`. Evidence
`/tmp/four-rotor-compact-cyclic-tt-repaired.zELuOK` supports:

```text
BOUNDED_CYCLIC_PHASE_TT_NATIVE_PAIR_CLOSURE_SATURATES_GRID17_CENTRAL_INTERFACE_AT_DEPTH5_WITH_ACTUAL_RESTORATION_AND_REUSE
```

Only logical TT and core cells are claimed; simultaneous SVD/workspace peak
payload is open. This closes generic cyclic TT factorization as the repair:

```text
COMPACT_CYCLIC_PHASE_TT_CENTRAL_RANK_SATURATES_DENSE_INTERFACE
```

The selected successor is an exact global-rotation quotient of the cyclic
phase law. It must remove the redundant `U(1)` coordinate by conservation,
preserve dense-reference parity and actual restoration/reuse, and report the
identical matched classical quotient rather than claiming leverage.

## Global rotation removes one phase coordinate exactly

For rotation-invariant programs, the phase carrier now stores three relative
angles instead of four absolute angles. The conjugate free phase derives
`n0 = -(n1+n2+n3) mod 17` from public total-momentum conservation. Direct
pair phases remain native in the relative coordinates.

Resident state falls exactly from 83,521 to 4,913 complex cells, a factor of
17. The fixed signature over depths `1,2,4,8,16,32,64` includes 78,608
carrier bytes, 78,880 retained plan bytes, 157,760 bytes at plan compilation,
236,368 explicit engine bytes, and 314,976 wrapper bytes including the
non-reloaded verification baseline.

The quotient's full lift agrees with independent dense execution within
`2.048e-14` through depth 64. Primary restoration is `8.614e-15`; unrelated
generation-two reuse restores at `2.310e-15`, and fresh/restored boundaries
agree within `7.994e-15`. Evidence
`/tmp/four-rotor-rotation-quotient-repaired.EGpI53` supports:

```text
BOUNDED_EXACT_GLOBAL_ROTATION_QUOTIENT_CYCLIC_PHASE_CARRIER_REDUCES_FOUR_ROTOR_STATE_BY_GRID_FACTOR_WITH_DEPTH_INDEPENDENT_MEMORY_ACTUAL_RESTORATION_AND_REUSE
```

The quotient is structural phase progress, but the identical classical
quotient has the same law and the remaining state grows as
`N^(rotors-1)`. No advantage, distinct phase resource, or Small Wall crossing
is established. The next mechanism must attack that relative-coordinate
growth or introduce a useful phase resource that the matched compact
classical method cannot immediately inherit.

## Total-momentum coordinate streaming removes the dense free plan

The quotient free update now derives
`n0 = -(n1+n2+n3) mod 17` one 17-cell momentum slice at a time instead of
retaining a `17^3` complex free-phase table. Across depths
`1,2,4,8,16,32,64`, its full lifted state remains within `2.048e-14` of the
independent dense execution and its resource signature is depth-independent.

The retained public plan falls from 78,880 to 408 bytes, a `193.333x`
reduction. Maximum explicit engine arrays fall from 236,368 to 118,592 bytes
and wrapper arrays from 314,976 to 197,200 bytes. The actual depth-32 carrier
restores within `8.614e-15`; unrelated generation-two reuse restores within
`2.310e-15` and agrees with fresh execution within `7.994e-15`.

Evidence `/tmp/four-rotor-streamed-momentum-coordinate-final.oba4A3`
supports:

```text
BOUNDED_TOPOLOGY_STREAMED_TOTAL_MOMENTUM_COORDINATE_PHASE_CLOSURE_ELIMINATES_DENSE_QUOTIENT_FREE_PLAN_WITH_ACTUAL_RESTORATION_AND_REUSE
```

The materialized `n0` slice is an unprojected topology-derived classical
coordinate, not a hidden or unresolved phase-resident port. The carrier
remains 4,913 complex cells, execution performs 18,496 coordinate closures,
and the matched classical streamed quotient is identical. This fixes retained
operator-plan growth but not:

```text
RELATIVE_COORDINATE_EXPONENTIAL_GROWTH_AND_MATCHED_CLASSICAL_QUOTIENT_IDENTITY
```

## Exchange symmetry changes fixed-grid rotor carrier growth to polynomial

For an explicitly exchange-symmetric, global-rotation-invariant non-affine
subfamily, the phase carrier now stores one unresolved cyclotomic amplitude
per cyclic necklace of occupation histograms. At grid 17 and four rotors,
4,845 histograms form 285 necklaces, reducing the prior 4,913-cell labelled
rotation quotient by another factor of `17.239`.

The native collision phase depends quadratically on occupation collisions.
The free law is a circulant quadratic chirp. Both commute with permutation
and global rotation. Each induced free coefficient is streamed from an exact
17-component cyclotomic permanent count; neither the `285^2` operator nor the
83,521-cell labelled wave nor an assignment list is retained.

At fixed grid, Burnside counting gives `O(R^16)` carrier growth. For `R < 17`
the free-orbit formula is `binomial(R+16,16)/17`; the analytic five-rotor
dimension is 1,197. This is a declared symmetric family and does not compress
the preceding open-chain program.

The depth-eight weighted norm error is `2.220e-16`; an independent labelled
verifier agrees within `1.346e-15`. Actual inverse restoration is
`7.457e-15`; unrelated generation-two reuse restores within `8.247e-15` and
agrees with fresh execution within `8.882e-16`.

Evidence `/tmp/four-rotor-necklace-orbit-final.RoJ1jK` supports:

```text
BOUNDED_EXCHANGE_SYMMETRIC_GLOBAL_ROTATION_NECKLACE_PHASE_CARRIER_CHANGES_FIXED_GRID_ROTOR_GROWTH_FROM_EXPONENTIAL_TO_POLYNOMIAL_WITH_STREAMED_EXACT_CYCLOTOMIC_FREE_CLOSURE_ACTUAL_RESTORATION_AND_REUSE
```

The carrier is 4,560 bytes and the maximum explicit engine payload is 19,829
bytes, but the depth-eight lifecycle streams 1,299,600 coefficients by
enumerating 530,236,800 permanent assignment terms. The matched classical
orbit recurrence is identical. The new obstruction is:

```text
STREAMED_NECKLACE_FREE_CLOSURE_QUADRATIC_TRANSITION_WORK_AND_MATCHED_CLASSICAL_ORBIT_IDENTITY
```

## Bosonic Givens closure removes permanent enumeration from the accepted path

The 285-cell resident necklace carrier now expands temporarily into the 4,845
permutation-symmetric occupation coefficients, applies a topology-compiled
136-rotation Givens factorization of the 17-mode chirp, and closes every
global-rotation orbit back to its necklace amplitude.

The accepted path retains no transition operator, materializes no labelled
83,521-cell wave or labelled assignment expansion, and enumerates zero
permanent assignment terms. It does materialize 77,520 bytes of occupation
scratch. This preserves the fixed-grid `O(R^16)` law but temporarily releases
the factor-17 global-rotation quotient.

One-step weighted state parity against the exact-cyclotomic permanent
predecessor is `2.570e-15`; the depth-eight boundary error is `4.663e-15`.
Actual restoration is `8.446e-15`; unrelated generation-two reuse restores
within `9.273e-15` and agrees with fresh execution within `6.162e-15`.

Evidence `/tmp/four-rotor-bosonic-givens-accepted.GMfQks` supports:

```text
BOUNDED_TOPOLOGY_COMPILED_BOSONIC_GIVENS_PHASE_CLOSURE_REPLACES_STREAMED_NECKLACE_TRANSITION_PERMANENTS_WITH_POLYNOMIAL_OCCUPATION_SCRATCH_ACTUAL_RESTORATION_AND_REUSE
```

The predecessor comparison enumerates 530,236,800 permanent terms; the
accepted path performs 15,917,440 heterogeneous polynomial block terms. Their
count ratio is not a total-work claim. The same-process warm lifecycle is
descriptively `2.951x` faster. Maximum explicit engine payload rises from
19,829 to 97,447 bytes.

The identical classical bosonic Givens method inherits the repair, so the
new obstruction is:

```text
POLYNOMIAL_OCCUPATION_SCRATCH_AND_MATCHED_CLASSICAL_BOSONIC_GIVENS_IDENTITY
```

## CATVM enforces custody of the factorized occupation intermediate

The automatically factorized closure now executes behind a non-dumpable
same-UID Linux service. After the diagonal and first 68 of 136 Givens
rotations, the backend retains the actual 4,845-cell occupation state across
the protocol boundary. Projection is denied without a valid boundary, and
the continuation consumes that same resident state before closing to the
285-cell necklace carrier.

Only the final seven-value invariant boundary is released. The backend then
applies the actual inverse to the same carrier allocation, restores within
`8.446e-15`, and the actual restored carrier executes an unrelated
generation-two program and restores within `9.273e-15`. Missing, wrong, and
applicably reordered inverses separate by more than `1.0`. The snapshot sham
reloads a 4,560-byte copy and is explicitly not accepted restoration.

Evidence `/tmp/catvm-bosonic-givens-accepted.j4xxOA` and the focused
independent review support:

```text
BOUNDED_CATVM_ENFORCED_TOPOLOGY_COMPILED_BOSONIC_GIVENS_HIDDEN_OCCUPATION_COMPOSITION_WITH_ACTUAL_INVERSE_RESTORATION_AND_REUSE
```

The ceiling is x86-64 little-endian Linux, same-UID `AF_UNIX`
`SOCK_SEQPACKET`, grid 17, four rotors, depth eight, tested nonzero chirp
schedule, complex128, and software only. The manifest binds the exact executed
binary. An ASan/UBSan rebuild also completes the transaction without a
diagnostic.

The direct arm is service-local and forward-only, not a warm direct-process
baseline. Kernel socket buffers and allocator metadata remain outside the
explicit 102,147-byte service-plus-packet payload bound. The matched classical
bosonic Givens method remains identical. This does not establish leverage,
Small Wall crossing, cross-UID secrecy, a distinct phase resource, physical
execution, or unbounded computation.

The next phase-machine repair is:

```text
SYMMETRY_PRESERVING_NECKLACE_GENERATOR_CLOSURE_WITHOUT_OCCUPATION_EXPANSION
```

It must eliminate the 4,845-cell occupation release without replacing it by
an equivalent retained operator or uncontrolled Krylov workspace.

## A Hermitian generator closes directly on the necklace quotient

The single-particle circulant free unitary now compiles to a 17-by-17
Hermitian logarithm. Its lifted `dGamma(H)` action is streamed directly among
canonical occupation necklaces, and a degree-64 Chebyshev recurrence applies
`exp(+/- i dGamma(H))` with three 285-cell work vectors.

The accepted path materializes zero occupation cells and no 285-by-285
transition operator. One-step state parity against bosonic Givens is
`2.617e-15`; the depth-eight boundary differs by `2.609e-15`. The complete
omitted Chebyshev tail is bounded by `6.697e-41`.

Primary actual-inverse restoration is `1.388e-14`. The actual restored
carrier executes an unrelated generation-two program, restoring within
`1.727e-14` and matching a fresh boundary within `1.277e-14`. Missing, wrong,
and applicably reordered controls remain material.

Evidence `/tmp/four-rotor-necklace-generator-final.P7tXmz` and the focused
independent review support:

```text
BOUNDED_SYMMETRY_PRESERVING_HERMITIAN_NECKLACE_GENERATOR_PHASE_CLOSURE_ELIMINATES_OCCUPATION_EXPANSION_WITH_ACTUAL_RESTORATION_AND_REUSE
```

Explicit engine payload falls from the bosonic Givens predecessor's 97,447
bytes to 33,470 bytes, a `2.912x` reduction. The comparison harness's
4,845-cell occupation scratch is separately disclosed and never attributed
to the accepted path. The generator lifecycle is about `3.36x` slower and
streams 16,868,352 terms, so this is a memory/closure repair rather than a
work advantage.

The matched compact classical Hermitian quotient recurrence is identical.
The next diagnostic is:

```text
MATCHED_COHERENT_DEPHASED_CLASSICAL_NECKLACE_GENERATOR_SMALL_WALL_TRIAD
```

It must test whether coherent interference changes the useful boundary while
retaining the best matched classical complex recurrence as the baseline. No
leverage may be inferred from avoided occupation expansion alone.

## Coherence changes the boundary but the compact classical arm is identical

The coherent necklace generator, an exact initial-and-each-step
necklace-basis dephasing sham, and the best matched compact classical complex
recurrence now execute one bounded resource diagnostic on the same depth-eight
public instance.

Coherent and dephased boundaries separate by `0.0235042`, while probability
normalization remains within `2.220e-15`. This establishes that initial and
interstep coherence affects the boundary. It does not isolate the collision
phase contribution.

The coherent and matched-classical arms are deliberately the same executable
complex recurrence. Their primary and reuse boundaries agree exactly; both
perform actual inverse restoration and actual restored-carrier reuse. The
irreversible dephased arm instead creates and reloads a 4,560-byte snapshot
and is not accepted restoration.

Evidence `/tmp/four-rotor-necklace-coherence-triad-final.rAd8AR` and the
focused independent review support:

```text
BOUNDED_MATCHED_COHERENT_INITIAL_AND_EACH_STEP_NECKLACE_DEPHASED_CLASSICAL_GENERATOR_SMALL_WALL_RESOURCE_DIAGNOSTIC
```

Resource and timing scopes are separated per arm. No timing comparison is
made. The dephased arm streams 265,118,400 permanent terms, so its smaller
probability state is not presented as a cheap matched baseline.

The result rules out dephased probability recurrence as an equivalent
mechanism for this boundary, but it does not distinguish phase-native software
from the identical compact classical complex recurrence. No advantage or
Small Wall crossing is established.

The phase-owned experiment selected at this historical point was:

```text
COHERENCE_DEPENDENT_OPEN_RELATIONAL_CATALYTIC_INFERENCE_ON_NECKLACE_CARRIER
```

It must integrate typed public evidence relations into unresolved phase
composition and preserve final-boundary-only inference, actual restoration,
reuse, and the identical compact-classical comparison.
It was subsequently completed at source head `65be0046`; do not restart it.

## Typed open observations now close into unresolved necklace phase state

Each public signature now defines a phase-valued open relation
`R(x,o) = omega^(strength * (feature(x)-o)^2)` between the unresolved
necklace state and either a collision-count or cyclic-separation observation
port. Six public observations close by native substitution without
enumerating their domains or materializing a relation table.

The closed relations interleave with noncommuting Hermitian necklace
generator updates. The 285 latent amplitudes remain unprojected until the
seven-class collision-hypothesis score is emitted. An independent
hypothesis-by-hypothesis aggregation agrees exactly with that boundary.

Primary actual-inverse restoration is `9.657e-15`. The actual restored carrier
executes an unrelated generation-two program, restores within `1.405e-14`,
and agrees with fresh execution within `8.438e-15`. Bypass, module-order,
dephasing, and inverse controls all separate materially.

Evidence `/tmp/four-rotor-necklace-open-observation-final-v2` and the focused
independent review support:

```text
BOUNDED_COHERENCE_DEPENDENT_TYPED_OPEN_OBSERVATION_RELATION_CLOSURE_AND_CATALYTIC_HYPOTHESIS_SCORING_ON_COMPACT_NECKLACE_PHASE_CARRIER_WITH_ACTUAL_RESTORATION_AND_REUSE
```

The accepted engine payload is 33,638 bytes and the complete explicit
lifecycle is 38,310 bytes. The accepted path materializes no relation table,
truth table, candidate assignments, or observation-domain expansion. The
dephased comparison separately streams 198,838,800 permanent terms.

This is direct-process evidence, so intermediate non-emission is observed but
no-smuggle custody is not enforced here. The score contract is bounded and
is not a Bayesian posterior, ground-truth accuracy result, general catalytic
inference, or learning result.

The identical compact classical complex recurrence reproduces the boundary
exactly. No advantage or Small Wall crossing is established.

## Clean-room four-rotor, necklace, and CATVM gate

Independent oracles reconstructed the immediate bundle from exact source head
`65be0046ae02c79ab8c3b3356ef68d891de19e53` without calling the production
projection. The public typed score matched exactly across alternate
observations, strengths, orders, and a different valid program family.
Global rotation, streamed momentum, the 285-cell exchange-symmetric necklace,
bosonic Givens closure, the Hermitian generator, coherence diagnostic, and
public observation result pass at their registry ceilings.

The original staged bosonic CATVM aggregate is `REJECTED_SOURCE_DEFECT`:
disconnect after `BEGIN` exited without inverse-restoring the borrowed
carrier. Its hidden 4,845-cell occupation custody, denied projection, normal
path response ordering, numerical restoration, and reuse are preserved.
A distinct mode-locked successor restores on disconnect, separates snapshot
and in-place modes, removes content-derived receipts, and passed independent
wrong/missing/reordered inverse attacks:

```text
classification        INDEPENDENTLY_VERIFIED_STRICT_SCOPE
restoration           NUMERICAL_PHYSICAL_STATE_RESTORATION
primary error         8.446068614196007e-15
reuse error           9.27279516256988e-15
fresh/reuse delta     6.161737786669619e-15
```

The generator's strongest compact classical recurrence is identical.
Coherence changes the tested boundary by `0.039045417878374566`, but that
causal dependence is not a resource absent from compact classical software.
Durable evidence is in
`FOUR_ROTOR_NECKLACE_CATVM_CLEANROOM_VERIFICATION.json`.

## Shared unresolved latent port and owner-bound CATVM repair

The resumed successor replaces public scalar substitution with one resident
two-cell coherent latent fiber jointly carried by all 285 necklace cells.
Four feature-controlled `Z/X/Y/X` consumers share it, with Hermitian necklace
updates interleaved. A coupling-only clean-room commutator has weighted state
distance `0.4080881046853783`, so the shared consumers genuinely
noncommute. The 570-complex joint carrier is never projected before the final
seven-bin boundary, and no relation table, assignment list, occupation
expansion, or dense 285-by-285 operator is materialized.

The first implementation's full custody claim is:

```text
REJECTED_SOURCE_DEFECT
```

`LatentModule.owner` was merely required to be nonzero. Four distinct wrong
nonzero owners were accepted with zero boundary and restored-state delta.
Outer lease/generation custody, hidden-stage behavior, numerical restoration,
disconnect cleanup, reuse, and the shared-carrier mechanics remain valid
subclaims.

A distinct repair checks all four primary and three reuse consumers against
static port owner `0x4c415431` before a carrier operation. CATVM separately
enforces the exact nonce-derived outer lease and restoration generation.
Independent attacks before and during residency denied wrong module owner,
outer lease, and generation with no boundary. Attacked and clean final and
reuse boundaries agree exactly.

```text
OWNER_BOUND_COHERENT_SHARED_LATENT_OBSERVATION_PORT_PHASE_CONTRACTION_ON_NECKLACE_CARRIER
CATVM_ENFORCED_OWNER_BOUND_COHERENT_SHARED_LATENT_OBSERVATION_PORT_PHASE_CONTRACTION_ON_NECKLACE_CARRIER
classification        INDEPENDENTLY_VERIFIED_STRICT_SCOPE
restoration           NUMERICAL_PHYSICAL_STATE_RESTORATION
primary error         6.743856375997441e-15
reuse error           1.0937531519571031e-14
fresh/reuse delta     5.9396931817445875e-15
```

The exact ceiling is Linux x86-64, same UID, one accepted Unix seqpacket
connection, fixed grid-17/four exchange-symmetric rotation-invariant rotors,
285 necklaces, 570 complex cells, a fixed four-module primary and
three-module reuse program, static owner binding, seven-bin boundary, and
complex128 software. The static tag is not cryptographic or a dynamic
program-owner interface.

The strongest compact classical implementation remains the identical
570-complex recurrence. No distinct phase resource, advantage, Small Wall
crossing, general catalytic inference, physical waveform execution, physical
bit replacement, or unbounded computation is established. The exact next
obstruction is:

```text
PHASE_NATIVE_RESOURCE_BEYOND_IDENTICAL_FIXED_TWO_CELL_570_COMPLEX_RECURRENCE
```

## Topology-rematerialized shared-latent depth

The public phase program is now compiled from `(variant, ordinal)` descriptors
instead of retaining a module tape. The inverse reconstructs the same
descriptors in reverse. Independent conformance checks cover direct depths
`1, 2, 4, 8, 16, 32` and an unrelated depth-11 reuse on the same restored
carrier.

```text
TOPOLOGY_REMATERIALIZED_OWNER_BOUND_SHARED_LATENT_PHASE_PROGRAM_FIXED_570_CARRIER_ACROSS_INCREASING_DEPTH
classification                         INDEPENDENTLY_VERIFIED_STRICT_SCOPE
restoration                            NUMERICAL_PHYSICAL_STATE_RESTORATION
fixed catalytic carrier                570 complex cells
retained module tape                   0 bytes
retained inverse history               0 bytes
maximum restoration error              9.351601280794374e-14
reuse restoration error                1.0927586480356206e-13
fresh/restored reuse boundary error    9.009459844833145e-14
```

The first CATVM depth service remains
`REJECTED_SOURCE_DEFECT`. Resident descriptor errors did not return the
resident state flag, rejected repeated initialization changed the accepted
nonce, and resident `STOP` returned before numerical inverse cleanup. Its
valid `CONTINUE` ordering, disconnect cleanup, hidden-stage behavior, and
same-backing reuse are preserved subclaims.

The distinct atomic-dispatch repair corrects those state transitions. As a
complete package it is also `REJECTED_SOURCE_DEFECT`, because the reported
primary and reuse peak totals omitted the live 289-complex generator matrix.
Its atomic response-ordering subclaim remains independently valid at the
recorded ceiling.

The separated accounting successor is:

```text
CATVM_ATOMIC_DISPATCH_AND_RESOURCE_ACCOUNTING_REPAIRED_TOPOLOGY_REMATERIALIZED_OWNER_BOUND_SHARED_LATENT_PHASE_PROGRAM_FIXED_570_CARRIER_AT_DEPTH32
classification                         INDEPENDENTLY_VERIFIED_STRICT_SCOPE
restoration                            NUMERICAL_PHYSICAL_STATE_RESTORATION
primary restoration error              4.951942189030858e-14
reuse restoration error                6.545926890278212e-14
fresh/restored reuse boundary error    4.829470157119431e-14
restoration generations                1, 2
baseline reload bytes                  0
accepted primary plus reuse protocol   740 bytes
```

The counted primary and reuse peaks are 2,569 and 3,139 complex cells
excluding persistent plan roots, or 2,586 and 3,156 including those roots.
They are enumerated complex-buffer counts rather than whole-process RSS
bounds. Allocator, native-library, operating-system, socket, controller, and
binary memory are not bounded by this result.

The strongest compact classical implementation is the identical 570-complex
recurrence and agrees with zero boundary error. Fixed logical cell count
therefore does not by itself provide a distinct phase resource. The next
phase-owned experiment should measure exact phase precision or another
resource that the logical cell count may conceal, while retaining the same
matched recurrence, restoration, and reuse requirements.

## Exact dyadic-cyclotomic phase precision

A distinct direct-process diagnostic replaces complex128 values with
canonical elements of `Z[ZETA8,1/2]`. It retains the public 285
necklace-descriptor set and two-cell latent fiber, while using public
necklace-index matchings rather than the prior Hermitian necklace generator.
The inverse rematerializes each `(variant, ordinal)` descriptor in reverse.

```text
BOUNDED_EXACT_DYADIC_CYCLOTOMIC_PHASE_PRECISION_GROWTH_ON_FIXED_570_CELL_NECKLACE_LATENT_CARRIER_WITH_EXACT_RESTORATION_AND_REUSE
classification                   INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification level               INDEPENDENT_ORACLE_REEXECUTION
restoration                      EXACT_ALGEBRAIC_RESTORATION
tested depths                    1, 2, 4, 8, 16, 32, 64
reuse depth                      23
logical phase cells              570
retained module tape             0 bytes
retained inverse history         0 bytes
baseline reload                  0 bytes
```

The depth measurements are:

```text
depth                         1      2      4      8      16      32      64
maximum numerator bits       1      1      1      4       7      15      31
maximum denominator power    1      2      3      5       9      17      33
logical payload bits      2,854  2,864  2,898  3,188   4,339  14,645  57,166
```

Both the depth-64 primary and unrelated depth-23 reuse restore the canonical
carrier exactly, and fresh/restored reuse boundaries agree. Backing identity
is claimed only for the outer vector of 570 `Phase` objects; dynamic
big-integer limb allocations are not stable or claimed.

The independent oracle reconstructs the 4,845 histograms and 285 necklaces
without importing the production backend. It matches all boundaries,
restorations, and forward norms over `F17` and `F41`. A separate
integer-arithmetic reexecution reproduces every reported precision tuple and
the reuse result. That reexecution is now durable in
`four_rotor_necklace_exact_phase_precision_integer_oracle.py`; the qualifier
executes it and byte-compares its result with
`EXACT_PHASE_PRECISION_INTEGER_ORACLE_RESULTS.json`.

The baseline payload is 2,851 bits, so the fixed object count conceals growing
exact coefficient width. Reported payload excludes big-integer
allocator/container overhead, temporary lifetime peak, whole-process RSS, and
bit-operation complexity. The strongest compact classical implementation is
the identical exact 570-cell recurrence.

This result establishes neither CATVM custody nor a distinct phase resource,
computational advantage, Small Wall crossing, physical waveform execution,
physical bit replacement, or unbounded computation. The exact next
obstruction is:

```text
FIXED_570_LOGICAL_PHASE_CELLS_HIDE_DEPTH_GROWING_EXACT_COEFFICIENT_WIDTH_AND_THE_IDENTICAL_CLASSICAL_RECURRENCE
```

## Fixed-width cyclotomic star quotient

A distinct exact diagnostic maps every one of the 570 logical phase cells
into four conjugate-paired residues over `F17 x F17 x F41 x F41`. It retains
the public 285 necklace descriptors, rematerializes public
`(variant, ordinal)` operations for the inverse, and preserves the declared
star norm exactly.

```text
BOUNDED_DUAL_PRIME_CONJUGATE_PAIR_CYCLOTOMIC_STAR_QUOTIENT_FIXED_570_CELL_PHASE_CLOSURE_ACROSS_DEPTH4096_WITH_EXACT_RESTORATION_AND_REUSE
classification                     INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification level                 INDEPENDENT_ORACLE_REEXECUTION
restoration                        EXACT_ALGEBRAIC_RESTORATION
tested depths                      1, 64, 256, 1024, 4096
reuse depth                        1537
logical phase cells                570
residues per phase cell            4
carrier inline bytes               4,560
restoration baseline inline bytes  4,560
declared payload subtotal bytes    19,492
retained module tape               0 bytes
retained inverse history           0 bytes
baseline reload                    0 bytes
```

The depth-4096 primary and unrelated depth-1537 reuse both restore exactly in
the quotient on the same outer carrier backing. Fresh/restored reuse
boundaries agree and restoration generations are `1, 2`. Wrong, missing, and
reordered inverses fail restoration; phase-disabled, topology-perturbed, and
null-carrier paths change the boundary. An independent implementation matches
the four-residue boundaries and exact Fraction bridges through depth 64
without importing the production backend.

The fixed width is not lossless analytic compression. The nonzero integer 697
maps to zero in every declared embedding. This establishes one kernel element,
not a complete kernel generator. The declared 19,492-byte subtotal excludes
allocator and heap metadata, the verification harness, binaries, and
whole-process RSS.

The strongest compact classical implementation is the identical four-residue
recurrence with zero boundary error. No CATVM custody, distinct phase resource,
computational advantage, Small Wall crossing, physical waveform execution,
physical bit replacement, or unbounded computation is established. The next
obstruction is:

```text
FIXED_WIDTH_CYCLOTOMIC_STAR_QUOTIENT_DISCARDS_NONZERO_ANALYTIC_KERNEL_697_AND_REMAINS_IDENTICAL_FINITE_FIELD_RECURRENCE
```

## Exact H-T analytic projective-orbit obstruction

The branch-native exact gates `H` and `T = diag(1, ZETA8)` already give a
transferable obstruction to a fixed finite-state lossless repair. For
`U = H T`, if `q` is the eigenvalue ratio, exact algebra gives:

```text
q + q^-1 = trace(U)^2 / det(U) - 2 = -1 - sqrt(2)/2
```

This value is not in the quadratic integer ring `Z[sqrt(2)]`. Since a root of
unity would make `q + q^-1` an algebraic integer, `q` is not a root of unity.
The initial basis vector is cyclic because `det[e0, Ue0] = 1/sqrt(2)` is
nonzero. Consequently the complete analytic projective orbit
`(H T)^n [e0]` is infinite, and no fixed finite state set can encode that
complete orbit injectively.

The theorem was independently classified
`INDEPENDENTLY_VERIFIED_TRANSFERABLE`. The executor evidence has the narrower
ceiling:

```text
BOUNDED_EXACT_HT_ANALYTIC_INFINITE_PROJECTIVE_ORBIT_REJECTS_FIXED_FINITE_STATE_LOSSLESS_PHASE_QUOTIENT_WITH_EXACT_RESTORATION_AND_REUSE
execution ceiling                 Linux x86_64, depths 1..64
logical phase cells               2
depth-64 maximum numerator bits   17
depth-64 maximum denominator      17
depth-64 logical payload bits     141
reuse                             H then T^3, depth 23
restoration                       EXACT_ALGEBRAIC_RESTORATION
```

The exact primary and unrelated reuse programs restore on the same outer
backing with generations `1, 2` and no baseline reload. Missing, wrong, and
reordered inverse controls fail. The independent Fraction implementation
reproduces every boundary, norm, and coefficient-height tuple. The strongest
compact classical method is the identical exact two-cell recurrence.

The illustrative values 1 and 698 agree in the four `F17/F41` embeddings but
are not a normalized transaction and are not used to prove the theorem.
The theorem does not rule out unbounded symbolic state, fixed-dimensional
continuous state with unbounded precision, controlled approximation,
lossy/output-specific quotients, other gate pairs, or other phase mechanisms.
It establishes no unbounded computation, CATVM custody, distinct phase
resource, computational advantage, Small Wall crossing, physical waveform
execution, or physical bit replacement.

The next obstruction is:

```text
LOSSLESS_INFINITE_HT_PROJECTIVE_ORBIT_REQUIRES_UNBOUNDED_SYMBOLIC_INFORMATION_OR_NONFINITE_CONTINUOUS_STATE_WHILE_THE_EXACT_COMPACT_CLASSICAL_RECURRENCE_REMAINS_IDENTICAL
```

## Stateful Pancharatnam transport retains public-endpoint-invisible phase

The next mechanism uses a resident U(1) fiber coordinate over public
Bloch-sphere paths. It does not select a canonical phase independently at
each base point. Instead, every edge chooses the target phase that makes its
overlap with the current carrier positive real. This stateful Pancharatnam
transport preserves path-dependent holonomy when a public closed path returns
to the same Bloch endpoint.

```text
BOUNDED_STATEFUL_PANCHARATNAM_GAUGE_TRANSPORT_ENDPOINT_INVISIBLE_HOLONOMY_PHASE_MEMORY_WITH_NUMERICAL_RESTORATION_AND_REUSE
classification                    INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification                      INDEPENDENT_ORACLE_REEXECUTION
restoration                       NUMERICAL_PHYSICAL_STATE_RESTORATION
claim ceiling                     Linux x86_64 direct process, complex128
carrier                           two complex cells
tested latitude segments          4, 8, 16, 32, 64, 128, 256, 512
primary restoration error         8.533e-16
reuse restoration error           4.722e-16
100-cycle maximum restoration     8.699e-16
predeclared tolerance             1e-10
retained edge history             0 bytes
```

The inverse regenerates the public vertices in reverse and restores the same
carrier variable before an unrelated 37-repetition spherical-rectangle reuse
program. Fresh and restored reuse boundaries differ by `1.355e-15`, and no
baseline reload occurs. Restoration generations `1, 2` are descriptive
bookkeeping rather than lease-enforced state.

The missing- and wrong-inverse controls fail. Reordering is inapplicable for
this commutative scalar U(1) update. Premature canonicalization removes the
holonomy; reversing path orientation conjugates it; changing the enclosed
area changes it; and a null carrier is rejected. “Endpoint-invisible” is
strictly limited to the identical public Bloch endpoint, not the full fiber
carrier or final overlap.

An independent 80-digit implementation reproduces the discrete latitude
formula, its decreasing continuous-limit error, both accepted boundaries,
unit norms, and restorations below `1e-70` at oracle precision. The compiled
package is deterministic and passes its undefined-behavior build.

The resource record counts inline payloads and public descriptors, not stack
temporaries, code, libm, allocator state, process RSS, or timing. The
identical one-double gauge-angle recurrence is the compact matched recurrence
used here; minimality is not claimed, and these fixed public paths also admit
closed-form product evaluation. Therefore no runtime advantage, distinct
phase resource, CATVM custody, Small Wall crossing, physical execution,
physical bit replacement, or unbounded computation is established.

The next obstruction is:

```text
PANCHARATNAM_HOLONOMY_IS_PUBLIC_ENDPOINT_INVISIBLE_AND_RESIDENT_BUT_FIXED_PUBLIC_PATHS_ADMIT_AN_EQUIVALENT_OR_CLOSED_FORM_COMPACT_CLASSICAL_PHASE_RECURRENCE
```

## A shared non-Abelian dark frame makes inverse order material

A two-column dark frame over a public bright ray in `C^3` generalizes the
scalar gauge carrier. Public `phi1` and `phi2` loops transport the actual
resident frame by a unitary polar correction. The transported edge overlap
is positive Hermitian, and the two closed-loop holonomies do not commute.

```text
BOUNDED_NONABELIAN_WILCZEK_ZEE_SHARED_PHASE_FRAME_NONCOMMUTING_HOLONOMY_COMPOSITION_WITH_NUMERICAL_RESTORATION_AND_REUSE
classification                    INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification                      INDEPENDENT_ORACLE_REEXECUTION
restoration                       NUMERICAL_PHYSICAL_STATE_RESTORATION
claim ceiling                     Linux x86_64 direct process, complex128
public chart                      alpha=1.0, beta=0.7
loop segments                     512
loop-order difference             1.9817209383 Frobenius
primary restoration error         4.158e-15
reuse restoration error           4.687e-15
100-cycle maximum restoration     1.684e-13
predeclared tolerance             1e-9
retained edge history             0 bytes
```

The primary public word applies `phi1` and then `phi2`; reversing the word
changes the 2x2 final boundary. The actual inverse must traverse `phi2`
backward before `phi1`. Reversing that inverse order fails restoration, so
the earlier U(1) applicability exception no longer applies. Wrong, missing,
and null-carrier controls also fail. The same restored carrier variable then
runs the unrelated word `phi1^-1, phi2, phi1`, matching a fresh carrier
within `5.434e-15`.

Independent SVD-polar reconstruction confirms bright/dark orthogonality,
positive-Hermitian edge overlap, noncommutation, correct inverse order, and
reuse without relying on the production inverse-square-root formula. An
80-digit oracle independently reproduces the finite-edge products and
restorations. The finite product is explicitly distinct from its continuous
connection limit.

Resource accounting remains scoped. Six complex128 carrier cells occupy 96
inline bytes, and the boundary matrix occupies 64 bytes. The 448-byte
per-edge figure is a named-object subtotal, not a peak; the optimized
transport function alone has a measured 784-byte stack frame. Complete caller
stack, code, libm, allocator, whole-process RSS, and timing remain unbounded.

The identical 2x2 matrix recurrence reproduces the accepted boundary, and the
fixed public loop modules also have closed-form products. The matrix state is
not claimed minimal. No generic CP2 path compiler, CATVM custody, distinct
phase resource, advantage, Small Wall crossing, physical execution, physical
bit replacement, or unbounded computation is established.

The next obstruction is:

```text
NONABELIAN_SHARED_PHASE_FRAME_ESTABLISHES_ORDER_SENSITIVE_NATIVE_HOLONOMY_BUT_AN_IDENTICAL_2X2_MATRIX_OR_CLOSED_FORM_CLASSICAL_RECURRENCE_REMAINS
```

## Root-locked phase-VM operation trace has a smaller Q3 bisimulation

The branch-native complex twin-rail phase VM was compared after every
operation with a separate uint8 Q3 transition system. The accepted diagnostic
covers all 243 input states on nine canonical legal wiring variants:
ROT residues 0, 1, and 2; ADD; MULADD with distinct and equal inputs; SWAP;
CSWAP; and PCSWAP. It does not claim every register placement.

```text
BOUNDED_ROOT_LOCKED_PHASE_VM_OPERATION_TRACE_CLASSICAL_BISIMULATION_WITH_NUMERICAL_RESTORATION_AND_REUSE
classification                    INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification                      INDEPENDENT_ORACLE_REEXECUTION
restoration                       NUMERICAL_PHYSICAL_STATE_RESTORATION
operation cases                   2,187
forward/inverse local checkpoints 4,374
total checkpoints                 4,414
compared relation cells           22,190
semantic trace FNV                f752af9c339230e9
maximum root distance             4.006e-14
maximum restoration error         2.616e-13
fresh/restored native boundary    6.642e-14
predeclared tolerance             2e-11
```

The primary 12-step chain restores the actual complex carrier before an
unrelated eight-step program reuses the same backing. Fresh and restored
native relation boundaries agree within tolerance, restoration generations
are `1, 2`, and no baseline reload occurs. Missing, wrong, and reordered
inverse controls fail.

The independent 80-digit oracle reconstructs the root interpolation formulas,
the Boolean-one indicator, both conditional-swap operations, both final
boundaries, and the symbolic trace without importing production code. The
FNV value is a deterministic replay commitment rather than
collision-resistant proof.

For the accepted semantic comparison, the native carrier uses two complex128
rails, or 32 heap bytes per register, while the matched implementation stores
one uint8 Q3 symbol per register and may pack each symbol into two bits. The
160- and 256-byte native figures are one-carrier payloads, not total verifier
memory. The verifier simultaneously holds 320 native heap bytes in an
exhaustive case and 768 native heap bytes in a chained case; symbolic buffers,
stack, code, allocator, and whole-process peak are not bounded.

This closes one ambiguity in the software lane. Any closed finite
deterministic software phase backend is classically identity-simulable by
storing the same machine state and applying the same deterministic
transition. The root-locked VM has the stronger result that a smaller
symbolic state suffices. This statement does not cover physical analog
resources, external oracles, nondeterministic resources, or
unbounded-precision models.

No CATVM custody, distinct phase resource, computational advantage, Small
Wall crossing, physical waveform execution, physical bit replacement, or
unbounded computation is established. The next obstruction is:

```text
FINITE_DETERMINISTIC_SOFTWARE_PHASE_BACKENDS_REMAIN_CLASSICALLY_IDENTITY_SIMULABLE_AND_THE_ROOT_LOCKED_VM_ADMITS_A_STRICTLY_SMALLER_Q3_RECURRENCE
```

## Two shared latent ports with one joint four-cell fiber

Two separately typed binary latent ports now inhabit one four-complex fiber
on each of the 285 exchange-symmetric rotation necklaces. The fixed primary
program contains four local and two controlled-phase joint modules. Both
ports remain resident and unprojected until the final seven-bin norm
boundary.

```text
OWNER_BOUND_TWO_SHARED_LATENT_PORT_JOINT_PHASE_CONTRACTION_ON_NECKLACE_CARRIER
CATVM_ENFORCED_OWNER_BOUND_TWO_SHARED_LATENT_PORT_JOINT_PHASE_CONTRACTION_ON_NECKLACE_CARRIER
classification                     INDEPENDENTLY_VERIFIED_SOURCE_LOCAL
verification level                 CLEANROOM_ADVERSARIAL_VERIFICATION
restoration                        NUMERICAL_PHYSICAL_STATE_RESTORATION
resident complex cells             1,140
primary / reuse modules            6 / 4
primary restoration error          8.6226248845102124e-15
reuse restoration error            1.7019059468819284e-14
fresh/restored reuse boundary      8.4376949871511897e-15
identity joint boundary effect     0.004308395186971431
separable joint boundary effect    0.0095970301579075268
```

The direct product fiber has zero determinant within `9.48e-22`. One joint
controlled phase raises the maximum per-necklace determinant to
`5.758e-6`, and its inverse restores within `2.889e-17`. An independent
small algebra oracle confirms nonfactorability, local/joint noncommutation,
and inverse restoration without importing the production recurrence.
Identity and one explicitly factorizable diagonal surrogate retain every
generator stage and still change the final boundary. Strength-only mutations
of each fixed joint consumer separately change that boundary. The controls
do not prove a lower bound against all separable programs.

CATVM binds every fixed primary and reuse module to exact current
`(id,type,owner,generation,lease)` records for both ports. It rejects stale
generation, wrong lease/type/owner, duplicate-port, and undermerge mutations
before carrier work. A staged response is receipt-only. The service releases
the final boundary after inverse execution, restoration verification, and
atomic generation advancement. Disconnect and authorized staged STOP restore
before exit or acknowledgement; denied staged STOP leaves the transaction
live. The actual restored backing runs the unrelated reuse program at
generation two with no baseline reload.

Resource figures cover declared complex-vector payloads and wire ABI only:
18,240 bytes for the carrier, 36,480 bytes for persistent baseline plus
carrier, 18,240 bytes of declared streamed generator/extraction scratch, a
32-byte request, and a 116-byte response. They exclude plan and program
objects, the harness, evidence, allocator, native libraries, OS state, and
whole-process peak.

The compact classical comparison reruns the identical 1,140-complex
recurrence and is classified `PACKAGE_SELF_REVIEW`, not separate reference
parity. No distinct phase resource, computational advantage, general
catalytic inference, Small Wall crossing, physical waveform execution,
physical bit replacement, or unbounded computation is established.

The next obstruction is:

```text
TWO_FULL_TUPLE_BOUND_SHARED_PORTS_AND_NONSEPARABLE_JOINT_CONSUMERS_GENERALIZE_RELATIONAL_GEOMETRY_BUT_THE_FIXED_1140_COMPLEX_RECURRENCE_REMAINS_AN_IDENTICAL_COMPACT_CLASSICAL_SOFTWARE_MODEL
```

## Bounded descriptor-compiled two-port program families

The two-port CATVM now compiles public flat program descriptors instead of
accepting only fixed source arrays. Three sealed slots execute six-, five-,
and seven-module families on the same restored 1,140-complex backing.

```text
CATVM_ENFORCED_BOUNDED_PUBLIC_DESCRIPTOR_COMPILED_TWO_SHARED_LATENT_PORT_PROGRAM_FAMILIES_WITH_FULL_TUPLE_CUSTODY_RESTORATION_AND_REUSE
classification                     INDEPENDENTLY_VERIFIED_SOURCE_LOCAL
verification level                 CLEANROOM_ADVERSARIAL_VERIFICATION
restoration                        NUMERICAL_PHYSICAL_STATE_RESTORATION
maximum restoration error          3.04574225268428e-14
maximum fresh/restored boundary    1.7541523789077473e-14
restoration generations            1, 2, 3
```

The compiler has no carrier or answer input. It derives owners, validates
both local port consumers and two joint consumers, derives the stage cut from
the first joint consumer, seals a forward-only topology, and derives inverse
order by reverse traversal. Every active consumer holds current full custody
tuples plus sealed program identity. Wrong tuple, stale program, malformed
descriptor, premature projection, staged substitution, disconnect, STOP,
null carrier, and inverse controls pass at the recorded ceiling.

The independent oracle covers public topology compilation only. The
identical 1,140-complex recurrence remains the strongest compact classical
method and has only package-local identity comparison here. Resource evidence
counts slot and bound-custody objects and explicitly excludes unmeasured
vector capacity, plan/topology allocation, allocator, native-library, OS, and
total process peak memory.

No generic scheduler, arbitrary program algebra, distinct phase resource,
computational advantage, Small Wall crossing, catalytic inference, physical
waveform execution, physical bit replacement, or unbounded computation is
established. The next obstruction is:

```text
BOUNDED_DESCRIPTOR_COMPILATION_REMOVES_FIXED_PROGRAM_ARRAYS_BUT_FLAT_MAX8_TWO_PORT_PROGRAMS_REMAIN_THE_IDENTICAL_1140_COMPLEX_CLASSICAL_RECURRENCE_AND_DO_NOT_GENERALIZE_PORT_ARITY_OR_GRAPH_TOPOLOGY
```

## Multi-port TT diagnostic finds full bounded numerical ranks

The tensor-factorized successor executes the 285-necklace carrier with two
through six binary latent ports. It applies deterministic public local
X/Y/Z modules and joint controlled-phase modules, restores by reverse public
topology, and runs an unrelated second program in the same logical carrier
container.

```text
claim
    BOUNDED_TENSOR_FACTORIZED_MULTI_PORT_NECKLACE_PHASE_DIAGNOSTIC_FINDS_FULL_TOLERANCE_DEFINED_CANONICAL_MATRICIZATION_RANK_AND_NO_TT_COMPACTION_WITH_RESTORATION_AND_LOGICAL_REUSE

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    SEPARATE_REFERENCE_PARITY

restoration
    INVERSE_PLUS_CANONICAL_NUMERICAL_QUOTIENT
```

The production path does not allocate an assignment-indexed dense tensor.
Nevertheless, its final canonical rank signatures are:

```text
ports 2  [4, 2]
ports 3  [8, 4, 2]
ports 4  [16, 8, 4, 2]
ports 5  [32, 16, 8, 4, 2]
ports 6  [64, 32, 16, 8, 4, 2]
```

A separate direct dense oracle reconstructs the same public programs without
importing the production backend or projection. It independently matches
final boundaries to at most `2.50e-15`, restores to at most `7.55e-15`, and
matches reuse boundaries to at most `5.39e-15`. Every final canonical cut has
full tolerance-defined numerical rank. The minimum retained singular value
at six ports is more than `1.3e10` times the declared cutoff.

Peak TT core cells are:

```text
1,160  2,364  4,900  10,484  23,700
```

The matched explicit dense state cells are:

```text
1,140  2,280  4,560  9,120  18,240
```

Thus this TT representation does not compact the tested family. Its first
bond reaches the full `2**ports` width, and its core payload exceeds the
dense reference in all five cases.

Correct inverse restoration upper bounds remain below `1.40e-13`; missing,
wrong, and reordered inverse controls separate. Reuse preserves the logical
carrier container, not fixed core array backing. Baseline and fresh parity
carrier cells are counted. TT, SVD, and generator temporary figures are
component maxima rather than a whole-process RSS bound.

The identical TT recurrence and separate direct dense recurrence are matched
references. No strongest compact classical method is established.

This result does not establish a universal rank lower bound, fixed-rank
multi-port closure, CATVM custody, distinct phase resource, computational
advantage, Small Wall crossing, catalytic inference, physical waveform
execution, replacement of physical bits with pi, or unbounded computation.

The next selected representation change is:

```text
EXACT_SYMMETRY_ADAPTED_CYCLOTOMIC_OPERATOR_SCHMIDT_QUOTIENT_FOR_MULTI_PORT_NECKLACE_PHASE_CLOSURE
```

The live obstruction is:

```text
GENERIC_NONCOMMUTING_MULTI_PORT_NECKLACE_PROGRAMS_SATURATE_EVERY_TESTED_CANONICAL_CUT_AND_TT_STORAGE_EXCEEDS_DENSE_SO_FIXED_RANK_REQUIRES_A_DIFFERENT_PHASE_ALGEBRA_OR_RESTRICTED_LAW_WITH_AN_INDEPENDENT_COMPACT_CLASSICAL_COMPARISON
```

## Exact F17 Gaussian quotient closes a restricted multi-port phase law

The representation-changing successor stores an exact quadratic Gaussian
phase function as `(A,b,c,sign)` over F17. Symmetric phase shears and
single-port Fourier transforms update that tuple without materializing the
`17**ports` phase vector.

```text
claim
    BOUNDED_EXACT_F17_GAUSSIAN_CYCLOTOMIC_MULTI_PORT_PHASE_QUOTIENT_WITH_NONCOMMUTING_FOURIER_SHEAR_CLOSURE_RESTORATION_AND_REUSE

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    SEPARATE_REFERENCE_PARITY

restoration
    EXACT_ALGEBRAIC_RESTORATION
```

At ports 2, 4, 8, 16, and 32, the resident F17-cell counts are:

```text
7  21  73  273  1,057
```

The corresponding fixed-width coefficient payloads, including the sign bit,
are:

```text
36  106  366  1,366  5,286 bits
```

Both tested program families contain `ports+3` public modules. The accepted
path retains no inverse history and reloads no baseline. Reverse public
modules exactly restore the symbolic coefficient state, preserve the same
quadratic and linear array backing, advance restoration generation, and
permit an unrelated second program on the same carrier.

A separate Python-integer oracle consumes the hashed public descriptors and
independently matches the exact boundary and both restorations at every
tested width. At ports 2 and 4, explicit complex phase vectors of 289 and
83,521 logical cells also match within the predeclared `2e-11` tolerance.
Compiler generation and the production missing, wrong, and reordered
controls are not independently regenerated.

Accounting distinguishes two accepted-path carriers from the third fresh
verification carrier. It includes coefficient payload, machine metadata,
serialized primary and reuse programs, Fourier scratch, projection scratch,
and determinant-search compilation for both programs. Python objects,
NumPy/native internals, allocator, OS, and whole-process peak remain
unbounded.

The identical exact Gaussian coefficient recurrence is a compact classical
implementation. No distinct phase resource, computational advantage, Small
Wall crossing, CATVM custody, necklace integration, non-Gaussian closure,
physical waveform execution, replacement of physical bits with pi, or
unbounded computation is established.

The next selected experiment is:

```text
EXACT_F17_NON_GAUSSIAN_CHARACTER_SUM_QUOTIENT_CLOSURE_DIAGNOSTIC_BEYOND_QUADRATIC_GAUSSIAN_CLASS
```

The live obstruction is:

```text
EXACT_F17_QUADRATIC_GAUSSIAN_PHASE_CLOSURE_HAS_POLYNOMIAL_FIXED_WIDTH_STATE_BUT_IS_IDENTICAL_TO_A_COMPACT_CLASSICAL_RECURRENCE_AND_DOES_NOT_COVER_NON_GAUSSIAN_PHASE_LAWS_OR_NECKLACE_CARRIER_INTEGRATION
```

## Exact one-cubic-latent character-sum quotient closes in a bounded chart

The non-Gaussian successor stores one unresolved cubic latent coordinate in
the exact F17 coefficient state `(A,b,l,c,a,d,e,sign)`. Its third finite
difference is nonzero, while public shears and Fourier transforms close
inside that tuple.

```text
claim
    BOUNDED_EXACT_F17_SINGLE_CUBIC_LATENT_CHARACTER_SUM_MULTI_PORT_PHASE_QUOTIENT_WITH_NATIVE_FOURIER_SHEAR_CLOSURE_RESTORATION_AND_REUSE

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    SEPARATE_REFERENCE_PARITY

restoration
    EXACT_ALGEBRAIC_RESTORATION
```

At ports 2, 4, 8, 16, and 32, resident F17 cells and semantic payload bits are:

```text
ports             2    4    8     16     32
F17 cells        12   28   84    292  1,092
payload bits     61  141  421  1,461  5,461
```

Every primary family has `ports+2` Fourier consumers of the live latent
coupling. The latent remains unresolved during forward composition. The
final boundary uses a fixed 17-value character trace and returns 16 canonical
cyclotomic coefficients.

An independent Python-list oracle matches exact boundaries, restoration, and
reuse at all widths. Dense complex parity covers ports 2 and 4. A distinct
port-2 dense construction retains the latent as an explicit axis through all
forward modules and matches the same boundary before exact reverse reuse.

The exact reverse preserves the same `A`, `b`, and `l` array backing, reaches
generation 2 and lease 3 after unrelated reuse, retains no inverse history,
and reloads no baseline. Missing, wrong, and reordered inverse controls fail
in the production package; compiler generation and those controls remain
source-local.

The package counts both accepted resident carriers, public program
descriptors, compiler search, execution scratch, final histogram and
cyclotomic output scratch, boundary serialization, restoration, and reuse.
Python/NumPy/native allocator and whole-process peaks remain unbounded.

This is a fixed one-latent trace and has an identical compact classical
coefficient recurrence. It does not establish multiple-latent closure,
machine-enforced custody, necklace integration, a distinct phase resource,
advantage, Small Wall crossing, catalytic inference, physical waveform
execution, physical bit replacement, or unbounded computation.

The next selected experiment is:

```text
EXACT_F17_INTERACTING_MULTI_CUBIC_LATENT_SIGNATURE_GROWTH_AND_TRACE_CLOSURE_DIAGNOSTIC
```

The live obstruction is:

```text
SINGLE_CUBIC_LATENT_PHASE_QUOTIENT_CLOSES_IN_O_PORTS_SQUARED_FIXED_WIDTH_STATE_WITH_FIXED17_TRACE_BUT_HAS_AN_IDENTICAL_CLASSICAL_RECURRENCE_AND_MULTIPLE_INTERACTING_LATENTS_MAY_REQUIRE_A_GROWING_INTERACTION_SIGNATURE_OR_17_TO_K_TRACE
```

## Interacting cubic latent variables expose final-trace growth

The next exact F17 diagnostic retains the complete degree-three latent
signature while public Gaussian/shear Fourier modules consume shared latent
couplings:

```text
claim
    BOUNDED_EXACT_F17_INTERACTING_MULTI_CUBIC_LATENT_SIGNATURE_REMAINS_POLYNOMIAL_WHILE_FINAL_CHARACTER_TRACE_GROWS_AS_17_TO_K_WITH_EXACT_RESTORATION_AND_REUSE

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    SEPARATE_REFERENCE_PARITY

restoration
    EXACT_ALGEBRAIC_RESTORATION
```

At fixed public width four, latent counts one through four require 28, 38,
52, and 71 resident F17 cells. The complete cubic signatures contain 1, 4,
10, and 20 coefficients. The generic final trace streams 17, 289, 4,913,
and 83,521 assignments respectively.

The exact resident signature is polynomial at this fixture ceiling. The
accepted projection is not compact in latent count: it streams every
assignment without materializing an assignment table. Accounting now includes
the integer histogram and canonical cyclotomic payloads, projection
temporaries, serialized boundaries, public program descriptors, compiler
mask scans, restoration, and reuse. Python/NumPy/native allocator and
whole-process peaks remain unbounded.

The separate Python-list reference matches exact boundaries, inverse
restoration, and unrelated reuse for every declared case. A dense
ports-two / latents-two oracle retains both latent axes through all public
modules. Production compiler generation and inverse controls remain
package-local.

The reverse sequence restores the exact coefficient state on the same array
backing, retains no inverse history, reloads no baseline, and supports an
unrelated second program. The mixed cubic signature is preinitialized and is
not changed by the public Gaussian modules. An identical compact classical
coefficient recurrence has the same generic final trace.

This result does not establish compact multiple-latent final closure, native
cubic-interaction updates, arbitrary non-Gaussian composition, CATVM custody,
a distinct phase resource, computational advantage, Small Wall crossing,
catalytic inference, physical waveform execution, physical bit replacement,
or unbounded computation.

The next selected phase-machine experiment is:

```text
TOPOLOGY_FACTORIZED_INTERACTING_CUBIC_LATENT_GRAPH_WITH_NATIVE_VARIABLE_ELIMINATION_OR_TRANSFER_CLOSURE_AVOIDING_17_TO_K_FINAL_TRACE
```

The live obstruction is:

```text
INTERACTING_CUBIC_LATENT_SIGNATURE_REMAINS_POLYNOMIAL_AT_FIXED_PUBLIC_WIDTH_BUT_GENERIC_FINAL_CHARACTER_TRACE_GROWS_AS_17_TO_K_AND_THE_IDENTICAL_COMPACT_CLASSICAL_RECURRENCE_HAS_THE_SAME_COST
```

## Exact cubic-chain transfer closes the declared path family

The topology-factorized successor composes exact F17 unary cubic and
nearest-neighbor mixed cubic path factors through a canonical `17 x 16`
integer cyclotomic transfer message.

```text
claim
    BOUNDED_EXACT_F17_TOPOLOGY_FACTORIZED_INTERACTING_CUBIC_LATENT_CHAIN_NATIVE_CYCLOTOMIC_TRANSFER_CLOSURE_REPLACES_17_TO_K_TRACE_WITH_LOGARITHMIC_REVERSIBLE_PEBBLE_STORAGE_EXACT_RESTORATION_AND_REUSE

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    SEPARATE_REFERENCE_PARITY

restoration
    EXACT_ALGEBRAIC_RESTORATION
```

At nodes 2, 3, 5, 9, 17, 33, and 65, the recursive reversible schedule uses
2, 3, 4, 5, 6, 7, and 8 message slots, or 544 through 2,176 logical integer
cells. Retain-all execution would use 544 through 17,680 cells. Forward
transfer applications are 1, 3, 9, 27, 81, 243, and 729.

The accepted path does not execute the global `17**nodes` assignment trace.
It does enumerate fixed local left-value, right-value, and cyclotomic-basis
domains inside each transfer. The final boundary is a 16-coefficient
numerator with public normalization metadata `17^(-nodes/2)`.

The actual recursive inverse rematerializes intermediate messages, clears the
final and every scratch message exactly, and removes the seed. The same
outer carrier, message containers, and row backing are preserved. An
unrelated program then consumes the actual restored carrier without baseline
reload or retained inverse history.

A separate oracle implements a two-message exact dynamic program and
retain-all inverse for every case, plus direct assignment enumeration at
nodes 2 and 3. It does not import or call the production transfer or
projection.

Exact payload width remains the obstruction. Maximum logical message payload
grows from 869 to 216,589 bits, and maximum single-coefficient signed width
grows from 3 to 136 bits. The identical exact two-message classical dynamic
program uses 544 integer cells and less transfer work. Periodic block-transfer
powering is untested, so this is not an established strongest
family-specific baseline.

The result does not establish arbitrary graph topology, fixed-width
arithmetic, constant reversible storage, CATVM custody, a distinct phase
resource, computational advantage, Small Wall crossing, catalytic inference,
physical waveform execution, replacement of physical bits with pi, or
unbounded computation.

The next selected experiment is:

```text
EXACT_CYCLOTOMIC_CONTENT_AND_PROJECTIVE_PHASE_GAUGE_QUOTIENT_ON_TOPOLOGY_FACTORIZED_CUBIC_CHAIN_TRANSFER_MESSAGES
```

It must test exact content or projective phase reduction without moving
answer-bearing scale or precision into hidden metadata, and compare against
the identical two-message dynamic program plus periodic block powering where
applicable.

The live obstruction is:

```text
CHAIN_TRANSFER_REMOVES_17_TO_K_TRACE_BUT_EXACT_INTEGER_PAYLOAD_WIDTH_GROWS_WITH_DEPTH_AND_THE_IDENTICAL_TWO_MESSAGE_CLASSICAL_DYNAMIC_PROGRAM_USES_LESS_STORAGE_AND_WORK_THAN_REVERSIBLE_PEBBLING
```

## Exact adaptive gauge and 17-content reduce, but do not bound, width

The exact successor uses the cyclotomic null relation
`1 + zeta + ... + zeta^16 = 0` to select a minimum-signed-bit omitted-root
chart independently for each latent-value row. It also extracts the maximal
common integer power of 17 from each resident message and retains the
exponent explicitly.

```text
claim
    BOUNDED_EXACT_F17_ADAPTIVE_OMITTED_ROOT_AND_17_CONTENT_CYCLOTOMIC_QUOTIENT_CHAIN_TRANSFER_REDUCES_INTEGER_PAYLOAD_WITH_EXACT_RESTORATION_AND_REUSE_BUT_RETAINS_DEPTH_GROWING_WIDTH

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    SEPARATE_REFERENCE_PARITY

restoration
    EXACT_ALGEBRAIC_RESTORATION
```

At nodes 2, 3, 5, 9, 17, 33, and 65, the unfactored reversible payload
peaks are:

```text
869  1,837  4,653  13,432  35,191  88,872  216,589 bits
```

Adaptive gauge plus exact 17-content reduces them to:

```text
937  1,885  2,851  5,580  12,760  30,652  76,433 bits
```

Maximum coefficient signed widths fall from
`3, 6, 10, 18, 34, 68, 136` to `3, 5, 5, 9, 14, 26, 50`. The resident
content exponents `0, 0, 1, 2, 5, 10, 21` match
`floor((nodes-1)/3)` in the declared cases. Exact `(1-zeta17)` extraction
does not reduce the tested final fixed-basis payloads.

The accepted inverse is exact subtraction on the actual resident message:
coefficients, the integer 17-content exponent, and the offset pivot register
are all uncomputed. Seed release uses the same operation. This repairs a
pre-seal draft that only validated and cleared the target. The same backing
is restored and reused by an unrelated program without baseline reload or
retained inverse history.

A separate fixed-basis exact oracle reconstructs the public descriptors and
matches semantic boundaries, factorized boundaries, maximal 17-content,
adaptive pivot minima, and retain-all restoration for every case. Direct
assignment enumeration additionally matches at nodes 2 and 3. The oracle
does not import or call the production compiler, transfer, gauge selector,
projector, or inverse.

Resource accounting includes the seed and inverse expected messages, pivot
metadata, a 49-cell combined regauge temporary, projection and reconstruction
buffers, content diagnostics, descriptors, restoration, reuse, and a fresh
verification carrier. Python objects, allocator, recursive stack,
growing-integer operations, OS state, and whole-process peak are not bounded.

The identical exact adaptive-gauge/content recurrence is the matched compact
classical method. Dense period-17 block powering is not executed because its
73,984 integer cells and at least 4,624 build transfer-equivalents exceed all
declared single-query streaming cases. The strongest family-specific method
is therefore still unestablished.

The quotient is a substantive exact payload reduction, not a fixed-width
closure. It does not establish CATVM custody, a distinct phase resource,
computational advantage, Small Wall crossing, catalytic inference, physical
waveform execution, physical bit replacement, or unbounded computation.

The next selected experiment is:

```text
EXACT_PERIOD17_BLOCK_KRYLOV_ORBIT_AND_PROJECTIVE_SCALE_ACCOUNTING_ON_ADAPTIVE_F17_CUBIC_CHAIN
```

The live obstruction is:

```text
EXACT_17_CONTENT_AND_ADAPTIVE_GAUGE_REDUCE_THE_TESTED_CHAIN_PAYLOAD_BUT_RESIDUAL_COEFFICIENT_WIDTH_GROWS_AND_THE_IDENTICAL_CLASSICAL_RECURRENCE_INHERITS_EVERY_REDUCTION
```

## Period-17 modular seed images do not yet give an exact quotient

The exact public coefficient schedule repeats every 17 edges. Applying that
block as a black-box map on the fixed 272-coordinate cyclotomic message gives
seed Krylov dimensions 241 for PRIMARY and 256 for REUSE over both `F41` and
`F73`.

```text
claim
    BOUNDED_F17_PERIOD17_CUBIC_CHAIN_BLOCK_MODULAR_SEED_KRYLOV_IMAGES_HAVE_DIMENSIONS241_AND256_WHILE_EXACT_ADAPTIVE_PROJECTIVE_CONTENT_RETAINS_RESIDUAL_WIDTH_GROWTH_WITH_EXACT_RESTORATION_AND_REUSE

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    SEPARATE_REFERENCE_PARITY

restoration
    EXACT_ALGEBRAIC_RESTORATION
```

The ranks are exact over the two tested finite fields and imply rational-rank
lower bounds only. No exact rational dimension, minimal polynomial, or
`Z[zeta17]` dependency has been established.

At periods 1, 2, 4, and 8, PRIMARY adaptive payload is
`3,462, 6,263, 13,205, 26,164` bits and REUSE payload is
`3,629, 6,567, 13,949, 27,689` bits. Maximum residual quotient width grows
from 15 to 99 bits for PRIMARY and 16 to 104 bits for REUSE, even after
extracting exact power-of-17 content. All residual coefficient gcds are one.

At period 4, the actual resident messages are exactly subtracted, the same
backing is restored, and an unrelated program matches fresh execution.
Generation and lease reach two; retained inverse history and baseline reload
are zero.

The separate oracle independently recompiles the public descriptors, uses
fixed-basis integer transfer and reverse-pivot elimination, and matches all
ranks and projective tuples. Explicit logical accounting includes production
metadata and restoration work plus oracle working cells. The modular-oracle
peak is 70,720 field cells; its final fixed/adaptive/semantic coexistence is
816 integer cells. Python object, allocator/native-library, bit-operation,
and whole-process peaks remain unbounded.

The identical exact 272-coordinate block map is the matched compact
classical implementation. This result does not establish a lifted exact
quotient, fixed width, CATVM custody, a distinct phase resource,
computational advantage, Small Wall crossing, catalytic inference, physical
waveform execution, physical bit replacement, or unbounded computation.

The next selected experiment is:

```text
EXACT_Z_ZETA17_LIFT_OF_PERIOD17_MODULAR_KRYLOV_DEPENDENCIES_WITH_RESOURCE_CAPPED_CRT_OR_STRUCTURAL_CERTIFICATE
```

The live obstruction is:

```text
THE_PERIOD17_BLOCK_HAS_STABLE_241_AND256_DIMENSIONAL_MODULAR_SEED_IMAGES_BUT_THE_DEPENDENCIES_ARE_NOT_YET_LIFTED_TO_AN_EXACT_Z_ZETA17_QUOTIENT_AND_EXACT_PROJECTIVE_CONTENT_DOES_NOT_STOP_RESIDUAL_WIDTH_GROWTH
```

## Native cyclotomic-module closure does not reduce scalar-Q order

The eight-prime exact-lift control combined the prior degree-241 and
degree-256 modular dependence coefficients through a 108-bit CRT modulus.
Direct streamed exact `Z[zeta17]` evaluation completed below the declared
16,384-bit vector cap. Both candidate residuals had all 272 cells nonzero,
with maximum signed widths 8,536 and 9,254 bits. The tested CRT candidates
therefore do not lift.

The structural successor compiles each public period-17 block into a
17-by-17 operator over `Q(zeta17)`. Its exact monic degree-17 characteristic
polynomial annihilates the entire operator.

```text
claim
    BOUNDED_EXACT_F17_PERIOD17_CUBIC_CHAIN_NATIVE_CYCLOTOMIC_17_STATE_BLOCK_MODULE_HAS_ORDER_AT_MOST17_CAYLEY_HAMILTON_CLOSURE_WITH_EXACT_RESTORATION_AND_REUSE_BUT_RETAINS_GROWING_INTEGER_WIDTH

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    SEPARATE_REFERENCE_PARITY

restoration
    EXACT_ALGEBRAIC_RESTORATION
```

The closure order is over `K = Q(zeta17)`. It is not a minimal-order result,
a scalar-`Q` order-17 result, or a lift of the previous modular
dependencies. Restriction of scalars expands each `K` coefficient to a
16-by-16 `Q`-linear map.

The production runtime certifies but does not execute the characteristic
recurrence. It applies the dense cyclotomic block under recursive reversible
pebbling. At periods 1, 2, 4, and 8, the carrier occupies
`544, 816, 1,088, 1,360` integer cells. Maximum resident coefficient signed
width is `36, 71, 141, 283` bits for PRIMARY and
`37, 72, 144, 289` bits for REUSE.

At period 8, coefficient-wise subtractive inverse restores the actual
borrowed carrier on the same backing. An unrelated REUSE program consumes
it, matches fresh execution, reaches generation and lease two, and leaves
all messages zero with no inverse history or baseline reload.

The separate tuple-ring oracle recompiles both public descriptors and
operators without production imports. It checks both whole-operator
annihilator identities, all eight boundaries, and exact subtraction. The
resource figures are named component-level logical cells, not exact process
peaks; SymPy internals, coexisting buffers, Python objects, allocator and
native-library overhead, bit-operation cost, and whole-process memory remain
unbounded.

The identical cyclotomic matrix execution and characteristic closure are
available to compact classical software. No distinct phase resource,
computational advantage, Small Wall crossing, CATVM custody, catalytic
inference, physical waveform execution, physical bit replacement, or
unbounded computation is established.

The next selected experiment is:

```text
EXECUTABLE_NATIVE_K_COEFFICIENT_CAYLEY_HAMILTON_RECURRENCE_WITH_FIXED_MESSAGE_WINDOW_RESTORATION_AND_WIDTH_ACCOUNTING
```

The live obstruction is:

```text
THE_NATIVE_Q_ZETA17_COEFFICIENT_ORDER_AT_MOST17_CLOSURE_IS_CERTIFIED_NOT_EXECUTED_DOES_NOT_LIFT_THE_PRIOR_Q_DEPENDENCIES_OR_ESTABLISH_A_SCALAR_Q_ORDER17_RECURRENCE_AND_RETAINS_GROWING_INTEGER_WIDTH_WITH_AN_IDENTICAL_CLASSICAL_RECURRENCE
```

## Executed native-K recurrence fixes slots, not exact bit storage

The degree-17 whole-operator characteristic has zero constant coefficient,
so it factors as `p(x) = x q(x)` with monic degree-16 `q` over
`K = Q(zeta17)`. Production now executes the reduced recurrence:
`A^n seed = A (x^(n-1) mod q)(A) seed` for every tested positive period.

```text
claim
    BOUNDED_EXACT_F17_PERIOD17_CUBIC_CHAIN_EXECUTED_NATIVE_Q_ZETA17_CAYLEY_HAMILTON_RECURRENCE_USES_FIXED_18_RESIDENT_MESSAGE_SLOTS_PLUS16_CYCLOTOMIC_COEFFICIENT_REGISTERS_ACROSS_PERIODS1_4_16_64_WITH_EXACT_RESTORATION_AND_REUSE_BUT_GROWING_INTEGER_WIDTH_AND_IDENTICAL_CLASSICAL_EXECUTION

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    SEPARATE_REFERENCE_PARITY

restoration
    EXACT_ALGEBRAIC_RESTORATION
```

The runtime builds `A seed` through `A^16 seed` once, computes 16
native-`K` coefficients by binary polynomial powering, and contracts them
into the final message. It uses 18 resident message slots and 16
cyclotomic coefficient registers at periods 1, 4, 16, and 64. It then
subtracts the output, coefficients, basis, and seed in reverse order.

Every transaction performs 16 forward and 16 inverse block applications.
The retained basis occupies 4,352 integer cells; seed plus basis occupies
4,624. Both compiled family operators occupy 9,248 integer cells and both
characteristics occupy 576 cells during the full run. There is no separate
inverse-operation log or baseline reload.

At period 64:

```text
                         PRIMARY       REUSE
recurrence payload       2,368,807     2,447,532 bits
direct two-message       1,221,725     1,246,736 bits
maximum signed width         2,266         2,313 bits
nonzero coefficient regs        15            16
```

The independent no-import tuple oracle uses sequential multiplication by
`x` modulo `q`, rather than production binary powering. It matches all
eight recurrence and direct boundaries and independently reproduces the
exact boundary payload, carrier peak payload, coefficient width, nonzero
slot/register, and direct two-message resource tuples.

The oracle also restores message and coefficient backing exactly and runs
PRIMARY then REUSE on the same carrier. Generation and lease reach two,
fresh REUSE agrees, and all state returns to zero.

PRIMARY has zero `q(0)`; REUSE has nonzero `q(0)`. Neither factor has a
certified cyclotomic-unit or integral-inverse status, so an integrally
reversible rolling window is not established. The identical cyclotomic
recurrence is available to compact classical software. No phase resource,
advantage, Small Wall crossing, CATVM custody, physical execution, or
unbounded computation follows.

The next selected experiment is:

```text
EXACT_REVERSIBLE_CYCLOTOMIC_UNIT_HEIGHT_REDUCTION_FOR_NATIVE_K_RECURRENCE_WITH_SCALE_LEDGER_ACCOUNTING
```

The live obstruction is:

```text
THE_EXECUTED_NATIVE_Q_ZETA17_RECURRENCE_FIXES_RESIDENT_SLOT_AND_REGISTER_COUNTS_IN_THE_TESTED_MECHANISM_BUT_EXACT_INTEGER_WIDTH_GROWS_AN_INTEGRALLY_REVERSIBLE_ROLLING_WINDOW_IS_NOT_ESTABLISHED_AND_COMPACT_CLASSICAL_SOFTWARE_EXECUTES_THE_IDENTICAL_RECURRENCE
```

## Cyclotomic-unit normalization reduces payload but does not bound width

The successor represents each resident message as a `Z[zeta17]` vector plus
a seven-integer exponent ledger for the declared cyclotomic units
`u_a = 1 + zeta + ... + zeta^(a-1)`, `a = 2..8`. A deterministic exact
search accepts only unit moves that reduce vector-payload-plus-ledger bits.

```text
claim
    BOUNDED_EXACT_SEVEN_GENERATOR_CYCLOTOMIC_UNIT_GAUGE_REDUCES_LEDGER_INCLUSIVE_CARRIER_PAYLOAD_FOR_TWO_PUBLIC_F17_PERIOD17_RECURRENCE_FAMILIES_ACROSS_PERIODS1_4_16_64_256_WITH_EXACT_VALID_PATH_RESTORATION_AND_REUSE_BUT_DOES_NOT_BOUND_INTEGER_WIDTH

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    SEPARATE_REFERENCE_PARITY

restoration
    EXACT_ALGEBRAIC_RESTORATION
```

All ten declared cases reduce counted carrier payload:

```text
period  family   unnormalized   unit gauge   reduction   max width
1       PRIMARY      1,306,953   1,291,419      15,534         558
1       REUSE        1,332,125   1,292,645      39,480         558
4       PRIMARY      1,335,548   1,319,724      15,824         558
4       REUSE        1,361,285   1,320,981      40,304         558
16      PRIMARY      1,451,089   1,433,338      17,751         558
16      REUSE        1,479,214   1,435,141      44,073         560
64      PRIMARY      2,368,807   2,337,094      31,713       2,228
64      REUSE        2,447,532   2,375,887      71,645       2,242
256     PRIMARY      5,850,470   5,767,292      83,178       8,931
256     REUSE        6,112,042   5,935,644     176,398       8,988
```

At period 256, four calls per family reach the 128-step search cap. The
result therefore does not establish a strict local minimum for every call
or a global optimum. Multiplicative independence of the seven generators is
also not certified.

The valid forward path reconstructs all basis and coefficient scales at the
final contraction. It then rematerializes and subtracts output,
coefficients, basis messages, seed, and ledgers on the original backing.
PRIMARY-to-REUSE restored-carrier reuse reaches generation and lease two
with no inverse log or baseline reload. Wrong and reordered inverses are
detected, but rejected attempts do not provide failure-atomic rollback.

The separate no-import oracle recompiles both public operators, checks the
complete annihilator identities, advances recurrence coefficients
sequentially by `x mod q`, and independently reproduces all ten boundaries,
payloads, widths, search counts, cap hits, restoration, and reuse. Direct
dense boundary parity is executed through period 64; period 256 is supported
by the separate recurrence and annihilator check.

Exact accounting covers declared carrier integers and ledgers, not Python
objects, allocator peak, transient bit-operation peak, or whole-process
storage. The identical recurrence and unit normalizer are available to
compact classical software. No CATVM custody, distinct phase resource,
computational advantage, Small Wall crossing, catalytic inference, physical
waveform execution, physical bit replacement, or unbounded computation is
established.

The next selected experiment is:

```text
EXACT_F17_PERIOD17_BOUNDARY_HEIGHT_LOWER_BOUND_FOR_LOSSLESS_DISCRETE_PHASE_STATE
```

The live obstruction is:

```text
THE_SEVEN_GENERATOR_UNIT_GAUGE_REDUCES_COUNTED_PAYLOAD_BUT_EXACT_INTEGER_WIDTH_STILL_GROWS_EIGHT_PERIOD256_SEARCHES_HIT_THE_DECLARED_CAP_AND_COMPACT_CLASSICAL_SOFTWARE_EXECUTES_THE_IDENTICAL_NORMALIZATION
```

## Exact pi-adic height rejects a fixed finite boundary alphabet

The next diagnostic derives an exact boundary-height certificate over
`Z[zeta17]`, with `pi = 1 - zeta17`.

```text
claim
    BOUNDED_EXACT_F17_PERIOD17_PI_ADIC_BOUNDARY_HEIGHT_CERTIFICATE_PROVES_UNBOUNDED_LOSSLESS_DISCRETE_BOUNDARY_ALPHABET_AND_WORST_CASE_OMEGA_LOG_N_HORIZON_CODE_WIDTH_FOR_TWO_PUBLIC_RECURRENCE_FAMILIES_WITH_EXACT_MESSAGE_PAYLOAD_RESTORATION_AND_REUSE_BUT_IDENTICAL_COMPACT_CLASSICAL_RECURRENCE

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    SEPARATE_REFERENCE_PARITY

restoration
    EXACT_ALGEBRAIC_RESTORATION
```

Both public period-17 cubic path families satisfy the exact lower bound

```text
L(n) = CEIL((272*n + 16)/3)
```

on boundary `pi`-valuation. The production gate checks both complete
characteristic identities, direct exact boundaries through period 17, all
three residue-class induction inequalities, and the normalized recurrence.

The normalized `F17` recurrence has:

```text
family   prefix   cycle length   nonzero outputs   exact density
PRIMARY       3          1,632             1,555       1555/1632
REUSE         0         14,688            13,826       6913/7344
```

Nonzero cycle outputs recur infinitely often, so infinitely many exact
boundaries have valuation exactly `L(n)`. Since `L` is strictly increasing,
their valuations and exact boundaries are distinct.

The no-production-import oracle uses the sealed separate descriptor/ring
kernel, recompiles both public operators, checks the supplied annihilators,
repeats exact `pi` division and induction, and uses Brent instead of Floyd
cycle detection. It reproduces both exact cycles and detects a weakened
induction coefficient and a perturbed normalized recurrence.

The encoding consequence is deliberately narrow. For an injective exact
boundary or valuation encoding through horizon `N`, when the decoder gets no
free period index, the worst-case code width through `N` is
`Omega(log N)`. No pointwise-at-every-period, generic machine-memory,
online-space, indexed-generator, or variable-length impossibility is
established. The `pi` exponent itself has an `O(log N)` ledger, and the
identical compact classical recurrence remains.

Message payloads restore exactly on the same backing at 16 periods, and
unrelated REUSE agrees with fresh execution. Generation and lease advance to
two, so full carrier-object equality and bounded repeated-use metadata are
not claimed. Named component counts are not exact temporary or process
peaks.

The next selected phase-machine experiment is:

```text
EXACT_REVERSIBLE_PI_CONTENT_LEDGER_NORMALIZED_NATIVE_K_RECURRENCE_WITH_RESIDUAL_HEIGHT_ACCOUNTING
```

It must move compulsory `pi` content into a counted reversible exponent
ledger during native recurrence execution and determine whether the exact
residual height remains unbounded. It must preserve exact message-payload
restoration and reuse, count metadata growth, and retain the identical
classical normalized recurrence.

The live obstruction is:

```text
THE_EXACT_BOUNDARY_PI_VALUATION_REJECTS_A_FIXED_FINITE_BOUNDED_WIDTH_EXACT_STATE_SET_BUT_ESTABLISHES_ONLY_A_WORST_CASE_LOGARITHMIC_HORIZON_CODE_WIDTH_WITHOUT_A_FREE_PERIOD_INDEX_THE_PI_EXPONENT_HAS_A_COMPACT_LOGARITHMIC_LEDGER_AND_THE_IDENTICAL_CLASSICAL_RECURRENCE_REMAINS
```

## Pi-content factoring increases resident payload in both tested bases

The executed successor stores each exact message and recurrence coefficient
as a `Z[zeta17]` residual plus its common nonnegative
`pi = 1 - zeta17` exponent. It factors that content during recurrence
arithmetic and rematerializes only the final exact boundary.

```text
claim
    BOUNDED_EXACT_PI_CONTENT_LEDGER_NORMALIZED_NATIVE_K_RECURRENCE_WORSENS_MATCHED_ZETA_BASIS_CARRIER_PAYLOAD_FOR_TWO_PUBLIC_F17_PERIOD17_FAMILIES_ACROSS_PERIODS1_4_16_64_256_WITH_EXACT_PAYLOAD_AND_LEDGER_RESTORATION_AND_REUSE_WHILE_PI_BASIS_VALUES_REMAIN_AUXILIARY_UNMATCHED_DIAGNOSTICS

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    SEPARATE_REFERENCE_PARITY

restoration
    EXACT_ALGEBRAIC_RESTORATION
```

The exact exponent ledger remains small, but the residual height grows. All
ten tested cases use more resident signed-integer payload than the identical
raw recurrence:

```text
period  family   raw zeta basis   zeta residual+ledger   auxiliary pi basis
1       PRIMARY       1,306,953               6,152,594            6,193,436
1       REUSE         1,332,125               6,213,406            6,253,922
4       PRIMARY       1,335,548               6,287,563            6,328,417
4       REUSE         1,361,285               6,349,688            6,390,206
16      PRIMARY       1,451,089               6,827,356            6,868,281
16      REUSE         1,479,214               6,895,056            6,935,581
64      PRIMARY       2,368,807              11,160,253           11,203,120
64      REUSE         2,447,532              11,433,973           11,476,565
256     PRIMARY       5,850,470              27,417,447           27,460,355
256     REUSE         6,112,042              28,372,693           28,415,308
```

At period 256, residual signed width reaches 42,337 bits for PRIMARY and
42,774 for REUSE. The exponent ledger uses 16 signed bits.

The durable exact-integer oracle imports no production implementation. It
independently recompiles both public operators, checks both annihilator
identities, uses sequential `x mod q` coefficient advancement, and matches
all exact decompositions, boundaries, valuations, raw baselines, and
resident payload/width tuples. A nonzero coefficient-ledger mutation changes
the boundary.

The valid inverse subtracts output, coefficients, basis, seed, and ledgers
on the original backing. Payload and ledgers return exactly to zero; the
unrelated REUSE program agrees with fresh execution. Generation and lease
advance to two, so full object equality and bounded metadata are not
claimed.

This is a finite negative diagnostic for one exact normalization. It is not
an all-integral-basis or asymptotic lower bound. The accepted worsening
comparison uses raw and normalized recurrences in the same zeta basis. The
pi-coordinate values are auxiliary only because no raw recurrence was
measured in that basis; the historical cross-basis worsening interpretation
is rejected. Python object overhead, allocator peak, internal multiplication
temporaries, and whole-process storage are not bounded. The identical compact
classical normalizer remains.

The next selected phase-machine experiment is:

```text
EXACT_MULTI_EMBEDDING_CYCLOTOMIC_UNIT_BALANCING_AFTER_PI_CONTENT_FACTORIZATION
```

The live obstruction is:

```text
COMPULSORY_PI_SCALE_HAS_A_COMPACT_LEDGER_BUT_DIVIDING_IT_EXPANDS_MATCHED_ZETA_BASIS_RESIDUAL_HEIGHT_THE_PI_BASIS_VALUES_ARE_AUXILIARY_UNMATCHED_DIAGNOSTICS_AND_THE_IDENTICAL_CLASSICAL_NORMALIZATION_REMAINS
```

## Exact multi-embedding unit balance repairs pi payload but remains above raw

The bounded successor adds a seven-entry cyclotomic-unit ledger after
pi-content factorization and chooses unit moves with the exact field trace
of `a * conjugate(a)`. This is the sum of squared magnitude over all sixteen
embeddings and is compared with exact integers.

```text
claim
    BOUNDED_EXACT_MULTI_EMBEDDING_TRACE_ENERGY_CYCLOTOMIC_UNIT_BALANCING_AFTER_PI_FACTORIZATION_REDUCES_RESIDENT_PI_RESIDUAL_PAYLOAD_FOR_TWO_PUBLIC_F17_PERIOD17_FAMILIES_AT_PERIODS1_AND64_BUT_REMAINS_ABOVE_THE_IDENTICAL_RAW_RECURRENCE_AT_PERIOD64_AND_DECLARED_LIVE_STATE_COUNTS_DUPLICATE_REMATERIALIZATION_WITH_EXACT_RESTORATION_AND_REUSE

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    SEPARATE_REFERENCE_PARITY

restoration
    EXACT_ALGEBRAIC_RESTORATION
```

```text
period family    raw       pi-only     balanced resident   declared live
1      PRIMARY   1,306,953  6,152,594            601,130       1,202,260
1      REUSE     1,332,125  6,213,406            603,011       1,206,022
64     PRIMARY   2,368,807 11,160,253          5,489,878      10,979,756
64     REUSE     2,447,532 11,433,973          5,699,705      11,399,410
```

All four balanced resident and declared-live counts improve on the pi-only
representation. Both period-64 counts remain above the identical raw
recurrence. The declared-live count includes a full public-topology
rematerialization beside the carrier without alias discounts; it is not a
whole-process peak.

The no-production-import oracle recompiles both public operators, verifies
both annihilator identities, uses sequential `x mod q` coefficient
advancement, and reproduces every boundary and forward/inverse resource
tuple. The inverse subtracts payload and both ledgers on the original
backing. PRIMARY-to-REUSE cross-family reuse is verified at period 1;
period-64 cases restore separately.

The greedy search has a 128-step cap. Fifty-three PRIMARY and fifty-eight
REUSE calls hit it at period 64, so local or global optimality is not
established. Multiplicative independence of the unit generators is not
certified. Python objects, allocator behavior, convolution temporaries,
native-library storage, and whole-process peak remain unbounded. The
identical recurrence and balancer remain available to classical software.

The next selected experiment is:

```text
EXACT_LOG_EMBEDDING_UNIT_LATTICE_CLOSEST_VECTOR_BALANCING_WITH_SEARCH_WORK_AND_LEDGER_ACCOUNTING
```

The live obstruction is:

```text
EXACT_TRACE_ENERGY_UNIT_BALANCING_REDUCES_PI_FACTORED_RESIDENT_PAYLOAD_BUT_RESIDENT_AND_DECLARED_LIVE_STATE_REMAIN_ABOVE_THE_RAW_RECURRENCE_AT_PERIOD64_MANY_CALLS_HIT_THE_DECLARED_CAP_AND_THE_IDENTICAL_CLASSICAL_BALANCER_REMAINS
```

## Fixed-precision unit-lattice center: carrier reduction, total-cost obstruction

The bounded successor proposes one global seven-entry unit move for every
nonzero normalization call. It evaluates norm embeddings with a fixed
65,536-bit cosine table and solves for the center in float64. This proposal
is approximate and is not a certified closest-vector solution. Exact integer
trace energy decides whether the proposed carrier update is accepted.

```text
claim
    BOUNDED_65536_BIT_LOG_EMBEDDING_UNIT_LATTICE_CENTER_PROPOSAL_WITH_EXACT_TRACE_ACCEPTANCE_REDUCES_PI_FACTORED_RESIDENT_AND_DUPLICATE_REMATERIALIZATION_LIVE_PAYLOAD_BELOW_THE_IDENTICAL_RAW_RECURRENCE_FOR_TWO_PUBLIC_F17_PERIOD17_FAMILIES_AT_PERIODS1_AND64_WITH_EXACT_RESTORATION_AND_REUSE_BUT_THE_CONSERVATIVE_DECLARED_NAMED_COMPONENT_MAXIMA_SUM_REMAINS_ABOVE_RAW_AND_THE_IDENTICAL_CLASSICAL_EXECUTION_REMAINS

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    SEPARATE_REFERENCE_PARITY

restoration
    EXACT_ALGEBRAIC_RESTORATION
```

```text
period family    raw       pi-only     resident   duplicate-live   named-maxima sum
1      PRIMARY   1,306,953  6,152,594   430,790          861,580        1,701,036
1      REUSE     1,332,125  6,213,406   430,336          860,672        1,701,260
64     PRIMARY   2,368,807 11,160,253   773,662        1,547,324        5,891,350
64     REUSE     2,447,532 11,433,973   782,790        1,565,580        5,942,297
```

Resident and duplicate-rematerialization live payload beat raw in all four
cases. The conservative sum of separately observed named component maxima
does not. It adds the live state, nominal high-precision table mantissas,
float64 matrix and scratch, fixed-precision scalar work, a 256-bit rolling
proposal commitment, and exact temporary maxima. This is not a simultaneous
process peak; Python, allocator, native-library, and whole-process storage
remain unbounded.

The separate oracle uses the sealed public-descriptor recurrence kernels,
sequential coefficient advancement, and a normal-equation center solve
instead of the production least-squares call. It reproduces proposal
commitments, exact boundaries, all declared integer precision/resource
tuples, inverse tuples, restoration, mutation response, and fresh-versus-
restored reuse.

The valid inverse rematerializes from public topology and restores exact
payload and both ledgers to zero on the original backing. Unrelated REUSE at
period 1 agrees with a fresh carrier. Direct-process generation and lease are
observed bookkeeping only.

The same proposal and exact acceptance test are available to compact
classical software. Certified embeddings or closest-vector optimality, fixed
integer width, total resource advantage, a distinct phase resource, Small
Wall crossing, CATVM custody, physical execution, physical bit replacement,
and unbounded computation are not established.

The next selected experiment is:

```text
COMPACT_CERTIFIED_UNIT_REDUCTION_WITHOUT_A_65536_BIT_EMBEDDING_TABLE_OR_EQUIVALENT_HIDDEN_PRECISION_COST
```

The live obstruction is:

```text
THE_GLOBAL_PROPOSAL_REDUCES_TESTED_CARRIER_PAYLOAD_BUT_REQUIRES_A_LARGE_FIXED_PRECISION_EMBEDDING_TABLE_AND_FLOATING_SOLVE_THE_CONSERVATIVE_NAMED_COMPONENT_MAXIMA_SUM_REMAINS_ABOVE_RAW_THE_IDENTICAL_COMPACT_CLASSICAL_METHOD_REMAINS_AND_NO_TOTAL_RESOURCE_ADVANTAGE_IS_ESTABLISHED
```

## Exact log-free 49-direction unit descent: corrected temporary-cost obstruction

The exact successor removes the 65,536-bit cosine table and floating solve.
It minimizes exact integer trace energy along seven generator directions and
all 42 public pair sum/difference directions. Exponential bracketing plus
binary search finds each exact line minimum, and every nonzero call finishes
an unchanged full sweep without a tested cap hit.

```text
classification  INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification    SEPARATE_REFERENCE_PARITY
restoration     EXACT_ALGEBRAIC_RESTORATION
```

```text
period family    raw       resident   duplicate-live   corrected named-maxima sum
1      PRIMARY   1,306,953   495,482          990,964                1,163,011
1      REUSE     1,332,125   470,924          941,848                1,109,835
64     PRIMARY   2,368,807   874,192        1,748,384                8,479,798
64     REUSE     2,447,532   865,136        1,730,272                8,512,822
```

Resident and duplicate-live payload beat the raw recurrence in all four
cases. The corrected period-64 sum remains far above raw after adding the
previously omitted live power result/factor pair, accepted move scale, and
ledger-derived unit scale maxima. The sum is not a simultaneous process
peak and does not bound internal multiplication, Python objects, allocator
state, or whole-process memory.

The separate sequential-coefficient oracle reproduces every boundary,
forward/inverse resource tuple, restoration check, and semantic mutation.
Each declared period-1 and period-64 case restores exact payload and ledgers
on the original backing. Cross-family restored-carrier reuse is checked only
at period 1.

The identical exact recurrence and search remain available to compact
classical software. No distinct phase resource, computational advantage,
Small Wall crossing, CATVM custody, physical waveform execution, physical
bit replacement, or unbounded computation is established.

The next experiment is:

```text
DEFERRED_EXACT_UNIT_LEDGER_SEARCH_WITH_SINGLE_NET_RESIDUAL_ACTION_AND_STREAMED_UNIT_MATERIALIZATION
```

The obstruction is:

```text
EXACT_LOG_FREE_SEARCH_REMOVES_THE_FIXED_PRECISION_TABLE_AND_FLOATING_SOLVE_BUT_ACCEPTED_VECTOR_AND_UNIT_MATERIALIZATION_WORK_PLUS_LIVE_POWER_AND_SCALE_TEMPORARIES_LEAVE_THE_CORRECTED_PERIOD64_NAMED_MAXIMA_SUM_ABOVE_RAW_WHILE_THE_IDENTICAL_CLASSICAL_EXECUTION_REMAINS
```

## Deferred exact unit-ledger action: large history-cost reduction, residual obstruction

The exact search now accumulates coordinate moves in the norm and seven-entry
unit ledger, then applies one net residual action after certification. Vector
addition uses one relative unit alignment, and boundary projection precedes
one scalar unit materialization.

```text
classification  INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification    SEPARATE_REFERENCE_PARITY
restoration     EXACT_ALGEBRAIC_RESTORATION
```

```text
period family    raw       resident-live   named-maxima sum   change from M110
1      PRIMARY   1,306,953        990,964          1,193,107           +30,096
1      REUSE     1,332,125        941,848          1,137,779           +27,944
64     PRIMARY   2,368,807      1,708,194          3,908,681        -4,571,117
64     REUSE     2,447,532      1,683,548          4,692,261        -3,820,561
```

The period-64 totals improve materially but remain above raw. The named totals
sum separately observed maxima for resident plus inverse-rematerialized state,
exact search power pairs, trial norms, energy pairs, the one net action,
relative alignment, scalar projection, and the fixed public unit table. They
are not simultaneous or whole-process peaks.

The independent oracle imports no production successor and uses sequential
coefficient advancement. It reproduces every boundary, forward and inverse
resource tuple, local-search certificate, exact restoration result, and the
period-1 cross-family restored-carrier reuse result. No accepted search move
materializes the full residual vector.

The identical deferred implementation remains available to compact classical
software. No distinct phase resource, computational advantage, Small Wall
crossing, CATVM custody, physical waveform execution, physical bit
replacement, or unbounded computation is established.

The next experiment is:

```text
TOPOLOGY_DERIVED_SINGLE_LIVE_VECTOR_HORNER_PHASE_RECURRENCE_WITH_STREAMED_INVERSE_REMATERIALIZATION_AND_MATCHED_RAW_HORNER_BASELINE
```

The obstruction is:

```text
DEFERRED_UNIT_LEDGER_SEARCH_REDUCES_PERIOD64_NAMED_MAXIMA_BY_3_82_TO_4_57_MBITS_BUT_RESIDENT_PLUS_DUPLICATE_INVERSE_REMATERIALIZATION_AND_EXACT_SEARCH_POWER_PAIRS_LEAVE_3_91_AND4_69_MBIT_TOTALS_ABOVE_2_37_AND2_45_MBIT_RAW_WHILE_THE_IDENTICAL_DEFERRED_CLASSICAL_EXECUTION_REMAINS
```

## Single-resident Horner recurrence removes retained multi-vector history

The topology-compiled Horner successor stores one final 17-element cyclotomic
vector in the borrowed carrier and retains zero inverse-history bytes. Its
inverse rematerializes the exact output from public topology and restores
payload and ledgers on the original backing for both public families at
periods 1 and 64. Period-1 primary-to-unrelated reuse agrees with a fresh
carrier.

```text
classification  INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification    SEPARATE_REFERENCE_PARITY
restoration     EXACT_ALGEBRAIC_RESTORATION
```

The independent oracle advances coefficients sequentially by `x mod q`
instead of production binary powering and reproduces boundaries, six named
phase-vector checkpoints, five named raw-vector checkpoints, inverse outputs,
restoration, and reuse.

```text
period family    phase named total   matched raw Horner   difference
1      PRIMARY              99,971               10,005      +89,966
1      REUSE               107,579               10,097      +97,482
64     PRIMARY           3,244,245            2,790,766     +453,479
64     REUSE             3,970,780            2,901,994   +1,068,786
```

Named checkpoints are not legal live intervals or a simultaneous process
peak. Python objects, allocator state, native-library storage, internal ring
multiplication, and whole-process memory are not bounded. The identical
normalized Horner execution remains available to compact classical software.

No distinct phase resource, computational advantage, Small Wall crossing,
CATVM custody, catalytic inference, physical waveform execution, physical bit
replacement, or unbounded computation is established.

The next experiment is:

```text
EXACT_MAXIMAL_REAL_SUBFIELD_UNIT_NORM_SEARCH_QUOTIENT_WITH_MATCHED_CYCLOTOMIC_AND_CLASSICAL_COST
```

The obstruction is:

```text
HORNER_HISTORY_RELEASE_REMOVES_THE_RETAINED_MULTI_VECTOR_OBSTRUCTION_BUT_EXACT_ARITHMETIC_WIDTH_AND_THE_IDENTICAL_NORMALIZED_CLASSICAL_HORNER_EXECUTION_REMAIN
```

## Exact maximal-real-subfield Horner search

The unit-search norm state and 98 signed direction norm factors now execute
in the degree-8 maximal real subfield. Production uses the integral
`(1,s1,...,s7)` basis. A separate power-basis oracle independently
reconstructs all factors, 9,604 products, traces, public Horner operators,
boundaries, named resources, inverse outputs, restoration, and reuse.

```text
classification  INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification    SEPARATE_REFERENCE_PARITY
restoration     EXACT_ALGEBRAIC_RESTORATION
hosted run      NOT ESTABLISHED
```

The retained warm table is 3,407 logical payload bits. The one-time compiler
transition is 11,861 bits, including 571 separately allocated predecessor
singleton norm factors. Search accounting releases the full aggregate norm
after conversion and explicitly includes the persistent real norm and energy.
Stale trial norms and energy scalars are released before later probes.

```text
period family    phase named total   matched raw Horner   difference
1      PRIMARY              93,479               10,005      +83,474
1      REUSE               101,169               10,097      +91,072
64     PRIMARY           3,466,215            2,790,766     +675,449
64     REUSE             3,840,016            2,901,994     +938,022
```

All phase named totals remain above raw. The identical normalized
real-subfield execution remains available to classical software. The stricter
M113 accounting is not promoted as a direct total-payload improvement over
M112.

Exact payload and ledger restoration on the original backing holds for both
families at periods 1 and 64. Primary-to-unrelated restored-carrier reuse is
established at period 1 only. Full object-state equality, bounded generation
or lease metadata, CATVM custody, a distinct phase resource, computational
advantage, Small Wall crossing, physical execution, physical bit replacement,
catalytic inference, and unbounded computation are not established.

The next experiment is:

```text
STREAMED_EXACT_REAL_SUBFIELD_AUTOCORRELATION_WITHOUT_FULL_AGGREGATE_NORM_AND_MATCHED_FULL_CYCLOTOMIC_BOUNDARY
```

The obstruction is:

```text
REAL_SUBFIELD_SEARCH_REMOVES_CONJUGATION_REDUNDANCY_FROM_NORM_FACTORS_BUT_PER_ELEMENT_AUTOCORRELATION_AND_THE_FINAL_CERTIFIED_ACTION_REMAIN_FULL_CYCLOTOMIC_ALL_PHASE_NAMED_TOTALS_REMAIN_ABOVE_RAW_AND_THE_IDENTICAL_CLASSICAL_REAL_SUBFIELD_EXECUTION_REMAINS
```

## Exact streamed real autocorrelation removes the full aggregate norm

M114 converts every temporary full cyclotomic Hermitian product immediately
into the degree-8 real subfield and accumulates it there. Neither production
nor the separate power-basis oracle constructs the predecessor's summed
degree-16 norm.

```text
classification  INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification    SEPARATE_REFERENCE_PARITY
restoration     EXACT_ALGEBRAIC_RESTORATION
hosted run      NOT ESTABLISHED
source head     7b0e6050c94619643175fe803c01a3d192f84134
```

The qualifier declares the two actual norm shapes rather than assuming every
call is a 17-cell carrier vector:

```text
period family   calls  17-cell  singleton  terms
1      PRIMARY      5        4          1     69
1      REUSE        5        4          1     69
64     PRIMARY     93       63         30   1101
64     REUSE       98       66         32   1154
```

```text
period family    streamed named total   matched raw Horner   difference
1      PRIMARY                  96,088               10,005      +86,083
1      REUSE                   103,778               10,097      +93,681
64     PRIMARY               3,466,223            2,790,766     +675,457
64     REUSE                 3,840,024            2,901,994     +938,030
```

The live metric now includes the persistent real accumulator with transient
conversion or addition state. M114 is therefore 2,609 bits above M113 at
period 1 and 8 bits above M113 at period 64 for each corresponding family.
This is a stricter named-live account, not a resource improvement.

All exact boundaries, inverse outputs, restoration on original backing, and
period-1 unrelated reuse agree independently. Each per-element Hermitian
product, the resident carrier, and the final certified action remain full
cyclotomic. Compact classical software can execute the identical streamed
algorithm. No distinct phase resource, advantage, Small Wall crossing, CATVM
custody, physical execution, physical bit replacement, catalytic inference,
or unbounded computation is established.

The held successor is:

```text
EXACT_REAL_SUBFIELD_HERMITIAN_TERM_GENERATOR_WITHOUT_PER_ELEMENT_FULL_CYCLOTOMIC_PRODUCT_OR_PHASE_NATIVE_NONCLASSICAL_TRACE_COUPLING
```

The exact resume obstruction is:

```text
THE_FULL_AGGREGATE_NORM_IS_GONE_BUT_EACH_HERMITIAN_TERM_THE_RESIDENT_CARRIER_AND_THE_CERTIFIED_ACTION_STILL_USE_THE_FULL_CYCLOTOMIC_REPRESENTATION_AND_COMPACT_CLASSICAL_SOFTWARE_CAN_USE_THE_SAME_STREAM
```

## Exact direct real Hermitian generator

M115 removes the remaining per-element degree-16 Hermitian products from the
accepted norm path. For canonical `x = sum(a_j zeta^j)` with `a_16 = 0`, it
streams the cyclic correlations `P_k = sum_j a_j a_(j+k mod 17)` and emits
the exact integral-real-basis tuple `(P_0-P_8, ..., P_7-P_8)`.

```text
classification  INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification    SEPARATE_REFERENCE_PARITY
restoration     EXACT_ALGEBRAIC_RESTORATION
hosted run      NOT ESTABLISHED
source head     d4e117ae06597abb683e50b0e2a466cbdc877c2d
```

The production schedule stores `P_8` and streams the other eight
correlations. Per term it records 136 integer multiplications, 127
accumulation additions, and 8 final subtractions. It creates no materialized
conjugate, full degree-16 product, full-to-real conversion, or degree-16 norm
scratch.

The separate oracle imports no production M115 module and reconstructs the
Hermitian output in a different representation: `x = A(y) + zeta B(y)` and
`x conjugate(x) = A^2 + yAB + B^2`. Across 10,811 term-schedule checks and
11,083 semantic calls, its boundaries, resources, restoration, and reuse
match production exactly.

```text
period family   calls  terms  integer products  direct named total  raw Horner  M115-raw
1      PRIMARY      5     69             9,384              93,790      10,005    83,785
1      REUSE        5     69             9,384             101,475      10,097    91,378
64     PRIMARY     93  1,101           149,736           3,324,441   2,790,766   533,675
64     REUSE       98  1,154           156,944           3,695,435   2,901,994   793,441
```

The corresponding reductions from M114 are 2,298, 2,303, 141,782, and
144,589 bits. The named totals are sums of separately observed maxima, not
simultaneous or whole-process peaks. All four remain above raw, and compact
classical software can execute the identical direct degree-8 bilinear map.

Exact payload and ledger restoration on the original backing holds for both
families at periods 1 and 64, without retained inverse history or baseline
reload. Unrelated cross-family reuse is established at period 1. Full object
equality and machine-enforced generation or lease custody are not established.

No distinct phase resource, computational advantage, Small Wall crossing,
CATVM custody, physical waveform execution, physical bit replacement,
catalytic inference, or unbounded computation is established.

The next experiment is:

```text
PHASE_NATIVE_NONCLASSICAL_TRACE_COUPLING_OR_EXACT_FULL_CARRIER_PHASE_QUOTIENT_WITH_BOUNDARY_LIFT
```

The obstruction is:

```text
THE_NORM_SEARCH_NO_LONGER_NEEDS_ANY_DEGREE16_HERMITIAN_PRODUCT_BUT_THE_PHASE_CARRIER_AND_CERTIFIED_ACTION_REMAIN_FULL_CYCLOTOMIC_AND_COMPACT_CLASSICAL_SOFTWARE_EXECUTES_THE_IDENTICAL_DIRECT_BILINEAR_MAP
```

## Exact quadratic-extension post-forward storage diagnostic

M116 represents each stored full-cyclotomic element exactly as
`A + zeta B`, with two eight-coordinate integral real-subfield lanes and
`zeta^2-s1*zeta+1=0`. The map has determinant `+1`, retains 16 independent
integer coordinates, and is an isomorphism rather than a quotient.

```text
classification  INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification    SEPARATE_REFERENCE_PARITY
restoration     EXACT_ALGEBRAIC_RESTORATION
hosted run      NOT ESTABLISHED
source head     01ff12d2984560cad59d0448125fe79b644fb225
```

The stored carrier, projection, and unit-ledger materialization use the pair
algebra. Pair multiplication uses three real-subfield products, and the
accepted forward path performs one split-to-full lift for the final scalar
boundary. The full forward Horner construction and full inverse
rematerialization remain; each is converted to the pair representation and
counted. No full resident vector is retained after forward conversion.

The initial resource comparison had mixed metadata-inclusive pair residency
with metadata-exclusive full-vector residency. Adversarial review rejected
that field before sealing. The repaired source includes pi and unit ledgers
on both sides, counts the empty carrier backing during forward conversion,
and includes normalized projection state plus pair-multiplication live work.
The qualifier seals the repaired fields.

```text
period family   pair resident  comparable full  pair-full  M116 total  M115 total  raw
1      PRIMARY          3,036            2,937         99      94,616      93,790   10,005
1      REUSE            5,452            5,254        198     102,400     101,475   10,097
64     PRIMARY        222,134          222,077         57   3,325,225   3,324,441 2,790,766
64     REUSE          211,318          211,202        116   3,696,278   3,695,435 2,901,994
```

All pair resident measurements and all named totals worsen. The named total
remains a conservative sum of component maxima rather than a simultaneous or
whole-process peak. Python object, allocator, native-library, bigint-internal,
and whole-process storage are excluded.

The separate oracle imports no production M116 module. It derives semantic
pair arithmetic in the real-subfield power basis and independently
reexecutes the integral `S`-basis schedule. All four boundaries, raw
boundaries, resource tuples, comparable residency deltas, inverse outputs,
and restoration results match. Exact restoration returns the actual original
backing to zero payload and ledgers without retained inverse history or
baseline reload. Period-1 unrelated reuse matches a fresh carrier in boundary
and declared resource signature. Generation and lease remain observed
direct-process metadata, not enforced CATVM custody.

The strongest matched classical execution is the identical two-by-eight
pair recurrence. This package establishes no dimension reduction, full
pair-native recurrence, distinct phase resource, computational advantage,
Small Wall crossing, CATVM custody, catalytic inference, physical waveform
execution, physical bit replacement, or unbounded computation.

The next experiment is:

```text
PHASE_NATIVE_NONCLASSICAL_TRACE_COUPLING_WITHOUT_FULL_CARRIER_RECONSTRUCTION
```

The obstruction is:

```text
THE_EXACT_TWO_BY_EIGHT_PAIR_CHANGES_POST_FORWARD_STORAGE_COORDINATES_WITHOUT_REDUCING_RANK_INCREASES_COMPARABLE_RESIDENT_AND_TOTAL_NAMED_PAYLOAD_RETAINS_FULL_FORWARD_AND_INVERSE_CONSTRUCTION_AND_HAS_AN_IDENTICAL_COMPACT_CLASSICAL_RECURRENCE
```

## Exact globally phase-covariant three-shear feedback diagnostic

M117 uses the M116 two-by-eight pair algebra as the resident primitive for a
three-step nonlinear feedback law. Each shear adds
`zeta^k * x_left * Tr_K/S(x_left*conjugate(x_right))` to a distinct target.
The public plan is `(0,1->2,k5)`, `(1,2->0,k11)`, `(2,0->1,k3)`.

```text
classification  INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification    SEPARATE_REFERENCE_PARITY
restoration     EXACT_ALGEBRAIC_RESTORATION
hosted run      NOT ESTABLISHED
source head     5e1b6b5bed88be896e920c7769a6935bd2395c87
```

All three shear pairs have executed nonzero commutators for both public seed
families. Three tested common root rotations preserve each boundary, while a
single-site phase change produces a different boundary. The accepted path
materializes neither a full cyclotomic carrier nor a full scalar lift.

```text
family    coupled boundary    no-shear boundary    named sum
PRIMARY   197                 16                   5,177 bits
REUSE     112                 -1                   5,123 bits
```

Reverse-order arithmetic inversion restores the actual zero-carrier backing
exactly, after which cross-family reuse agrees with fresh execution in
boundary, rank, and arithmetic signature. A wrong arithmetic inverse and a
same-order inverse fail restoration. The independent degree-16 power-basis
oracle and separate two-by-eight reference agree after every forward and
inverse step for the main families, phase mutations, global rotations, and an
alternate plan.

The zero cross-cell diagnostic is declared erasure, not executed physical
dephasing. Named totals count 272 integer carrier coordinates, public program
and plan payload, the retained 17-root table, seed buffers, named coupling
work, and the final boundary. They exclude internal multiplication scratch,
Python and allocator storage, bigint/native-library internals, compilation
work, and whole-process peaks.

The strongest compact classical implementation is the identical fixed-rank
two-by-eight integer recurrence. The exact nonlinear phase coupling therefore
does not establish a distinct phase resource, advantage, Small Wall crossing,
CATVM custody, inference, physical execution, bit replacement, or unbounded
computation.

The next experiment is:

```text
VARIABLE_RANK_PHASE_COUPLING_WITH_COMPACT_CLASSICAL_TENSOR_BASELINE
```

The obstruction is:

```text
FIXED_RANK_RELATIVE_HERMITIAN_TRACE_FEEDBACK_REMAINS_AN_IDENTICAL_COMPACT_CLASSICAL_POLYNOMIAL_RECURRENCE
```

## Exact four-site interleaved phase coupling rank diagnostic

M118 executes two public width-four, local-dimension-two programs on sixteen
actual `Q(zeta_17)` phase cells. Four public `K2,2` diagonal phase gates are
interleaved with one local unimodular shear. Only one final cyclotomic scalar
is projected.

```text
classification  INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification    INDEPENDENT_ORACLE_REEXECUTION
restoration     EXACT_ALGEBRAIC_RESTORATION
hosted run      NOT ESTABLISHED
source head     30de9ca9b0e663d0ab8367db7a91d7aab727cbe4
```

Both public families have exact natural TT ranks

```text
[1,1,1] -> [2,2,1] -> [2,2,1] -> [2,2,1] -> [2,4,2] -> [2,4,2].
```

All three final two-versus-two cuts have rank four, and all final one-site
cuts have rank two. An independent full power-basis oracle reproduces the
rank trace, exact boundaries, inverse, and controls without importing M118 or
the M116 pair backend. Its `F103` order-seventeen-root images give nonzero
final determinant residues `(59,42,86)` for `PRIMARY` and `(90,65,78)` for
`REUSE`.

```text
family    final boundary                                  phase sum  compact sum
PRIMARY   [9,2,1,4,2,1,0,1,2,0,0,2,0,0,0,0]             1,761      643 bits
REUSE     [9,0,2,0,3,0,4,0,2,1,0,0,0,1,0,2]             1,754      621 bits
```

Exact reverse-order gates followed by public seed unload return the actual
backing to zero with no retained inverse history or snapshot. The unrelated
reuse program consumes the same backing and matches fresh execution in
boundary, rank trace, and every nonmetadata arithmetic statistic.

The strongest matched classical method compiles the public treewidth-two
topology to a coalesced sparse character sum: eight nonconstant roots for
`PRIMARY`, seven for `REUSE`, zero general pair multiplications, and four
live phase cells. It reproduces the exact boundary without resident tensor
expansion and uses less named logical payload. The phase benchmark measures
a restoring transaction while the compact benchmark measures boundary-only
evaluation; those timings are not operation-matched and are not evidence of
advantage. Named sums remain sums of component maxima, not simultaneous or
whole-process peaks, and exclude rank-verifier, Python, allocator, bigint,
and native-library storage.

Earlier `Q(zeta_5)` work already established bounded exact TT-rank growth.
M118 is an F17 integration calibration, not a general TT or growing-family
result. It establishes no compact variable-rank closure, distinct phase
resource, computational advantage, Small Wall crossing, CATVM custody,
inference, physical execution, bit replacement, or unbounded computation.

The next experiment is:

```text
RESIDENTLY_GENERATED_NONPUBLIC_COUPLING_TOPOLOGY_OR_GROWING_TREEWIDTH_PHASE_CLOSURE_WITH_OPTIMAL_CLASSICAL_TENSOR_BASELINE
```

The obstruction is:

```text
PUBLIC_LOW_TREEWIDTH_PHASE_TOPOLOGY_RETAINS_THE_COMPLETE_BOUNDARY_LAW_WITHOUT_RESIDENT_TENSOR_EXPANSION_DESPITE_EXACT_TT_RANK_GROWTH
```

## Exact runtime-weighted F17 grid factor closure diagnostic

M119 compiles public square-grid topology before binding two deterministic
runtime-weight families. It stores exact unary and pair factors on actual
`Q(zeta_17)` phase cells and avoids a dense `2^(n^2)` global amplitude tensor
on the accepted path.

```text
classification  INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification    INDEPENDENT_ORACLE_REEXECUTION
restoration     EXACT_ALGEBRAIC_RESTORATION
hosted run      NOT ESTABLISHED
source head     af8a41ae5acf179eac57576ce58a75a55d2d2d1e
```

```text
n                              2       3        4
actual factor cells           24      66      128
row-interface messages         4       8       16
exact Q(zeta17) rank           4       8       16
dense global cells avoided    16     512   65,536
total transfer transitions    16     128      768
```

Independent direct enumeration, Gray-code energy-delta enumeration, and row
character transfer reproduce all six final boundaries. Independent
order-seventeen `F103` determinant images certify the actual row-interface
ranks. The accepted transaction projects only the final cyclotomic scalar.
Reverse actions recover the exact factor seed; seed unload restores the
original backing to zero. The held-out family reuses that same backing and
matches fresh execution without retained history or snapshot reload.

Projection now validates both topology fingerprint and family ownership.
Controls reject premature projection, mismatched projection custody, missing
or wrong inverse, noncommuting inverse reordering, mutation, null carrier, and
a false rank cap. A separable zero-weight diagnostic has rank one, and removal
of one interface edge changes the boundary and halves the certified rank.

The evaluated compact classical set contains the identical exact row
transfer, a Gray-delta 17-bin histogram, and a three-order observed MTBDD
sweep. The MTBDD builds the full assignment tree and claims no order
optimality. The set is not exhaustive or Pareto-optimal. Broader matchgate or
holographic reductions remain unresolved. Warm timings compare a restoring
phase transaction against boundary-only classical paths and are not
operation-matched.

The result is a bounded growing-treewidth calibration and negative resource
diagnostic. It establishes no compact closure across growing treewidth,
distinct phase resource, advantage, Small Wall crossing, CATVM custody,
inference, physical execution, bit replacement, or unbounded computation.

The next experiment is:

```text
PHASE_NATIVE_SEPARATOR_QUOTIENT_OR_COUPLING_RESOURCE_THAT_AVOIDS_BOTH_TWO_TO_THE_N_INTERFACE_MESSAGES_AND_TWO_TO_THE_N_SQUARED_GLOBAL_ENUMERATION_AGAINST_THE_EVALUATED_COMPACT_CLASSICAL_BASELINE_SET
```

The obstruction is:

```text
EXACT_PHASE_FACTOR_STORAGE_REMAINS_COMPACT_BUT_FINAL_CLOSURE_HAS_NO_LEVERAGE_THE_IDENTICAL_TRANSFER_USES_TWO_TO_THE_N_SEPARATOR_MESSAGES_WHILE_THE_17_BIN_GRAY_DELTA_HISTOGRAM_TRADES_CHARACTER_ACCUMULATOR_MEMORY_FOR_TWO_TO_THE_N_SQUARED_GLOBAL_ASSIGNMENTS
```

## Exact resident-factor Kronecker-butterfly grid closure repair

M120 preserves M119 and repairs its missed row-kernel application in a
distinct source package at scientific head
`9ddc7fc623ec28bb6557e865fdadca5b38d73241`.

```text
classification       INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification         INDEPENDENT_ORACLE_REEXECUTION
factor restoration   EXACT_ALGEBRAIC_RESTORATION
transient buffers     NO_RESTORATION_CLAIM
hosted run            NOT ESTABLISHED
```

For each vertical column, the accepted path validates the actual resident
factor as `[[1,1],[1,zeta17^J]]` and consumes its resident root in an
MSB-indexed Kronecker butterfly. All six M119 boundaries are preserved.

```text
n                              2       3       4
factor cells                  24      66     128
exact interface rank           4       8      16
row-interface messages         4       8      16
nontrivial root actions         4      24      96
butterfly additions             8      48     192
global tensor cells avoided    16     512  65,536
```

This removes M119's `(n-1)4^n` source/target work from the accepted closure.
The new exact counts are `(n-1)n2^(n-1)` resident-root actions and twice as
many additions. No dense transfer matrix or source/target pair enumeration is
present. Row-factor construction and pointwise diagonal work remain separate
and counted.

The independent oracle imports neither production nor the phase backend. It
reconstructs every boundary in a separate 16-integer power basis from a
Gray-code character histogram, reexecutes exact counts, and forms explicit
`F103` matrices with ranks `4,8,16`. Setting one separator weight to zero
halves each rank. A separate exact dense-vector check agrees with every
butterfly interface.

Only the final scalar receives a full lift. The actual borrowed factor carrier
is reversed and seed-unloaded to its original zero backing, then reused by the
held-out family on the same backing. The transient butterfly frontiers are
discarded projection work and have `NO_RESTORATION_CLAIM`; no CATVM or
catalytic custody of them is claimed.

The strongest evaluated generic row recurrence is now the identical exact
`2^n`-message classical butterfly. Its diagonal generation is not
operation-matched to the resident-factor path, and timings are observational.
ADD/MPS, matchgate, holographic, symmetry, and boundary-specific reductions
remain unexhausted.

The repair therefore changes the work law but not the scientific resource
position. Exact interface rank and message storage remain `2^n`, with an
identical compact classical recurrence. No separator quotient, distinct phase
resource, advantage, Small Wall crossing, CATVM custody, inference, physical
execution, bit replacement, or unbounded computation is established.

The initially selected fixed four-bit cubic plus mixing successor was skipped
after applicability review because it recombined mechanisms already present
in the F5 cubic/Fourier, F17 cubic-latent, and M118 interleaved phase/shear
packages without changing the growing separator law. The next package tested
that obstruction directly.

## Exact generic-runtime F17 linear-separator quotient no-go

M121 broadens the public runtime interface to arbitrary nonzero F17 unary and
edge descriptors. Two distinct generated descriptors per size execute through
the actual M120 factor carrier at `n=2,3,4`. Every boundary agrees with the
identical exact butterfly, reverse phase actions plus seed unload restore the
original zero backing exactly, and the held-out descriptor reuses that same
backing with fresh-carrier parity.

```text
classification                 INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification                   INDEPENDENT_ORACLE_REEXECUTION
factor restoration             EXACT_ALGEBRAIC_RESTORATION
transient projection buffers   NO_RESTORATION_CLAIM
hosted run                      NOT ESTABLISHED
source head                     1963fd2ba12e68008af5824496fc1b3c75ca6495
```

At the penultimate-to-final-row separator, the legal continuation matrix is
the product of

```text
tensor_i [[1,zeta17],[1,zeta17^2]]
a nonzero horizontal-row phase diagonal
tensor_i [[1,1],[1,zeta17^J_i]], J_i != 0.
```

All three factors are invertible over `Q(zeta17)`. Hence a uniform fixed
linear encoder that must accept arbitrary field messages and preserve every
legal nonzero final-row continuation is injective and requires at least
`2^n` field coordinates. The accepted certificate is formula-only, emits
`n=1,...,16`, and materializes neither dense rank matrices nor the
continuation family.

An independent oracle imports neither production nor the phase backend. It
reconstructs all six exact boundaries with separate butterfly and
Gray-histogram recurrences. Explicit verification-only matrices over `F103`
and `F137` have continuation, vertical, and combined ranks `4,8,16`.
Zeroing one vertical weight or duplicating one continuation choice gives
`2,4,8`. A rank-`2^n-1` coordinate-drop encoder has a nonzero kernel vector
detected by a legal continuation.

This rejects only uniform fixed `Q(zeta17)`-linear separator compression for
the broadened legal continuation family. It does not reject nonlinear,
program-dependent, or restricted signatures, nor ADD/MTBDD, exact MPS/tensor,
matchgate/Pfaffian, holographic, global, or boundary-specific contractions.
It establishes no total complexity lower bound, distinct phase resource,
advantage, Small Wall crossing, CATVM custody, inference, physical execution,
bit replacement, or unbounded computation.

The selected successor is:

```text
NONLINEAR_CANONICAL_PHASE_SEPARATOR_CHART_WITH_RESIDENT_UPDATES_ON_GROWING_HETEROGENEOUS_WIDTHS_AGAINST_ALL_ORDER_ADD_MTBDD_EXACT_MPS_TENSOR_AND_MATCHGATE_HOLOGRAPHIC_BASELINES
```

The surviving obstruction is:

```text
GENERIC_NONZERO_RUNTIME_GRID_CONTINUATIONS_REQUIRE_FULL_TWO_TO_THE_N_COORDINATES_FOR_ANY_UNIFORM_FIXED_Q_ZETA17_LINEAR_SEPARATOR_ENCODER_SO_THE_NEXT_REPAIR_MUST_CHANGE_REPRESENTATION_CLASS_AND_RETAIN_BROADER_GLOBAL_ALGORITHM_CONTROLS
```

## Nonlinear projective and canonical TT separator charts do not compact the generic grid message

M122 executes an exact gauge-fixed tensor-train row carrier over `Q(zeta17)`
for two descriptors at `n=2,3,4`. The accepted path applies the resident
vertical, unary, and horizontal phase updates without materializing the
`2^n` row vector. It projects one final scalar and then reverses the actual
schedule. The same backing, exact canonical state, generation, and lease are
restored before a held-out descriptor is reused with fresh-carrier boundary
and rank parity.

```text
classification                 INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification                   INDEPENDENT_ORACLE_REEXECUTION
carrier restoration            EXACT_ALGEBRAIC_RESTORATION
transient buffers              NO_RESTORATION_CLAIM
source head                     e8c8e45ab7308197f553fb44522ca4664356e171
```

Projective normalization retains `2^n-1` exact ratios plus the answer-bearing
scale. The fixed zero-index pivot is nonzero for the declared family, but the
coordinate count does not fall. At `n=4`, projective denominator width reaches
91 bits.

The generic exact discriminator has final ranks `[2,4,2]`, 40 raw core cells,
24 gauge dimensions, 16 effective coordinates, and 15 fully reduced
projective EVDD nodes under the exact all-order subset optimization. The dense
message has 16 coordinates and its full binary decision tree has 15
nonterminal nodes. Neither nonlinear chart compacts it.

Dual-field resident and dense recurrences agree through `n=8`. The natural
rank profile reaches `[2,4,8,16,8,4,2]`; effective coordinates reach 256, raw
cores 680 cells, and named scratch 768 field cells. `F103` certifies maximal
ranks across every bit-order subset at each tested width. An `F137` `n=8`
degeneracy is preserved, so exact `Q(zeta17)` execution beyond `n=4` and
two-field all-order maximality are not claimed. Wider decision-diagram results
are quasi-reduced layered counts unless redundant-test deletion is separately
certified.

The direct zero-field planar-Ising/Pfaffian positive control passes; a
single-site field defect and the generic discriminator reject that direct
route. Broader matchgate, holographic, global, and boundary-specific methods
remain open. The identical exact TT recurrence is classical. Its exact zero
tests, division, RREF, pivoting, and gauge fixing are not native phase
primitives, and scratch coefficient payload is estimated rather than measured
as a process peak.

No general nonlinear no-go, CATVM custody, distinct phase resource,
computational advantage, Small Wall crossing, physical execution, bit
replacement, or unbounded computation follows.

The selected successor is:

```text
EXACT_F17_PLANAR_FREE_FERMION_MATCHGATE_PHASE_COVARIANCE_CLOSURE_FOR_ZERO_FIELD_GRID_SIGNATURES_WITH_SPARSE_EXTERNAL_FIELD_DEFECT_GROWTH_RESTORATION_REUSE_AND_IDENTICAL_PFAFFIAN_BASELINE
```

The surviving obstruction is:

```text
GENERIC_F17_GRID_MESSAGES_SATURATE_PROJECTIVE_TT_AND_ALL_ORDER_N4_DECISION_DIAGRAM_CHARTS_WHILE_SOFTWARE_ZERO_TEST_DIVISION_AND_GAUGE_FIXING_ARE_NOT_PHASE_PRIMITIVES_SO_THE_NEXT_REPAIR_MUST_CHANGE_THE_RESIDENT_PHASE_UPDATE_OR_RESTRICT_THE_SIGNATURE_FAMILY_WITHOUT_MOVING_GROWTH_INTO_DEFECT_ENUMERATION
```

## Exact planar free-fermion terminal Pfaffian closure and sparse-defect obstruction

M123 restricts the M122 grid family to the declared open planar zero-field
sector and changes the closure law. Public topology compilation builds a
trivalent terminal graph and a GF(2) face orientation without inspecting the
answer. Exact terminal entries remain resident until the final boundary is
closed through one square-root-free Pfaffian.

```text
classification                     INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification                       INDEPENDENT_ORACLE_REEXECUTION
resident carrier restoration       EXACT_ALGEBRAIC_RESTORATION
dense Pfaffian projection buffers  NO_RESTORATION_CLAIM
source head                        8a2d459c3706a3ac1187ae876006c6c9af1a36ee
hosted run                         NOT ESTABLISHED
```

A separate binary residue histogram and bounded-face cycle-space recurrence
reproduce every exact `Q(zeta17)` boundary at `n=2,3,4`, including the
declared `n=4` defect counts `k=0,1,2,4`. The actual terminal and scalar
carrier reverses exactly with no retained inverse history or snapshot reload,
and an unrelated program reuses the same restored backing with fresh-carrier
boundary parity. Dual-field `F103/F137` runs through `n=8` establish only
source-audited structural behavior, not independent exact-number parity.

The implemented sparse-field path streams one even sector at a time. Its
sector count is `1` for `k=0,1` and `2^(k-1)` thereafter, with no retained
sector tuple. This is an implemented upper bound, not a general lower bound.
The M120 `O(n^2 2^n)` general-field row transfer remains a matched fallback
whose memory does not depend on `k`.

At `n=4`, the terminal graph has 72 vertices, the sparse resident carrier has
96 field cells, and each dense Pfaffian projection accounts for 10,378 named
logical field cells. The reported dense bit payload applies the maximum
observed field-element width to the caller matrix, elimination copy, and ten
named live field slots. It is not a measured allocator or whole-process peak
and excludes Python containers, SymPy/native internals, and bigint workspace.

The identical terminal Pfaffian is the strongest implemented classical
baseline; known covariance or sparse planar Pfaffian implementations are also
admitted. Therefore the zero-field closure establishes no distinct phase
resource or advantage. Dense external-field compaction, arbitrary matchgate
or holographic closure, CATVM custody, Small Wall crossing, physical execution,
bit replacement, and unbounded computation remain unestablished.

The selected successor is:

```text
EXACT_SHARED_PHASE_PARITY_LEDGER_FOR_SPARSE_NON_GAUSSIAN_FIELD_INSERTIONS_WITHOUT_EVEN_SECTOR_ENUMERATION_OR_A_PROOF_THAT_THE_LEDGER_RANK_MUST_GROW
```

The surviving obstruction is:

```text
ZERO_FIELD_FREE_FERMION_CLOSURE_IS_IDENTICAL_TO_COMPACT_CLASSICAL_GAUSSIAN_SOFTWARE_WHILE_TRUE_EXTERNAL_FIELDS_SPLIT_INTO_2_TO_THE_K_MINUS_1_EVEN_GAUSSIAN_SECTORS_AND_NO_PHASE_NATIVE_OPERATION_CURRENTLY_CLOSES_THEIR_SUM_WITHOUT_MOVING_THAT_GROWTH
```

## Exact shared phase-parity ledger closes only the two-row separator

M124 replaces M123's even-sector enumeration on declared two-row open ladders
with a public column-interleaved four-state exact recurrence.

```text
classification                 INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification                   INDEPENDENT_ORACLE_REEXECUTION
resident carrier restoration   EXACT_ALGEBRAIC_RESTORATION
transient recurrence frontier  NO_RESTORATION_CLAIM
source head                    ae36a130352ba92f4272858cc00bde13c6336b90
```

Independent `Q(zeta17)` power-basis execution agrees for both public families
at widths `1,2,4,8,16,32,64`; `F103/F137` structural parity agrees through
128. Bounded independent assignment and occurrence-expanded oracles agree
through width four. Grouped ordering grows to rank 16 at width four, while
column interleaving holds the optimum observed maximum rank at four. The exact
PRIMARY width-two native defect signature has a nonzero
Grassmann--Pluecker delta; this does not classify other exact cases.

The accepted path uses `10W-4` resident factor cells, four frontier cells, at
most 19 named transient field cells for width greater than one, and `O(W)`
field operations with bit complexity reported separately. It neither retains
the `4^W` signature nor enumerates even sectors. Final-only projection, exact
same-backing reversal, generation advancement, and unrelated width-16 restored
reuse all pass without history or snapshot reload.

The fixed rank is explained by the two-row four-state separator and is
identical to the strongest compact classical transfer. No growing-treewidth
closure, distinct phase resource, advantage, CATVM custody, or Small Wall
crossing follows.

## Paired phase bases close a growing planar matchgate family but cancel publicly

M125 uses exact paired `zeta17` bases satisfying `T S^T = I`. Shared native
edge indices therefore close by identity, while the compact native degree-four
exact-one generators have nonzero components in both fermion parities. After
the public basis cancellation, the core is a weighted planar perfect-matching
problem evaluated by a Kasteleyn determinant.

```text
classification                   INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification                     INDEPENDENT_ORACLE_REEXECUTION
resident carrier restoration     EXACT_ALGEBRAIC_RESTORATION
transient determinant buffers    NO_RESTORATION_CLAIM
source head                      ae36a130352ba92f4272858cc00bde13c6336b90
```

Independent exact weighted-matching recursion agrees for both families at
`n=2,4,6`, reconstructing perfect-matching counts `2,36,6728`. A separate
modular row-profile recurrence agrees for both fields through `n=12`. Direct
native Holant enumeration is confined to the bounded `n=2` control. Public
Kasteleyn signing and reference-matching sign calibration do not inspect the
answer.

Resident state is `2N(N-1)+8` field cells. The `N^2/2` determinant has
`N^4/4` cells; caller plus elimination copy and five named scalars use
`N^4/2+5` named transient cells. Accepted work is `O(N^6)` field operations,
with exact bit complexity and logical-payload exclusions stated separately.
Final-only projection, exact same-backing reversal, generation advancement,
and unrelated `n=6` restored reuse pass without history or snapshot reload.

This is a bounded growing-treewidth holographic matchgate closure, not general
Holant closure or a distinct phase resource. The strongest compact classical
method performs the identical public paired-basis compilation and determinant.

The selected successor is:

```text
RESIDENTLY_GENERATED_OR_COHERENCE_DEPENDENT_PHASE_BASIS_COUPLING_WITHOUT_A_PUBLIC_CLASSICAL_HOLOGRAPHIC_CANCELLATION_OR_A_MATCHED_NO_GO
```

The obstruction is:

```text
THE_PUBLIC_PAIRED_PHASE_BASIS_CLOSES_THE_DECLARED_GROWING_TREEWIDTH_MATCHGATE_FAMILY_BUT_CANCELS_EXACTLY_TO_THE_IDENTICAL_COMPACT_CLASSICAL_KASTELEYN_DETERMINANT
```

## Coherent shared latent basis control remains a fixed classical direct sum

M126 replaces M125's one public paired-basis cancellation with a resident
17-coordinate latent port consumed by two grid-wide basis-mismatch modules.
Exact Fourier and quadratic-chirp updates separate the modules. Basis-control
and chirp mutations change the final boundary, and Fourier/chirp order changes
the latent state.

```text
classification                   INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification                     INDEPENDENT_ORACLE_REEXECUTION
resident carrier restoration     EXACT_ALGEBRAIC_RESTORATION
transient determinant buffers    NO_RESTORATION_CLAIM
source head                      16ad7bec40126f9b60531225135f78c2bc3ac111
hosted run                       NOT ESTABLISHED
```

Each latent sector closure is streamed directly into resident state. The
accepted path materializes no 17-value boundary vector, edge-local signature
table, or edge-assignment table, and projects only the final `w[0]` scalar.
Exact inverse rematerialization restores the same backing and advances its
generation. Unrelated reuse agrees with fresh execution in boundary and
resource signature, without snapshot reload or retained inverse history.

Independent exact execution agrees for both public families at `n=2,4,6`;
dual-field structural execution agrees through `n=10`. A separate row-profile
matching recurrence, independently reconstructed 34-coordinate latent law,
bounded `n=2` native-Holant enumeration, and basis-identity controls all agree.

Resident state is `4N(N-1)+34` field cells. Named transaction transient state
is `max(N^4/2+5,34)` field cells; final boundary payload bits and JSON bytes
are reported. Full exact bit complexity and whole-process memory are not
established.

The coherent shared port is causal, but its fixed 17 coordinates are the
identical classical direct sum. The strongest matched baseline is the same
34-coordinate recurrence plus 17 compact Kasteleyn closures per module. No
CATVM custody, arbitrary latent geometry, distinct phase resource, advantage,
Small Wall crossing, physical execution, bit replacement, or unbounded
computation follows.

The selected successor is:

```text
GROWING_SHARED_LATENT_PHASE_GEOMETRY_WITH_NATIVE_CONVOLUTION_OR_EXACT_RANK_REDUCTION_AGAINST_THE_STRONGEST_GROUP_ALGEBRA_CLASSICAL_BASELINE
```

The obstruction is:

```text
COHERENCE_DEPENDENT_BASIS_CONTROL_AVOIDS_ONE_PUBLIC_SCALAR_CANCELLATION_BUT_A_FIXED17_STATE_LATENT_PORT_IS_THE_IDENTICAL_CLASSICAL_DIRECT_SUM_SO_THE_NEXT_ROUTE_MUST_GROW_OR_CLOSE_LATENT_GEOMETRY_WITHOUT_GROWING_CLASSICAL_SECTOR_RANK
```

## Exchange symmetry grows latent geometry but does not close its rank

M127 replaces the fixed 17-state direct sum with the degree-`k`
exchange-symmetric sector. The accepted carrier uses exact occupation orbits
of dimension

```text
H(k) = binomial(k+16,16) = 17,153,969,4845 for k=1,2,3,4.
```

```text
classification                   INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification                     INDEPENDENT_ORACLE_REEXECUTION
resident carrier restoration     EXACT_ALGEBRAIC_RESTORATION
transient compiler/grid buffers  NO_RESTORATION_CLAIM
source head                      8d0c04b02460d1f7b5e37c03183cd6c4c77addbc
hosted run                       NOT ESTABLISHED
```

The exact public DFT17 plan is lifted through streamed occupation blocks.
Two grid-wide modules consume the unresolved orbit geometry through
non-sum-only power sums, with exact lifted Fourier and chirp updates between
them. Production projects only one final scalar and materializes no labelled
`17^k` tensor, dense `H(k) x H(k)` operator, orbit-boundary vector,
assignment table, relation table, or inverse history.

Independent labelled-tensor and row-profile recurrences reproduce every
sealed boundary and exact reverse. Deterministic symmetric-vector tests also
reproduce the lifted DFT17, and exhaustive controls confirm that `p1,...,pk`
identify the declared multisets while `p1` alone overmerges at `k>=2`.
Exchange symmetry is essential; the result does not compress the original
labelled open-chain family.

The same backing restores to exact zero, advances its restoration generation,
and supports unrelated fresh/restored reuse without snapshot reload. Named
resources include two `H(k)` resident vectors plus 48 grid weights, 578
retained plan coefficients, `18H(k)` public topology integers, 1,496 compiler
field cells, 133 transaction transient field cells, and 48 transaction
transient integers. Full exact bit complexity and whole-process memory remain
excluded.

Exact PRIMARY resident payload rises from 5,637 bits at `k=1` to 66,389 bits
at `k=2`, with maximum resident numerator width rising from 15 to 28 bits.
The strongest compact classical method executes the identical `H(k)` orbit
recurrence. Thus both orbit rank and arithmetic height remain obstructions;
no distinct phase resource, advantage, Small Wall crossing, CATVM custody,
physical waveform execution, physical bit replacement, or unbounded
computation is established.

The selected successor is:

```text
EXACT_RELATION_PRESERVING_RANK_REDUCTION_OR_NATIVE_PHASE_CONVOLUTION_ACROSS_GROWING_EXCHANGE_SYMMETRIC_DEGREE_WITHOUT_MOVING_H_K_GROWTH_INTO_PRECISION_OR_HISTORY
```

The obstruction is:

```text
EXCHANGE_SYMMETRY_REPLACES_LABELLED_17_TO_THE_K_GROWTH_BY_EXACT_H_K_ORBIT_GROWTH_AND_NON_SUM_ONLY_CONTROLS_BUT_THE_RESIDENT_RANK_STILL_GROWS_AS_BINOMIAL_K_PLUS_16_CHOOSE_16_ARITHMETIC_WIDTH_ALSO_GROWS_AND_COMPACT_CLASSICAL_SOFTWARE_EXECUTES_THE_IDENTICAL_RECURRENCE
```

## Exchange-symmetric selectable phase-module algebra is linearly irreducible

M128 is a distinct extension of M127, not a retroactive reinterpretation. It
adds independently selectable `p1,...,pk` phase-character primitives and
both orientations of adjacent-mode shears to either `H(k)` orbit module for
the declared `k=1,2,3,4` cases.

```text
claim
    BOUNDED_EXACT_F17_POWER_SUM_CHARACTER_AND_BIDIRECTIONAL_MODE_SHEAR_PHASE_MODULE_HAS_NO_NONTRIVIAL_LINEAR_QUOTIENT_BELOW_H_K_ON_EACH_DECLARED_EXCHANGE_SYMMETRIC_K1_TO_K4_ORBIT_MODULE_WITH_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    INDEPENDENT_ORACLE_REEXECUTION

restoration
    resident carrier      EXACT_ALGEBRAIC_RESTORATION
    transient/certificate NO_RESTORATION_CLAIM

source head
    aa641f93a89c6311f469e222ae4f94e1ecad960f
```

For `k<17`, Newton identities make the complete `p1,...,pk` signature
injective on occupations. Exact character orthogonality yields every
coordinate projector. Bidirectional adjacent-mode shears connect the
occupation graph and, after coordinate projection, isolate its nonzero
one-particle transitions. Products along the streamed predecessor tree
generate all matrix units. Thus any uniform exact linear quotient preserving
every declared selectable primitive and the final nonzero occupation
functional has dimension at least
`H(k)=17,153,969,4845` for `k=1,2,3,4`.

Independent exact `Q(zeta17)` transactions agree at `k=1,2`; independent
`F103/F137` transactions agree through `k=4`. Final-only projection, actual
reverse order, exact zero restoration on the same backing, generation
advancement, and unrelated `k=2` reuse all pass. Fresh/restored reuse agrees
across the complete reported deterministic resource signature.

The accepted carrier retains `48+2H(k)` field cells and streams character,
shear, and grid work. No character table, dense `H(k) x H(k)` operator,
labelled tensor, relation table, serialized intermediate, inverse history, or
certificate edge list is accepted-path state. Certificate signature sets and
union-find storage are reported separately. Full exact bit complexity,
Python/native container and allocator storage, bigint internals, and
whole-process peak remain excluded.

The strongest compact classical baseline is the identical `H(k)`-coordinate
exact recurrence. M128 rejects only the uniform exact linear quotient route
for the new primitive algebra. It does not reject nonlinear,
program-restricted, integrable, approximate, or unextended-M127 quotients and
does not establish CATVM custody, a distinct phase resource, advantage, Small
Wall crossing, physical execution, physical bit replacement, or unbounded
computation.

The next experiment is:

```text
NONLINEAR_OR_PROGRAM_RESTRICTED_INTEGRABLE_PHASE_CONVOLUTION_CHART_WITH_EXACT_COMPOSITION_WITHOUT_H_K_COORDINATE_MATERIALIZATION_OR_HIDDEN_PRECISION_HISTORY_GROWTH
```

## Coherent Veronese chart closes the fixed integrable word but not the grid coupling

M129 implements the nonlinear program-restricted route left open by M128.
The accepted phase carrier stores 17 exact coefficients `v_j` for
`(sum_j v_j x_j)^k`. The fixed public `p1,...,p4` character operations are
diagonal updates of those coefficients, and each declared adjacent-mode shear
is the exact one-particle substitution `v_row += zeta17^e v_pivot`.

```text
claim
    BOUNDED_EXACT_PROGRAM_RESTRICTED_F17_COHERENT_VERONESE_PHASE_CHART_CLOSES_FIXED_P1_TO_P4_CHARACTERS_AND_BIDIRECTIONAL_MODE_SHEARS_IN_17_RESIDENT_COORDINATES_ACROSS_GROWING_EXCHANGE_SYMMETRIC_DEGREE_WITH_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_SEALED_WORDS_HAVE_A_STRONGER_TWO_SCALAR_WARM_CLASSICAL_BASELINE

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    INDEPENDENT_ORACLE_REEXECUTION

restoration
    resident carrier  EXACT_ALGEBRAIC_RESTORATION
    transient buffers NO_RESTORATION_CLAIM

source head
    f787e13100715219b8d39bd113f5aaade3695013
```

Exact `Q(zeta17)` execution passes at `k=4,8,16,32`; structural `F103/F137`
execution passes through `k=128`. The resident count stays at 17 field cells
and 762 counted exact payload bits while the implicit `H(k)` rises from 4,845
at `k=4` to 687,917,389,635,036,844,569 at `k=128`. The final scalar is
`k*v0^(k-1)*v1`; its exact payload grows from 197 to 1,545 bits through the
exact `k=32` scope and is counted.

The forward chart is committed by streaming one serialized field record at a
time and is never returned. Actual reverse execution restores exact zero on
the same backing, advances generation, and supports an unrelated `k=16`
reuse after `k=8` with fresh boundary and full reported resource-signature
agreement. No snapshot or inverse history is used.

The independent oracle expands all `H(4)=4,845` occupation coordinates in
`Q(zeta17)`, `F103`, and `F137`, separately checks all 40 primitives and their
inverses in both finite fields, reconstructs the streamed commitments, and
hard-gates the compiled classical baseline. Actual M128 `k=2` grid states
have nonzero rank-one catalecticant minors 58 and 46; those bounded witnesses
prove that the grid injections leave this chart. They do not establish grid
exit for every degree.

For descriptor-runtime programs, compact classical software executes the
identical 17-coordinate recurrence. For the fixed sealed words, a stronger
warm baseline compiles the public word through 17 working cells, retains only
`(v0,v1)`, and evaluates the boundary with two retained field cells. M129
therefore establishes a bounded nonlinear chart closure, not a distinct phase
resource or advantage.

The next experiment is:

```text
EXACT_BOUNDED_SECANT_OR_GAUSSIAN_COHERENT_PHASE_CHART_FOR_ONE_SHARED_NONINTEGRABLE_ORBIT_COUPLING_WITH_RANK_GROWTH_AND_STRONGEST_COMPACT_CLASSICAL_BASELINE
```

The obstruction is:

```text
THE_RANK1_COHERENT_VERONESE_CHART_REMOVES_H_K_RESIDENCY_FOR_AN_INTEGRABLE_CHARACTER_AND_ONE_PARTICLE_SHEAR_SUBPROGRAM_BUT_M127_GRID_ORBIT_SHEARS_AND_GENERIC_SUPERPOSITIONS_LEAVE_THE_CHART_AND_EACH_SEALED_PUBLIC_WORD_COMPILES_TO_AN_EVEN_SMALLER_TWO_SCALAR_WARM_CLASSICAL_BOUNDARY_RECURRENCE
```

## One involutive superposition closes on a rank-two coherent secant

M130 implements the smallest exact chart broadening supported by the M129
obstruction. With `R` the mode-zero/mode-one swap on the symmetric sector, the
coupling is `C_eta=I+eta*R` and its exact inverse is
`(I-eta*R)/(1-eta^2)`. It sends the rank-one seed `x0^k` to the rank-two
secant `x0^k+eta*x1^k`.

```text
claim
    BOUNDED_EXACT_F17_INVOLUTIVE_COHERENT_SUPERPOSITION_COUPLING_EXPANDS_ONE_RANK1_VERONESE_PHASE_STATE_TO_A_RANK2_SECANT_SHARED_BY_TWO_NONCOMMUTING_MODULES_WITH_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_A_COMPILED_FOUR_DYNAMIC_SCALAR_CLASSICAL_RECURRENCE

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    INDEPENDENT_ORACLE_REEXECUTION

restoration
    resident carrier  EXACT_ALGEBRAIC_RESTORATION
    transient buffers NO_RESTORATION_CLAIM

source head
    e728386bef818bf92a497cf6ff1f183971e36e8c
```

Exact `Q(zeta17)` transactions pass at `k=4,8,16,32`; structural `F103/F137`
transactions pass through `k=128`. The accepted carrier uses two weights and
two 17-coordinate coherent vectors, or 36 resident field cells. The inverse
coupling uses at most four logical component records, or 72 field cells,
before exact duplicate cancellation. Two fixed 20-primitive modules have a
boundary-visible noncommuting order and consume the actual resident secant.

Only the final `(k-1,1,0,...,0)` occupation scalar is projected. Reverse
module execution followed by the exact coupling inverse restores exact zero
on the same backing, advances generation, and supports unrelated `PRIMARY
k=8` then `REUSE k=16` execution. Fresh/restored boundaries and complete
reported resource signatures agree without snapshot reload or inverse
history.

The independent oracle separately reconstructs the public compiler, coupling,
two-component recurrence, rank-two derivative minor, projection, inverse,
duplicate merge, and warm baseline. Full `H(4)=4,845` occupation reexecution
agrees in `Q(zeta17)`, `F103`, and `F137`; all 16 declared transaction
commitments and boundaries agree.

The first independent review rejected two overstatements before the scientific
commit. Four distinct generated terms after a second coupling do not prove
minimal secant rank four, so the repaired evidence leaves minimal rank and the
repeated-coupling rank law unestablished. The public weights can also be folded
into the mode-one endpoints, so the strongest fixed-word warm baseline retains
four total scalars, not six. That repaired baseline is independently sealed
and boundary matched. Descriptor-runtime classical software executes the
identical 36-cell recurrence.

Exact payload-height maxima remain package-local. Python containers, native
allocators, bigint internals, hashlib state, bit-operation complexity, and
whole-process peaks are excluded. M130 does not establish repeated-coupling
fixed-rank closure, M127 grid closure, arbitrary secant input closure, Gaussian
closure, CATVM custody, a distinct phase resource, computational advantage,
Small Wall crossing, physical waveform execution, physical bit replacement,
or unbounded catalytic computation.

The next experiment is:

```text
EXACT_SECOND_NONCOMMUTING_COUPLING_CATALECTICANT_RANK_LOWER_BOUND_OR_GAUSSIAN_CLOSURE_DIAGNOSTIC_WITHOUT_COMPONENT_ASSIGNMENT_ENUMERATION
```

The obstruction is:

```text
ONE_INVOLUTIVE_SUPERPOSITION_BROADENS_RANK1_TO_RANK2_BUT_THE_SECANT_RECURRENCE_IS_CLASSICALLY_IDENTICAL_FIXED_WORDS_COMPILE_TO_FOUR_TOTAL_FOLDED_ENDPOINT_SCALARS_AND_A_SECOND_COUPLING_GENERATES_FOUR_DISTINCT_TERMS_WITHOUT_YET_PROVING_MINIMAL_SECANT_RANK_ABOVE_TWO
```

## A second interleaved coupling has exact normalized secant rank four

M131 executes the selected M130 catalecticant diagnostic without component
assignment enumeration.

```text
claim
    BOUNDED_EXACT_SECOND_INTERLEAVED_INVOLUTIVE_COHERENT_SUPERPOSITION_COUPLING_WITH_MODULE_ORDER_NONCOMMUTATION_FORCES_CATALECTICANT_RANK4_ON_THE_DECLARED_F17_SECANT_PROGRAM_AND_CLOSES_ON_A_FOUR_COMPONENT_72_CELL_PHASE_CARRIER_WITH_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_AN_EIGHT_TOTAL_FOLDED_SCALAR_WARM_CLASSICAL_RECURRENCE

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    INDEPENDENT_ORACLE_REEXECUTION

restoration
    resident carrier  EXACT_ALGEBRAIC_RESTORATION
    transient buffers NO_RESTORATION_CLAIM

source head
    7fc59e806cc49298c44cd9d16ea2e309358d7de4
```

The second `I+eta*R` coupling is interleaved after module A. The two coupling
operators commute with each other because both are polynomials in the same
reflection `R`; the verified noncommutation is the coupling/module-A order. A
fixed normalized `4x4` catalecticant minor is nonzero in every declared case.
Each coherent component contributes a rank-one outer product, so the minor
gives rank at least four and the explicit four-component chart gives rank at
most four. The exact normalized divided-power secant rank is four. Ordinary
symmetric Waring interpretation is restricted to `Q(zeta17)` at
`k=4,8,16,32`; dual-field cases through `k=128` are structural.

The accepted carrier holds four weights plus four 17-coordinate vectors, or
72 field cells. The inverse coupling has at most eight logical component
records, or 144 field cells, before exact duplicate cancellation. Only the
final `(k-1,1,0,...,0)` occupation scalar is projected. Reverse module B,
second coupling, module A, and first coupling execution restores exact zero
on the same backing. Unrelated `PRIMARY k=8` then `REUSE k=16` execution
advances generation to two and agrees with a fresh carrier in boundary and
reported resource signature without snapshot reload or inverse history.

The independent oracle separately reconstructs both couplings, both module
halves, projection, inverse, duplicate merge, and the strongest warm baseline.
Its catalecticant determinant is a separate 24-term Leibniz expansion. Full
`H(4)=4,845` occupation reexecution agrees in `Q(zeta17)`, `F103`, and
`F137` through both couplings and modules and the complete reverse path. All
16 transaction commitments and all 16 rank certificates agree.

The strongest fixed-word warm classical baseline folds each public weight
into its mode-one endpoint and retains eight total scalars after 72 compiler
working cells. Descriptor-runtime classical software uses the identical
four-component 72-cell recurrence. Exact payload-height tuples remain
package-local; Python containers, allocator/native-library storage, bigint
internals, hashlib state, bit-operation complexity, and whole-process peaks
remain excluded.

M131 does not establish a third or unbounded coupling-rank law, fixed-rank
unbounded-depth closure, general Gaussian closure, arbitrary secant input
closure, CATVM custody, a distinct phase resource, computational advantage,
Small Wall crossing, physical execution, physical bit replacement, or
unbounded catalytic computation.

The next experiment is:

```text
EXACT_SYMBOLIC_ITERATED_COHERENT_COUPLING_SECANT_RANK_GROWTH_LAW_OR_FIXED_RANK_GAUSSIAN_PHASE_CHART_NO_GO_WITHOUT_COMPONENT_ENUMERATION
```

The obstruction is:

```text
THE_SECOND_INTERLEAVED_COUPLING_NOW_HAS_AN_EXACT_NORMALIZED_CATALECTICANT_RANK4_CERTIFICATE_AND_A72_CELL_CLOSED_TRANSACTION_BUT_EACH_ADDITIONAL_GENERIC_COUPLING_CAN_DOUBLE_GENERATED_COMPONENTS_AND_THE_FIXED_WORD_ALREADY_COMPILES_TO_AN_EIGHT_TOTAL_SCALAR_CLASSICAL_BOUNDARY_RECURRENCE
```

## Iterated affine-reflection couplings have exact exponential secant rank

M132 resolves the M131 rank-law question for one declared public family.

```text
claim
    BOUNDED_EXECUTION_AND_EXACT_SYMBOLIC_VANDERMONDE_CERTIFICATE_FOR_ITERATED_NONCOMMUTING_AFFINE_REFLECTION_SUPERPOSITION_COUPLING_SECANT_RANK_TWO_TO_THE_M_GROWTH_WITH_EXACT_RESTORATION_AND_REUSE_BUT_EXPLICIT_EXPONENTIAL_PHASE_COMPONENT_ENUMERATION_AND_MATCHED_R_WEIGHT_FULL_STATE_OR_TWO_SCALAR_FINAL_BOUNDARY_CLASSICAL_RECURRENCES

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    INDEPENDENT_ORACLE_REEXECUTION

restoration
    resident carrier  EXACT_ALGEBRAIC_RESTORATION
    transient buffers NO_RESTORATION_CLAIM

source head
    105d1a1944b975382ad1d084ac7bbfd30d317824
```

At level `j`, the one-particle affine involution sends slope `t` to
`a_j-t`, where `a_j=2^j-1`. The public support induction is exactly
`S_j=S_(j-1) union (a_j-S_(j-1))={0,...,2^j-1}`. The distinct reflections
do not commute. Each coupling `I+eta_j R_j` has exact inverse
`(I-eta_j R_j)/(1-eta_j^2)`.

For `r=2^m` and `k=2r-2`, the normalized `r x r` Hankel catalecticant
factors as `V diag(w) V^T`. Its determinant is the nonzero product of the
weights and squared Vandermonde differences. The lower bound `r` and explicit
`r`-component upper bound establish exact normalized divided-power secant
rank `2^m`. Production executes `m=1..6`, or ranks `2..64` and degrees
`2,6,14,30,62,126`, in exact `Q(zeta17)` and structurally in `F103/F137`.
Ordinary symmetric Waring interpretation is restricted to `Q(zeta17)`;
finite-field applicability requires `r<=p`.

The accepted carrier explicitly enumerates all `r` coherent components. It
uses `18r` resident field cells and at most `36r` logical coupling-transient
field cells. No separate truth table or assignment buffer is retained; that
narrow fact does not make the execution symbolic or compact. The symbolic
part is the rank certificate.

Only the final degree-one mode-one occupation is projected. Reverse execution
restores exact zero on the same backing without snapshot reload or inverse
history. Unrelated `PRIMARY m=3` then `REUSE m=4` execution advances
generation to two and matches fresh boundary and resource signatures.

The strongest matched full-state classical representation independently
stores `r` atomic weights on public support `0..r-1`; the implementation uses
`3r/2` named old-plus-new update cells. The separately checked dense moment
recurrence uses `2r-1` cells but is not the strongest compact baseline.
Final-boundary-only execution uses two dynamic moments, and a sealed word can
cache one scalar. The explicit phase chart therefore uses substantially more
field cells and establishes no advantage or distinct phase resource.

M132 does not establish an arbitrary interleaved-coupling rank law, a
fixed-rank closure, general Gaussian no-go, CATVM custody, Small Wall
crossing, physical waveform execution, physical bit replacement, or
unbounded catalytic computation. Exact payload-height tuples and complete
bit complexity remain outside the independently verified scope.

The next experiment is:

```text
EXACT_FIXED_RANK_GAUSSIAN_OR_STABILIZER_PHASE_CHART_FOR_NONCOMMUTING_SUPERPOSITION_COUPLINGS_OR_A_TRANSFERABLE_NO_GO_THAT_FORCES_A_DIFFERENT_NATIVE_PHASE_RESOURCE
```

The obstruction is:

```text
THE_DECLARED_NONCOMMUTING_AFFINE_REFLECTION_COUPLING_FAMILY_HAS_AN_EXACT_TWO_TO_THE_M_SECANT_RANK_LAW_AND_REQUIRES_EXPONENTIAL_EXPLICIT_PHASE_COMPONENTS_WHILE_ITS_STRONGEST_MATCHED_FULL_STATE_CLASSICAL_RECURRENCE_USES_ONLY_TWO_TO_THE_M_WEIGHTS_AND_ITS_FINAL_BOUNDARY_COLLAPSES_TO_TWO_CLASSICAL_MOMENTS
```

## The declared Gray-ordered weight tensor has exact bond dimension two

M133 removes M132's explicit component enumeration for the special weight
tensor, but it does not compact the associated coherent polynomial.

```text
claim
    BOUNDED_EXACT_GRAY_CODE_BOND2_PHASE_FACTOR_CHART_COMPRESSES_THE_DECLARED_ITERATED_NONCOMMUTING_AFFINE_REFLECTION_SUPERPOSITION_COMPONENT_WEIGHTS_FROM_TWO_TO_THE_M_EXPLICIT_COMPONENTS_TO_TWO_M_RESIDENT_PHASE_FACTOR_CELLS_ACROSS_DEPTH128_WITH_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_IS_AN_IDENTICAL_PUBLIC_MARKOV_FACTOR_CLASSICAL_RECURRENCE_AND_DOES_NOT_COMPACT_THE_GENERAL_COHERENT_POLYNOMIAL_OR_ESTABLISH_A_STABILIZER_RESOURCE

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    INDEPENDENT_ORACLE_REEXECUTION

restoration
    resident factor carrier EXACT_ALGEBRAIC_RESTORATION
    transient buffers       NO_RESTORATION_CLAIM

source head
    bd76fb590c0653a26de42a2db83c9dd89c4b5d90
```

Writing component index `t` in binary with bits `x_j` and setting
`x_(m+1)=0`, the reflected-copy weights satisfy
`w_m(t)=product_j eta_j^(x_j XOR x_(j+1))`. This is an exact open-boundary
nearest-neighbor MPS. Each internal edge has matrix
`[[1,eta_j],[eta_j,1]]` and nonzero determinant `1-eta_j^2`, so its minimal
maximum bond dimension is exactly two for declared depths at least two. The
independent oracle directly reconstructs small exact tensors and confirms
rank two across every internal cut.

The accepted `Q(zeta17)` path executes depths
`1,2,4,8,16,32,64,128`; `F103/F137` structural parity covers depths one
through six. It stores `2m` resident factor cells, with `m` nontrivial public
phase factors, and never expands the `2^m` component weights, catalecticant,
or dense operator. The selected boundary is contracted from resident factors
with four named working cells.

Each resident site is actually coupled from `(1,0)` to `(1,eta_j)`, consumed,
and inverted in reverse order to `(1,0)` before unseeding to exact zero on the
same backing. Unrelated depth-eight `PRIMARY` then depth-sixteen `REUSE`
execution agrees with a fresh carrier without snapshot or inverse history.
The reported reuse counter is package-local instrumentation, not CATVM
custody generation.

The first independent review found that lease ownership was recorded but not
enforced. The scientific source was repaired before sealing: contraction and
inverse now validate carrier type, stage, depth, family, and recomputed lease.
Executed null, wrong projection-owner, and wrong inverse-owner attacks pass.
The reviewer reexecuted the qualifier and a direct owner mutation and found
no second claim-breaking defect.

The strongest matched full-signature classical state is the identical `m`
exact factors. The selected final boundary uses two dynamic moments, while a
sealed public word compiles to three nonzero lower-triangular transfer
entries. Exact carrier and boundary payloads, descriptor bytes, and
commitment bytes are reproduced, but Python containers and allocator,
bigint/native/hash internals, bit-operation complexity, and whole-process
peak remain excluded.

M133 does not compact the general coherent binary form: M132's `2^m` secant
rank remains. Higher moments do not generally close on the two stored
boundary scalars. This package establishes neither arbitrary-boundary
closure, conventional stabilizer or Gaussian classification, CATVM custody,
a distinct phase resource, advantage, Small Wall crossing, physical
execution, physical bit replacement, nor unbounded catalytic computation.

The next experiment is:

```text
EXACT_HIGHER_ORDER_GRAY_PHASE_FACTOR_COUPLING_OR_COHERENT_MOMENT_PORT_CLOSURE_BEYOND_PAIRWISE_BOND2_WITH_MATCHED_CLASSICAL_TENSOR_BASELINE
```

The obstruction is:

```text
THE_SPECIAL_GRAY_ORDERED_COMPONENT_WEIGHT_TENSOR_HAS_EXACT_BOND2_AND_LINEAR_FACTOR_STORAGE_BUT_THE_ASSOCIATED_COHERENT_BINARY_FORM_RETAINS_SECANT_RANK_TWO_TO_THE_M_ARBITRARY_HIGHER_MOMENT_BOUNDARIES_DO_NOT_CLOSE_ON_TWO_SCALARS_AND_THE_IDENTICAL_PUBLIC_CLASSICAL_FACTOR_RECURRENCE_REMAINS
```

## Overlapping cubic factors raise exact bond dimension to three, not beyond a compact classical chain

M134 replaces M133's pairwise Gray factor law with the smallest growing
one-dimensional pure-cubic Boolean interaction.

```text
claim
    BOUNDED_EXACT_OVERLAPPING_NONAFFINE_CUBIC_BOOLEAN_PHASE_FACTOR_CHAIN_HAS_MINIMAL_MPS_BOND3_AND_TWO_M_RESIDENT_FACTOR_CELLS_ACROSS_DEPTH128_WITH_FINAL_ONLY_PARTITION_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_THE_IDENTICAL_THREE_SCALAR_CLASSICAL_RECURRENCE_AND_FIXED_LOGICAL_RANK_HIDES_GROWING_EXACT_BOUNDARY_WIDTH

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    INDEPENDENT_ORACLE_REEXECUTION

restoration
    resident factor carrier EXACT_ALGEBRAIC_RESTORATION
    transient buffers       NO_RESTORATION_CLAIM

source head
    15b9f81e999455d7156dc2d3a7d55b20a3d47fae
```

For public bits `x_0,...,x_(m+1)`, factor `j` is
`theta_j^(x_j x_(j+1) x_(j+2))`.  Its multiplicative third interaction
invariant equals the nonidentity `theta_j`, so it is not a unary or pairwise
Gray factorization on the same bits.  The two factors crossing an interior cut
give a rank-three matrix: two rows coincide and a three-by-three minor is, up
to sign, `(theta_left-1)(theta_right-1)`.  Full-tensor exact elimination
independently gives cut ranks `[2,2]` at depth one and
`[2,3,...,3,2]` at every checked larger depth.

The accepted path executes exact `Q(zeta17)` depths
`1,2,4,8,16,32,64,128` and dual-field structural depths one through eight.
It uses `2m` resident exact field cells and no accepted assignment tensor,
expanded component weights, or dense transfer.  Each factor pair is actually
coupled from `(1,0)` to `(1,theta_j)`, consumed, and inverted in reverse before
unseeding exact zero on the same backing.  Unrelated depth-eight then
depth-sixteen reuse agrees with fresh execution without snapshot or retained
inverse history.  Active use enforces type, stage, depth, family, and a
recomputed lease; the restoration generation is package-local, not CATVM
custody.

The independent oracle uses a separate four-state last-two-bit recurrence and
direct exact assignment enumeration through depth eight.  It also reconstructs
local and full-tensor ranks over `Q(zeta17)`, `F103`, and `F137`, exact
boundaries, payload tuples, commitments, and matched classical signatures.
Controls cover identity rank collapse, bond-two rejection, phase and schedule
perturbations, wrong/missing/reordered inverse, premature projection, null and
wrong-owner use, unavailable snapshot, and primary/reuse lease separation.

The selected partition boundary closes on the exact three-state recurrence
`A'=A+B`, `B'=A+C`, `C'=A+theta*C`, then `Z=2A+B+C`.  The strongest matched
classical selected-boundary implementation is this identical recurrence.  The
full signature needs only the `m` public factors; an arbitrary input in the
three-state chart needs three coefficients; and the fixed transaction can
cache one scalar.  Resident payload grows from 66 to 8448 bits and final
boundary payload from 36 to 1828 bits across the declared exact depths.

This is strict one-dimensional chain closure, not arbitrary cubic-hypergraph
or arbitrary-boundary closure.  It establishes no CATVM custody, distinct
phase resource, computational advantage, Small Wall crossing, physical
waveform execution, physical bit replacement, or unbounded computation.

The next experiment is:

```text
EXACT_SHARED_LATENT_BRANCHING_CUBIC_PHASE_FACTOR_HYPERTREE_CLOSURE_OR_SEPARATOR_RANK_GROWTH_WITH_MATCHED_CLASSICAL_HYPERTREE_BASELINE
```

The obstruction is:

```text
THE_DECLARED_NONAFFINE_CUBIC_PHASE_CHAIN_CLOSES_AT_EXACT_BOND3_BUT_THE_PHASE_CARRIER_STORES_TWICE_THE_PUBLIC_FACTOR_SIGNATURE_THE_SELECTED_BOUNDARY_HAS_AN_IDENTICAL_THREE_SCALAR_CLASSICAL_RECURRENCE_AND_EXACT_BOUNDARY_WIDTH_GROWS_WITH_DEPTH
```

## Two shared latent coordinates require an exact rank-four junction but not more than four classical scalars

M135 changes the separator geometry rather than extending M134's chain.  It
uses two shared Boolean latent coordinates `h,k`; each branch couples one
local anchor and two local leaves through `alpha^(h*u*v)` and
`beta^(k*u*w)`.  The branch map has a nonzero exact four-by-four minor
`(alpha-1)^2(beta-1)^2`.  Independent direct construction of the complete
two-branch eight-by-eight boundary tensor gives exact rank four over
`Q(zeta17)`, `F137`, and `F239`.

```text
claim
    BOUNDED_EXACT_TWO_SHARED_LATENT_NONAFFINE_CUBIC_PHASE_CYCLE_WITH_INTERLEAVED_WALSH_TRANSPORT_NONCOMMUTING_WITH_BRANCH_DIAGONALS_HAS_CERTIFIED_RANK4_JUNCTION_CLOSURE_ON_A_FOUR_CELL_RESIDENT_PORT_WITH_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_THE_IDENTICAL_FOUR_SCALAR_CLASSICAL_RECURRENCE

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    INDEPENDENT_ORACLE_REEXECUTION

restoration
    factor and shared-port carrier EXACT_ALGEBRAIC_RESTORATION
    transient buffers               NO_RESTORATION_CLAIM

source head
    74684d968304da3886a4e4309a35993bf3cc291f
```

Alternating unnormalised Walsh transports on the `h` and `k` axes occur only
between consumer groups.  Disabling transport, applying it before the
consumer, merging `h=k`, copying instead of sharing the port, or perturbing a
phase changes the final boundary.  Only the final scalar is projected.  The
lawful inverse consumes the actual resident factors and reverses every
diagonal and Walsh action before restoring factor and port state to exact zero
on the same backing.  Unrelated restored-carrier reuse agrees with fresh
execution without snapshot or retained inverse history.

The focused review required nested backing identity, projection count in the
zero predicate, complete inverse-order controls, and exact classical/phase
port-payload parity.  The repaired qualifier independently passes twenty
transactions and full lawful, reordered, wrong-Walsh, missing, wrong-factor,
ownership, projection, null, overmerge, undershare, and semantic attacks.

Exact branch counts are `2,4,8,16,32,64`; dual-field structural branch counts
are two through eight.  At `b` branches the carrier holds `4b` factor cells,
including `2b` public nontrivial phases, plus four shared-port cells.  The
factor payload grows from 264 to 8448 bits and the maximum exact port payload
from 205 to 13,915 bits.  The identical four-scalar classical recurrence has
the same measured port payload; its full signature stores only the `2b`
public factors, an arbitrary port input compiles to four row coefficients,
and the fixed transaction can cache one scalar.

The result is restricted to this declared two-latent cycle and scalar final
boundary.  It is not arbitrary cubic-hypergraph closure, arbitrary port
arity, CATVM custody, a distinct phase resource, computational advantage,
Small Wall crossing, physical execution, bit replacement, or unbounded
catalytic computation.

The next experiment is:

```text
EXACT_GROWING_SHARED_LATENT_CUBIC_PORT_SEPARATOR_RANK_LAW_OR_RELATION_PRESERVING_PHASE_QUOTIENT_WITH_MATCHED_CLASSICAL_JUNCTION_BASELINE
```

The obstruction is:

```text
THE_TWO_SHARED_LATENT_CUBIC_CYCLE_REQUIRES_EXACT_RANK4_BUT_CLOSES_ON_THE_IDENTICAL_FOUR_SCALAR_CLASSICAL_RECURRENCE_THE_PHASE_CARRIER_STORES_TWICE_THE_PUBLIC_FACTOR_SIGNATURE_AND_PHASE_AND_CLASSICAL_PORT_PAYLOAD_WIDTH_GROW_IDENTICALLY_WITH_DEPTH
```

## M136: Growing typed cubic separator rank no-go

M136 replaces larger rank-four fixtures with a symbolic all-`k` separator
law.  For `k` individually typed shared Boolean latent ports, the anchor-one
left and right branch minors are Kronecker products of
`[[1,1],[1,theta_i]]`.  Every accepted `theta_i` is nonidentity and the typed
Walsh separator is invertible, so the exact two-sided separator rank is
`2^k`.

```text
claim
    EXACT_K_TYPED_SHARED_LATENT_CUBIC_BRANCH_MAPS_WITH_INVERTIBLE_PHASE_WALSH_INTERLEAVING_HAVE_SEPARATOR_RANK_TWO_TO_THE_K_AND_REJECT_UNIFORM_EXACT_LINEAR_RELATION_QUOTIENTS_BELOW_TWO_TO_THE_K_WHILE_BOUNDED_DESCRIPTOR_PHASE_FACTOR_CARRIERS_RESTORE_AND_REUSE_WITHOUT_PORT_EXPANSION_BUT_LOW_TREEWIDTH_CLASSICAL_FACTOR_CONTRACTION_REMAINS

classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    INDEPENDENT_ORACLE_REEXECUTION

restoration
    resident cubic-factor carrier EXACT_ALGEBRAIC_RESTORATION
    transient buffers              NO_RESTORATION_CLAIM

source head
    e3af2700c46fec0e6dea330d9dc133b8144754eb
```

The production package executes exact `Q(zeta17)` arities
`1,2,4,8,16,32,64` and dual-field structural arities `1,2,4,8,16`.  The
independent oracle builds dense exact matrices only for verification through
`k=6` over `Q(zeta17)`, `F103`, and `F137`.  All left, right, Walsh, and
two-sided ranks equal `2^k`; left identity, right identity, and singular
transport mutations reduce rank.  All typed configurations are separated by
legal right continuations.

The accepted transaction stores `2k` public phases in `4k` exact paired
factor cells.  Its symbolic rank certificate uses the actual resident values
without materialising any exponential port, assignment family, dense minor,
or determinant value.  The actual inverse restores the same carrier backing
to exact zero, and unrelated restored-carrier reuse agrees with fresh
execution without snapshot or inverse history.

The theorem rejects only fixed or uniform exact field-linear quotients that
must preserve arbitrary incoming messages and the full declared typed
continuation family.  It is not a general storage lower bound.  The strongest
matched classical implementation contracts this local family in `O(k)`
two-by-two Kronecker factors; general descriptors retain the
`poly(descriptor size) * 2^treewidth` baseline.  The phase path therefore has
no established resource or advantage.

The next experiment is:

```text
EXACT_CUBIC_PHASE_POLYNOMIAL_WALSH_DERIVATIVE_BINOMIAL_FACTOR_CHART_MERGING_OR_RANK_GROWTH_WITH_MATCHED_CLASSICAL_TREEWIDTH_BASELINE
```

The obstruction is:

```text
THE_DECLARED_K_TYPED_CUBIC_BRANCH_FAMILY_HAS_AN_EXACT_TWO_TO_THE_K_UNIFORM_LINEAR_PORT_LOWER_BOUND_BUT_RANK_ALONE_DOES_NOT_BLOCK_THE_IDENTICAL_O_K_KRONECKER_CLASSICAL_CONTRACTION_SO_A_REPAIR_MUST_USE_A_NONLINEAR_PROGRAM_DEPENDENT_PHASE_CHART_OR_CHANGE_THE_NATIVE_UPDATE_LAW_WITHOUT_MOVING_EXPONENTIAL_STATE_ELSEWHERE
```

No nonlinear or program-dependent quotient no-go, operational exponential
phase-port closure, arbitrary cubic-hypergraph closure, CATVM custody,
distinct phase resource, computational advantage, Small Wall crossing,
physical execution, physical bit replacement, or unbounded catalytic
computation is established.

## M137: Exact rank-one cubic/Walsh derivative group-algebra closure

M137 follows the M136 linear-quotient obstruction with a nonlinear
program-dependent chart.  Public topology defines
`Q_n=sum_i z_(2i)z_(2i+1) mod 17`; exact Boolean-polynomial
canonicalization proves that every accepted cubic phase derivative is a
scalar multiple of this one signature.  One unresolved typed Walsh port then
closes in two 17-coordinate rows of `K[C17]`.

```text
Q(zeta17) branch pairs       1     2     4     8    16     32     64     128
phase/Walsh rounds           2     4     8    16    32     64    128     256
resident field cells        34    34    34    34    34     34     34      34
maximum payload bits      1096  1120  1616  3313  7607  16305  33763   68588
```

Twenty exact and dual-field structural transactions execute final-only
projection, exact inverse restoration on the same backing, and no-snapshot
reuse.  The inverse schedule is rematerialized from public topology and no
inverse history is retained.  The package-local restoration counter is not
a CATVM custody generation.

The separate oracle reconstructs all public fingerprints, boundaries, hidden
state commitments, inverse seeds, and exact payload tuples.  A distinct
17-residue multiplicity recurrence matches every transaction, and twelve
bounded direct Boolean-assignment sums through six branch pairs match both
implementations.  The focused review classification is
`INDEPENDENTLY_VERIFIED_STRICT_SCOPE` at
`INDEPENDENT_ORACLE_REEXECUTION`.

An independent second quadratic derivative has rank two, expands the
canonical full `C17^2` chart to 578 cells, and is rejected by the 34-cell
compiler.  This does not prove a universal 578-cell lower bound for every
alternative exact representation.  The strongest compact classical method
is the identical 34-coordinate group-algebra or 17-residue character
recurrence with the same exact payload law.  No distinct phase resource,
advantage, Small Wall crossing, CATVM custody, physical execution, physical
bit replacement, or unbounded computation is established.

The next experiment is:

```text
EXACT_TWO_SIGNATURE_CUBIC_WALSH_DERIVATIVE_C17_SQUARED_SEPARATION_RANK_GROWTH_OR_LOW_RANK_FACTOR_CHART_WITH_MATCHED_CLASSICAL_17_BY_17_RECURRENCE
```

It tests whether alternating two-signature updates admit an exact low
separation-rank phase factor chart, or whether separation rank grows toward
the canonical 17-by-17 coefficient surface against the identical strongest
classical recurrence.

## M138: Two-signature separation-rank and streamed-classical obstruction

M138 executes two independent disjoint quadratic derivative signatures on
one unresolved typed Walsh port in the canonical 578-cell
`K[C17 x C17]^2` chart. Exact `Q(zeta17)` cases and structural `F103/F137`
cases cover 62 transactions across `PRIMARY`, `REUSE`, and `ALTERNATE`
families. Each declared family reaches exact 17-by-17 matrix separation rank
17 at a tested ceiling, rejecting only uniform matrix-factor caps below 17
for these families.

The separate oracle reconstructs public descriptors, full coefficient
surfaces, ranks, final scalar boundaries, 289-residue streamed boundaries,
commitments, resident payload tuples, and inverse seeds. Six small direct
assignment sums agree over two finite fields. Classification is
`INDEPENDENTLY_VERIFIED_STRICT_SCOPE` at
`INDEPENDENT_ORACLE_REEXECUTION`.

Adversarial review repaired three accounting/comparison defects and one
control defect before sealing. The out-of-place Walsh update has 578 scratch
cells simultaneously live with the 578-cell carrier, for a 1,156-cell logical
peak. Projection declares 20 persistent named cells separately from its
289-cell rank buffer. A real wrong-order inverse on the projected resident
carrier fails exact restoration.

```text
Q(zeta17) branch pairs             1      2      4      8     16     32     64      128
resident payload bits          18504  18528  18744  19369  21392  25842  40028   110637
resident plus scratch bits     37004  37040  37364  38323  41501  49067  76295   216450
```

The matched classical comparison is a memory/work frontier. Retain-both
reproduces the full diagnostic; an executed 289-residue stream reproduces
every final scalar with eight named dynamic exact fields plus 17 public
integer counts; and an explicitly unexecuted conservative 320-field
rematerialized construction can reproduce ranks and commitment with more
work. No streamed payload tuple or dominant classical point is claimed.

The resident carrier restores exact canonical zero on the same backing and
supports unrelated reuse without snapshot or inverse history. The package-
local restoration count is not CATVM custody. Full separation rank does not
prove a universal 578-cell minimum or a general rank-r lower bound. No
distinct phase resource, advantage, Small Wall crossing, physical execution,
physical bit replacement, or unbounded computation is established.

The next mechanism must change the native phase coupling law or use a
nonlinear relation quotient beyond additive `C17` signature axes; adding more
signature axes or dense group-algebra cells would repeat the diagnosed
obstruction.

## M139: Nonlinear anisotropic radial phase quotient

M139 changes the coupling law from additive signature axes to the nonlinear
anisotropic norm relation `Q(x,y)=x^2-3y^2` on `F17^2`.  Its 289 public
coordinates form 17 exact norm shells of sizes `1,18,...,18`.  Quartic radial
phases and the declared normalized anisotropic phase Fourier transform close
exactly on one 17-cell unresolved radial carrier and do not commute on the
tested programs.

Seven exact `Q(zeta17)` transactions cover depths through 64; 48 structural
transactions cover three public families over `F103` and `F137` through depth
128.  An independent implementation reconstructs the geometry, schedules,
forward/inverse recurrence, commitments, resident payload maxima, and selected
dense 289-coordinate controls.  All selected dense controls match every one
of the 17 shell values and the final boundary.  Classification is
`INDEPENDENTLY_VERIFIED_STRICT_SCOPE` at
`INDEPENDENT_ORACLE_REEXECUTION`.  Update-scratch payload tuples remain
package-local.

```text
exact depth                         1     2     4     8     16     32     64
maximum resident payload bits   2203  2243  4364  8709  17552  35394  70992
resident field cells              17    17    17    17     17     17     17
update scratch cells              20    20    20    20     20     20     20
resident plus update live         37    37    37    37     37     37     37
```

Projection uses six scratch cells, for 23 carrier-plus-projection cells.  The
accepted compiler retains no coordinate table, assignment expansion, or truth
table, but it does retain a 289-cell public Fourier kernel.  Public compilation
visits 5,202 coordinates per algebra.  Exhaustive verification visits 83,810,
uses 867 dense control buffer cells, and is outside the accepted transaction.
The whole three-algebra verification package retains 1,734 kernel-plus-dense-
buffer cells, excluding arithmetic expression temporaries, Python and native
storage, bigint internals, and whole-process memory.

The exact inverse restores canonical zero on the same carrier backing and an
unrelated program reuses it without snapshot or inverse history.  The
restoration counter is package-local and does not establish CATVM custody.

The strongest executed compact classical baseline is the identical 17-cell
radial recurrence with the same 289-cell kernel, scratch law, and
`O(289*depth)` work.  M139 therefore establishes no distinct phase resource,
advantage, Small Wall crossing, physical execution, physical bit replacement,
or unbounded computation.

The next obstruction is the retained public kernel together with growing
exact payload and the identical classical recurrence.  The selected repair is:

```text
EXACT_MATRIX_FREE_ANISOTROPIC_RADIAL_PHASE_FOURIER_FROM_KLOOSTERMAN_GENERATOR_WITH_MATCHED_CLASSICAL_RECURRENCE
```

## M140: Matrix-free anisotropic radial phase Fourier

M140 removes M139's retained 289-cell public Fourier kernel by factoring the
anisotropic radial transform through 16 nonzero `F17` character parameters.
The accepted path performs 272 source and 272 target contractions per
Fourier, retains only the exact `17^-1` generator scalar, and neither calls a
per-entry kernel generator nor enumerates the 289 coordinates.

Seven exact `Q(zeta17)` transactions through depth 64 and 48 structural
transactions over `F103` and `F137` through depth 128 restore exactly.  The
same backing supports an unrelated reuse without snapshot or inverse history.
The independent oracle reconstructs schedules, factorized forward/inverse
execution, commitments, and resident payload maxima; checks all 289 entries
over both finite fields and selected exact entries against direct coordinate
sums; executes two dense 289-coordinate controls; and rejects missing-term
and wrong-sign mutations.  Classification is
`INDEPENDENTLY_VERIFIED_STRICT_SCOPE` at
`INDEPENDENT_ORACLE_REEXECUTION`.

```text
exact depth                         1     2     4     8     16     32     64
maximum resident payload bits   2203  2243  4364  8709  17552  35394  70992
resident field cells              17    17    17    17     17     17     17
update scratch cells              38    38    38    38     38     38     38
resident plus update live         55    55    55    55     55     55     55
```

Including the retained generator scalar gives 56 counted field cells.
Projection uses six scratch cells, for 23 carrier-plus-projection cells.  The
verification-only coordinate kernels and dense controls are outside this
accepted resource path.  Python/native allocation, bigint workspace, hashlib
internals, and whole-process memory remain excluded.

The strongest executed compact classical baseline is the identical
matrix-free 17-coordinate recurrence with the same `O(544*depth)` work and
scratch law.  M139's retained-kernel recurrence remains a relevant frontier
point: 289 retained kernel cells, 20 update scratch cells, and
`O(289*depth)` warm products.  Removing the kernel is therefore not evidence
of a computational advantage.

The current obstruction is depth-growing exact payload together with the
identical compact classical recurrence.  The selected next experiment is:

```text
EXACT_HEIGHT_STABLE_ANISOTROPIC_RADIAL_PHASE_COUPLING_OR_NO_GO_WITH_MATRIX_FREE_CLASSICAL_FRONTIER
```

M140 establishes neither CATVM custody, a general nonlinear quotient, a
distinct phase resource, computational advantage, Small Wall crossing,
physical execution, physical bit replacement, nor unbounded computation.

## M141: Shared-scale exact-height diagnostic

M141 represents each exact M140 final state using 272 integral cyclotomic
coefficients, one shared base-17 denominator exponent, and one greatest-common
`1-zeta17` exponent.  It reconstructs the exact 17-cell resident state before
final projection and inverse.  All 21 cases span three public program families
and depths `1,2,4,8,16,32,64`.

The shared denominator reduces repeated-denominator final-state accounting by
more than half in every declared case.  This does not solve exact height.  All
tested depth-2-and-above states have zero common pi content.  At depth 64 the
shared-denominator payload is 35,425, 35,369, and 35,380 bits for `PRIMARY`,
`REUSE`, and `ALTERNATE`; maximum residual signed width is 133, 134, and 133
bits, and the denominator exponent is 32.

The independent retained-kernel recurrence reproduces all commitments,
boundaries, denominator powers, pi valuations, widths, and payload records in
252 comparisons, restores every exact seed, and detects a one-bit payload
mutation.  Classification is `INDEPENDENTLY_VERIFIED_STRICT_SCOPE` at
`INDEPENDENT_ORACLE_REEXECUTION`.

The resident carrier has `EXACT_ALGEBRAIC_RESTORATION`; transient scale and
oracle buffers have `NO_RESTORATION_CLAIM`.  Same-backing unrelated reuse uses
no snapshot or inverse history.  The identical shared-scale-normalized
matrix-free 17-coordinate recurrence remains the strongest compact classical
baseline.

This is a bounded representation obstruction, not a universal height theorem
or an optimality result.  It establishes no CATVM custody, distinct phase
resource, computational advantage, Small Wall crossing, physical execution,
physical bit replacement, or unbounded computation.  The next selected test
is exact global cyclotomic-unit balancing on this anisotropic shared-scale
state, with every unit ledger and rematerialization cost matched in the
identical classical normalization.

## M142: Global cyclotomic-unit balance bounded no-go

```text
BOUNDED_EXACT_LOG_FREE_49_DIRECTION_AND_FIXED_PRECISION_LOG_EMBEDDING_GLOBAL_CYCLOTOMIC_UNIT_BALANCERS_AGREE_ON_ALL_21_COMMON_PI_FACTORED_ANISOTROPIC_F17_RADIAL_FINAL_STATES_REPAIR_THE_DEPTH1_PI_RESIDUAL_BLOWUP_BUT_SELECT_THE_IDENTITY_AT_EVERY_DEPTH2_PLUS_CASE_THROUGH_DEPTH64_WHILE_EXACT_RECONSTRUCTION_RESTORATION_REUSE_AND_THE_IDENTICAL_CLASSICAL_BALANCING_PATH_REMAIN
```

The exact log-free 49-direction unit line descent and a separate 65,536-bit
log-embedding proposal with exact acceptance agree on every one of the 21
M141 final states.  At depth 1 they select unit ledger
`[1,1,1,1,1,1,1]` and repair the common-pi residual artifact:

```text
family       M141 pi residual bits   balanced all-ledger bits
PRIMARY                         3579                        542
REUSE                           3553                        554
ALTERNATE                       3558                        599
```

Every tested depth-2-through-64 state selects the identity ledger under both
balancers.  The ledger-inclusive depth-64 payload remains 35,433 bits for
`PRIMARY`, 35,377 for `REUSE`, and 35,388 for `ALTERNATE`.  Thus this declared
global-unit route does not bound the exact residual height.

The independent retained-kernel oracle matches 231 declared fields, verifies
98 adjacent signed unit directions for each of 21 states, restores every
exact seed, and detects a unit-ledger mutation.  Classification is
`INDEPENDENTLY_VERIFIED_STRICT_SCOPE` at
`INDEPENDENT_ORACLE_REEXECUTION`.  The neighborhood certificates do not prove
global cyclotomic-unit optimality or a lower bound for every exact
representation.

The resident 17-cell carrier has `EXACT_ALGEBRAIC_RESTORATION`; unit-search
and scale buffers have `NO_RESTORATION_CLAIM`.  Exact reconstruction occurs
before final-only projection and inverse, and unrelated same-backing reuse
uses neither snapshot nor inverse history.  Accounting includes the 272
integer coefficients, nine scale/unit ledger integers, 49 compiled exact
directions, and the 8,177-bit unit table.  The strongest compact classical
baseline is the identical shared-scale and exact 49-direction balancing
recurrence.

M142 establishes no CATVM custody, distinct phase resource, computational
advantage, Small Wall crossing, physical execution, physical bit replacement,
or unbounded computation.  The next experiment must change the primitive
phase state/update law rather than extend this exhausted scale-and-unit gauge
fixture family.

## M143: Numerical unit-phasor-pair chart

Claim:

```text
BOUNDED_NUMERICAL_UNIT_PHASOR_PAIR_CHART_REPRESENTS_NORMALIZED_ANISOTROPIC_F17_RADIAL_PHASE_FOURIER_STATE_IN_FIXED34_PHASE_ANGLE_COORDINATES_ACROSS21_CASES_THROUGH_DEPTH4096_WITH_HISTORY_FREE_NUMERICAL_PHYSICAL_STATE_RESTORATION_AND_REUSE_BUT_REQUIRES_TRANSCENDENTAL_CANONICAL_CHART_ARITHMETIC_AND_HAS_THE_IDENTICAL17_COMPLEX_COMPACT_CLASSICAL_RECURRENCE
```

M143 stores each normalized radial coefficient as two `float64` unit-phasor
angles.  Across 21 declared cases in three families through depth 4096, the
carrier stays at 34 angles or 272 resident bytes.  Maximum boundary error
against the identical matrix-free 17-complex recurrence is `2.242e-11`,
maximum final-state error is `1.658e-12`, and maximum transaction restoration
error is `5.764e-12`, all below predeclared tolerances.

The actual inverse acts on the resident angle array without reset, reload, or
history.  Unrelated depth-1537 same-backing reuse differs from fresh execution
by `1.646e-12`; 100 consecutive depth-64 cycles restore within `8.189e-12`.
The correct class is `NUMERICAL_PHYSICAL_STATE_RESTORATION`, not exact
algebraic restoration.

An independent long-double oracle reconstructs the radial Fourier from all
4,913 public coordinate visits, checks involution to `3.152e-19`, reexecutes
all 21 cases and both reuse controls, and detects a boundary mutation.  The
result is `INDEPENDENTLY_VERIFIED_STRICT_SCOPE` at
`INDEPENDENT_ORACLE_REEXECUTION`.

The matched compact baseline also uses 272 resident bytes and the identical
17-complex matrix-free recurrence.  The phase chart adds exponentials,
magnitudes, arguments, and arccosines around each Fourier.  The current
obstruction is therefore the explicit complex decode and transcendental
recharting, not resident-cell growth in the tested cases.

M143 establishes no exact semantics, unbounded numerical stability, CATVM
custody, distinct phase resource, advantage, Small Wall crossing, physical
execution, physical bit replacement, or unbounded computation.  The next
test is a direct angle-domain interference update or a bounded no-go against
the identical complex recurrence; increasing fixture depth alone is not the
next experiment.

## M144: Streamed native-angle radial interference

Claim:

```text
BOUNDED_STREAMED_ANGLE_DOMAIN_ANISOTROPIC_F17_RADIAL_INTERFERENCE_CONSUMES_FIXED34_RESIDENT_PHASE_ANGLES_WITHOUT17_COMPLEX_STATE_DECODE_OR_RETAINED_DENSE_KERNEL_ACROSS21_CASES_THROUGH_DEPTH4096_WITH_HISTORY_FREE_NUMERICAL_RESTORATION_AND_REUSE_BUT_USES_TWO_CARTESIAN_ACCUMULATORS_PER_OUTPUT_AND_IS_STRICTLY_MORE_TRIGONOMETRIC_WORK_THAN_THE_IDENTICAL17_COMPLEX_MATRIX_FREE_RECURRENCE
```

M144 pairs the public nonzero character indices `p` and `-p` to form a real
radial kernel.  It consumes the 34 resident phase angles directly, streams one
output through two Cartesian scalar accumulators, and returns that output to
the two-angle chart.  No 17-complex accepted state or dense kernel is retained.

Across the 21 declared cases through depth 4096, maximum boundary error is
`2.607e-11` against the matrix-free baseline and `2.012e-11` against the
streamed-complex baseline.  Maximum transaction restoration is `2.102e-12`.
Unrelated depth-1537 same-backing reuse differs from fresh execution by
`8.638e-13`; 100 same-backing depth-64 cycles restore within `3.752e-12`.
The accepted class is `NUMERICAL_PHYSICAL_STATE_RESTORATION`; no reset,
reload, snapshot, or inverse history is used.

The independent long-double oracle imports no production or predecessor
module.  It reconstructs the dense real kernel from 2,312 public pair visits,
checks character pairing and involution, performs 474 declared comparisons,
enforces the production restoration/reuse fields, independently repeats both
reuse paths, and detects a kernel mutation.  The result is
`INDEPENDENTLY_VERIFIED_STRICT_SCOPE` at
`INDEPENDENT_ORACLE_REEXECUTION`.

The native path reaches 1,018 named live bytes including program.  The
work-minimizing matrix-free complex baseline reaches 1,770 named live bytes,
but the equal-memory streamed real-kernel complex baseline reaches only 986.
It also avoids the native path's 1,156 input-phasor and 51 chart trigonometric
calls per Fourier.  Removing the decoded full state therefore does not
establish a representation or work advantage.

The next selected test is topology-compiled local polar Givens phase coupling
without global Cartesian accumulation, or a bounded no-go against matched
complex Givens execution.  M144 establishes no exact semantics, unbounded
numerical stability, CATVM custody, distinct phase resource, computational
advantage, Small Wall crossing, physical execution, physical bit replacement,
or unbounded catalytic computation.

## M145: Topology-compiled local polar Givens coupling

Claim:

```text
BOUNDED_TOPOLOGY_COMPILED_LOCAL_POLAR_GIVENS_PHASE_COUPLING_FACTORS_WEIGHTED_ANISOTROPIC_F17_RADIAL_INTERFERENCE_IN136_TWO_CELL_COUPLERS_WITH_FIXED34_RESIDENT_PHASE_ANGLES_ACROSS21_CASES_THROUGH_DEPTH4096_WITH_HISTORY_FREE_NUMERICAL_RESTORATION_AND_REUSE_BUT_REQUIRES_A2312_BYTE_PUBLIC_PLAN_LOCAL_CARTESIAN_REGISTERS_AND_HAS_THE_IDENTICAL_COMPLEX_GIVENS_RECURRENCE
```

Public shell normalization converts the radial transform to a real symmetric
orthogonal operator.  A deterministic QR schedule compiles 136 two-cell
Givens rotations and 17 signs.  The accepted path applies these directly to
the resident phase pairs and retains neither a 17-complex state nor a dense
kernel.

Across 21 declared cases through depth 4096, maximum boundary error against
the identical complex-Givens recurrence is `7.280e-11`, maximum state error is
`5.641e-12`, and maximum single-transaction restoration error is `6.853e-12`.
Unrelated depth-1537 same-backing reuse differs from fresh execution by
`7.230e-13`; 100 same-backing depth-64 cycles restore within `1.368e-11`.
The supported restoration class is `NUMERICAL_PHYSICAL_STATE_RESTORATION`
for the declared seeded carrier, not exact algebraic restoration.

The independent oracle reconstructs the weighted operator in long-double
arithmetic, independently compiles and executes the float64 plan, performs 289
case, field, resource, and mutation checks, and detects omitted weighting and
plan or inverse corruption.  The result is
`INDEPENDENTLY_VERIFIED_STRICT_SCOPE` at
`INDEPENDENT_ORACLE_REEXECUTION`.

The public plan occupies 2,312 retained bytes and compiles within a 4,960-byte
named full-lifecycle peak.  Warm native execution reaches 3,055 named bytes,
while the identical in-place 17-complex Givens baseline reaches 2,959 and
performs no Fourier input-phasor or chart trigonometry.  The local phase law
therefore removes global Cartesian accumulation but establishes no storage or
work advantage.

The next test is a gauge-transported phase-only local Givens lift on the
weighted unit sphere, or a bounded zero-fiber obstruction against such a lift.
M145 establishes no exact semantics, unbounded numerical stability, CATVM
custody, distinct phase resource, computational advantage, Small Wall
crossing, physical execution, physical bit replacement, or unbounded
catalytic computation.

## M149: Exact C17 phase-fiber port convolution

Claim:

```text
BOUNDED_EXACT_C17_ONE_HOT_PHASE_FIBER_PORT_CONVOLUTION_CONSUMES_ONE_RESIDENT_PORT_ACROSS16_TARGETS_AND_RECIPROCAL_NONCOMMUTING_UPDATES_WITHOUT_SCALAR_ANGLE_OR_EXPONENT_READOUT_RELATION_TABLE_ASSIGNMENT_EXPANSION_OR_RETAINED_PLAN_ON_FIXED51_BY17_PHASE_ORBIT_COORDINATES_THROUGH_DEPTH4096_WITH_EXACT_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_AN_EXECUTED_IDENTICAL51_RESIDUE_RECURRENCE_WITH17X_LESS_RESIDENT_STATE
```

M149 stores 51 logical C17 phase factors as 867 one-hot orbit coordinates.
One resident hub port directly controls 16 target triplets through cyclic
convolution, and the updated target ports then control reciprocal hub updates.
OUT and IN do not commute. The accepted update path performs no resident angle
or exponent readout and uses no relation table, assignment expansion, or
retained plan.

All 21 cases through depth 4096 match an executed 51-residue recurrence
exactly in final expanded bytes and final cyclotomic boundary. All 867 carrier
cells restore exactly on the same backing. Unrelated depth-1537 reuse and 100
depth-64 reuse cycles also restore exactly without snapshot, inverse history,
restoration baseline, or post-inverse canonicalization.

The independent bitmask oracle imports neither production nor M145. It
separately reconstructs the public schedule, one-hot group action, compact
residue recurrence, byte commitments, boundaries, controls, inverse, and both
reuse paths. All 323 comparisons pass. Classification is
`INDEPENDENTLY_VERIFIED_STRICT_SCOPE` at
`INDEPENDENT_ORACLE_REEXECUTION`; resident restoration is
`EXACT_ALGEBRAIC_RESTORATION`, and transient buffers have
`NO_RESTORATION_CLAIM`.

The phase-orbit carrier uses 867 resident bytes and a 1,479-byte named warm
peak. Its forward step performs 179 logical convolutions or 51,731 coordinate
multiplications. The matched residue recurrence uses 51 resident bytes, 179
modular additions, and a 276-byte named warm peak. The pure C17 phase orbit
therefore has an exact 17-times-smaller classical quotient and establishes no
distinct resource or advantage.

The rejected raw complex-coordinate attempt first exceeds its predeclared
unit-norm tolerance at depth 64 because radial floating error becomes control
magnitude. No normalization was inserted or misclassified as exact inverse
restoration.

The next experiment must introduce coherent superposition across multiple
C17 phase fibers with native interference closure, or establish a matched
fixed-rank classical factor no-go. M149 establishes no coherent multi-orbit
superposition, general relational contraction, CATVM custody, computational
advantage, Small Wall crossing, physical execution, physical bit replacement,
or unbounded catalytic computation.

## M146: Stateful weighted three-phasor gauge lift

Claim:

```text
BOUNDED_STATEFUL_WEIGHTED_THREE_PHASOR_GAUGE_CHART_LIFTS_LOCAL_F17_GIVENS_COUPLING_ACROSS_BASE_ZERO_IN_FIXED51_RESIDENT_PHASE_ANGLES_FOR_THE_DECLARED_MAGNITUDE_ENVELOPE_ACROSS21_CASES_THROUGH_DEPTH4096_WITH_HISTORY_FREE_NUMERICAL_RESTORATION_AND_REUSE_BUT_ADDS17_GAUGE_CELLS_RETAINS_LOCAL_CARTESIAN_RECHARTING_AND_HAS_THE_IDENTICAL_SMALLER_COMPLEX_GIVENS_RECURRENCE
```

The M145 two-phasor chart cannot be lifted injectively through an actual
zero/nonzero Givens transition: the input fiber is uncountable and the generic
output fiber has four elements.  M146 adds one resident gauge phasor per
coordinate with weight `1/32`.  For base magnitude at most `15/16`, the
remaining two-phasor chart is representable for every gauge; at base zero its
radius is `1/31`.  Local Givens operations transport the gauges by an
invertible public counter-rotation.

The carrier has 51 resident `float64` phase angles.  Across 21 cases through
depth 4096, maximum boundary error against the matrix-free compact comparison
is `7.313e-11`; maximum single-transaction restoration error is `1.375e-11`.
The same backing is restored and reused without snapshot, inverse history, or
retained restoration baseline.  The class is
`NUMERICAL_PHYSICAL_STATE_RESTORATION`; transient buffers have
`NO_RESTORATION_CLAIM`.

An oracle independent of production and predecessor code reconstructs the
weighted operator in long-double arithmetic, compiles a separate QR plan,
reexecutes the 51-angle carrier, and independently checks the zero-fiber
repair, controls, and reuse.  All 233 comparisons pass.  Classification is
`INDEPENDENTLY_VERIFIED_STRICT_SCOPE` at
`INDEPENDENT_ORACLE_REEXECUTION`.

The accepted path reaches 3,223 named warm bytes and still uses 24 local
Cartesian/chart scratch cells.  Executed compact classical comparisons reach
3,055 named warm bytes for the identical Givens plan, 1,767 for the
matrix-free work frontier, and 983 for the streamed real-kernel memory
frontier.  M146 therefore repairs phase-fiber loss but does not establish a
storage or work advantage.

The next experiment must make the gauge causally visible to a later
noncommuting phase module and distinguish equal-base/different-gauge carriers,
or show that the mechanism collapses to a matched compact 51-scalar
recurrence.  M146 establishes no global unit-sphere chart, exact semantics,
CATVM custody, distinct phase resource, computational advantage, Small Wall
crossing, physical execution, physical bit replacement, or unbounded
catalytic computation.

## M147: Fiber-active gauge-dependent phase shear

Claim:

```text
BOUNDED_FIBER_ACTIVE_GAUGE_DEPENDENT_PHASE_SHEAR_MAKES_EQUAL_BASE_DIFFERENT_GAUGE_F17_CARRIERS_BOUNDARY_DISTINGUISHABLE_WHILE_INTERLEAVING_NONCOMMUTING_PHASE_AND_LOCAL_GIVENS_MODULES_ON_FIXED51_RESIDENT_PHASE_ANGLES_ACROSS21_CASES_THROUGH_DEPTH4096_WITH_HISTORY_FREE_NUMERICAL_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_AN_EXECUTED_MATCHED17_COMPLEX_PLUS17_GAUGE_SCALAR_RECURRENCE
```

M147 applies a gauge-dependent diagonal phase shear before each local Givens
transform.  The shear preserves base magnitude, leaves the actual gauge
resident, and uses no relation table.  Phase/shear and shear/Givens order
mutations both change the boundary.

The equal-base/different-gauge witness starts with base agreement
`5.888e-16`, then separates final states by `0.0825` and boundaries by
`0.0493`.  Both actual 51-angle carriers restore.  This establishes that the
base-only 17-complex quotient cannot represent the declared variable-gauge
carrier family.

All 21 cases through depth 4096 match an executed 17-complex-plus-17-gauge
scalar recurrence.  Maximum state and boundary errors are `5.361e-12` and
`6.908e-11`; maximum single-transaction restoration error is `2.839e-11`.
Unrelated depth-1537 and 100-cycle depth-64 reuse pass on the same backing
without snapshot, retained history, or restoration baseline.

The independent implementation imports neither M147 nor M146, shares only
the established M145 public program and plan compiler, and separately
reexecutes the chart, shear, inverse, witness, scalar recurrence, controls,
and reuse in 146 comparisons.  Classification is
`INDEPENDENTLY_VERIFIED_STRICT_SCOPE` at
`INDEPENDENT_ORACLE_REEXECUTION`.

The 51-angle path reaches 3,223 named warm bytes and retains 24 local
Cartesian/chart scratch cells.  The matched 51-scalar recurrence reaches
3,191 named warm bytes and avoids phase-chart decode and reencode.  Thus the
gauge is causally active but no distinct phase resource is established.

The next experiment is a reversible triangular shared-gauge angle coupling
that removes Cartesian recharting, or a bounded bisimulation to the identical
51-angle recurrence.  M147 establishes no optimal classical lower bound,
CATVM custody, computational advantage, Small Wall crossing, physical
execution, physical bit replacement, or unbounded catalytic computation.

## M148: Direct phase-angle triangular shared-gauge coupling

Claim:

```text
BOUNDED_DIRECT_PHASE_ANGLE_TRIANGULAR_SHARED_GAUGE_COUPLING_USES_ONE_RESIDENT_GAUGE_ACROSS16_TARGETS_AND_RECIPROCAL_NONCOMMUTING_HUB_UPDATES_WITHOUT_COMPLEX_DECODE_CARTESIAN_SCRATCH_CHART_RECONSTRUCTION_OR_RETAINED_GIVENS_PLAN_ON_FIXED51_PHASE_ANGLES_ACROSS21_CASES_THROUGH_DEPTH4096_WITH_HISTORY_FREE_NUMERICAL_RESTORATION_AND_REUSE_BUT_IS_BYTE_BISIMULATED_BY_AN_EXECUTED_IDENTICAL51_ANGLE_CLASSICAL_RECURRENCE
```

M148 applies only common translations to resident phase triplets.  One
rotating hub gauge controls 16 target triplets, then those updated target
gauges control the hub.  The OUT and IN layers do not commute.  Updates use no
complex-base decode, Cartesian chart scratch, chart reconstruction, retained
Givens plan, relation table, or assignment expansion.

Two carriers with decoded base agreement `5.888e-16` and gauge separation
`0.714` reach final decoded-state separation `0.0131` and boundary separation
`0.0462` under the same public depth-4 program.  Both actual carriers restore
within `2.704e-15`; the gauge fiber is therefore causally relevant for this
declared family.

All 21 cases through depth 4096 reproduce the separately executed identical
51-angle recurrence byte-for-byte.  Maximum boundary difference is
`4.826e-15`, maximum transaction restoration is `9.860e-13`, unrelated
depth-1537 reuse agrees with fresh execution within `1.714e-13`, and 100
same-backing depth-64 cycles restore within `2.780e-13`.  The resident carrier
has 51 `float64` angles (408 bytes), no retained public plan, four update
scratch scalars, and a 750-byte named warm peak including the public program.

The independent oracle imports neither M148 nor M146 and shares only M145's
established phase exponent and shell weights.  It separately reconstructs
the seed chart, triangular schedule, commitments, boundaries, inverse,
witness, mutation controls, reuse, conditioning attack, and matched recurrence
in 129 comparisons.  Classification is
`INDEPENDENTLY_VERIFIED_STRICT_SCOPE` at
`INDEPENDENT_ORACLE_REEXECUTION`; the carrier restoration class is
`NUMERICAL_PHYSICAL_STATE_RESTORATION`, while transient buffers have
`NO_RESTORATION_CLAIM`.

The initially attempted strength `5/32` is rejected from the accepted path:
its depth-4096 inverse error reaches `1.925`.  The accepted `1/32` strength
restores within `9.860e-13` for the same attack.  This is a bounded numerical
envelope, not unbounded stability.

The direct phase primitive removes M147's recharting overhead, but the exact
byte bisimulation and equal resource law establish no distinct phase
resource or advantage.  The next experiment is:

```text
UNRESOLVED_MULTI_CONSUMER_PHASE_FIBER_PORT_CONTRACTION_WITHOUT_SCALAR_GAUGE_READOUT_OR_BOUNDED_COLLAPSE_TO_MATCHED_COMPACT_PHASE_FACTOR_RECURRENCE
```

It must keep one typed phase-fiber port unresolved across multiple
noncommuting consumers and close it only at the boundary without relation
tables or assignment expansion.  M148 establishes no optimal classical lower
bound, general relational contraction, exact algebraic semantics, unbounded
numerical stability, CATVM custody, computational advantage, Small Wall
crossing, physical execution, physical bit replacement, or unbounded
catalytic computation.

## M150: F103[C17] superposition interference and spectral factor no-go

Claim:

```text
BOUNDED_EXACT_F103_C17_PHASE_ORBIT_SUPERPOSITION_CONVOLUTION_SHEARS_KEEP_ONE_MULTI_COORDINATE_RESIDENT_PORT_UNPROJECTED_ACROSS16_TARGETS_AND_RECIPROCAL_NONCOMMUTING_CONSUMERS_WITH_EXACT_DESTRUCTIVE_INTERFERENCE_RESTORATION_AND_REUSE_THROUGH_DEPTH1024_BUT_FACTORIZE_EXACTLY_INTO_AN_EXECUTED17_MODE_CLASSICAL_RECURRENCE_WITH_EQUAL_RESIDENT_DIMENSION_AND_LOWER_CONVOLUTION_WORK
```

M150 removes M149's one-hot restriction. Each of 51 logical phase factors is a
general 17-coordinate element of `F103[C17]`. One resident port controls 16
target triangular convolution shears; updated target slot-one factors then
control reciprocal hub shears. OUT and IN do not commute. The path performs no
resident angle, exponent, or scalar readout and exposes only the final
17-coordinate boundary.

All 18 family/depth cases through depth 1024 match an executed 17-mode F103
number-theoretic recurrence in every final coefficient cell and boundary. The
accepted cases contain 36,893 exact cancellation events. Every 867-cell
carrier restores exactly on the same backing without snapshot, inverse
history, retained baseline, or post-inverse canonicalization. Unrelated
depth-613 reuse matches fresh state, boundary, and resource signature, and 100
depth-16 cycles restore exactly.

The pure-Python oracle imports neither production nor M149. It reconstructs
the three schedules, 17 modal recurrences, inverse NTT, all 867-byte
commitments, boundaries, support counts, inverse controls, mutations, and
reuse. All 126 comparisons pass. Classification is
`INDEPENDENTLY_VERIFIED_STRICT_SCOPE` at
`INDEPENDENT_ORACLE_REEXECUTION`; resident restoration is
`EXACT_ALGEBRAIC_RESTORATION`, while transient buffers have
`NO_RESTORATION_CLAIM`.

Since 17 divides 102, the C17 group algebra splits into 17 distinct F103
characters. The coefficient and spectral paths both retain 867 one-byte field
coordinates, but the depth-1024 convolution work is 18,939,904 coefficient
multiplications versus 1,114,112 modal multiplications. Named warm live
storage is 1,243 versus 1,226 bytes. Thus multi-coordinate superposition and
exact interference are real bounded mechanisms, but this entire primitive
algebra still has an exact classical factorization. No optimal classical
recurrence is claimed.

The next experiment interleaves convolution with a reversible
coefficientwise quadratic phase shear. Its purpose is to break simultaneous
character diagonalization and determine whether the resulting mode mixing
closes natively or merely collapses to the identical 17-coordinate classical
recurrence. M150 establishes no complex or physical coherence, general
relational contraction, CATVM custody, distinct phase resource, computational
advantage, Small Wall crossing, physical execution, physical bit replacement,
or unbounded catalytic computation.

## M151: C17 quadratic spectral-mode mixing no-go

Claim:

```text
BOUNDED_EXACT_F103_C17_INTERLEAVED_CONVOLUTION_AND_COEFFICIENTWISE_QUADRATIC_PHASE_SHEARS_BREAK17_INDEPENDENT_SPECTRAL_MODE_CLOSURE_WHILE_KEEPING_ONE_RESIDENT_MULTI_COORDINATE_PORT_UNPROJECTED_ACROSS_16_CONSUMERS_WITH_EXACT_RESTORATION_AND_REUSE_THROUGH_DEPTH1024_BUT_COLLAPSE_TO_EXECUTED_COUPLED17_MODE_AND_IDENTICAL867_COORDINATE_CLASSICAL_RECURRENCES
```

M151 interleaves M150's cyclic-convolution shears with a reversible
coefficientwise quadratic shear. One resident hub slot supplies the shared
quadratic term to 16 target consumers. The port remains unprojected; the path
uses no scalar or exponent readout, relation table, assignment expansion,
retained plan, inverse history, or restoration baseline.

The quadratic shear genuinely breaks independent character-mode evolution.
For the declared transform, pointwise coefficient square becomes inverse-17
times circular self-convolution of the 17 modes. The independent-mode square
sham changes the final state, and an explicit two-mode witness produces a
third mode.

All 18 cases across three public families and depths 1, 4, 16, 64, 256, and
1024 match a fully coupled 17-mode recurrence in all 867 cells and at the
boundary. The pure-Python independent oracle imports neither production nor
M150, uses no NumPy, separately executes the 867-coordinate coefficient and
coupled 17-mode recurrences, and passes 180 comparisons. Classification is
`INDEPENDENTLY_VERIFIED_STRICT_SCOPE` at
`INDEPENDENT_ORACLE_REEXECUTION`.

All 867 carrier cells restore exactly on the same backing. Unrelated
depth-613 reuse matches fresh execution, and 64 depth-8 cycles restore
exactly. Resident restoration is `EXACT_ALGEBRAIC_RESTORATION`; transient
buffers have `NO_RESTORATION_CLAIM`. No snapshot reload or post-inverse
canonicalization is used.

The accepted runs contain 10,601 exact quadratic cancellation events. Both
evaluated paths retain 867 bytes and reach 1,369 named warm bytes. At depth
1024 the coefficient path counts 18,957,312 convolution-plus-quadratic core
multiplications, while the coupled spectral path counts 1,410,048. The oracle
also executes the identical 867-coordinate classical coefficient recurrence.
No optimal classical baseline is claimed.

The selected successor is a typed open `F103[C17]` phase-relation bialgebra:
native convolution composition plus Hadamard intersection on a shared
unresolved port, with final-only closure, exact restoration/reuse, and the
strongest compact classical recurrence. M151 establishes no general
relational contraction, CATVM custody, distinct phase resource,
computational advantage, Small Wall crossing, physical execution, physical
bit replacement, or unbounded catalytic computation.
