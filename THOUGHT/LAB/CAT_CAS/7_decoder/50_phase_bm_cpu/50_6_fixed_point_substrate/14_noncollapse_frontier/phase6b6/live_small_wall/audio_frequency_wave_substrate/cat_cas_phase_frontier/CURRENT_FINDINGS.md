# CAT_CAS Phase Frontier Lab

This directory is intentionally a mutable research surface. It is not a frozen
package, promotion packet, or new stopping point.

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
