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

## Enforced CATVM open-intermediate composition

The branch-native Boolean/F3 phase engine now runs behind its first enforced
machine boundary. A carrier-owning Linux `SOCK_SEQPACKET` service is
non-dumpable before allocation, locks a private anonymous mapping, rejects
forked mappings and core dumps, unlinks its single-client socket, and installs
a post-accept seccomp allowlist. The IPC-only controller is not linked to the
phase core.

The accepted transaction is:

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
before restoration and therefore survives outside inverse history.

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

This establishes:

```text
CATVM_OPEN_INTERMEDIATE_COMPOSITION_ESTABLISHED_ON_PHASE_BACKEND
```

only for the bounded tested Linux userspace boundary against its ordinary
same-UID controller. It does not establish root/kernel or microarchitectural
secrecy, arbitrary topology, compact wide-interface relations, general
holographic relational computation, computational advantage, physical
waveform or silicon computation, Small Wall crossing, or unlimited catalytic
computation. The relational frontier remains reusable typed modules whose
open phase-resident outputs compose recursively into larger modules and wider
interfaces without host expansion or decode.

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
