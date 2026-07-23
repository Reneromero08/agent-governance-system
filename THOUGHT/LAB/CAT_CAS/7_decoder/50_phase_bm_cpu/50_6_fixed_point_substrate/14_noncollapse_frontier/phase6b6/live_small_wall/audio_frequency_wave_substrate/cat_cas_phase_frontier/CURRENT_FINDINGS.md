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

On the target process's available CPUs `0-1`, seven descriptive matched runs
gave a one-thread phase median of `114,396,958 ns` and a compact scalar median
of `12,369,630 ns`, a `9.248x` phase penalty. The two-thread phase median was
slower, not faster. DVFS was unchanged and uncontrolled. The construction
therefore establishes finite spatial logical parallelism and a race-clean C
runtime, not software speed advantage, physical parallelism, or C5.

## Next active work

The next move must stop paying one conventional complex-polynomial evaluation
per spatial gate. Interpreter removal and CPU threading are now measured, and
neither changes that underlying resource law:

1. construct a genuinely global phase interaction whose carrier performs a
   whole layer without a host loop over gates;
2. transfer the fixed twin-rail construction to a flagship global operator
   without an equivalent compact classical recurrence;
3. identify a physical phase operation whose parallel work is not paid again by
   the conventional host;
4. continue direct-metal probes only when they test a genuinely different
   physical coupling mechanism, not another software-renamed solver.
