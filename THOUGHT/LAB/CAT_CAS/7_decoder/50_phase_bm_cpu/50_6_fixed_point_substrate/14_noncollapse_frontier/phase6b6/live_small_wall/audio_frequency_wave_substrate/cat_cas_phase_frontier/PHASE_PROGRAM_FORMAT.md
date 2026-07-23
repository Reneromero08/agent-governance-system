# Mutable CAT_CAS phase-program format

This is a deliberately small public instruction interface for the mutable
phase-computing frontier. It is not a frozen language or an architecture
commitment.

```text
CATCAS_PHASE_PROGRAM 1
REGISTERS <count>
SET <register> <symbol>
PASSES <positive count>
ROT <target> <amount>
ADD <source> <target>
MULADD <left> <right> <target>
SWAP <left> <right>
CSWAP <control> <left> <right>
END
```

Symbols are elements of `F3`, represented at the load and observation
boundaries as `0`, `1`, or `2`. Unspecified input registers start at zero.
`SET` and the optional `PASSES` line precede native instructions. `PASSES`
defaults to one and repeats the complete instruction body without expanding
the public program or retaining an inverse history.

During execution, a register is the relative phase of two borrowed complex
rails. The native operators are:

```text
ROT(t,c):       z[t] <- z[t] * omega^c
ADD(s,t):       z[t] <- z[t] * z[s]
MULADD(a,b,t):  z[t] <- z[t] * F(z[a],z[b])
SWAP(a,b):      exchange both rails
CSWAP(c,a,b):   exchange a,b only when c carries phase symbol 1
```

`F` is the fixed roots-of-unity interpolation polynomial satisfying
`F(omega^a, omega^b) = omega^(a*b)`. `ADD` and `MULADD` forbid input/target
aliasing so their inverse is mechanically determined by the unchanged input
relations. The public instruction stream is traversed backward after the
boundary latch. No generated inverse history or carrier snapshot is retained.

`CSWAP` does not decode its control. It constructs the phase indicator

```text
g(c) = c^2 * F(c,c)^2 = omega^(2c + 2c^2)
```

which is `omega` only for phase symbol `1` and unity for symbols `0` and `2`.
It then applies `g(c)*(b-a)` and its negative simultaneously to the two target
relations using `F`. Thus `CSWAP` is a self-inverse Fredkin operation over the
Boolean control subset with a total, identity action for control symbol `2`.

The format is plain text because program representation is not yet the
scientific bottleneck. It can be replaced without changing the native phase
operators.

Every input is nevertheless byte-custodied. The native engine, separate
reference, and circuit compiler hash the exact bytes supplied to them. Raw
lines are length-aware, capped at 4,095 bytes including any line terminator,
and reject embedded NUL bytes. Transfers used for evidence must preserve
source bytes exactly; text adapters that add, remove, or translate line
endings change the program identity.

## Boolean relations as a boundary subset

Boolean values may use phase labels `0` and `1` without changing the native
carrier law. Over `F3`:

```text
NOT(a)     = 1 + 2a
AND(a,b)   = ab
XOR(a,b)   = a + b + ab
OR(a,b)    = a + b + 2ab
NAND(a,b)  = 1 + 2ab
```

These identities are valid for `a,b` in `{0,1}`. They compile directly to
`ROT`, `ADD`, and `MULADD`; therefore NAND-universal Boolean circuits are a
boundary-compatible subset of the phase program algebra. Native recurrence
still operates on complex relations, not decoded bits.

## Stored-program predication

A program-enable phase `p`, one-hot program-counter phase `q`, and zero
workspace phase `e` can activate a Fredkin slot without host branching:

```text
MULADD q p e
CSWAP e a b
MULADD q p e
MULADD q p e
```

The first operation forms `e = q*p`. The two trailing `MULADD` operations add
the same product twice, returning `e` to zero in `F3`. A fixed sweep can
therefore visit every slot while only the phase-resident selected slot changes
data. The program counter, program bits, enable workspace, and data all remain
inside the forward state and are restored by the public inverse traversal.

## Fredkin circuit compiler

`fredkin_phase_compiler.c` translates a small public circuit description:

```text
CATCAS_FREDKIN_CIRCUIT 1
WIRES <count>
SET <wire> <symbol>
CYCLES <positive count>
FREDKIN <control> <left> <right> <enabled-symbol>
END
```

The compiler does not execute the circuit. For `N` declared gates it emits
`N` one-hot phase-counter registers, `N` phase-resident enable symbols, two
workspace registers, the data wires, and one fixed native schedule. Each slot
computes `enable = pc * program`, computes the routed control relation,
performs `CSWAP`, uncomputes both workspaces, and advances the one-hot phase
counter. `PASSES = N * CYCLES` applies the same local fabric repeatedly.

The committed routed-network circuit compiles byte-for-byte to
`programs/compiled_routed_network.holo`. The circuit source and generated
program remain public inputs; neither compiler nor generated schedule contains
an answer oracle or decoded data-dependent branch.
