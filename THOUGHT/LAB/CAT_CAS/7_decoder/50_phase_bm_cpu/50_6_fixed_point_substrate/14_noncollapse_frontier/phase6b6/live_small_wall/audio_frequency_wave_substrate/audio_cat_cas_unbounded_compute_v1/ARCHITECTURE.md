# Compact toroidal path-sum architecture

## Bottleneck attacked

`audio_phase_native_computer_v1` proves reusable programmable phase state, but
its register machine follows one resolved execution path. This successor attacks
the next bottleneck directly: a compact phase geometry must aggregate a
classically exploding path set without materializing those paths.

## Public problem

A `.holo` program supplies:

```text
cyclic residue modulus M
odd phase moduli p_1 ... p_k
binary-choice weights w_1 ... w_n
one target residue t
```

The classical path set contains every include/exclude choice, hence \(2^n\)
paths. The requested result is the number of paths ending at `t`, modulo the
product of the phase moduli.

The compiler validates and maps each public weight to one cyclic shift. It does
not execute the recurrence, inspect an answer, enumerate a path, or carry an
expected result.

## Native state

For phase modulus \(p\), layer \(j\), and cyclic residue \(s\), the relative
carrier phase is

```text
z[p,j,s] = exp(2*pi*i*count[j,s]/p).
```

The state lives on a finite product torus. It contains one modular path-count
relation per residue, phase modulus, and reversible time layer. It contains no
complete path mode.

## Global operator

One weight `w` executes simultaneously over every residue:

```text
z[j+1,s] *= lock_p(z[j,s] * z[j,s-w]).
```

Phase multiplication performs modular addition. `lock_p` is a fixed,
label-free injection-lock dynamic:

```text
z <- z * exp(-i * Im(z^p) / p)
```

applied three times. It suppresses floating phase drift without producing or
selecting a classical residue. Removing it becomes a decisive long-trajectory
control.

Each new layer is written by a triangular torus shear while its source layer
remains unresolved. The operator acts globally on the cyclic geometry. CPU code
orchestrates array operations, but no decoded count feeds native evolution.

## Catalytic closure

After the last shear, only the target relations are multiplied into a small
surviving phase latch. The program then runs its own shears in reverse and
removes the carrier-specific seed rotation:

```text
borrow dirty carrier
-> seed relative phase geometry
-> execute global shears
-> latch target phase
-> reverse compiled shears
-> remove seed
-> restore carrier
-> decode latch at CollapseBoundary
-> pass actual restored carrier to another .holo program
```

No per-instruction phase factors are retained. The inverse is derived from the
compiled public program. The result latch remains outside the reversed carrier
history.

## Resource law

For fixed cyclic width `M`, phase-modulus count `k`, and program length `n`:

```text
complete classical paths:        2^n
compact .holo source:             O(n)
native phase relations:           O(n*k*M)
forward phase work:               O(n*k*M)
restoration work:                 O(n*k*M)
complete path modes:              0
```

Thus path-work leverage against explicit include/exclude expansion grows
exponentially. The same recurrence has a conventional dynamic-programming
implementation with \(O(nM)\) work. This package therefore tests compact
unresolved phase computation and growing path-work leverage; it does not claim
an advantage over the best compact classical algorithm.

## Claim boundary

The maximum intended claim is:

```text
BOUNDED_SOFTWARE_COMPACT_TOROIDAL_PATH_SUM_REFERENCE_ONLY
```

The experiment cannot establish physical execution, energy advantage,
universal computation, fixed-size unbounded information, advantage over
classical dynamic programming, hardware bit replacement, or a Wall crossing.
