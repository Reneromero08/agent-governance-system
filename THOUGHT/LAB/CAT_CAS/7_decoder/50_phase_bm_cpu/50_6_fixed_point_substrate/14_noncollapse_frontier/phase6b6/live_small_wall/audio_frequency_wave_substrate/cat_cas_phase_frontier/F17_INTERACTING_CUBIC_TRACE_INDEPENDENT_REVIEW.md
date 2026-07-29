# F17 interacting cubic-latent trace independent review

## Decision

```text
classification:
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification_level:
    SEPARATE_REFERENCE_PARITY

restoration_class:
    EXACT_ALGEBRAIC_RESTORATION
```

The accepted scope is limited to fixed public width four with one through four
latent variables, plus the dense ports-two / latents-two oracle case, over
`F17`. The package tests two public algorithmic program families. It does not
establish a general multiple-latent closure algorithm.

## Independent reconstruction

The separate oracle does not import the production diagnostic and does not call
its compiler, execution backend, or projection function. It reconstructs the
exact coefficient recurrence with Python lists for every declared case. It also
materializes both latent axes in a dense complex parity calculation for the
ports-two / latents-two case. The exact reference agrees on both final
boundaries and restored states, and the dense check stays within the
predeclared tolerance.

The public topology compiler has no final-boundary, trace, or answer input.
Its row-mask and latent-mask scans are now counted explicitly. The accepted
path does not materialize an assignment table, but its projection streams all
`17**latents` assignments. Histogram integer payload, canonical cyclotomic
integer payload, serialized boundary size, program descriptors, and projection
temporary slots are reported. Python/NumPy allocator and native-library process
peaks remain outside the bounded accounting.

## Preserved result

The resident signature grows polynomially at this ceiling:

```text
latents:                    1     2      3       4
resident F17 field cells:  28    38     52      71
cubic coefficient cells:   1     4     10      20
streamed trace work:       17   289   4913   83521
```

Forward execution keeps the latent variables unresolved in the coefficient
state. The actual reverse sequence restores the exact borrowed arrays without
retained inverse descriptors or a baseline reload. A different program then
reuses the same restored carrier backing and matches a fresh carrier.

## Controls and ceilings

Missing and wrong inverse controls produce exact semantic-state mismatches.
The applicable reordered control is rejected by a zero public Gaussian pivot.
These controls are package-local; the independent oracle reconstructs the
positive recurrence, boundary, restoration, reuse, and dense latent-axis
parity.

The mixed cubic signature is preinitialized and unchanged by the public
Gaussian/shear modules. Therefore this result does not establish native cubic
interaction updates, arbitrary non-Gaussian composition, compact final trace
closure, or a lower bound on all compact classical methods. An identical exact
compact classical coefficient recurrence and generic trace exists.

It also does not establish machine-enforced CATVM custody, a distinct phase
resource, computational advantage, a Small Wall crossing, physical waveform
execution, replacement of physical bits with pi, or unbounded computation.
