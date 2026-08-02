# I2D Physical Reversibility Gap Contract

## Result

```text
I2D_PHYSICAL_REVERSIBILITY_GAP_CONTRACT_COMPLETE
OBSERVABLE_AND_BACKEND_PREREQUISITES_UNMET
PHYSICAL_PACKAGE_NOT_FREEZE_READY
NO_PHYSICAL_REVERSIBLE_GENERATOR_ESTABLISHED
NO_PHYSICAL_INVERSE_ESTABLISHED
I2E_PROSPECTIVE_MEASUREMENT_HARNESS_SKELETON_NEXT
NO_LIVE_EXECUTION_AUTHORIZED
FAMILY10H_TRANSPORT_OPERATOR_NOT_ESTABLISHED
FAMILY10H_HOLONOMY_NOT_ESTABLISHED
SMALL_WALL_NOT_CROSSED
```

I2C proved a protocol distinction, not a hardware result. The many-to-one ownership model
can show order dependence, carrier-coupled output, and endpoint return while failing
injectivity and both inverse laws. A hypothetical permutation model passes the desired
laws synthetically.

I2D freezes the minimum evidence needed to tell which class a future physical operation
belongs to.

## 1. Current capability boundary

Established or retained:

```text
D_single scalar q calibration
D_local local primary-minus-sham differential
compile-only persistent post-source runtime interface
byte-preserving ownership-intent extraction target
synthetic overwrite and reversible reference models
```

Not established:

```text
stable second carrier axis
frozen multivariate physical state
nondestructive repeated tomography
physical operator backend
receiver-side generator pair
left or right inverse
causal independent accumulator
R2 restoration
```

The current physical package is therefore not freeze-ready.

## 2. Minimum state observable

Logical bytes are required for custody but are not dispositive. The physical state must
include enough independent receiver observables to distinguish at least three public
carrier codewords and retain held-out rank at least two.

Required components are:

```text
logical byte digest
D_single scalar q calibration
D_local primary-minus-sham coordinate
multi-observer coherence vector
overlap-stratified line response
timing and probe vector
public codeword discrimination record
```

### Multi-observer coherence vector

At minimum, the same carrier must be observed from:

```text
declared home core
remote core A
remote core B
matched route controls
```

The vector must be frozen before acquisition and must separate carrier response from
mapping, route, temperature, frequency, process, and schedule custody.

### Overlap-stratified response

Every observation must retain these line strata separately:

```text
A-only support
A intersection B
B-only support
outside the union
```

A genuine pair effect must localize according to the frozen transport model. A uniform
or first-touch response across all strata is a confound.

### Repeated tomography

The same physical carrier must be measured after each primitive. Measurement disturbance
must be calibrated prospectively. A sequence of fresh carriers cannot establish a state
map or inverse law.

## 3. Public starting states

At least three answer-blind public starting states are required. They need not literally
be hidden hardware ownership labels `H`, `A`, and `B`; they must be experimentally
prepared codewords that the frozen receiver state vector distinguishes across held-out
sessions, mappings, and delays.

The state family must be large enough to expose map collisions. Testing an operation only
on the all-home baseline can make a destructive reclaim appear inverse.

## 4. Injectivity screen

For every candidate generator `T`, prepare distinct public states `s_i` and require:

```text
s_i not equivalent to s_j
implies
T(s_i) not equivalent to T(s_j)
```

within the frozen error law.

Any observed collision kills the reversible-map claim even if:

- the operation is byte-preserving;
- AB and BA differ;
- a reclaim operation returns the baseline endpoint;
- an accumulator retains an order signal.

This is the direct physical translation of the I2C overwrite counterexample.

## 5. Two-sided inverse tests

For each candidate `T` and candidate `T_inverse`, require both:

```text
T_inverse(T(s)) ~ s
T(T_inverse(s)) ~ s
```

for every held-out starting-state stratum.

A valid test must cross:

```text
starting codeword
session
mapping
route
delay
amplitude
line-set placement
replicate
```

Wrong core, route, direction, line set, amplitude, partial inverse, and inverse order must
fail under the same frozen equivalence law.

A same-core store cannot be called inverse merely because it is quiet. A home-core
reclaim cannot be called inverse merely because it returns one baseline.

## 6. Directional backend receipts

Each physical operation must produce a receipt binding:

```text
carrier identity
acting core
prior-owner proxy
destination-owner proxy
route
line set
operation count
amplitude
start and end barriers
logical-byte digest before and after
state-tomography receipt before and after
```

The backend must demonstrate that the intended physical action occurred. A public
operator label without directional evidence is insufficient.

## 7. Post-source custody

The source must not inherit the realized word, challenge seed, operator schedule, or
accumulator target. Fork-only code-path separation is insufficient because the child
inherits the parent address space.

A future package must use an explicit capability boundary such as:

```text
exec-based preparation-only source
separate preparation payload mapping
post-waitpid receiver entropy
closed source descriptors and IPC
zero surviving source helpers
```

## 8. Composition and commutator laws

Only after injectivity and two-sided inverses pass may the package test:

```text
held-out composition
powers
contractible words
commutator
inverse commutator
commutator followed by inverse
```

Required nulls include:

```text
disjoint support
commuting pair
carrier off
amplitude zero
word-only replay
reference only
time-matched sham
```

The commutator and inverse commutator must reverse the frozen orientation observable.
Aggregate significance cannot rescue a failed replicate or failed null.

## 9. Independent accumulator

The accumulator must receive its output from carrier transport. It must not compute the
public word locally.

Required interventions:

```text
carrier coupling removed -> output collapses
carrier off -> output collapses
word-only replay -> output collapses
reference-only path -> output collapses
accumulator coupling removed -> output collapses
```

After the output freezes and the accumulator decouples, the inverse commutator must
restore the complete carrier state while the output remains.

## 10. R2 boundary

A valid R2 candidate requires:

```text
forward carrier displacement above use floor
correct inverse return within deadline
wrong inverses rejected
natural relaxation rejected
destructive reset rejected
same object retained
independent output retained
system plus environment accounted
both fresh replicates passing
```

Logical-byte equality or endpoint scalar equality alone is R0 custody, not R2.

## 11. REV matrix

The executable contract freezes fifteen mandatory tests:

```text
REV-01 public multi-state preparation
REV-02 frozen multivariate state vector
REV-03 nondestructive repeated tomography
REV-04 persistent object identity
REV-05 post-source challenge secrecy
REV-06 directional backend receipts
REV-07 injectivity collision screen
REV-08 left inverse law
REV-09 right inverse law
REV-10 wrong-inverse rejection
REV-11 held-out composition law
REV-12 commutator orientation and nulls
REV-13 causal independent accumulator
REV-14 inverse restoration with retained output
REV-15 system-plus-environment accounting
```

Every row is currently unpassed.

## 12. Freeze decision

The contract is complete, but these physical package fields remain unfrozen:

```text
numerical thresholds
state vector
operator topology
backend implementation
accumulator implementation
challenge custody
```

Therefore:

```text
physical_package_freeze_ready = false
live_execution_authorized = false
```

The next gate is:

```text
I2E_PROSPECTIVE_MEASUREMENT_HARNESS_SKELETON
```

I2E may define schemas, compile-only interfaces, fixtures, and fail-closed validation for
the required measurements. It may not invent thresholds, contact the target, or promote
a physical mechanism.
