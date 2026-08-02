# Prospective Restoration R2 Contract

## Status

```text
PROSPECTIVE_ONLY
NO_EQUIVALENCE_THRESHOLD_FROZEN
NO_LIVE_ACQUISITION_AUTHORIZED
R2_NOT_ESTABLISHED
```

This contract defines what a future Family 10h holonomy candidate must restore. It does
not set numerical thresholds before carrier tomography identifies stable coordinates.
No threshold may be retrofitted after scientific acquisition.

## 1. Restoration object

Separate the measured carrier from validity metadata.

Carrier state:

\[
S_{carrier}=(S_{PMU},S_{probe},S_{timing},S_{other}).
\]

Custody state:

```text
mapping identity
policy identity
CPU identities
process identity
address-layout identity
schedule identity
temperature envelope
measurement-window identity
```

R2 is adjudicated on `S_carrier`. Custody state determines whether two records belong to
a valid matched stratum.

## 2. Required measurements

For each fresh preparation and loop word, collect:

```text
S_baseline_pre
S_baseline_time_matched
S_after_forward
S_after_declared_inverse
S_after_wrong_inverse
S_after_wrong_order_inverse
S_after_natural_relaxation
S_after_destructive_reset
S_carrier_off
```

The measurement operation and its disturbance ceiling must be frozen before acquisition.
Every queried carrier is fresh unless nondestructive reuse is independently established.

## 3. Forward displacement

A candidate catalytic loop must genuinely use the carrier. Require a prospectively
frozen displacement floor:

\[
d(S_{after\ forward},S_{baseline\ pre})>\delta_{use}.
\]

If the carrier does not move above the use floor, later equality of hashes or observables
is ceremonial restoration and cannot support a catalytic claim.

## 4. Inverse restoration

A declared inverse sequence must return the carrier within one frozen multivariate
region:

\[
d_R(S_{after\ inverse},S_{baseline\ matched})\leq\epsilon_R.
\]

The distance `d_R`, covariance estimator, hierarchy, and margin `epsilon_R` must be
frozen using public calibration data only.

Both fresh replicates must pass independently. An aggregate cannot rescue a failed
replicate.

## 5. Time boundary

Freeze:

```text
forward completion deadline
inverse start deadline
inverse completion deadline
natural-relaxation observation grid
```

Restoration must occur because of the declared inverse within the inverse deadline.
Overlap reached only after the natural-relaxation window is not R2.

## 6. Independent accumulator retention

Let `A_out` be the accumulator observable. A valid catalytic result requires:

\[
S_{after\ inverse}\sim S_{baseline},
\]

while:

\[
A_{out}\neq A_{null}
\]

under the frozen output law.

Carrier restoration with no retained independent output is not computation. Output
retention accompanied by carrier nonrestoration is not catalytic closure.

## 7. Mandatory negative controls

The gate must reject:

```text
wrong inverse
correct inverse in wrong order
inverse with wrong public amplitude
partial inverse
natural relaxation
process restart
destructive reset
fresh carrier substitution
carrier-off
reference-only accumulator
word-only accumulator replay
```

The wrong controls must be evaluated under the same distance, margin, deadlines, and
replicate law as the correct inverse.

## 8. Environment boundary

Inventory every known environment coupled to the operation:

```text
cache and coherence state
page state
frequency and policy state
thermal state
receiver and source process state
measurement register state
accumulator state
allocated memory
open descriptors
```

A future package must classify each field as:

```text
restored carrier degree
retained output degree
matched custody field
external bath requiring separate accounting
```

Unmeasured gain, loss, heat, or hidden memory prevents an R2 claim.

## 9. Equivalence calibration

A valid calibration set must include:

```text
repeated untouched baselines
measurement-only baselines
time-matched baselines
carrier-off baselines
known reversible calibration loops
known irreversible disturbances
```

The equivalence region must accept untouched and known reversible returns while rejecting
known irreversible disturbances at the frozen error rate.

## 10. Exclusive outcomes

```text
R2_CUSTODY_INVALID
R2_NO_FORWARD_CARRIER_USE
R2_RESTORATION_NOT_ESTABLISHED
R2_RESTORATION_CANDIDATE
```

`R2_RESTORATION_CANDIDATE` requires forward use, correct-inverse return, negative-control
rejection, retained accumulator output, and both fresh replicates passing.

It does not establish physical holonomy, the Native Catalytic Fiber Pushforward, or a
Small Wall crossing.
