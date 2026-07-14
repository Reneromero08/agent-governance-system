# Final Gain-Covariant OrbitState Confirmation Contract

This contract freezes the final prospective confirmation law after the retry-one
gain-covariant offline audit. It does not authorize target contact and does not
promote `SMALL_WALL_CROSSED`.

## Identity

Frozen predecessor evidence:

```text
official_retained_class = ORBITSTATE_INDEPENDENT_COUPLING_CANDIDATE
retrospective_gain_covariant_class = PRIVATE_ORBITSTATE_GAIN_COVARIANT_GEOMETRY_ESTABLISHED
predecessor_transaction_run_id = orbitstate_independent_v2_1
predecessor_commit = 5b7b1338921cb6d5b3ce10a958f4262039592065
```

New prospective confirmation identity:

```text
science_package_id = orbitstate_gain_covariant_confirmation_v1_0
transaction_run_id = orbitstate_gain_covariant_confirmation_v1_0
public_randomization_seed = orbitstate-gain-covariant-final-confirmation-public-seed-5b7b1338-a7daa611
```

No previous raw receiver evidence, feature rows, stage receipts, source receipts, or
private adjudication output may enter this prospective confirmation.

## Fixed Source Geometry

The confirmation keeps the same OrbitState source formulas:

```text
N = 256
d = 23
fold(d) = 233
M = 2048
quantization_scale = 1536
phi = 2*pi*23/256

Q_d = 1536 * exp(+i*phi)
Q_fold = 1536 * exp(-i*phi)
Q_polarity = -1536 * exp(+i*phi)
```

The physical carrier remains the ordinary Linux Family 10h Change-to-Dirty PMU
coordinate with the same source and receiver boundary:

```text
source_core = 4
receiver_core = 5
receiver_feature_extraction_must_be_public_only = true
private_source_map_opens_only_after_receiver_feature_hash_freeze = true
```

The same public schedule shape is retained:

```text
fresh_replicates = 2
conditions = 9
public_decoder_phases = 4
mapping_leg_records = 144
independent_component_windows = 288
stage_receipts = 2016
source_receipts = 288
```

## Control-Only Carrier Gain

Carrier gain is estimated separately inside each replicate. The estimator may use only
the predeclared controls:

```text
post_projection expected complex vector = 1536*cos(phi) + 0i
equal_orbit_odd_zero expected complex vector = 1536 + 0i
```

For each replicate:

```text
g_post = Re(Z_post_projection) / (1536*cos(phi))
g_equal = Re(Z_equal_orbit) / 1536
g_control = mean(g_post, g_equal)
```

Hard gates:

```text
g_post > 0
g_equal > 0
relative_error(g_post, g_equal) <= 0.25
```

The estimator must not use `Z_d`, `Z_fold`, `Z_polarity_inversion`, private target
classification, or any post-run fitted target gain.

## Gain-Normalized Private Geometry

For each replicate, predict:

```text
Z_d_predicted = g_control * Q_d
Z_fold_predicted = g_control * Q_fold
Z_polarity_predicted = g_control * Q_polarity
```

Hard gates:

```text
complex_relative_error(Z_d, Z_d_predicted) <= 0.25
complex_relative_error(Z_fold, Z_fold_predicted) <= 0.25
complex_relative_error(Z_polarity_inversion, Z_polarity_predicted) <= 0.25
complex_relative_error(Z_fold, conjugate(Z_d)) <= 0.25
complex_relative_error(Z_polarity_inversion, -Z_d) <= 0.25
min(abs(Im(Z_d)), abs(Im(Z_fold))) > g_control * 456
```

Both fresh replicates must pass. Aggregate values are reported as a check, but aggregate
success cannot rescue a failed replicate.

## Strong-Signal Phase Controls

Each condition-phase mapping pair is partitioned by the frozen source receipt value:

```text
strong if abs(q_theta) >= 256
near_zero if abs(q_theta) < 256
```

For every strong pair cell:

```text
relative_error(map0.logical_response, map1.logical_response) <= 0.25
relative_error(map0.physical_a_minus_b, -map1.physical_a_minus_b) <= 0.25
logical_response * q_theta > 0 for both mapping legs
```

There is no allowance for retrospective strong-signal exceptions. A single strong
mapping, reversal, or sign failure blocks confirmation.

## Near-Zero Absolute Law

The near-zero law freezes one absolute physical count bound before the next run:

```text
absolute_near_zero_bound = 152
```

The bound is applied to separately named statistics, not by comparing one-leg and
two-leg quantities as if they were identical:

```text
abs(each near_zero raw logical_response) <= 152
abs(logical_pair_average) <= 152
abs(physical_reversal_average) <= 152
abs(decoded_first_harmonic_imaginary_null) <= 152
```

For source-off and query-off controls, both real and imaginary decoded first-harmonic
components must also satisfy:

```text
abs(decoded_first_harmonic_real) <= 152
abs(decoded_first_harmonic_imaginary) <= 152
```

For post-projection and equal-orbit, the real component is a gain control and is not a
near-zero statistic; the imaginary component remains a near-zero statistic.

The bound is not derived from a same-sized held-out sample maximum, and no post-run
threshold revision is allowed.

## Custody and No-Smuggle Gates

The confirmation is invalid unless all custody gates pass:

```text
receiver_feature_hash_frozen before private unblinding
receiver feature extraction contains no private source-map or source-receipt fields
source_core = 4 for every source receipt
receiver_core = 5 for every receiver window
source_cpu_before = source_cpu_after = 4
receiver_cpu_before = receiver_cpu_after = 5
receiver_feedback_used_to_select_q = false
positive_work + negative_work + dummy_work = 4096
PMU group unmultiplexed for every receiver window
PMU event IDs are positive and distinct
process scans return 0 with no forbidden process hits
effective UID = 0
AuthenticAMD family 16 target
cores 4 and 5 online
temperature below 68 C
policy4 and policy5 restored exactly
copy-back manifest verifies every copied file
temporary remote root removed only after verified copy-back
```

## Final Confirmation Rule

The final confirmation class may be emitted only when:

```text
both fresh replicates pass control gain agreement
both fresh replicates pass gain-normalized d/fold/polarity geometry
both fresh replicates pass fold conjugacy and polarity inversion
both fresh replicates pass gain-scaled odd magnitude
all strong-signal phase controls pass
all near-zero controls pass the frozen absolute law
source formula law passes
feature-freeze law passes
all custody and no-smuggle laws pass
no target-derived gain fitting occurs
no post-run threshold revision occurs
```

No target contact is authorized by this contract. A separate frozen implementation,
manifest, source bundle, self-test, transport simulation, and read-only audit are
required before any live authorization.
