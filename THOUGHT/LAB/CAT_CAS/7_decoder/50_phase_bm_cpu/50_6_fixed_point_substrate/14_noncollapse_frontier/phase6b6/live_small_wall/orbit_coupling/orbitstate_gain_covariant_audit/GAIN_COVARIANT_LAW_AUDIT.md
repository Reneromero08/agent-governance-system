# Gain-Covariant OrbitState Law Audit

This is an offline retrospective audit of committed retry-one evidence. It does not alter the official target class and does not promote `SMALL_WALL_CROSSED`.

## Official Boundary

- Official retained class: `ORBITSTATE_INDEPENDENT_COUPLING_CANDIDATE`
- Retrospective gain-covariant class: `PRIVATE_ORBITSTATE_GAIN_COVARIANT_GEOMETRY_ESTABLISHED`
- Evidence root: `D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\14_noncollapse_frontier\phase6b6\live_small_wall\orbit_coupling\runs\orbitstate_independent_v2_1`

## Unit-Gain Defect

The original law computes `expected_phi = 2*pi*23/256` and `expected_re = 1536*cos(expected_phi)`, then compares `Re(Z_d)` and `Re(Z_fold)` directly against that source-domain value. The receiver measures Change-to-Dirty physical counts, not source work units. Without an independently frozen unit-gain calibration, that law tests `physical_count == source_work` as an unstated assumption.

- Expected source-domain real component: `1297.6950762235501`
- Unit-gain law failed: `True`

## Control-Only Gains

- `replicate_0`: g_post=1.89412755356, g_equal=1.81298828125, g_control=1.85355791741, agreement=0.0428372799719
- `replicate_1`: g_post=1.87043169422, g_equal=1.89876302083, g_control=1.88459735753, agreement=0.0149209386868
- `aggregate`: g_post=1.88227962389, g_equal=1.85587565104, g_control=1.86907763747, agreement=0.014027656951

## Gain-Calibrated Private Geometry

### replicate_0
- `Z_d`: gain=1.89962395468, angle_error_rad=0.00183683062851, control_error=0.0253837184301, orthogonal_residual=5.35955157019
- `Z_fold`: gain=1.90386821658, angle_error_rad=0.00182777726252, control_error=0.0275508457997, orthogonal_residual=5.34505100119
- `Z_polarity_inversion`: gain=1.88526392038, angle_error_rad=0.000808333696853, control_error=0.0173208118008, orthogonal_residual=2.34074524603
### replicate_1
- `Z_d`: gain=1.85541657837, angle_error_rad=0.000810022537121, control_error=0.0167431861174, orthogonal_residual=2.30849982403
- `Z_fold`: gain=1.87743064368, angle_error_rad=0.000701652003572, control_error=0.00424540916404, orthogonal_residual=2.02337769812
- `Z_polarity_inversion`: gain=1.85604462178, angle_error_rad=0.00175059110302, control_error=0.0178731808141, orthogonal_residual=4.99073820796
### aggregate
- `Z_d`: gain=1.87752026652, angle_error_rad=0.00132947114008, control_error=0.0053340794669, orthogonal_residual=3.83402569711
- `Z_fold`: gain=1.89064943013, angle_error_rad=0.000571905995136, control_error=0.0117676201803, orthogonal_residual=1.66083665154
- `Z_polarity_inversion`: gain=1.87065427108, angle_error_rad=0.0012757832089, control_error=0.00164937344311, orthogonal_residual=3.665741727

## Phase-Transfer Partition

- Strong pair cells: `48`
- Strong logical mapping failures: `2`
- Strong physical reversal failures: `2`
- Strong sign failures: `0`
- Near-zero pair cells: `24`
- Near-zero individual response bound violations: `15`
- Near-zero pair residual violations: `8`

The two strong-signal mapping/reversal failures are preserved explicitly:
- replicate 1, pre_projection_d, phase 1: map0=1268.0, map1=1721.0, logical_rel=0.263219058687, physical_rel=0.263219058687
- replicate 1, source_polarity_inversion_d, phase 3: map0=1271.0, map1=1740.0, logical_rel=0.269540229885, physical_rel=0.269540229885

## Near-Zero Law Audit

The retained raw-leg law bounds single randomized windows, while the decoded null controls are first-harmonic cancellations across phases and mapping legs. The two quantities are not the same statistic and should not share one absolute ceiling without a prospective derivation.

Prospective repair: freeze one absolute count bound before execution, use the same bound only for algebraically identical one-leg quantities and paired residual quantities, and do not compare one-leg and two-leg quantities to the same ceiling without an explicit prospective derivation.

## Conclusion

This audit explains the unit-gain defect and establishes the gain-covariant decoded geometry retrospectively. It does not rewrite the official class, does not remove phase-level failures, and does not authorize a new live transaction.
