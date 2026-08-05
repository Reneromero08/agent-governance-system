# Continuous-S1 Streamed Jacobi Moment Rematerialization Review

Decision: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration classification: `EXACT_ALGEBRAIC_RESTORATION`

## Reconstructed mechanism

This successor changes the M210 runtime moment law without changing its
continuous-`S1`, generic Gaussian-rational center family or its requested
first-harmonic `Q^24` boundary.  Instead of retaining the thirteen power sums,
it retains the actual 24-center source and its 34-record public operation
word.  For each required moment `m=0..12`, it scans that word backward,
materializes the effective power sum in one Gaussian-rational scratch cell,
consumes it into the universal Jacobi-log series, repeats the scan, and
subtracts the value exactly.  The scratch cell is zero before the next moment.

Only the final 25-coefficient, 22,609-payload-bit first-harmonic jet is
projected.  This is a
direct-process exact formal-series experiment.  It does not establish an
enforced hidden-intermediate or no-smuggle boundary.

The initial pushed draft at
`18213589522aa72aa3e58817f2fdd604e6cd4662` computed but omitted that projected
boundary payload from its sealed accounting.  The repaired source and seal add
it explicitly; the earlier Git object remains unchanged.

## Independent execution

The oracle imports no CAT_CAS module.  It reconstructs both public center
families and operation words with tuple-based rational complex arithmetic.
It computes each boundary twice: once by forward effective-center
reconstruction and direct factor-by-factor sparse convolution, and again by
an independently written reverse-scan Jacobi-log recurrence with an explicit
add/subtract scratch value.  Both exact algorithms reproduce the production
primary and unrelated-reuse commitments and their 22,609-bit and 16,427-bit
projected boundary payloads.

The oracle also independently reproduces the full primary resource tuple:
26 public-word scans, 884 operation-record visits, 624 source-center visits,
576 nonzero power evaluations, 13 scratch writes and inverse writes, 110
universal-log cells, 86 weighted-log cells carrying 172,243 payload bits, a
325-cell exponential peak carrying 2,480,635 payload bits, and 106,586 formal
series products.  The primary source is 24 cells carrying 763 payload bits.
Python fraction objects, allocator/interpreter/native-library state,
serialization, timing and whole-process peaks remain excluded, not zero.

## Restoration and controls

The actual scratch value is algebraically uncomputed after every consumption;
the source is never mutated by the transaction.  The same source and scratch
list backings survive restoration, and an unrelated 17-center family consumes
those restored backings at restoration generation two.  Fresh and restored
reuse boundaries agree without snapshot reload or retained inverse history.

Wrong operation type, null source, dirty scratch and semantic center
perturbation controls discriminate.  This one-shot direct-process transaction
does not add CATVM lease, response-ordering or machine-enforced custody claims.

## Strict ceiling

```text
FORMAL_CONTINUOUS_S1_WRAPPED_GAUSSIAN_Q_GENERIC_GAUSSIAN_RATIONAL_UNIT_CENTER_FAMILIES0_1_PRIMARY_CENTER_COUNT24_REUSE_CENTER_COUNT17_FIRST_HARMONIC_QJET_ORDER24_STREAMED_MOMENT_REMATERIALIZATION_DIRECT_PROCESS_ONLY
```

The repair removes M210's retained 13-cell, 47,209-bit moment vector, but it
does not remove the retained 24-center source or the much larger exact
projection series.  It trades retained moment state for repeated public-source
work.  The identical reverse-scan moment and formal-series recurrence is a
compact classical implementation, while the direct-factor recurrence remains
a second exact matched baseline.

No fixed-rank unbounded-precision closure, CATVM custody, distinct phase
resource, computational advantage, Small Wall crossing, physical waveform or
silicon execution, replacement of physical bits with pi, catalytic inference,
or unbounded computation is established.
