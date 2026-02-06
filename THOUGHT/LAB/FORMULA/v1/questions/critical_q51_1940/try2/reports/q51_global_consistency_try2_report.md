# Q51 Global Consistency Check — Try2 Results

## Inputs
- Phase arithmetic: results/results.json
- Zero signature: results/q51_zero_signature_try2_results.json
- Berry phase: results/q51_berry_phase_try2_results.json
- Reported Cramer's V: 0.27

## Aggregate Metrics
- Phase fraction passing: 1.000
- Phase mean error (rad): 0.4588
- Zero mean |S|/n: 0.0904
- Zero uniform fraction (chi-square p > 0.05): 0.000
- Berry mean |Δγ| global: 0.0000
- Berry mean |Δγ| local: 0.0000
- Berry mean quant score (1/8): 1.0000

## Circularity Scan
- Direct data-flow detected: False
- Shared dataset URL: True
- Shared model list: True
- PCA-based projection in all tests: True

## Implicit Assumptions / Dependencies
- All tests rely on PCA-based projections; independence is limited by shared coordinate choices.
- Phase interpretations assume PCA axes encode meaningful angular structure (not guaranteed).
- Cramer's V is treated as a reported input (not recomputed here).

## Consistency Check (Hostile)
- Can all results be true simultaneously: True
- Phase arithmetic supported by metrics: True
- Zero-signature supported by metrics: False
- Roots-of-unity supported by zero-signature metrics: False
- Berry holonomy supported by invariance+quantization: True

## Strongest Claim Forced
The intersection forces only projection-level regularities supported by the metrics: phase-difference analogies pass the fixed threshold, and loop γ is numerically stable under the tested basis transforms. Zero-signature cancellation is not supported by the mean |S|/n threshold in this run, so it cannot be forced by the intersection.

## Interpretive (Not Forced)
- Exact 8th-roots-of-unity phase structure
- Underlying complex multiplication structure for analogies
- Topological Berry phase / holonomy in embedding space

