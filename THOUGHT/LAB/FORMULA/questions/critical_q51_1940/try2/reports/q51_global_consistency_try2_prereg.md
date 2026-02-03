# Pre-Registration: Q51 Global Consistency Check (Try2)

Date: 2026-02-03

## HYPOTHESIS
No direct circularity exists between the four tests, and the four results can be simultaneously true without logical contradiction.
Any stronger inference (roots-of-unity structure or topological holonomy) is not forced by their intersection.

## PREDICTION
A dependency scan will show shared inputs (dataset + PCA projections) but no direct data-flow between test outputs and other tests.
Consistency checks will show no mutual contradictions among the reported metrics.

## FALSIFICATION
If any test directly consumes another test's output as input, or if computed metrics are mutually exclusive under the fixed thresholds,
the hypothesis is falsified.

## DATA SOURCE
- Phase arithmetic results: results/results.json
- Zero-signature results: results/q51_zero_signature_try2_results.json
- Berry phase results: results/q51_berry_phase_try2_results.json
- Reported Cramer's V (octant–phase association): 0.27 (as given in prompt)
- Underlying external dataset: https://download.tensorflow.org/data/questions-words.txt

## SUCCESS THRESHOLD
- direct_dependency_found = False
- contradictions_count = 0

## FIXED PARAMETERS
- Phase fraction passing min: 0.6
- Phase mean error max (rad): 0.7853981633974483
- Zero-signature mean |S|/n max: 0.05
- Zero-signature uniform p min: 0.05
- Zero-signature uniform fraction min: 0.5
- Berry mean |Δγ| max: 0.2
- Berry quant score min: 0.8

## Anti-Patterns Guardrail
- No synthetic data generation
- No parameter search or post-hoc threshold changes
- Only existing external-data results are used

## Notes
- This check is adversarial: results are treated as hostile evidence against each other.
- All findings (pass/fail) will be reported.
