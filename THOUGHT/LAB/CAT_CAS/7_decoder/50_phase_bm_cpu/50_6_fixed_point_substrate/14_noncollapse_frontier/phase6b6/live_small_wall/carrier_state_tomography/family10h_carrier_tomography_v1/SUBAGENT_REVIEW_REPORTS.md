# Family 10h Carrier Tomography Subagent Review

Package: `family10h_carrier_tomography_v1_0`

Review mode: read-only, offline, no target contact, no PMU execution.

## Reviewer Verdicts

| Role | Agent ID | Final verdict |
|---|---|---|
| Physical carrier-state auditor | `019f62c5-3e70-7c92-bdde-9b54e6f9a6e6` | `MATERIAL_BLOCKER` |
| Experimental-design/operator auditor | `019f62c5-63d2-7e70-8831-ffc8b97226c6` | `MATERIAL_BLOCKER` |
| Custody/evidence auditor | `019f62c5-77f1-7243-b71d-c048adee0144` | final latest-state response not received before decision; earlier custody blockers were repaired and revalidated locally |
| Claim-boundary adjudicator | `019f62c5-9ad3-7b40-ac8a-2e993c9161b2` | `NO_MATERIAL_BLOCKER` |

## Material Blockers

### PHYS-REVIEW-01

The physical reviewer found that temperature sampling is path-stable but not yet bound to an approved CPU sensor identity. The live target code chooses the first valid `hwmon*/temp*_input`; a non-CPU sensor could satisfy the threshold. Minimal repair is to bind a manifest-approved CPU sensor identity, such as `k10temp` name plus label/path, and reject non-CPU sensor substitutions.

### OPER-REVIEW-01

The operator reviewer found that evidence samples collapse active query observations into one contrast sample labeled `query_A`. This blocks per-query and query-order operator identification.

### OPER-REVIEW-02

The operator reviewer found that mapping and delay holdouts are restricted replicate holdouts because training still contains the same mapping and delay levels. This does not satisfy a strict held-out mapping or held-out delay prediction claim.

### OPER-REVIEW-03

The operator reviewer found that cross-validated codeword classification is still asserted in the response-matrix summary rather than derived as a gate.

### OPER-REVIEW-04

The operator reviewer found that lifetime reporting still does not fully implement the contract-level state-lifetime vocabulary and variation decomposition expected for final authorization.

## Parent Disposition

Because at least one independent material blocker remains, this package is not frozen for authorization. The correct package decision is:

```text
FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_BLOCKED
```

No live execution is authorized.
