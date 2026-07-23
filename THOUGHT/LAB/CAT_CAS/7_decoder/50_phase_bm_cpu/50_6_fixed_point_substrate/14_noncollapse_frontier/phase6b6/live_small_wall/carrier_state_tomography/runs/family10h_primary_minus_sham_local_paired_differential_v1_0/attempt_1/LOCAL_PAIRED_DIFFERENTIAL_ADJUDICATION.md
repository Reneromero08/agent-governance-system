# Local Paired Differential Adjudication

Result class: `FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_CUSTODY_INVALID`
Scientific claim: `PUBLIC_POST_SOURCE_LOCAL_PRIMARY_MINUS_SHAM_DIFFERENTIAL_NOT_ESTABLISHED`
Archive SHA-256: `61cd5c7994ca28e9e954bcf50559b9a7b6122bee6e34e2bd3b6600be90412519`
Adjudication SHA-256: `f8bd832ed585cf4d5a47cd725592f93d5d9981d4564ca0c2bca3bbb8293bb2e1`

## Round Results

| Round | R_primary | R_sham | D | max abs generic | generic/D | envelope | sham<0 diagnostic | q99 warnings |
|---:|---:|---:|---:|---:|---:|---|---|---|

This adjudication uses the frozen prospective local paired differential law only. It gates on `D = R_primary - R_sham`, `R_primary > 0`, one-factor matched stratum ordering, and the existing `0.25 * D` generic-control envelope.

`R_sham < 0` is reported as a diagnostic, not a gate. Complete six-factor crossed cells are also diagnostic only because each retained crossed cell is sparse.

Claim boundary: no full carrier-state tomography, physical relational memory, catalytic borrowing, R2 restoration, or Small Wall crossing is established by this package.
