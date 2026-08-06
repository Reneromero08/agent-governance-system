# Source Reproduction Report

Status: unverified transfer candidate; not canonical; no Small Wall promotion.

Frozen source commit:
`c0cee6a9475d35bc64c90ec30567826bcf3c9e9a`

This report covers source-branch reproduction only. It does not count the
source qualifier and a copied rerun as independent verification.

## Method

The four source packages were executed from the clean detached source
worktree frozen in `SOURCE_RECEIPT.json`. Each successful diagnostic was
repeated in a distinct fresh output directory after removing prior build
products. Required-input deletion tests were run from disposable source
copies. Standard output, standard error, exit status, generated-file lists,
and generated-file hashes are preserved under
`raw_logs/source_reproduction/`. Complete successful output trees are
preserved under `raw_outputs/source_reproduction/`.

The complete reproduction hash manifest is
`SOURCE_REPRODUCTION_FILE_HASHES.sha256`:

```text
sha256 9033fa8ee6ab3479b5adf381ef88927d18c103ffd2c80d867c9b072c5cf4c840
entries 1820
```

Environment:

```text
Linux 7.0.0-28-generic x86_64
gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0
GNU bash 5.3.9(1)-release
jq 1.8.1
glibc 2.43
```

## Results

| Candidate | Exact declared-path result | Diagnostic result | Missing-input control | Provisional source classification |
|---|---:|---:|---:|---|
| A: CATVM open intermediate composition | rc 126 | two runs rc 0 | rc 2 | `SOURCE_NOT_REPRODUCED` |
| B: reversible shared DAG | rc 126 | two runs rc 0 | rc 1 | `SOURCE_NOT_REPRODUCED` |
| C: fixed-schema QANF obstruction | two runs rc 0 | not needed | rc 1 | `SOURCE_REPRODUCED` |
| D: Boolean suffix quotient | two runs rc 0 | not needed | rc 1 | `SOURCE_REPRODUCED` with stale tracked-result provenance |

### Candidate A

The source-declared direct command was:

```text
./qualify_catvm_phase.sh . /tmp/ags-catvm-phase-a-run1
```

It failed with rc 126:

```text
ionice: failed to execute ./qualify_catvm_phase.sh: Permission denied
```

The frozen Git tree records the qualifier as non-executable. Running the same
script explicitly through its declared shell:

```text
sh qualify_catvm_phase.sh . /tmp/ags-catvm-phase-a-shell
sh qualify_catvm_phase.sh . /tmp/ags-catvm-phase-a-shell-run2
```

passed twice and generated 63 files per run. After removal of
`catvm_primary.aspr`, the qualifier failed with rc 2 and did not accept an
existing result. This shows that the implementation can run after a packaging
repair, but the exact frozen source package does not reproduce as declared.

Classification: `SOURCE_NOT_REPRODUCED`.

### Candidate B

The exact top-level command was:

```text
bash qualify_catvm_rank2_phase.sh /tmp/ags-catvm-rank2-b-run1
```

It progressed through compilation and fault controls, then failed with rc 126
when it invoked the tracked non-executable nested qualifier:

```text
qualify_internal_rematerializing_general_multi_dag_affine_phase.sh:
Permission denied
```

The `Bad system call` emitted earlier in the run is the expected seccomp
fault-control process and is not the packaging failure.

A disposable source copy with only the tracked shell scripts made executable
passed twice:

```text
diagnostic source copy; chmod u+x *.sh;
bash qualify_catvm_rank2_phase.sh <fresh-output>
```

Each successful run generated 326 files. Removing
`general_multi_dag_affine_topology.txt` caused rc 1. The qualifier therefore
does not merely trust the tracked result, but the frozen package still fails
on its exact path.

Classification: `SOURCE_NOT_REPRODUCED`.

### Candidate C

Both fresh commands passed:

```text
bash qualify_small_wall_qanf_obstruction.sh /tmp/ags-qanf-c-run1
bash qualify_small_wall_qanf_obstruction.sh /tmp/ags-qanf-c-run2
```

Each run generated 301 files. After excluding the declared aggregate timing
observations, the output was deterministic. Removing
`quadratic_anf_chain_primary.qanf` caused rc 1. The qualifier rebuilt its
binaries and results and did not accept a pre-existing result JSON.

Classification: `SOURCE_REPRODUCED`.

This classification says nothing yet about whether the selected compact
baseline is strongest, whether accounting is matched, or whether the claimed
obstruction survives independent reconstruction.

### Candidate D

Both fresh commands passed:

```text
bash qualify_algebraic_boolean_tt_suffix_quotient.sh /tmp/ags-quotient-d-run1
bash qualify_algebraic_boolean_tt_suffix_quotient.sh /tmp/ags-quotient-d-run2
```

Each run generated 179 files. Apart from the declared evidence-directory
path, the outputs were deterministic. Removing the included
`algebraic_series_parallel_phase.c` caused compilation failure and rc 1. The
qualifier rebuilt the result and did not accept the tracked result JSON as
authority.

The tracked source result is nevertheless stale relative to the frozen tree:
its recorded phase-source hash is
`1a8678...`, while the frozen source file hash is `281e1d...`. The frozen
commit added the `QTT_EMBEDDED_MAIN` wrapper without refreshing the tracked
result. A fresh qualifier run succeeds, so this is a tracked provenance
defect rather than a fresh execution failure.

Classification: `SOURCE_REPRODUCED` with stale tracked-result provenance.

## Freshness and fail-open findings

- Successful outputs were generated in distinct previously absent output
  directories.
- Generated binaries and results were rebuilt on both passes.
- All four required-input deletion or corruption controls failed closed.
- No qualifier used a tracked result JSON as the acceptance oracle.
- A and B have executable-bit packaging defects in the frozen Git tree.
- D has a stale tracked result/source hash binding.
- Reproduction does not cure the source protocol defect already identified
  for A, the fixed-topology scheduler specialization in B, the baseline
  challenge in C, or the family restriction in D.

## Evidence map

- Exact commands and exit codes:
  `raw_logs/source_reproduction/*/{command.txt,exit_code.txt}`
- Standard output and error:
  `raw_logs/source_reproduction/*/{stdout.json,stdout.log,stderr.log}`
- Generated file inventories and hashes:
  `raw_logs/source_reproduction/*/{generated_files.txt,generated_sha256.txt}`
- Complete successful artifacts:
  `raw_outputs/source_reproduction/candidate_[a-d]_run[1-2]/`
- Aggregate hash binding:
  `SOURCE_REPRODUCTION_FILE_HASHES.sha256`

## Scope limit

These results establish only source reproducibility status. They do not
establish no-smuggle safety, mathematical correctness, inverse restoration,
machine-boundary custody, transfer relevance, quotient minimality, or a
physical Family 10h claim.
