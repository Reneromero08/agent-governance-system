# CAT_CAS Cold-Agent Phase-Lock Evaluation

The control-plane validator proves that required files, authority order, branch context,
capability lineage, and receipt fields are structurally present. It cannot prove that a
model reconstructed the architecture correctly.

Use `phase_lock_cases.jsonl` for periodic cold-start evaluation across agent families.
Each evaluator receives only:

```text
MISSION.md
AGENTS.md
the generated task packet
one case prompt
```

It must produce a completed `PHASE_LOCK_RECEIPT.json` and a short answer to the case.

## Critical pass conditions

A release candidate fails if any agent:

- treats restoration as the compute advantage by itself;
- describes `.holo` only as compression or an answer file;
- hides candidate enumeration inside phase language;
- rejects a lawful final classical verifier;
- misses the registered audio branch context;
- claims physical audio computing or a Small Wall crossing from architecture-only work;
- accepts witness-conditioned initialization;
- cannot distinguish `flagship_compute` from `external_product`.

Store evaluation transcripts with:

```text
agent identity
model version
context digest
case ID
receipt
answer
verdict
reviewer
```

The lab is phase-lock-safe only after repeated cold starts pass without owner correction.
