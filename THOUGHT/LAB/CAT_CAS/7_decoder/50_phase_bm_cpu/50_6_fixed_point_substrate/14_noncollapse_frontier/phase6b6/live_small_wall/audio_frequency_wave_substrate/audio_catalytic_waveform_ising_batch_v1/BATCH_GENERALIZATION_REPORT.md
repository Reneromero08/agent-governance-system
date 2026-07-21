# Catalytic Waveform-Ising Frozen Batch Generalization

**Decision:** `CATALYTIC_WAVEFORM_ISING_BATCH_GENERALIZATION_PARTIAL`

**Claim ceiling:** `BOUNDED_SOFTWARE_CARRIER_CAUSAL_CATALYTIC_ISING_REFERENCE_ONLY`

## Custody and ordering

- Freeze commit: `b6b53493722aeca5cc8cc38bb41f9e9be66afb68`
- Pre-oracle evidence commit: `bb259b5a32ccfa9505d0fe7c61cbec0e39a57c3c`
- Ordered batch SHA-256: `4109d430789b8fb3912ad606b78311855e89e40b422fb3ecec9b84f5818c0c12`
- Pre-oracle evidence SHA-256: `1d05b0ab68751d34dd7199e6fcb94b12a9a340c2c3e36deebeb685747145b236`
- Pre-oracle trace SHA-256: `cfabefe4a2b5dba035ea75b92358d7741351b2f4b78694a57fc9f493f3b32cae`
- Prospective authority SHA-256: `ed537759d47cc69f0844a913113a2736a6ac8345550c06339be1bb253d3aee35` (11011 bytes)
- Oracle calls in the pre-oracle runner: `0`
- Final oracle trace hash: `61393d93b8cfad2e9f9b3f7d1308bd06c53d36d6c28911ca4a6de7194d57ff84`

All sixteen waveform executions, raw projections, restorations, restored-carrier reuses,
and controls were sealed before the exact 32-state oracles were opened.

## Outcomes

```text
batch size                     16
unique optima                  11
accepted correct               0
accepted incorrect             0
raw correct below gate         8
raw incorrect                  3
non-unique optimum             5
non-unique raw matches         4 / 5
uninterpretable                0
overall raw optimum agreement  8 / 11
```

The prospectively frozen promotion gate did not pass. Failed promotion checks:
`accepted_correct_count_min, accepted_correct_rate_min, all_causality_restoration_reuse_and_controls_pass, unique_optimum_instance_count_min`.

## Coherence and restoration

```text
coherence min/median/mean/max   0.357481815857 / 0.920339253433 / 0.855441233786 / 0.986836958025
restoration error max           1.51434583348e-14
reuse input error max           0.0
reuse restoration error max     1.34263562739e-14
```

## Controls

```text
batch-law materiality all       16 / 16
strict all controls              7 / 16
carrier content                  16 / 16
geometry combined               16 / 16
removed transform               7 / 16
removed operator                16 / 16
no lock                         16 / 16
wrong query                     16 / 16
wrong inverse                   16 / 16
omitted inverse step            16 / 16
omitted restoration             16 / 16
```

The stricter predecessor control requires both a material history change and a material
complex-response change. Its failures are preserved. The separately frozen batch law
accepts a material history or response change and is reported independently; no machine
constant, query, threshold, instance, or result was altered.

## Concrete evidence repairs

- The first pre-oracle qualification incorrectly treated strict predecessor-control
  failure as uninterpretable even when the frozen batch contract's material-history-or-
  response law passed. The qualification-only distinction was repaired and every
  pre-oracle trajectory was rerun before any oracle opened. Strict failures remain.
- The exact user-supplied experiment contract, which prospectively defined the three-way
  decision lattice, is now preserved byte-for-byte as `BATCH_EXPERIMENT_AUTHORITY.txt`
  and bound by its byte count and SHA-256.
- Non-unique optima are explicitly separated from unique successes and failures.

## Interpretation

This batch measures bounded software generalization of the unchanged carrier-causal
waveform machine. It does not establish scale, computational advantage, physical
computation, hardware bit replacement, or a Wall crossing.
