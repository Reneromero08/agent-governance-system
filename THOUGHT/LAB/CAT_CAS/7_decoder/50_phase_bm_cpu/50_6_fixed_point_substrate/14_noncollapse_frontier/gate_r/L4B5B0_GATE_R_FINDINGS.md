# Gate R Findings Appendix

## F1 — State partition

The base `S1_contextual` includes sender mode, route, core identity, TSC origin, and capture-window index inside the state vector. This creates input/context leakage and duplicate route conditioning.

**Repair:** effective state is measured response only. Executed control, nuisance context, and session gauge are separate objects.

## F2 — Measured-state notation

The base equation names latent `x`, but available instruments expose only I/Q/ring response plus context.

**Repair:** operators predict the next measured equivalence-class state `s_(t+1)` and cannot claim hidden-state identification.

## F3 — Driven transport versus persistence

Step and impulse stages include recovery windows, but the base design has no mandatory sender-off classification.

**Repair:** post-drive readout occurs with the sender disabled. Results are classified as `PERSISTENT_STATE_CANDIDATE` or `DRIVEN_RELATIONAL_TRANSPORT_ONLY`. Driven-only blocks memory/restoration claims.

## F4 — Tone/path confounding

The retained campaign used fixed tone order. Forward/reverse alone is insufficient because tone identity and path position remain partially confounded.

**Repair:** require FWD, REV, two frozen random orders, and an order-label sham; report both physical-tone and execution-order coordinates.

## F5 — Session gauge

Phase 6B.5D attributes most compact residual variation to session and identifies route `4:5`, seed `4` as a scalar-coordinate outlier with preserved relational invariants.

**Repair:** estimate a complex session gauge from preamble only, freeze it before evaluated rows, retain seed 4, and compare raw/gauge-normalized/session-lookup baselines.

## F6 — Diagnostic collapse

Prepared-state classification is useful for distinguishability but does not identify transition dynamics.

**Repair:** classification remains diagnostic; operator acceptance requires held-out trajectory prediction, persistence classification, and no-smuggle gates.

## Accepted design elements

- disjoint session-level train/validation/test partitions;
- training-only normalization;
- validation-only model/history selection;
- mean, persistence, input-only, route-only, time-index, shuffled-input, and random-linear baselines;
- affine to compact-nonlinear operator ladder;
- held-out one-step and rollout prediction;
- route/session stability analysis;
- explicit G1–G10 and F1–F10 structure;
- artifact schemas and SHA-bound review mechanics;
- explicit non-authorization of restoration and full-state claims.

## Technical conclusion

The base design is scientifically promising but incomplete in exactly the places exposed by the carrier result. The binding addendum closes those gaps without rewriting the sealed artifact. The combined bundle is technically accepted at the measured-response predictive-observability ceiling, pending project-owner ratification.
