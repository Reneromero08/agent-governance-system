# Vector ELO Lab (Lane E)

**Status:** Active (Implementation Phase)

ELO-based scoring for vectors, files, symbols, and ADRs. Systemic intuition via competitive ranking, Ebbinghaus forgetting curve, and LITE pack filtering.

## Contents

| File | Purpose |
|------|---------|
| `ROADMAP.md` | ELO system phases E.1-E.6 (logging, engine, pruning, LITE packs, annotation, dashboard) |
| `CHANGELOG.md` | ELO implementation history |
| `experiments/` | ELO convergence benchmarks and invariant discovery |

## Key Features

- **ELO Scoring**: Standard ELO update (K=16/8) with 4 tiers (HIGH/MEDIUM/LOW/VERY_LOW)
- **Forgetting Curve**: Ebbinghaus decay (30-day half-life, 800 floor)
- **R-Gated ELO**: Echo chamber prevention via reward thresholding
- **Memory Pruning**: Archives VERY_LOW content stale >30 days
- **LITE Packs**: ELO-tiered content selection for token-efficient context
- **Search Annotation**: ELO as metadata only (zero weight on ranking)

## Dependencies

- **Lane P** (LLM Packer): LITE packs depend on packer updates
- **Lane M** (Cassette Network): ELO scores stored in cassettes
- **Lane S** (SPECTRUM): ELO logging integrated with audit trail

## Sibling Labs

- `THOUGHT/LAB/COMPRESSION/` — SCL/SPC symbolic compression (Phase 5)
- `THOUGHT/LAB/EIGEN_ALIGNMENT/` — Cross-model eigenvalue alignment protocol
