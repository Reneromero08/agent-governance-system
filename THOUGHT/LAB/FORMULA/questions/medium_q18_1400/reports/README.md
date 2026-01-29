# Q18: Intermediate Scales

**Question:** Does R = E/sigma work at molecular, cellular, neural scales?

**Status:** IN PROGRESS

## Overview

This experiment tests whether the R formula (Evidence/Dispersion) works at biological scales between quantum and semantic. We test at four scales:

1. **Neural** (Tier 1): EEG data, cross-modal binding, temporal prediction
2. **Molecular** (Tier 2): Protein structures, binding mutations, folding
3. **Cellular** (Tier 3): Single-cell expression, perturbation response, phase transitions
4. **Gene Expression** (Tier 4): Bulk transcriptomics, cross-species transfer, essentiality

## Key Hypotheses

### H1: Scale Invariance
R is an intensive quantity (like temperature) - same value at all scales.
- **Prediction:** CV < 0.3 across all biological scales

### H2: 8e Conservation Law
The semiotic conservation law Df x alpha = 8e holds at biological scales.
- **Prediction:** Df x alpha = 21.746 +/- 15% at each scale

### H3: Blind Cross-Scale Transfer
R formula works without scale-specific tuning.
- **Prediction:** r > 0.3 for predictions across scales without retuning

## Directory Structure

```
q18/
├── README.md                    # This file
├── shared/
│   ├── biological_r.py          # Core R computation for biological data
│   ├── falsification_gauntlet.py # Adversarial test framework
│   └── conservation_law_detector.py # 8e detection
├── tier1_neural/
│   ├── test_*.py                # Neural scale tests
│   ├── adversarial/             # Adversarial attacks on neural R
│   └── results/neural_report.json
├── tier2_molecular/
│   ├── test_*.py                # Molecular scale tests
│   ├── adversarial/             # Adversarial protein sequences
│   └── results/molecular_report.json
├── tier3_cellular/
│   ├── test_*.py                # Cellular scale tests
│   ├── adversarial/             # Edge case cells
│   └── results/cellular_report.json
├── tier4_gene_expression/
│   ├── test_*.py                # Gene expression tests
│   ├── adversarial/             # Synthetic transcriptomes
│   └── results/gene_report.json
├── cross_scale/
│   ├── test_scale_invariance.py
│   ├── test_8e_universality.py
│   ├── test_blind_transfer.py
│   └── results/integration_report.json
└── receipts/
    └── *.json                   # Execution receipts
```

## Success Criteria

**Q18 is ANSWERED AFFIRMATIVELY if:**
1. At least 2/3 cross-modal tests pass
2. 8e constant appears at 3+ scales (CV < 15%)
3. At least 1 blind cross-scale transfer works (r > 0.3)
4. CV across scales < 0.3 (intensivity)

**Q18 is FALSIFIED if:**
1. NO cross-modal tests pass
2. 8e does NOT appear outside semantic scale
3. ALL blind transfers fail
4. CV > 0.5 across scales

## Execution

Run tests via pytest:
```bash
pytest q18/tier1_neural/ -v
pytest q18/tier2_molecular/ -v
pytest q18/tier3_cellular/ -v
pytest q18/tier4_gene_expression/ -v
pytest q18/cross_scale/ -v
```

## References

- Q7: Multi-scale composition (proved R scale-invariant at linguistic scales)
- Q39: Homeostatic regulation (proved universal dynamics across architectures)
- Q48-50: Semiotic conservation law (Df x alpha = 8e in semantic space)
