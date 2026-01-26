# Cassette Network Test Fixtures

**Status:** Production
**Migrated:** 2026-01-25
**Source:** CAPABILITY/TESTBENCH/cassette_network/

## Overview

Production-ready tests for the Cassette Network semantic infrastructure.

## Directory Structure

```
cassette_network/
├── ground_truth/          # Retrieval accuracy validation
│   ├── fixtures/
│   │   └── retrieval_gold_standard.json (v1.2.0)
│   └── test_retrieval_accuracy.py
├── adversarial/           # Negative controls & robustness
│   ├── fixtures/
│   │   ├── negative_controls.json (v1.1.0)
│   │   └── semantic_confusers.json
│   └── test_negative_controls.py
├── compression/           # H(X|S) ~ 0 proof
│   ├── fixtures/
│   │   └── task_parity_cases.json (v1.0.0)
│   └── test_compression_proof.py
├── determinism/           # Reproducibility tests
│   └── test_determinism.py
├── conftest.py            # Shared pytest fixtures
└── benchmark_success_metrics.py
```

## Running Tests

```bash
cd LAW/CONTRACTS/fixtures/cassette_network
python -m pytest -v
```

## Research Tests (Not Migrated)

The following remain in CAPABILITY/TESTBENCH/cassette_network/ for ongoing research:
- `cross_model/` - Eigenstructure alignment experiments
- `qec/` - Quantum error correction hypothesis testing

## Fixture Versions

| Fixture | Version | Model | Notes |
|---------|---------|-------|-------|
| retrieval_gold_standard.json | 1.2.0 | all-MiniLM-L6-v2 | Ground truth doc_ids |
| negative_controls.json | 1.1.0 | all-MiniLM-L6-v2 | Calibrated thresholds |
| task_parity_cases.json | 1.0.0 | - | Deterministic tasks |
