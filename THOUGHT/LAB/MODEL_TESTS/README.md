# Model Testing Framework

## Overview

Tool-augmented testing framework for evaluating LLM reasoning limits with practical tools.

## Directory Structure

```
MODEL_TESTS/
|-- README.md                           # This file
|-- scripts/                            # Python test scripts
|   |-- tool_executor.py                # Original math-only version
|   |-- tool_executor_v2.py             # Extended with web/file access
|   |-- wtf_tests.py                    # WTF-tier extreme tests
|   |-- oracle_bridge.py                # AI-to-AI communication bridge
|   |-- reasoning_limits_test.py        # 28-test reasoning suite
|   |-- test_tools.py                   # Tool testing utilities
|-- docs/                               # Documentation & reports
|   |-- FINDINGS.md                     # Key findings summary
|   |-- SESSION_SUMMARY.md              # Session notes
|   |-- ORACLE_SETUP.md                 # Oracle bridge setup guide
|   |-- WTF_TEST_REPORT.md              # WTF test results (v1.1)
|   |-- nemotron-3-nano-30b-benchmark-report.md
|-- logs/                               # Test execution logs
|   |-- tool_test_1.log
|   |-- tool_test_2.log
|-- nemotron-3-nano-30b-outputs/        # Test outputs
    |-- benchmarks/                     # Numbered benchmark outputs (01-35)
    |-- wtf-tests/                      # WTF test outputs
    |-- temp/                           # Temporary/debug files
```

## Core Executors

### tool_executor.py
- **Location:** scripts/tool_executor.py
- Persistent state: YES
- Tools: math, numpy, sympy, itertools, fractions
- Use case: Pure computation benchmarks

### tool_executor_v2.py
- **Location:** scripts/tool_executor_v2.py
- Persistent state: YES
- Tools: Python REPL + web search + Wikipedia + Grokipedia + file access
- Timeout: 300s (for complex reasoning)
- Features: REPL prompt stripping, action_input JSON support

## Quick Start

### Install Dependencies

```bash
pip install duckduckgo-search beautifulsoup4 wikipedia-api
```

### Run Tests

```bash
cd scripts

# Math-only test
python tool_executor.py "Factor 2^67 - 1"

# Web-enabled test
python tool_executor_v2.py "What is the current price of Bitcoin?"

# WTF-tier test
python wtf_tests.py
```

## Test Categories

### 1. Standard Benchmarks (01-35)
Located in `nemotron-3-nano-30b-outputs/benchmarks/`
- Physics: Schwarzschild radius, QFT vacuum, black holes
- Math: Riemann, Godel, Galois, P vs NP
- AGS-specific: Semiotic field theory analysis

### 2. WTF-Tier Tests
Located in `nemotron-3-nano-30b-outputs/wtf-tests/`
- Mathematical nightmares (Collatz, modular hell)
- Logic paradoxes (Liar's paradox variants)
- Meta-reasoning (capability probes)
- Edge cases (floating point precision)

## Key Results

### Infrastructure Score: 8/10

After fixes applied (v1.1):
- Timeout: 180s -> 300s
- REPL prompt stripping: Enabled
- action_input JSON: Supported

### Cognitive Score: 10/10

Model demonstrates:
- Graduate-level mathematical reasoning
- PhD-level formal logic
- Correct meta-cognition about problem difficulty

See `docs/WTF_TEST_REPORT.md` for full details.

## Philosophy

**Goal:** Find reasoning limits, not computational limits.

- DO test "can model choose sympy vs web search"
- DO test "can model chain multiple tools"
- DO test "can model recover from errors"
- DO test "can model verify its own work"
- DON'T test "can sympy factor huge numbers"
