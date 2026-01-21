# MARKET_BOT Lab Experiment

**Status:** EXPERIMENTAL
**Created:** 2026-01-21

## Overview

A catalytic market bot that combines:
- **Compass AGI** for paradigm/regime detection
- **cat_chat** for context-aware reasoning
- **Paper trading** for safe experimentation

## Architecture

```
News Headlines ──► Paradigm Detector ──► Regime (SHIFT/STABLE/TRANS)
                          │
                          ▼
Market Data ────► cat_chat Context ────► Reasoning Engine
                          │
                          ▼
                   Strategy Decision
                          │
                          ▼
                   Paper Trader ──► P&L Tracking
```

## Key Insight

The bot doesn't predict price. It predicts **when prediction works**.

- **STABLE regime**: Use momentum/trend strategies
- **SHIFT regime**: Go defensive, ignore technical signals
- **TRANSITIONAL**: Reduce size, tighten stops

## Files

| File | Purpose |
|------|---------|
| `market_bot.py` | Core bot with cat_chat integration |
| `data_sources.py` | News and price data (mock + real) |
| `paper_trader.py` | Simulated trading with P&L |
| `run_experiment.py` | Demo runner |

## Usage

```bash
# Run the demo
python run_experiment.py

# Run with specific scenario
python run_experiment.py --scenario crisis
```

## How cat_chat Helps

cat_chat provides **paradigm-aware context management**:

1. **During STABLE**: Hydrate focused context (recent trends, technicals)
2. **During SHIFT**: Hydrate diverse context (historical crises, warnings)
3. **Adaptive threshold**: E-threshold adjusts based on regime

This means the bot reasons differently in different regimes:
- Stable: "What does technical analysis say?"
- Shift: "What happened in similar historical crises?"

## Disclaimer

This is a LAB EXPERIMENT for research purposes only.
Not financial advice. Paper trading only.
