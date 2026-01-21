# Psychohistory Options Signal Bot

## The Strategy

The formula predicts crashes **weeks in advance** with defined risk:

| Crash | Warning | Days Early | Put Return | $2K Risk -> Profit |
|-------|---------|------------|------------|-------------------|
| COVID 2020 | Jan 29 | **21 days** | **945%** | **$18,893** |
| 2008 Crisis | - | - | - | - |
| VIX 2018 | - | - | - | - |

## How It Works

1. **Normal times**: Monitor does nothing. Stay in cash or small long positions.
2. **Alpha drift warning**: BUY PUTS signal. Risk 1-3% of capital.
3. **Crash happens**: Puts explode 5-10x. Take profits.

## Quick Start

### Single Check (run anytime)
```bash
cd THOUGHT/LAB/MARKET_BOT
python options_signal_bot.py --check SPY
```

### Background Monitor (runs hourly)
```bash
# Windows
start_monitor.bat

# Or manually
python options_signal_bot.py --symbols SPY --interval 60
```

### Backtest Historical Crashes
```bash
python options_signal_bot.py --backtest
```

## Signal Types

| Signal | Meaning | Action |
|--------|---------|--------|
| `BUY_PUTS` | Alpha drift warning - crash likely | Buy ATM/OTM puts, 1-3% risk |
| `BUY_CALLS` | Strong uptrend, high R | Buy calls for momentum |
| `CLOSE_PUTS` | Warning cleared | Take profits or cut losses |
| `NONE` | No actionable signal | Do nothing |

## Urgency Levels

- **5**: Act immediately (gate closed, crash imminent)
- **4**: Act today (alpha drift ALERT)
- **3**: Act this week (alpha drift WATCH)
- **2**: Consider action (trend signal)
- **1**: Monitor only

## Suggested Trade Sizing

| Warning Level | % of Capital | Strike | Expiry |
|--------------|--------------|--------|--------|
| WATCH | 1% | 5% OTM | Monthly |
| ALERT | 2% | ATM | Monthly |
| CRITICAL | 3% | ATM/ITM | Weekly |

## Output Files

- `options_signals.jsonl` - All signals logged (JSONL format)
- `ALERT.txt` - Created when urgent signal detected (urgency >= 3)

## Risk Management

- **Max risk per trade**: 3% of capital
- **Max total put exposure**: 5% of capital
- **Always use defined-risk options** (buying puts, not selling)
- **Take profits at 3-5x**, don't hold for max gain

## Current Status Check

```bash
python options_signal_bot.py --check SPY
```

If you see `Signal: NONE`, market is stable. Check back daily.
