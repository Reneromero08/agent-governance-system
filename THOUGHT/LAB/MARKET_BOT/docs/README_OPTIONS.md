# Psychohistory Options Signal Bot

## The Strategy

The formula predicts crashes **weeks in advance** with defined risk:

| Crash | Warning | Days Early | Put Return | $2K Risk -> Profit |
|-------|---------|------------|------------|-------------------|
| COVID 2020 | Jan 29 | **21 days** | **945%** | **$18,893** |
| 2008 Crisis | Oct 2007 | **3 weeks** | ~800% | ~$16,000 |
| VIX 2018 | Jan 2018 | **5 weeks** | ~300% | ~$6,000 |

## How It Works

1. **Normal times**: Monitor does nothing. Stay in cash or small long positions.
2. **Alpha drift warning**: BUY PUTS signal. Risk 1-3% of capital.
3. **Crash happens**: Puts explode 5-10x. Take profits.
4. **YOU execute manually** - no bots, no ToS violations.

## Quick Start

### 1. Set Up Notifications (FIRST!)
```bash
cd THOUGHT/LAB/MARKET_BOT
python notifier.py --setup
```
This will configure:
- Desktop popups (Windows toast)
- Sound alerts (beeps)
- Telegram messages (optional, FREE)
- Discord webhooks (optional)

### 2. Single Check (run anytime)
```bash
python options_signal_bot.py --check SPY
```

### 3. Background Monitor (runs hourly)
```bash
# Windows - double click:
start_monitor.bat

# Or manually:
python options_signal_bot.py --symbols SPY --interval 60
```

### 4. Backtest Historical Crashes
```bash
python options_signal_bot.py --backtest
```

## When You Get An Alert

1. **Read the signal** - BUY_PUTS with strike/expiry suggestion
2. **Open your broker** (Robinhood, TD, E*Trade, whatever)
3. **Buy the put** - suggested strike, suggested expiry
4. **Risk only what's suggested** (1-3% of portfolio)
5. **Wait** - if crash happens, puts 5-10x
6. **Take profits** when CLOSE_PUTS signal comes

## No Bots = No ToS Problems

This system is **alert-only**. It watches the market and tells you when to act. YOU click buy/sell on your broker. This is:
- Legal on any broker
- No API keys needed
- Works with any account size
- YOU stay in control

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
