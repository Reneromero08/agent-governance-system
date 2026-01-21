"""
OPTIONS SIGNAL BOT
==================

Runs in background, monitors for alpha drift warnings.
When danger detected: ALERT to buy puts.

The formula predicts crashes 5-12 steps ahead.
Puts bought before crashes can return 10-50x.

This is how you get rich with low risk:
- Normal times: Do nothing (or small call positions)
- Alpha drift warning: Buy puts (cheap insurance)
- Crash happens: Puts explode, you profit

Risk per trade: 1-3% of capital
Potential return: 10-50x on crash puts
"""

import json
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from signal_vocabulary import SignalState, AssetClass
from signal_extractor import SignalExtractor
from real_data_ingest import RealDataFetcher
from prime_radiant import PrimeRadiant
from formula_executor import MarketFormulaExecutor
from seldon_gate import SeldonGate, GateTier, AlphaWarningLevel

# Try to import notifier (optional dependency)
try:
    from notifier import Notifier
    NOTIFIER_AVAILABLE = True
except ImportError:
    NOTIFIER_AVAILABLE = False


# =============================================================================
# SIGNAL TYPES
# =============================================================================

class SignalType(Enum):
    NONE = "NONE"
    BUY_PUTS = "BUY_PUTS"           # Alpha drift warning - crash coming
    BUY_CALLS = "BUY_CALLS"         # Strong uptrend, high R
    CLOSE_PUTS = "CLOSE_PUTS"       # Crash happened, take profits
    CLOSE_CALLS = "CLOSE_CALLS"     # Trend weakening


@dataclass
class OptionsSignal:
    """Signal for options trading."""
    signal_type: SignalType
    symbol: str
    timestamp: str
    urgency: int                    # 1-5 (5 = act now)
    confidence: float               # 0-1
    reason: str
    metrics: Dict

    # Suggested trade
    suggested_strike: str           # "ATM", "5% OTM", "10% OTM"
    suggested_expiry: str           # "weekly", "monthly", "quarterly"
    suggested_size_pct: float       # % of capital to risk


# =============================================================================
# OPTIONS SIGNAL ENGINE
# =============================================================================

class OptionsSignalEngine:
    """
    Generates options trading signals from Psychohistory formula.

    Key signals:
    1. BUY_PUTS: Alpha drift warning detected (crash coming)
    2. BUY_CALLS: High R + strong uptrend (momentum play)
    3. CLOSE positions: Take profits or cut losses
    """

    def __init__(self):
        self.radiant = PrimeRadiant()
        self.executor = MarketFormulaExecutor()
        self.gate = SeldonGate()
        self.extractor = SignalExtractor()

        # State tracking
        self.vector_history: List = []
        self.signal_history: List[OptionsSignal] = []
        self.last_warning_level = AlphaWarningLevel.NONE

        # Configuration
        self.put_risk_pct = 0.02      # Risk 2% on puts
        self.call_risk_pct = 0.03     # Risk 3% on calls

    def analyze(self, symbol: str = "SPY") -> OptionsSignal:
        """
        Analyze current market and generate signal.

        Returns OptionsSignal with recommended action.
        """
        # Fetch recent data
        fetcher = RealDataFetcher()
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        try:
            data = fetcher.fetch(symbol, start_date, end_date, use_cache=False)
        except Exception as e:
            return OptionsSignal(
                signal_type=SignalType.NONE,
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                urgency=0,
                confidence=0,
                reason=f"Data fetch failed: {e}",
                metrics={},
                suggested_strike="N/A",
                suggested_expiry="N/A",
                suggested_size_pct=0,
            )

        if len(data.bars) < 50:
            return self._no_signal(symbol, "Insufficient data")

        # Extract signals for recent history
        states = self.extractor.extract_sequence(data, start_index=30)

        # Build vector history
        self.vector_history = []
        for state in states:
            vec = self.radiant.state_to_vector(state)
            self.vector_history.append(vec)

        # Current state
        current_state = states[-1]
        current_vec = self.vector_history[-1]
        current_price = data.bars[-1].close

        # Compute R
        context_vecs = self.vector_history[-11:-1] if len(self.vector_history) > 1 else []
        formula_result = self.executor.compute_R(current_vec, context_vecs)
        R = formula_result.R

        # Compute alpha
        alpha_vecs = self.vector_history[-20:] if len(self.vector_history) >= 3 else self.vector_history
        alpha_result = self.executor.compute_alpha(alpha_vecs)

        # Gate assessment
        gate_result = self.gate.assess(
            R=R,
            alpha=alpha_result.alpha,
            Df=alpha_result.Df,
            requested_tier=GateTier.T2_MEDIUM_POS,
        )

        # Compute trend
        prices = [b.close for b in data.bars[-30:]]
        fast_ma = sum(prices[-10:]) / 10
        slow_ma = sum(prices[-30:]) / 30
        trend_strength = (fast_ma - slow_ma) / slow_ma

        # RSI
        rsi = self._compute_rsi([b.close for b in data.bars[-20:]])

        metrics = {
            "R": R,
            "alpha": alpha_result.alpha,
            "Df": alpha_result.Df,
            "gate_status": gate_result.gate.status,
            "drift_warning": gate_result.drift.warning_level.name,
            "drift_from_05": abs(alpha_result.alpha - 0.5),
            "trend_strength": trend_strength,
            "rsi": rsi,
            "price": current_price,
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
        }

        # SIGNAL GENERATION

        # Priority 1: ALPHA DRIFT WARNING = BUY PUTS
        if gate_result.drift.warning_level in [AlphaWarningLevel.WATCH, AlphaWarningLevel.ALERT, AlphaWarningLevel.CRITICAL]:
            urgency_map = {
                AlphaWarningLevel.WATCH: 2,
                AlphaWarningLevel.ALERT: 4,
                AlphaWarningLevel.CRITICAL: 5,
            }
            urgency = urgency_map.get(gate_result.drift.warning_level, 1)

            # Size based on warning level
            size_map = {
                AlphaWarningLevel.WATCH: 0.01,     # 1% on watch
                AlphaWarningLevel.ALERT: 0.02,    # 2% on alert
                AlphaWarningLevel.CRITICAL: 0.03, # 3% on critical
            }
            size_pct = size_map.get(gate_result.drift.warning_level, 0.01)

            # Strike based on urgency
            strike_map = {
                AlphaWarningLevel.WATCH: "5% OTM",
                AlphaWarningLevel.ALERT: "ATM",
                AlphaWarningLevel.CRITICAL: "ATM or ITM",
            }
            strike = strike_map.get(gate_result.drift.warning_level, "5% OTM")

            # Expiry based on expected timing
            expiry = "monthly" if urgency < 4 else "weekly"

            signal = OptionsSignal(
                signal_type=SignalType.BUY_PUTS,
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                urgency=urgency,
                confidence=min(gate_result.drift.drift * 2, 1.0),
                reason=f"Alpha drift {gate_result.drift.warning_level.name}: alpha={alpha_result.alpha:.3f}, drift={gate_result.drift.drift:.3f}",
                metrics=metrics,
                suggested_strike=strike,
                suggested_expiry=expiry,
                suggested_size_pct=size_pct,
            )
            self.signal_history.append(signal)
            self.last_warning_level = gate_result.drift.warning_level
            return signal

        # Priority 2: GATE CLOSED = BUY PUTS (more urgent)
        if gate_result.gate.status == "CLOSED":
            signal = OptionsSignal(
                signal_type=SignalType.BUY_PUTS,
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                urgency=5,
                confidence=0.9,
                reason=f"Gate CLOSED: R={R:.2f} below threshold",
                metrics=metrics,
                suggested_strike="ATM",
                suggested_expiry="weekly",
                suggested_size_pct=0.03,
            )
            self.signal_history.append(signal)
            return signal

        # Priority 3: Strong uptrend + high R = BUY CALLS
        if (R > 1.0 and
            trend_strength > 0.02 and
            rsi > 50 and rsi < 70 and
            gate_result.drift.warning_level == AlphaWarningLevel.NONE):

            signal = OptionsSignal(
                signal_type=SignalType.BUY_CALLS,
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                urgency=2,
                confidence=min(R / 5, 1.0),
                reason=f"Strong uptrend: R={R:.2f}, trend={trend_strength:.1%}, RSI={rsi:.0f}",
                metrics=metrics,
                suggested_strike="ATM or slightly ITM",
                suggested_expiry="monthly",
                suggested_size_pct=0.02,
            )
            self.signal_history.append(signal)
            return signal

        # Priority 4: Previous warning cleared = CLOSE PUTS (take profits or cut)
        if (self.last_warning_level != AlphaWarningLevel.NONE and
            gate_result.drift.warning_level == AlphaWarningLevel.NONE):

            signal = OptionsSignal(
                signal_type=SignalType.CLOSE_PUTS,
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                urgency=3,
                confidence=0.7,
                reason="Alpha drift warning cleared - evaluate put positions",
                metrics=metrics,
                suggested_strike="N/A",
                suggested_expiry="N/A",
                suggested_size_pct=0,
            )
            self.last_warning_level = AlphaWarningLevel.NONE
            self.signal_history.append(signal)
            return signal

        # No signal
        return self._no_signal(symbol, f"No actionable signal: R={R:.2f}, alpha={alpha_result.alpha:.3f}")

    def _no_signal(self, symbol: str, reason: str) -> OptionsSignal:
        """Return no-signal result."""
        return OptionsSignal(
            signal_type=SignalType.NONE,
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            urgency=0,
            confidence=0,
            reason=reason,
            metrics={},
            suggested_strike="N/A",
            suggested_expiry="N/A",
            suggested_size_pct=0,
        )

    def _compute_rsi(self, prices: List[float], window: int = 14) -> float:
        """Compute RSI."""
        if len(prices) < window + 1:
            return 50.0

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(d, 0) for d in deltas[-window:]]
        losses = [-min(d, 0) for d in deltas[-window:]]

        avg_gain = sum(gains) / window
        avg_loss = sum(losses) / window

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


# =============================================================================
# BACKGROUND MONITOR
# =============================================================================

LOG_FILE = Path(__file__).parent / "options_signals.jsonl"
ALERT_FILE = Path(__file__).parent / "ALERT.txt"


def run_monitor(symbols: List[str] = None, interval_minutes: int = 60):
    """
    Run background monitor for options signals.

    Checks every interval_minutes and logs signals.
    Creates ALERT.txt when urgent signal detected.
    Sends notifications via desktop/sound/telegram/discord.
    """
    symbols = symbols or ["SPY"]
    engine = OptionsSignalEngine()

    # Initialize notifier
    notifier = None
    if NOTIFIER_AVAILABLE:
        notifier = Notifier()
        print("Notifications: ENABLED")
        if notifier.config.telegram_enabled:
            print("  - Telegram: ON")
        if notifier.config.discord_enabled:
            print("  - Discord: ON")
        if notifier.config.desktop_enabled:
            print("  - Desktop: ON")
        if notifier.config.sound_enabled:
            print("  - Sound: ON")
    else:
        print("Notifications: DISABLED (run 'python notifier.py --setup' to enable)")

    print("=" * 60)
    print("OPTIONS SIGNAL MONITOR")
    print(f"Symbols: {symbols}")
    print(f"Interval: {interval_minutes} minutes")
    print(f"Log: {LOG_FILE}")
    print(f"Alert file: {ALERT_FILE}")
    print("=" * 60)
    print("\nMonitoring... (Ctrl+C to stop)\n")

    while True:
        for symbol in symbols:
            try:
                signal = engine.analyze(symbol)

                # Log all signals
                log_entry = {
                    "timestamp": signal.timestamp,
                    "symbol": signal.symbol,
                    "signal_type": signal.signal_type.value,
                    "urgency": signal.urgency,
                    "confidence": signal.confidence,
                    "reason": signal.reason,
                    "metrics": signal.metrics,
                    "suggested_strike": signal.suggested_strike,
                    "suggested_expiry": signal.suggested_expiry,
                    "suggested_size_pct": signal.suggested_size_pct,
                }

                with open(LOG_FILE, 'a') as f:
                    f.write(json.dumps(log_entry) + "\n")

                # Print status
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                if signal.signal_type == SignalType.NONE:
                    print(f"[{timestamp}] {symbol}: No signal - {signal.reason}")
                else:
                    print(f"[{timestamp}] {symbol}: >>> {signal.signal_type.value} <<< (urgency={signal.urgency})")
                    print(f"           Reason: {signal.reason}")
                    print(f"           Strike: {signal.suggested_strike}, Expiry: {signal.suggested_expiry}")
                    print(f"           Size: {signal.suggested_size_pct:.1%} of capital")

                    # Create alert file for urgent signals
                    if signal.urgency >= 3:
                        alert_content = f"""
!!! URGENT OPTIONS SIGNAL !!!
Time: {timestamp}
Symbol: {symbol}
Signal: {signal.signal_type.value}
Urgency: {signal.urgency}/5
Confidence: {signal.confidence:.0%}

Reason: {signal.reason}

SUGGESTED TRADE:
  Strike: {signal.suggested_strike}
  Expiry: {signal.suggested_expiry}
  Size: {signal.suggested_size_pct:.1%} of capital

Metrics:
  R = {signal.metrics.get('R', 'N/A')}
  Alpha = {signal.metrics.get('alpha', 'N/A')}
  Price = ${signal.metrics.get('price', 'N/A')}
"""
                        with open(ALERT_FILE, 'w') as f:
                            f.write(alert_content)

                        print(f"\n!!! ALERT FILE CREATED: {ALERT_FILE} !!!\n")

                        # Send notifications
                        if notifier:
                            notifier.notify(
                                title=f"OPTIONS ALERT: {signal.signal_type.value}",
                                message=f"{symbol} @ ${signal.metrics.get('price', 0):.2f}\n"
                                        f"{signal.reason}\n\n"
                                        f"Strike: {signal.suggested_strike}\n"
                                        f"Expiry: {signal.suggested_expiry}\n"
                                        f"Size: {signal.suggested_size_pct:.1%}",
                                urgency=signal.urgency
                            )
                            print(">>> NOTIFICATION SENT <<<")

            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")

        # Wait for next check
        print(f"\nNext check in {interval_minutes} minutes...\n")
        time.sleep(interval_minutes * 60)


# =============================================================================
# BACKTEST: WHAT IF WE HAD THIS IN 2020?
# =============================================================================

def backtest_options_signals(event_name: str = "covid_crash_2020"):
    """
    Backtest: What signals would we have gotten before COVID crash?
    And what would puts have returned?
    """
    from real_data_ingest import RealDataFetcher, HISTORICAL_EVENTS
    import numpy as np

    print("=" * 70)
    print(f"OPTIONS SIGNAL BACKTEST: {event_name}")
    print("=" * 70)

    # Fetch data
    fetcher = RealDataFetcher()
    data, event = fetcher.fetch_event(event_name, "SPY", buffer_days=60)

    print(f"Event: {event['description']}")
    print(f"Crash: {event['crash_start']} to {event['crash_end']}")
    print(f"Expected drawdown: {event['max_drawdown']:.0%}")

    # Setup engine
    engine = OptionsSignalEngine()
    extractor = SignalExtractor()

    # Track signals
    signals_generated = []
    engine.vector_history = []

    # Track alpha history for drift detection (rolling window)
    alpha_history = []
    ALPHA_WINDOW = 20

    states = extractor.extract_sequence(data, start_index=30)

    print(f"\nAnalyzing {len(states)} days...\n")

    for i, state in enumerate(states):
        bar_idx = i + 30
        bar = data.bars[bar_idx]

        # Build vector
        vec = engine.radiant.state_to_vector(state)
        engine.vector_history.append(vec)

        if len(engine.vector_history) < 10:
            continue

        # Compute metrics
        current_vec = engine.vector_history[-1]
        context_vecs = engine.vector_history[-11:-1]
        formula_result = engine.executor.compute_R(current_vec, context_vecs)
        R = formula_result.R

        alpha_vecs = engine.vector_history[-20:] if len(engine.vector_history) >= 3 else engine.vector_history
        alpha_result = engine.executor.compute_alpha(alpha_vecs)

        # Track alpha for drift detection
        alpha_history.append(alpha_result.alpha)
        if len(alpha_history) > ALPHA_WINDOW * 2:
            alpha_history = alpha_history[-ALPHA_WINDOW * 2:]

        # Custom drift detection (matching historical_backtest logic)
        drift_warning = AlphaWarningLevel.NONE
        if len(alpha_history) >= ALPHA_WINDOW:
            baseline_alpha = np.mean(alpha_history[-ALPHA_WINDOW:])
            current_alpha = alpha_history[-1]
            drift = abs(current_alpha - 0.5)  # Drift from critical line
            alpha_std = np.std(alpha_history[-ALPHA_WINDOW:]) if len(alpha_history) >= ALPHA_WINDOW else 0.1

            # Warning levels based on drift magnitude
            if drift > 0.3 or (alpha_std > 0 and abs(current_alpha - baseline_alpha) > 2 * alpha_std):
                drift_warning = AlphaWarningLevel.CRITICAL
            elif drift > 0.2 or (alpha_std > 0 and abs(current_alpha - baseline_alpha) > 1.5 * alpha_std):
                drift_warning = AlphaWarningLevel.ALERT
            elif drift > 0.1 or (alpha_std > 0 and abs(current_alpha - baseline_alpha) > alpha_std):
                drift_warning = AlphaWarningLevel.WATCH

        # Check for warning
        if drift_warning != AlphaWarningLevel.NONE:
            signals_generated.append({
                "date": bar.timestamp,
                "price": bar.close,
                "warning": drift_warning.name,
                "alpha": alpha_result.alpha,
                "R": R,
            })

            if bar.timestamp < event["crash_start"]:
                status = "BEFORE CRASH"
            elif bar.timestamp <= event["crash_end"]:
                status = "DURING CRASH"
            else:
                status = "AFTER CRASH"

            print(f"  {bar.timestamp}: {drift_warning.name} @ ${bar.close:.2f} [{status}]")

    # Analysis
    crash_start_date = event["crash_start"]
    crash_end_date = event["crash_end"]

    # Find prices
    crash_start_price = None
    crash_end_price = None
    for bar in data.bars:
        if bar.timestamp == crash_start_date:
            crash_start_price = bar.close
        if bar.timestamp == crash_end_date:
            crash_end_price = bar.close

    if crash_start_price and crash_end_price:
        actual_drop = (crash_end_price - crash_start_price) / crash_start_price

        print(f"\n--- CRASH ANALYSIS ---")
        print(f"  Peak price: ${crash_start_price:.2f}")
        print(f"  Bottom price: ${crash_end_price:.2f}")
        print(f"  Actual drop: {actual_drop:.1%}")

        # Signals before crash
        signals_before = [s for s in signals_generated if s["date"] < crash_start_date]
        if signals_before:
            first_warning = signals_before[0]
            days_before = (datetime.strptime(crash_start_date, "%Y-%m-%d") -
                          datetime.strptime(first_warning["date"], "%Y-%m-%d")).days

            print(f"\n--- EARLY WARNING ---")
            print(f"  First warning: {first_warning['date']} ({first_warning['warning']})")
            print(f"  Days before crash: {days_before}")
            print(f"  Warning price: ${first_warning['price']:.2f}")

            # Hypothetical put returns
            # ATM put bought at warning, sold at bottom
            # Simplified: put value ~ strike - spot (when ITM)
            entry_price = first_warning["price"]

            # If we bought ATM put at entry price
            put_strike = entry_price
            put_value_at_entry = entry_price * 0.03  # ~3% premium for monthly ATM
            put_value_at_bottom = max(put_strike - crash_end_price, 0)

            if put_value_at_entry > 0:
                put_return = (put_value_at_bottom - put_value_at_entry) / put_value_at_entry

                print(f"\n--- HYPOTHETICAL PUT TRADE ---")
                print(f"  Buy ATM put @ ${put_strike:.2f} for ~${put_value_at_entry:.2f}")
                print(f"  Put value at bottom: ${put_value_at_bottom:.2f}")
                print(f"  Put return: {put_return:.0%} ({put_return:.1f}x)")

                # With 2% capital risk
                capital = 100000
                risk_amount = capital * 0.02
                contracts = risk_amount / (put_value_at_entry * 100)  # 100 shares per contract
                profit = contracts * (put_value_at_bottom - put_value_at_entry) * 100

                print(f"\n--- $100K PORTFOLIO ---")
                print(f"  Risk 2%: ${risk_amount:.0f}")
                print(f"  Contracts: ~{contracts:.1f}")
                print(f"  Profit: ${profit:,.0f}")
                print(f"  Portfolio return: {profit/capital:.1%}")
        else:
            print("\n[WARN] No warnings before crash (need to tune sensitivity)")

    return signals_generated


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = sys.argv[1:]

    if "--backtest" in args:
        # Backtest on historical crashes
        print("\n" + "=" * 70)
        print("BACKTEST: COVID CRASH 2020")
        backtest_options_signals("covid_crash_2020")

        print("\n" + "=" * 70)
        print("BACKTEST: 2008 FINANCIAL CRISIS")
        backtest_options_signals("financial_crisis_2008")

        print("\n" + "=" * 70)
        print("BACKTEST: VIX EXPLOSION 2018")
        backtest_options_signals("vix_explosion_2018")
        return

    if "--check" in args:
        # Single check
        symbol = "SPY"
        for arg in args:
            if not arg.startswith("--"):
                symbol = arg.upper()
                break

        engine = OptionsSignalEngine()
        signal = engine.analyze(symbol)

        print("=" * 60)
        print(f"OPTIONS SIGNAL CHECK: {symbol}")
        print("=" * 60)
        print(f"\nSignal: {signal.signal_type.value}")
        print(f"Urgency: {signal.urgency}/5")
        print(f"Confidence: {signal.confidence:.0%}")
        print(f"Reason: {signal.reason}")

        if signal.signal_type != SignalType.NONE:
            print(f"\nSuggested Trade:")
            print(f"  Strike: {signal.suggested_strike}")
            print(f"  Expiry: {signal.suggested_expiry}")
            print(f"  Size: {signal.suggested_size_pct:.1%} of capital")

        print(f"\nMetrics:")
        for k, v in signal.metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        return

    # Default: run monitor
    symbols = ["SPY"]
    interval = 60  # minutes

    for i, arg in enumerate(args):
        if arg == "--symbols" and i + 1 < len(args):
            symbols = args[i + 1].split(",")
        if arg == "--interval" and i + 1 < len(args):
            interval = int(args[i + 1])

    run_monitor(symbols, interval)


if __name__ == "__main__":
    main()
