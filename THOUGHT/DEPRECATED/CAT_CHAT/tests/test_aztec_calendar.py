"""
Test: Do AZTEC CALENDAR cycles predict market behavior?

The Aztec/Mesoamerican calendar system had specific cycles:
1. Tonalpohualli - 260-day sacred calendar (13 x 20)
2. Xiuhpohualli - 365-day solar calendar
3. Calendar Round - 52 years (18,980 days) when both align
4. Venus cycle - 584 days (synodic period of Venus)
5. Trecena - 13-day cycle within Tonalpohualli

If time cycles affect reality, these ancient calendars might capture
something modern Western calendars miss.
"""

import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from scipy import stats, signal


# ============================================================================
# AZTEC CALENDAR CALCULATIONS
# ============================================================================

# Reference date: The "creation" date in the Long Count calendar
# Most commonly accepted correlation: August 11, 3114 BCE (Gregorian)
# We'll use a more practical reference: the start of a known Calendar Round
AZTEC_EPOCH = datetime(1519, 8, 13)  # Fall of Tenochtitlan - known date in both systems


def get_tonalpohualli_day(date):
    """
    Get position in the 260-day Tonalpohualli cycle.
    Returns 0-259 (day number in sacred calendar).
    """
    if hasattr(date, 'to_pydatetime'):
        date = date.to_pydatetime()
    days_since = (date - AZTEC_EPOCH).days
    return days_since % 260


def get_trecena(date):
    """
    Get the trecena (13-day period) within Tonalpohualli.
    Returns 0-19 (which trecena of the 20).
    """
    tonalpohualli_day = get_tonalpohualli_day(date)
    return tonalpohualli_day // 13


def get_day_sign(date):
    """
    Get the day sign (0-19) in the Tonalpohualli.
    There are 20 day signs that cycle.
    """
    tonalpohualli_day = get_tonalpohualli_day(date)
    return tonalpohualli_day % 20


def get_day_number(date):
    """
    Get the day number (1-13) in the current trecena.
    """
    tonalpohualli_day = get_tonalpohualli_day(date)
    return (tonalpohualli_day % 13) + 1


def get_venus_phase(date):
    """
    Get position in Venus synodic cycle (584 days).
    Venus was extremely important to Mesoamerican astronomy.
    Returns 0-583 (day in Venus cycle).
    """
    # Venus inferior conjunction reference: January 9, 2022
    venus_reference = datetime(2022, 1, 9)
    venus_cycle = 584  # days

    if hasattr(date, 'to_pydatetime'):
        date = date.to_pydatetime()

    days_since = (date - venus_reference).days
    return days_since % venus_cycle


def get_calendar_round_position(date):
    """
    Get position in the 52-year Calendar Round (18,980 days).
    When Tonalpohualli (260) and Xiuhpohualli (365) realign.
    """
    if hasattr(date, 'to_pydatetime'):
        date = date.to_pydatetime()
    days_since = (date - AZTEC_EPOCH).days
    return days_since % 18980


# Day sign names (for reference)
DAY_SIGNS = [
    'Cipactli (Crocodile)', 'Ehecatl (Wind)', 'Calli (House)', 'Cuetzpalin (Lizard)',
    'Coatl (Serpent)', 'Miquiztli (Death)', 'Mazatl (Deer)', 'Tochtli (Rabbit)',
    'Atl (Water)', 'Itzcuintli (Dog)', 'Ozomatli (Monkey)', 'Malinalli (Grass)',
    'Acatl (Reed)', 'Ocelotl (Jaguar)', 'Cuauhtli (Eagle)', 'Cozcacuauhtli (Vulture)',
    'Ollin (Earthquake)', 'Tecpatl (Flint)', 'Quiahuitl (Rain)', 'Xochitl (Flower)'
]


def run_aztec_calendar_test():
    """Test if Aztec calendar cycles predict market behavior."""

    if not YFINANCE_AVAILABLE:
        print("Installing yfinance...")
        import subprocess
        subprocess.run(['pip', 'install', 'yfinance', '-q'])
        import yfinance as yf
    else:
        import yfinance as yf

    print("Fetching S&P 500 data (1993-2024)...")
    spy = yf.download('SPY', start='1993-01-01', end='2024-12-31', progress=False)
    print(f"Got {len(spy)} trading days (~{len(spy)/252:.1f} years)")

    # Calculate returns
    close_prices = spy['Close'].values.flatten()
    spy_returns = np.diff(close_prices) / close_prices[:-1]
    dates = spy.index[1:]  # Skip first date (no return)

    print("\n" + "=" * 70)
    print("AZTEC CALENDAR CYCLES VS S&P 500")
    print("=" * 70)

    results = {}

    # =========================================================================
    # TEST 1: Tonalpohualli (260-day sacred calendar)
    # =========================================================================
    print("\n--- TEST 1: TONALPOHUALLI (260-day sacred cycle) ---")

    tonalpohualli_days = np.array([get_tonalpohualli_day(d) for d in dates])

    # Bin into 26 periods of 10 days each for statistical power
    n_bins = 26
    bin_size = 260 // n_bins
    tonalpohualli_bins = tonalpohualli_days // bin_size

    bin_returns = {}
    for b in range(n_bins):
        mask = tonalpohualli_bins == b
        bin_returns[b] = spy_returns[mask]

    # ANOVA test
    f_stat, p_value = stats.f_oneway(*[bin_returns[b] for b in range(n_bins)])

    print(f"Testing {n_bins} phases of the 260-day cycle")
    print(f"ANOVA F-statistic: {f_stat:.3f}")
    print(f"p-value: {p_value:.4e}")
    print(f"SIGNIFICANT: {'YES ***' if p_value < 0.05 else 'NO'}")
    results['tonalpohualli_p'] = p_value

    # Show best and worst periods
    mean_returns = [(b, np.mean(bin_returns[b])*100) for b in range(n_bins)]
    mean_returns.sort(key=lambda x: x[1], reverse=True)
    print(f"\nBest period:  Phase {mean_returns[0][0]} ({mean_returns[0][1]:+.3f}%/day)")
    print(f"Worst period: Phase {mean_returns[-1][0]} ({mean_returns[-1][1]:+.3f}%/day)")

    # =========================================================================
    # TEST 2: Trecena (13-day cycle)
    # =========================================================================
    print("\n--- TEST 2: TRECENA (13-day cycle) ---")

    day_numbers = np.array([get_day_number(d) for d in dates])

    trecena_returns = {}
    for day in range(1, 14):
        mask = day_numbers == day
        trecena_returns[day] = spy_returns[mask]

    f_stat_tr, p_value_tr = stats.f_oneway(*[trecena_returns[d] for d in range(1, 14)])

    print(f"ANOVA F-statistic: {f_stat_tr:.3f}")
    print(f"p-value: {p_value_tr:.4e}")
    print(f"SIGNIFICANT: {'YES ***' if p_value_tr < 0.05 else 'NO'}")
    results['trecena_p'] = p_value_tr

    # Show returns by day number
    print(f"\n{'Day':<6} {'Mean Return':>12} {'N':>8}")
    print("-" * 30)
    for day in range(1, 14):
        mean_ret = np.mean(trecena_returns[day]) * 100
        n = len(trecena_returns[day])
        print(f"{day:<6} {mean_ret:>+11.3f}% {n:>8}")

    # =========================================================================
    # TEST 3: Day Signs (20 signs)
    # =========================================================================
    print("\n--- TEST 3: DAY SIGNS (20 Aztec day signs) ---")

    day_signs = np.array([get_day_sign(d) for d in dates])

    sign_returns = {}
    for sign in range(20):
        mask = day_signs == sign
        sign_returns[sign] = spy_returns[mask]

    f_stat_sign, p_value_sign = stats.f_oneway(*[sign_returns[s] for s in range(20)])

    print(f"ANOVA F-statistic: {f_stat_sign:.3f}")
    print(f"p-value: {p_value_sign:.4e}")
    print(f"SIGNIFICANT: {'YES ***' if p_value_sign < 0.05 else 'NO'}")
    results['day_sign_p'] = p_value_sign

    # Show best and worst signs
    sign_means = [(s, np.mean(sign_returns[s])*100, DAY_SIGNS[s]) for s in range(20)]
    sign_means.sort(key=lambda x: x[1], reverse=True)
    print(f"\nBest sign:  {sign_means[0][2]} ({sign_means[0][1]:+.3f}%/day)")
    print(f"Worst sign: {sign_means[-1][2]} ({sign_means[-1][1]:+.3f}%/day)")

    # =========================================================================
    # TEST 4: Venus Cycle (584 days)
    # =========================================================================
    print("\n--- TEST 4: VENUS CYCLE (584-day synodic period) ---")

    venus_phases = np.array([get_venus_phase(d) for d in dates])

    # Bin into 8 phases (like the Aztec Venus cycle divisions)
    n_venus_bins = 8
    venus_bin_size = 584 // n_venus_bins
    venus_bins = venus_phases // venus_bin_size

    venus_returns = {}
    for b in range(n_venus_bins):
        mask = venus_bins == b
        venus_returns[b] = spy_returns[mask]

    f_stat_v, p_value_v = stats.f_oneway(*[venus_returns[b] for b in range(n_venus_bins)])

    print(f"Testing {n_venus_bins} phases of the Venus cycle")
    print(f"ANOVA F-statistic: {f_stat_v:.3f}")
    print(f"p-value: {p_value_v:.4e}")
    print(f"SIGNIFICANT: {'YES ***' if p_value_v < 0.05 else 'NO'}")
    results['venus_p'] = p_value_v

    # Venus phase names (approximate Aztec interpretation)
    venus_phases_names = [
        'Morning Star Rise', 'Morning Star Peak', 'Superior Conjunction', 'Evening Star Rise',
        'Evening Star Peak', 'Inferior Conjunction', 'Underworld Transit', 'Rebirth'
    ]

    print(f"\n{'Phase':<25} {'Mean Return':>12} {'N':>8}")
    print("-" * 50)
    for b in range(n_venus_bins):
        mean_ret = np.mean(venus_returns[b]) * 100
        n = len(venus_returns[b])
        print(f"{venus_phases_names[b]:<25} {mean_ret:>+11.3f}% {n:>8}")

    # =========================================================================
    # TEST 5: Spectral analysis at Aztec frequencies
    # =========================================================================
    print("\n--- TEST 5: SPECTRAL POWER AT AZTEC FREQUENCIES ---")

    # Convert to trading days (252/year)
    aztec_periods_calendar = {
        'Trecena': 13,
        'Veintena': 20,
        'Tonalpohualli': 260,
        'Venus': 584,
        'Xiuhpohualli': 365,
    }

    # Approximate trading day equivalents
    aztec_periods_trading = {
        'Trecena': 9,           # 13 * (252/365) ~ 9
        'Veintena': 14,         # 20 * (252/365) ~ 14
        'Tonalpohualli': 180,   # 260 * (252/365) ~ 180
        'Venus': 403,           # 584 * (252/365) ~ 403
        'Xiuhpohualli': 252,    # 365 * (252/365) = 252
    }

    # Compute PSD
    freqs, psd = signal.welch(spy_returns, fs=252, nperseg=min(1024, len(spy_returns)//4))
    psd = np.asarray(psd).flatten()
    freqs = np.asarray(freqs).flatten()

    # Get power at each Aztec frequency
    print(f"{'Cycle':<20} {'Period':>10} {'Trading':>10} {'Power':>15} {'Rank':>8}")
    print("-" * 70)

    aztec_powers = []
    for name, period in aztec_periods_trading.items():
        # Find nearest frequency
        target_freq = 252 / period  # cycles per year
        idx = np.argmin(np.abs(freqs - target_freq))
        power = psd[idx]
        aztec_powers.append((name, period, power))

    # Rank by power
    all_powers = psd[1:]  # Skip DC
    for name, period, power in aztec_powers:
        rank = np.sum(all_powers > power) + 1
        total = len(all_powers)
        percentile = (1 - rank/total) * 100
        cal_period = [k for k, v in aztec_periods_trading.items() if v == period][0]
        cal_days = aztec_periods_calendar[cal_period]
        print(f"{name:<20} {cal_days:>10}d {period:>10}d {power:>15.2e} {percentile:>7.1f}%ile")

    # =========================================================================
    # TEST 6: Autocorrelation at Aztec lags
    # =========================================================================
    print("\n--- TEST 6: AUTOCORRELATION AT AZTEC LAGS ---")

    def autocorr_at_lag(x, lag):
        n = len(x)
        if lag >= n:
            return 0
        return np.corrcoef(x[:n-lag], x[lag:])[0, 1]

    def autocorr_significance(r, n):
        se = 1 / np.sqrt(n)
        z = r / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return p

    print(f"{'Cycle':<20} {'Lag':>8} {'Autocorr':>12} {'p-value':>12} {'Sig?':>8}")
    print("-" * 65)

    for name, lag in aztec_periods_trading.items():
        if lag < len(spy_returns) - 10:
            r = autocorr_at_lag(spy_returns, lag)
            p = autocorr_significance(r, len(spy_returns))
            sig = "YES ***" if p < 0.05 else "no"
            print(f"{name:<20} {lag:>8} {r:>12.4f} {p:>12.4e} {sig:>8}")
            results[f'{name}_autocorr_p'] = p

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: AZTEC CALENDAR VS S&P 500")
    print("=" * 70)

    significant = []
    if results['tonalpohualli_p'] < 0.05:
        significant.append('Tonalpohualli (260-day)')
    if results['trecena_p'] < 0.05:
        significant.append('Trecena (13-day)')
    if results['day_sign_p'] < 0.05:
        significant.append('Day Signs (20)')
    if results['venus_p'] < 0.05:
        significant.append('Venus Cycle (584-day)')

    for name in aztec_periods_trading.keys():
        key = f'{name}_autocorr_p'
        if key in results and results[key] < 0.05:
            significant.append(f'{name} autocorrelation')

    if significant:
        print("\nSIGNIFICANT AZTEC CYCLES DETECTED:")
        for s in significant:
            print(f"  * {s}")
        print("\nTHE AZTECS KNEW SOMETHING")
    else:
        print("\nNO SIGNIFICANT AZTEC CYCLES DETECTED")
        print("The sacred calendar does not predict S&P 500 returns.")

    # Bonferroni correction note
    n_tests = 4 + len(aztec_periods_trading)
    bonferroni_threshold = 0.05 / n_tests
    print(f"\nNote: With Bonferroni correction (n={n_tests}), threshold = {bonferroni_threshold:.4f}")

    bonferroni_sig = []
    for name, p in results.items():
        if p < bonferroni_threshold:
            bonferroni_sig.append(name)

    if bonferroni_sig:
        print(f"Still significant after Bonferroni: {bonferroni_sig}")
    else:
        print("Nothing survives Bonferroni correction.")

    return results


if __name__ == "__main__":
    run_aztec_calendar_test()
