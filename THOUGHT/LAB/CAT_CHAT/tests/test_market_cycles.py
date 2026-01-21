"""
Test: Do CYCLICAL PATTERNS exist in market data?

Proper spectral analysis instead of folk astrology:
1. Fourier decomposition of market returns
2. Look for statistically significant periodicities at ANY frequency
3. Test if cycles exist at lunar (~21 trading days), monthly, quarterly, annual
4. Compare to random noise baseline

This is how quantitative researchers actually test for market cycles.
"""

import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from scipy import stats, signal
from scipy.fft import fft, fftfreq


def run_cycle_detection():
    """Detect ANY cyclical patterns in market data using spectral analysis."""

    if not YFINANCE_AVAILABLE:
        print("Installing yfinance...")
        import subprocess
        subprocess.run(['pip', 'install', 'yfinance', '-q'])
        import yfinance as yf
    else:
        import yfinance as yf

    # Fetch LONG history for better frequency resolution
    print("Fetching S&P 500 data (1990-2024, 35 years)...")
    spy = yf.download('SPY', start='1993-01-01', end='2024-12-31', progress=False)
    print(f"Got {len(spy)} trading days (~{len(spy)/252:.1f} years)")

    # Calculate daily returns
    close_prices = spy['Close'].values.flatten()  # Ensure 1D
    returns = np.diff(close_prices) / close_prices[:-1]
    returns = returns[~np.isnan(returns)]  # Remove any NaN
    n = len(returns)

    print("\n" + "=" * 70)
    print("SPECTRAL ANALYSIS OF S&P 500 RETURNS")
    print("=" * 70)

    # =========================================================================
    # TEST 1: Power Spectral Density (Welch's method - more robust)
    # =========================================================================
    print("\n--- TEST 1: POWER SPECTRAL DENSITY ---")

    # Welch's method gives smoother estimate
    freqs, psd = signal.welch(returns.flatten(), fs=252, nperseg=min(1024, n//4))
    psd = np.asarray(psd).flatten()  # Ensure 1D
    freqs = np.asarray(freqs).flatten()

    # Convert frequency to period (in trading days)
    periods = 1 / freqs[1:]  # Skip DC component
    psd_no_dc = psd[1:]

    # Find peaks in PSD (use psd_no_dc for analysis)
    peak_indices, properties = signal.find_peaks(psd_no_dc, height=np.median(psd_no_dc)*2, distance=5)

    print(f"{'Period (days)':<20} {'Power':>15} {'Significance':>15}")
    print("-" * 55)

    # Known cycle periods to check
    known_cycles = {
        'Lunar': 21,        # ~29.5 calendar days = ~21 trading days
        'Monthly': 21,      # Also ~21 trading days
        'Quarterly': 63,    # ~3 months
        'Semi-annual': 126, # ~6 months
        'Annual': 252,      # 1 year
        'Presidential': 1008, # 4 years
    }

    significant_peaks = []
    for idx in peak_indices:
        period = periods[idx]
        power = psd_no_dc[idx]

        # Check if near a known cycle
        cycle_name = None
        for name, days in known_cycles.items():
            if abs(period - days) / days < 0.15:  # Within 15%
                cycle_name = name
                break

        if cycle_name:
            significant_peaks.append((period, power, cycle_name))
            print(f"{period:>8.1f} days      {power:>15.2e}   {cycle_name}")
        elif power > np.median(psd_no_dc) * 5:  # Strong peak
            significant_peaks.append((period, power, "Unknown"))
            print(f"{period:>8.1f} days      {power:>15.2e}   UNKNOWN CYCLE")

    if not significant_peaks:
        print("No significant peaks found above noise floor.")

    # =========================================================================
    # TEST 2: Autocorrelation at specific lags
    # =========================================================================
    print("\n--- TEST 2: AUTOCORRELATION AT CYCLE LAGS ---")

    def autocorr_at_lag(x, lag):
        """Compute autocorrelation at specific lag."""
        n = len(x)
        return np.corrcoef(x[:n-lag], x[lag:])[0, 1]

    def autocorr_significance(r, n, alpha=0.05):
        """Test if autocorrelation is significant (two-tailed)."""
        # Under null hypothesis, r ~ N(0, 1/sqrt(n))
        se = 1 / np.sqrt(n)
        z = r / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return p < alpha, p, z

    print(f"{'Cycle':<15} {'Lag (days)':<12} {'Autocorr':>10} {'p-value':>12} {'Significant':>12}")
    print("-" * 65)

    test_lags = {
        'Lunar/Monthly': 21,
        'Bi-weekly': 10,
        'Weekly': 5,
        'Quarterly': 63,
        'Semi-annual': 126,
        'Annual': 252,
    }

    significant_autocorr = []
    for name, lag in test_lags.items():
        if lag < n - 10:
            r = autocorr_at_lag(returns, lag)
            is_sig, p, z = autocorr_significance(r, n)
            sig_str = "YES ***" if is_sig else "no"
            print(f"{name:<15} {lag:<12} {r:>10.4f} {p:>12.4e} {sig_str:>12}")
            if is_sig:
                significant_autocorr.append((name, lag, r, p))

    # =========================================================================
    # TEST 3: Compare to random walk (Monte Carlo significance)
    # =========================================================================
    print("\n--- TEST 3: MONTE CARLO SIGNIFICANCE TEST ---")
    print("Comparing to 1000 random walks with same volatility...")

    # Generate random walks with same volatility as SPY
    n_simulations = 1000
    volatility = np.std(returns)

    # For each simulation, compute PSD and find max power
    max_powers_random = []
    for _ in range(n_simulations):
        random_returns = np.random.normal(0, volatility, n)
        _, psd_random = signal.welch(random_returns, fs=252, nperseg=min(1024, n//4))
        max_powers_random.append(np.max(psd_random[1:]))  # Skip DC

    # What's the 95th percentile of max power under null?
    threshold_95 = np.percentile(max_powers_random, 95)
    threshold_99 = np.percentile(max_powers_random, 99)
    actual_max = np.max(psd_no_dc)

    print(f"Actual max spectral power: {actual_max:.2e}")
    print(f"95th percentile (random):  {threshold_95:.2e}")
    print(f"99th percentile (random):  {threshold_99:.2e}")

    if actual_max > threshold_99:
        print("RESULT: Significant cyclical pattern (p < 0.01)")
    elif actual_max > threshold_95:
        print("RESULT: Marginally significant (p < 0.05)")
    else:
        print("RESULT: No significant cycles above random noise")

    # =========================================================================
    # TEST 4: Specific lunar cycle test (proper circular statistics)
    # =========================================================================
    print("\n--- TEST 4: LUNAR CYCLE (Circular Statistics) ---")

    # Get lunar phase for each trading day
    def get_moon_phase(date):
        known_new_moon = datetime(2000, 1, 6, 18, 14)
        lunar_cycle = 29.53059
        if hasattr(date, 'to_pydatetime'):
            date = date.to_pydatetime()
        days_since = (date - known_new_moon).total_seconds() / 86400
        phase = (days_since % lunar_cycle) / lunar_cycle
        return phase * 2 * np.pi  # Convert to radians

    phases = np.array([get_moon_phase(d) for d in spy.index[1:]])  # Skip first (NaN return)

    # Rayleigh test: are returns clustered at specific phases?
    # High returns clustered? Low returns clustered?

    # Weight phases by absolute returns (volatility clustering)
    abs_returns = np.abs(returns)

    # Circular mean of phases weighted by |returns|
    weighted_x = np.sum(abs_returns * np.cos(phases)) / np.sum(abs_returns)
    weighted_y = np.sum(abs_returns * np.sin(phases)) / np.sum(abs_returns)
    resultant_length = np.sqrt(weighted_x**2 + weighted_y**2)

    # Compare to uniform distribution (no lunar effect)
    # Under uniform, resultant_length ~ sqrt(2/(n*pi)) for large n
    n_days = len(phases)
    expected_r = np.sqrt(2 / (n_days * np.pi))

    # Rayleigh test statistic
    z = n_days * resultant_length**2
    p_rayleigh = np.exp(-z)  # Approximate p-value

    print(f"Days analyzed: {n_days}")
    print(f"Resultant length (clustering): {resultant_length:.4f}")
    print(f"Expected under uniform: {expected_r:.4f}")
    print(f"Rayleigh test p-value: {p_rayleigh:.4e}")

    if p_rayleigh < 0.05:
        mean_phase = np.arctan2(weighted_y, weighted_x)
        mean_phase_deg = np.degrees(mean_phase) % 360
        print(f"SIGNIFICANT lunar clustering at phase {mean_phase_deg:.1f} degrees")
        print(f"  (0=new moon, 180=full moon)")
    else:
        print("NO significant lunar phase clustering")

    # =========================================================================
    # TEST 5: Day-of-week effect (known market anomaly)
    # =========================================================================
    print("\n--- TEST 5: DAY-OF-WEEK EFFECT (Known Anomaly) ---")

    spy_with_dow = spy.copy()
    spy_with_dow['Return'] = spy_with_dow['Close'].pct_change().values.flatten()
    spy_with_dow['DayOfWeek'] = spy_with_dow.index.dayofweek

    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    dow_returns = {}

    print(f"{'Day':<12} {'Mean Return':>15} {'Std':>12} {'N':>8}")
    print("-" * 50)

    for dow in range(5):
        day_returns = spy_with_dow[spy_with_dow['DayOfWeek'] == dow]['Return'].dropna()
        dow_returns[dow] = day_returns
        print(f"{dow_names[dow]:<12} {day_returns.mean()*100:>14.3f}% {day_returns.std()*100:>11.2f}% {len(day_returns):>8}")

    # ANOVA test for day-of-week effect
    f_stat, p_anova = stats.f_oneway(*[dow_returns[d].values for d in range(5)])
    print(f"\nANOVA F-statistic: {f_stat:.3f}")
    print(f"ANOVA p-value: {p_anova:.4e}")
    print(f"Day-of-week effect: {'SIGNIFICANT' if p_anova < 0.05 else 'NOT SIGNIFICANT'}")

    # =========================================================================
    # TEST 6: Month-of-year effect (January effect, etc.)
    # =========================================================================
    print("\n--- TEST 6: MONTH-OF-YEAR EFFECT ---")

    spy_with_month = spy.copy()
    spy_with_month['Return'] = spy_with_month['Close'].pct_change().values.flatten()
    spy_with_month['Month'] = spy_with_month.index.month

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_returns = {}

    print(f"{'Month':<8} {'Mean Return':>15} {'Std':>12} {'N':>8}")
    print("-" * 45)

    for month in range(1, 13):
        m_returns = spy_with_month[spy_with_month['Month'] == month]['Return'].dropna()
        month_returns[month] = m_returns
        print(f"{month_names[month-1]:<8} {m_returns.mean()*100:>14.3f}% {m_returns.std()*100:>11.2f}% {len(m_returns):>8}")

    # ANOVA for month effect
    f_stat_month, p_month = stats.f_oneway(*[month_returns[m].values for m in range(1, 13)])
    print(f"\nANOVA F-statistic: {f_stat_month:.3f}")
    print(f"ANOVA p-value: {p_month:.4e}")
    print(f"Month-of-year effect: {'SIGNIFICANT' if p_month < 0.05 else 'NOT SIGNIFICANT'}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: CYCLICAL PATTERNS IN S&P 500")
    print("=" * 70)

    all_significant = []

    if significant_peaks:
        all_significant.append(f"Spectral peaks: {[f'{p[0]:.0f}-day ({p[2]})' for p in significant_peaks]}")

    if significant_autocorr:
        all_significant.append(f"Autocorrelation: {[f'{a[0]} (lag {a[1]})' for a in significant_autocorr]}")

    if p_rayleigh < 0.05:
        all_significant.append("Lunar phase clustering")

    if p_anova < 0.05:
        all_significant.append("Day-of-week effect")

    if p_month < 0.05:
        all_significant.append("Month-of-year effect (seasonality)")

    if all_significant:
        print("\nSIGNIFICANT CYCLES DETECTED:")
        for s in all_significant:
            print(f"  * {s}")
    else:
        print("\nNO SIGNIFICANT CYCLICAL PATTERNS DETECTED")
        print("Market returns appear consistent with random walk.")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if p_anova < 0.05 or p_month < 0.05:
        print("Known calendar anomalies (day-of-week, month-of-year) ARE detectable.")
        print("These are REAL cycles caused by human behavior (trading patterns, tax timing).")

    if p_rayleigh < 0.05:
        print("Lunar cycle shows phase-locked volatility clustering.")
        print("This could be coincidence or weak human-driven effect.")
    else:
        print("Lunar cycle shows NO significant effect on market returns.")

    print("\nBOTTOM LINE:")
    if any([p_anova < 0.05, p_month < 0.05]):
        print("CYCLES EXIST in markets, but they're CALENDAR cycles, not celestial cycles.")
        print("Human behavior (weekends, tax season, holidays) creates real periodicity.")
    else:
        print("Market appears to follow random walk with no detectable cyclical structure.")

    return {
        'spectral_peaks': significant_peaks,
        'autocorr': significant_autocorr,
        'lunar_p': p_rayleigh,
        'dow_p': p_anova,
        'month_p': p_month,
    }


if __name__ == "__main__":
    run_cycle_detection()
