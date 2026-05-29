"""
Test: Do Astrological Events Predict Real Market Data?

Uses REAL S&P 500 data and REAL astronomical calculations to test:
1. Lunar phases (full moon vs new moon)
2. Mercury retrograde periods

This is the ultimate test of the Aztec Calendar hypothesis.
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

from scipy import stats


# MERCURY RETROGRADE DATES (real astronomical data from ephemeris)
MERCURY_RETROGRADE_PERIODS = [
    ('2019-03-05', '2019-03-28'),
    ('2019-07-07', '2019-07-31'),
    ('2019-10-31', '2019-11-20'),
    ('2020-02-17', '2020-03-10'),
    ('2020-06-18', '2020-07-12'),
    ('2020-10-14', '2020-11-03'),
    ('2021-01-30', '2021-02-21'),
    ('2021-05-29', '2021-06-22'),
    ('2021-09-27', '2021-10-18'),
    ('2022-01-14', '2022-02-04'),
    ('2022-05-10', '2022-06-03'),
    ('2022-09-09', '2022-10-02'),
    ('2022-12-29', '2023-01-18'),
    ('2023-04-21', '2023-05-14'),
    ('2023-08-23', '2023-09-15'),
    ('2023-12-13', '2024-01-02'),
    ('2024-04-01', '2024-04-25'),
    ('2024-08-05', '2024-08-28'),
    ('2024-11-26', '2024-12-15'),
]


def get_moon_phase(date):
    """Calculate moon phase (0=new, 0.5=full) using real astronomy."""
    # Known new moon: Jan 6, 2000
    known_new_moon = datetime(2000, 1, 6, 18, 14)
    lunar_cycle = 29.53059  # days (synodic month)

    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')
    elif hasattr(date, 'to_pydatetime'):
        date = date.to_pydatetime()

    days_since = (date - known_new_moon).total_seconds() / 86400
    phase = (days_since % lunar_cycle) / lunar_cycle
    return phase


def classify_moon(phase):
    """Classify moon phase into categories."""
    if phase < 0.125 or phase >= 0.875:
        return 'new'
    elif 0.375 <= phase < 0.625:
        return 'full'
    else:
        return 'other'


def is_mercury_retrograde(date):
    """Check if date falls during Mercury retrograde."""
    if hasattr(date, 'strftime'):
        date_str = date.strftime('%Y-%m-%d')
    else:
        date_str = str(date)[:10]

    for start, end in MERCURY_RETROGRADE_PERIODS:
        if start <= date_str <= end:
            return True
    return False


def run_market_astrology_test():
    """Main test: Do astrological events predict market behavior?"""

    if not YFINANCE_AVAILABLE:
        print("Installing yfinance...")
        import subprocess
        subprocess.run(['pip', 'install', 'yfinance', '-q'])
        import yfinance as yf
    else:
        import yfinance as yf

    # Fetch S&P 500 data (5 years)
    print("Fetching S&P 500 data (2019-2024)...")
    spy = yf.download('SPY', start='2019-01-01', end='2024-12-31', progress=False)
    print(f"Got {len(spy)} trading days")

    # Calculate daily returns and volatility
    spy['Return'] = spy['Close'].pct_change()
    spy['Volatility'] = spy['Return'].rolling(5).std()  # 5-day rolling volatility

    # Add moon phase
    spy['MoonPhase'] = [get_moon_phase(d) for d in spy.index]
    spy['MoonType'] = [classify_moon(p) for p in spy['MoonPhase']]

    # Add Mercury retrograde
    spy['MercuryRetrograde'] = [is_mercury_retrograde(d) for d in spy.index]

    print("\n" + "=" * 70)
    print("REAL MARKET DATA vs ASTROLOGICAL EVENTS")
    print("=" * 70)

    # Remove NaN values
    spy_clean = spy.dropna()

    results = {}

    # TEST 1: Full Moon vs New Moon returns
    print("\n--- TEST 1: LUNAR PHASE AND RETURNS ---")
    full_moon_returns = spy_clean[spy_clean['MoonType'] == 'full']['Return']
    new_moon_returns = spy_clean[spy_clean['MoonType'] == 'new']['Return']
    other_returns = spy_clean[spy_clean['MoonType'] == 'other']['Return']

    print(f"Full moon days:  n={len(full_moon_returns):4d}, mean return={full_moon_returns.mean()*100:+.3f}%")
    print(f"New moon days:   n={len(new_moon_returns):4d}, mean return={new_moon_returns.mean()*100:+.3f}%")
    print(f"Other days:      n={len(other_returns):4d}, mean return={other_returns.mean()*100:+.3f}%")

    t_stat, p_value = stats.ttest_ind(full_moon_returns, new_moon_returns)
    print(f"T-test (full vs new): t={t_stat:.3f}, p={p_value:.4f}")
    sig_lunar_returns = p_value < 0.05
    print(f"SIGNIFICANT: {'YES' if sig_lunar_returns else 'NO'}")
    results['lunar_returns_p'] = p_value

    # TEST 2: Full Moon vs New Moon volatility
    print("\n--- TEST 2: LUNAR PHASE AND VOLATILITY ---")
    full_moon_vol = spy_clean[spy_clean['MoonType'] == 'full']['Volatility']
    new_moon_vol = spy_clean[spy_clean['MoonType'] == 'new']['Volatility']
    other_vol = spy_clean[spy_clean['MoonType'] == 'other']['Volatility']

    print(f"Full moon days:  mean volatility={full_moon_vol.mean()*100:.3f}%")
    print(f"New moon days:   mean volatility={new_moon_vol.mean()*100:.3f}%")
    print(f"Other days:      mean volatility={other_vol.mean()*100:.3f}%")

    t_stat_vol, p_value_vol = stats.ttest_ind(full_moon_vol.dropna(), new_moon_vol.dropna())
    print(f"T-test (full vs new): t={t_stat_vol:.3f}, p={p_value_vol:.4f}")
    sig_lunar_vol = p_value_vol < 0.05
    print(f"SIGNIFICANT: {'YES' if sig_lunar_vol else 'NO'}")
    results['lunar_vol_p'] = p_value_vol

    # TEST 3: Mercury Retrograde Returns
    print("\n--- TEST 3: MERCURY RETROGRADE AND RETURNS ---")
    retro_returns = spy_clean[spy_clean['MercuryRetrograde'] == True]['Return']
    normal_returns = spy_clean[spy_clean['MercuryRetrograde'] == False]['Return']

    print(f"Mercury retrograde:   n={len(retro_returns):4d}, mean return={retro_returns.mean()*100:+.3f}%")
    print(f"Mercury direct:       n={len(normal_returns):4d}, mean return={normal_returns.mean()*100:+.3f}%")

    t_stat_merc, p_value_merc = stats.ttest_ind(retro_returns, normal_returns)
    print(f"T-test: t={t_stat_merc:.3f}, p={p_value_merc:.4f}")
    sig_merc_returns = p_value_merc < 0.05
    print(f"SIGNIFICANT: {'YES' if sig_merc_returns else 'NO'}")
    results['mercury_returns_p'] = p_value_merc

    # TEST 4: Mercury Retrograde and Volatility
    print("\n--- TEST 4: MERCURY RETROGRADE AND VOLATILITY ---")
    retro_vol = spy_clean[spy_clean['MercuryRetrograde'] == True]['Volatility']
    normal_vol = spy_clean[spy_clean['MercuryRetrograde'] == False]['Volatility']

    print(f"Mercury retrograde:   mean volatility={retro_vol.mean()*100:.3f}%")
    print(f"Mercury direct:       mean volatility={normal_vol.mean()*100:.3f}%")
    vol_ratio = retro_vol.mean()/normal_vol.mean()
    print(f"Volatility ratio (retro/direct): {vol_ratio:.2f}x")

    t_stat_merc_vol, p_value_merc_vol = stats.ttest_ind(retro_vol.dropna(), normal_vol.dropna())
    print(f"T-test: t={t_stat_merc_vol:.3f}, p={p_value_merc_vol:.4f}")
    sig_merc_vol = p_value_merc_vol < 0.05
    print(f"SIGNIFICANT: {'YES' if sig_merc_vol else 'NO'}")
    results['mercury_vol_p'] = p_value_merc_vol

    # TEST 5: Cumulative performance
    print("\n--- TEST 5: CUMULATIVE PERFORMANCE ---")
    retro_cum = (1 + retro_returns).prod() - 1
    normal_cum = (1 + normal_returns).prod() - 1
    print(f"Total return during Mercury retrograde periods: {retro_cum*100:+.1f}%")
    print(f"Total return during Mercury direct periods:     {normal_cum*100:+.1f}%")

    # Annualized
    retro_annual = ((1 + retro_returns).prod()) ** (252/len(retro_returns)) - 1
    normal_annual = ((1 + normal_returns).prod()) ** (252/len(normal_returns)) - 1
    print(f"Annualized return (retrograde): {retro_annual*100:+.1f}%")
    print(f"Annualized return (direct):     {normal_annual*100:+.1f}%")

    # SUMMARY
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    significant_effects = []
    if sig_lunar_returns:
        significant_effects.append(f"Lunar returns: p={p_value:.4f}")
    if sig_lunar_vol:
        significant_effects.append(f"Lunar volatility: p={p_value_vol:.4f}")
    if sig_merc_returns:
        significant_effects.append(f"Mercury returns: p={p_value_merc:.4f}")
    if sig_merc_vol:
        significant_effects.append(f"Mercury volatility: p={p_value_merc_vol:.4f}")

    if significant_effects:
        print("STATISTICALLY SIGNIFICANT EFFECTS FOUND:")
        for e in significant_effects:
            print(f"  * {e}")
    else:
        print("NO STATISTICALLY SIGNIFICANT EFFECTS (p < 0.05)")

    # Even if not significant, report effect sizes
    print("\n--- EFFECT SIZES (regardless of significance) ---")
    print(f"Full moon vs new moon return diff: {(full_moon_returns.mean() - new_moon_returns.mean())*100:+.3f}%")
    print(f"Retrograde vs direct return diff:  {(retro_returns.mean() - normal_returns.mean())*100:+.3f}%")
    print(f"Retrograde volatility multiplier:  {vol_ratio:.2f}x")

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if any([sig_lunar_returns, sig_lunar_vol, sig_merc_returns, sig_merc_vol]):
        print("THE AZTECS WERE ONTO SOMETHING")
        print("Astrological cycles show statistically significant market correlations.")
    else:
        print("THE EMBEDDING SPACE LIED (or detected something else)")
        print("No significant correlation between astrological events and S&P 500.")
        print("\nPossible explanations:")
        print("  1. Embedding captures how humans TALK about these, not reality")
        print("  2. Effect exists but is too small to detect in 5 years")
        print("  3. Effect was arbitraged away by believers")
        print("  4. Need different market (crypto? emerging markets?)")

    return results


if __name__ == "__main__":
    run_market_astrology_test()
