"""
porsche_ferrari_quant_analysis.py
==================================================================

Porsche AG (P911.DE) vs. Ferrari N.V. (RACE.MI) relative-value analysis.
This is the exact analytical backbone behind the flagship research
report: same tests, same assumptions, same numbers - just written as
plain, flat, procedural Python instead of a class-based framework.

The question this script answers: has the P911/RACE price ratio been
mean-reverting around a stable long-run average (a statistical
arbitrage opportunity), or has the relationship between the two stocks
structurally broken down (a value trap)?

Five steps, run top to bottom:
    1. fetch_price_data()      -> pull daily prices, build the log ratio
    2. run_statistical_tests() -> ADF test (mean-reverting?) + Chow test
                                   (did the relationship break?)
    3. calculate_half_life()   -> OLS regression -> Ornstein-Uhlenbeck
                                   speed of reversion, plus a 1,000-draw
                                   bootstrap confidence interval
    4. run_scenarios()         -> bear / base / bull valuation bridge,
                                   plus a 2-way sensitivity grid
    5. backtest_spread()       -> historically, did "cheap" mean revert?

Install dependencies:
    pip install numpy pandas scipy statsmodels yfinance

Reproducibility
----------------
Locking start_date/end_date is NOT enough on its own: Yahoo Finance
quietly revises historical adjusted closing prices over time (dividend
and split adjustments get recomputed retroactively), so re-pulling "the
same" date range on two different days can still return slightly
different numbers - we saw this firsthand (1,007 vs. 1,008 trading
days on two live pulls of an identical date range).

So this script ships in two modes, set by ANALYSIS_MODE below:

    "reproducible" (default) - loads a frozen CSV snapshot of prices
        checked into this repo (DATA_SNAPSHOT_PATH). No network call at
        all, so the numbers are bit-for-bit identical forever, on any
        machine, in any year - this is what the PDF report's numbers
        are locked to.
    "live" - ignores the snapshot and pulls fresh data from Yahoo
        Finance. Use this to see how the thesis looks with more recent
        data; the numbers will then legitimately differ from the PDF
        report (see the Methodological Appendix footnote) - that's
        expected, not a bug.

To (re)build the snapshot the "reproducible" mode reads from, run once
with an internet connection and commit the CSV it writes:

    df = fetch_price_data(mode="live", end_date=AS_OF_DATE)
    df[["P911", "RACE"]].to_csv(DATA_SNAPSHOT_PATH)
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import yfinance as yf

# ----------------------------------------------------------------------
# REPRODUCIBILITY MODE - flip this one constant, nothing else
# ----------------------------------------------------------------------
ANALYSIS_MODE = "reproducible"     # "reproducible" (frozen snapshot) or "live" (fresh yfinance pull)
AS_OF_DATE = "2026-09-21"          # the exact end date the PDF report's numbers are locked to
START_DATE = "2022-09-30"
DATA_SNAPSHOT_PATH = "data/p911_race_prices_snapshot.csv"


# ----------------------------------------------------------------------
# STEP 1: Pull the price data and build the spread
# ----------------------------------------------------------------------
def fetch_price_data(
    start_date=START_DATE, end_date=AS_OF_DATE, mode=ANALYSIS_MODE, snapshot_path=DATA_SNAPSHOT_PATH
):
    """
    Get daily closing prices for Porsche (P911.DE) and Ferrari (RACE.MI)
    and compute the log price ratio.

    mode="reproducible": read the frozen CSV snapshot checked into this
    repo. Never touches the network, so it can't be affected by Yahoo
    Finance revising historical prices - this is what makes the PDF
    report's numbers reproducible forever.

    mode="live": download fresh data from Yahoo Finance, start_date to
    end_date (pass end_date=None for "today"). The numbers you get back
    will then legitimately drift slightly from the PDF report over time
    - that's expected vendor-side data revision, not a bug.
    """
    if mode == "reproducible":
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(
                f"No snapshot found at '{snapshot_path}'. Build one first by running, "
                f"once, with an internet connection:\n"
                f"    df = fetch_price_data(mode='live', end_date=AS_OF_DATE)\n"
                f"    df[['P911', 'RACE']].to_csv('{snapshot_path}')\n"
                f"...then commit that CSV to the repo, or pass mode='live' to skip "
                f"the snapshot entirely."
            )
        print(f"Loading frozen snapshot: {snapshot_path}  (mode='reproducible', no network call)")
        df = pd.read_csv(snapshot_path, index_col=0, parse_dates=True)

    elif mode == "live":
        pull_end_date = end_date or pd.Timestamp.today().date().isoformat()
        print(f"Downloading P911.DE and RACE.MI prices (LIVE): {start_date} to {pull_end_date} ...")
        raw = yf.download(
            ["P911.DE", "RACE.MI"],
            start=start_date,
            end=pull_end_date,
            auto_adjust=True,
            progress=False,
        )["Close"]
        df = raw.rename(columns={"P911.DE": "P911", "RACE.MI": "RACE"}).dropna()

    else:
        raise ValueError(f"mode must be 'reproducible' or 'live', got {mode!r}")

    # Everything below operates on the LOG of the price ratio, not the raw
    # ratio. Log differences are additive over time, which is exactly what
    # the ADF, Chow, and OU tests below assume about the data.
    df["log_ratio"] = np.log(df["P911"] / df["RACE"])

    print(f"Got {len(df)} trading days of data: {df.index.min().date()} to {df.index.max().date()}")
    return df


# ----------------------------------------------------------------------
# STEP 2: Is the spread mean-reverting, and did it break at some point?
# ----------------------------------------------------------------------
def run_statistical_tests(df, break_date="2024-06-01"):
    """
    Run two tests on the log price ratio:

    ADF test (Augmented Dickey-Fuller): is this series mean-reverting, or
    does it wander like a random walk with no pull-back force (a "unit
    root")? p-value < 0.05 means we can call it mean-reverting.

    Chow test: did the statistical relationship between the two stocks
    change at `break_date`? We compare one straight-line trend fit
    through the WHOLE sample against two separate trend lines fit before
    and after the break. p-value < 0.05 means yes, something changed.
    """
    log_ratio = df["log_ratio"].dropna()

    # --- ADF test: statsmodels already implements this correctly, so we
    # just call it rather than hand-rolling the Dickey-Fuller critical
    # values ourselves. ---
    adf_result = adfuller(log_ratio, autolag="AIC", result_object=False)
    adf_statistic = adf_result[0]
    p_value_adf = adf_result[1]

    # --- Chow test: statsmodels has no built-in for this, so we build it
    # from two simple OLS fits. ---
    before = log_ratio[log_ratio.index < break_date]
    after = log_ratio[log_ratio.index >= break_date]

    def line_fit_error(series):
        """Fit a straight line (y = a + b*t) by OLS, return leftover squared error."""
        t = np.arange(len(series))
        design_matrix = np.column_stack([np.ones(len(t)), t])
        coeffs, _, _, _ = np.linalg.lstsq(design_matrix, series.values, rcond=None)
        residuals = series.values - design_matrix @ coeffs
        return np.sum(residuals ** 2)

    ssr_pooled = line_fit_error(log_ratio)                       # 1 line, whole sample
    ssr_split = line_fit_error(before) + line_fit_error(after)   # 2 lines, split at the break

    k = 2                    # parameters per line: an intercept and a slope
    n = len(log_ratio)
    numerator = (ssr_pooled - ssr_split) / k
    denominator = ssr_split / (n - 2 * k)
    chow_f_statistic = numerator / denominator
    chow_p_value = 1 - stats.f.cdf(chow_f_statistic, k, n - 2 * k)

    print("\n--- Step 2: Statistical Tests ---")
    print(f"ADF statistic: {adf_statistic:.4f} | p-value: {p_value_adf:.4f}")
    print(f"  -> {'Mean-reverting' if p_value_adf < 0.05 else 'NOT mean-reverting (random walk)'}")
    print(f"Chow F-statistic: {chow_f_statistic:.2f} | p-value: {chow_p_value:.4f} "
          f"(break tested at {break_date})")
    print(f"  -> {'Structural break confirmed' if chow_p_value < 0.05 else 'No evidence of a break'}")

    return {
        "adf_statistic": adf_statistic,
        "p_value_adf": p_value_adf,
        "chow_f_statistic": chow_f_statistic,
        "chow_p_value": chow_p_value,
    }


# ----------------------------------------------------------------------
# STEP 3: How fast would the spread revert, and how much do we trust that?
# ----------------------------------------------------------------------
def calculate_half_life(df, n_bootstrap=1000, random_state=7):
    """
    Estimate the Ornstein-Uhlenbeck (OU) speed of reversion using a plain
    OLS regression, then bootstrap it 1,000 times to see how much we
    should actually trust that single point estimate.

    Once you discretize the OU process to daily steps, it collapses to
    an AR(1) regression:
        X_t = a + b * X_(t-1) + error
    Regressing today's log-ratio on yesterday's gives us `a` and `b`, and
    there's a direct algebraic formula that converts them into an
    annualized speed of reversion (kappa) and a half-life.
    """
    log_ratio = df["log_ratio"].dropna().values
    dt = 1 / 252  # ~252 trading days per year

    def fit_ar1(x_yesterday, x_today):
        """One OLS regression of today's value on yesterday's value -> (intercept, slope)."""
        design_matrix = np.column_stack([np.ones(len(x_yesterday)), x_yesterday])
        coeffs, _, _, _ = np.linalg.lstsq(design_matrix, x_today, rcond=None)
        return coeffs[0], coeffs[1]

    def solve_ou_params(intercept, slope):
        """Convert AR(1) coefficients into kappa, theta, and half-life (in trading days)."""
        kappa = (1 - slope) / dt
        theta = intercept / (kappa * dt) if kappa != 0 else np.mean(log_ratio)
        if kappa > 0:
            half_life_days = (np.log(2) / kappa) / dt
        else:
            half_life_days = np.inf  # kappa <= 0 means "no pull-back force at all"
        return kappa, theta, half_life_days

    # --- Point estimate, using the full sample ---
    x_today = log_ratio[1:]
    x_yesterday = log_ratio[:-1]
    intercept, slope = fit_ar1(x_yesterday, x_today)
    kappa, theta, spread_half_life = solve_ou_params(intercept, slope)

    # --- Bootstrap: how much should we trust that point estimate? ---
    # A single historical price path is only ONE possible realization of
    # the underlying random process that generated it. To see how noisy
    # our kappa estimate really is, we resample (yesterday, today) pairs
    # WITH replacement 1,000 times and refit the same regression on each
    # resample. The spread of kappa across those 1,000 refits is our
    # confidence interval - a simple, standard way to get a CI without
    # assuming a textbook formula for one.
    rng = np.random.default_rng(random_state)
    n_obs = len(x_today)
    boot_kappas = np.empty(n_bootstrap)
    boot_half_lives = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n_obs, size=n_obs)  # draw n_obs indices, with replacement
        boot_intercept, boot_slope = fit_ar1(x_yesterday[sample_idx], x_today[sample_idx])
        boot_kappa, _, boot_half_life = solve_ou_params(boot_intercept, boot_slope)
        boot_kappas[i] = boot_kappa
        boot_half_lives[i] = boot_half_life

    kappa_ci95 = (np.percentile(boot_kappas, 2.5), np.percentile(boot_kappas, 97.5))

    # A resample with kappa <= 0 implies "no reversion at all" for that
    # draw, so its half-life is infinite. We exclude those before taking
    # percentiles of the half-life (mixing in infinities breaks the
    # percentile calculation) and report how large that excluded share is
    # separately - that share is itself an important result: it tells you
    # how often "no reversion" is a live possibility in this data, not
    # just a rounding artifact.
    frac_bootstrap_non_reverting = np.mean(boot_kappas <= 0)
    finite_half_lives = boot_half_lives[np.isfinite(boot_half_lives)]
    if len(finite_half_lives) >= 20:
        half_life_ci95 = (np.percentile(finite_half_lives, 2.5), np.percentile(finite_half_lives, 97.5))
    else:
        half_life_ci95 = (np.nan, np.nan)  # too few finite draws for a meaningful CI

    print("\n--- Step 3: Ornstein-Uhlenbeck Half-Life (point estimate + bootstrap CI) ---")
    print(f"kappa (speed of reversion, annualized): {kappa:.4f}  |  "
          f"95% CI: ({kappa_ci95[0]:.4f}, {kappa_ci95[1]:.4f})")
    print(f"theta (equilibrium level, log-ratio units): {theta:.4f}  (ratio level: {np.exp(theta):.4f})")
    print(f"Half-life: {spread_half_life:,.0f} trading days (~{spread_half_life / 21:.1f} months)  |  "
          f"95% CI: ({half_life_ci95[0]:,.0f}, {half_life_ci95[1]:,.0f}) trading days")
    print(f"Bootstrap resamples implying NO reversion (kappa <= 0): "
          f"{frac_bootstrap_non_reverting:.1%} of {n_bootstrap}")

    return {
        "kappa": kappa,
        "theta": theta,
        "spread_half_life": spread_half_life,
        "kappa_ci95": kappa_ci95,
        "half_life_ci95": half_life_ci95,
        "frac_bootstrap_non_reverting": frac_bootstrap_non_reverting,
    }


# ----------------------------------------------------------------------
# STEP 4: Bear / base / bull fundamental valuation
# ----------------------------------------------------------------------
def run_scenarios():
    """
    Build a 3-state (bear / base / bull) valuation bridge for both
    stocks, plus a 2-way sensitivity grid for Porsche.

    The bridge works in two steps:
      1. Flex each company's EBITDA for the scenario's China-revenue
         growth assumption and a margin adjustment.
      2. Convert flexed EBITDA into a share price:
         implied price = (EBITDA x exit multiple - net debt) / shares.

    All EBITDA/margin/multiple assumptions below are illustrative
    research inputs, not live sell-side consensus - swap in current
    numbers before using this for a real investment decision.
    """
    # Baseline financials (EUR millions, except shares, which are in millions)
    porsche = {"base_ebitda": 6_860.0, "base_margin": 0.194, "net_debt": 4_100.0,
               "shares": 911.0, "china_share": 0.15}
    ferrari = {"base_ebitda": 2_970.0, "base_margin": 0.39, "net_debt": 2_000.0,
               "shares": 180.0, "china_share": 0.09}

    scenarios = pd.DataFrame([
        {"scenario": "Bear", "probability": 0.25, "china_growth": -0.20, "margin_delta_bps": -150,
         "porsche_multiple": 3.5, "ferrari_multiple": 22.0},
        {"scenario": "Base", "probability": 0.50, "china_growth": -0.05, "margin_delta_bps": 0,
         "porsche_multiple": 5.0, "ferrari_multiple": 27.0},
        {"scenario": "Bull", "probability": 0.25, "china_growth": 0.05, "margin_delta_bps": 100,
         "porsche_multiple": 7.0, "ferrari_multiple": 32.0},
    ])

    def flex_ebitda(company, china_growth, margin_delta_bps):
        """
        Flex a company's EBITDA for one scenario, in two steps:
        (a) apply the China revenue growth rate ONLY to the China-exposed
            share of revenue - everything else stays flat, and
        (b) shift the blended EBITDA margin by a basis-point adjustment,
            applied to the resulting total revenue.
        This keeps the "China exposure" story tied to each company's
        actual revenue mix, instead of flexing 100% of revenue by the
        same growth rate regardless of geography.
        """
        implied_revenue = company["base_ebitda"] / company["base_margin"]
        china_revenue = implied_revenue * company["china_share"]
        non_china_revenue = implied_revenue - china_revenue
        flexed_china_revenue = china_revenue * (1 + china_growth)
        flexed_revenue = non_china_revenue + flexed_china_revenue
        flexed_margin = company["base_margin"] + margin_delta_bps / 10_000
        return flexed_revenue * flexed_margin

    def implied_price(ebitda, multiple, net_debt, shares):
        """implied share price = (EBITDA x exit multiple - net debt) / shares outstanding"""
        return (ebitda * multiple - net_debt) / shares

    porsche_prices, ferrari_prices, implied_ratios = [], [], []
    for _, row in scenarios.iterrows():
        porsche_ebitda = flex_ebitda(porsche, row["china_growth"], row["margin_delta_bps"])
        ferrari_ebitda = flex_ebitda(ferrari, row["china_growth"], row["margin_delta_bps"])
        porsche_price = implied_price(
            porsche_ebitda, row["porsche_multiple"], porsche["net_debt"], porsche["shares"]
        )
        ferrari_price = implied_price(
            ferrari_ebitda, row["ferrari_multiple"], ferrari["net_debt"], ferrari["shares"]
        )
        porsche_prices.append(porsche_price)
        ferrari_prices.append(ferrari_price)
        implied_ratios.append(porsche_price / ferrari_price)

    scenarios["porsche_price"] = porsche_prices
    scenarios["ferrari_price"] = ferrari_prices
    scenarios["implied_pr_ratio"] = implied_ratios

    # Probability-weighted expected value across the three scenarios.
    expected_porsche_price = (scenarios["porsche_price"] * scenarios["probability"]).sum()
    expected_ferrari_price = (scenarios["ferrari_price"] * scenarios["probability"]).sum()
    expected_ratio = (scenarios["implied_pr_ratio"] * scenarios["probability"]).sum()

    print("\n--- Step 4: Bear / Base / Bull Scenarios ---")
    print(scenarios.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(
        f"\nProbability-weighted expected value -> "
        f"Porsche: EUR {expected_porsche_price:,.2f} | "
        f"Ferrari: EUR {expected_ferrari_price:,.2f} | "
        f"Ratio: {expected_ratio:.3f}"
    )

    # --- 2-way sensitivity grid: Porsche's implied price across a range
    # of EBITDA growth rates (rows) and exit EV/EBITDA multiples (columns).
    # This grid deliberately uses a SIMPLER flat EBITDA growth flex (not
    # the China-revenue bridge above), because its job is different: it
    # isolates pure operating-leverage risk, independent of any one
    # scenario's specific China narrative. ---
    growth_range = [-0.15, -0.05, 0.0, 0.05, 0.15]
    multiple_range = [3.5, 4.5, 5.0, 5.5, 6.5, 7.0]
    sensitivity_grid = pd.DataFrame(index=growth_range, columns=multiple_range, dtype=float)
    sensitivity_grid.index.name = "ebitda_growth"
    sensitivity_grid.columns.name = "exit_ev_ebitda"

    for growth in growth_range:
        flexed_ebitda = porsche["base_ebitda"] * (1 + growth)
        for multiple in multiple_range:
            sensitivity_grid.loc[growth, multiple] = implied_price(
                flexed_ebitda, multiple, porsche["net_debt"], porsche["shares"]
            )

    print("\nPorsche sensitivity grid (rows = EBITDA growth, cols = exit EV/EBITDA), implied price EUR:")
    print(sensitivity_grid.to_string(float_format=lambda v: f"{v:,.1f}"))

    return {"scenarios": scenarios, "sensitivity_grid": sensitivity_grid}


# ----------------------------------------------------------------------
# STEP 5: Did "cheap" historically mean revert? (+ risk metrics)
# ----------------------------------------------------------------------
def backtest_spread(df, z_window=252, entry_low=-2.0, entry_high=-1.5, break_date="2024-06-01"):
    """
    Three things, since they all come out of the same daily price data:

    (a) Historical backtest: every time the rolling Z-score of the log
        ratio dropped into the "cheap Porsche" band (entry_low to
        entry_high), check 6 and 12 months later whether the ratio
        closed at least half the gap back to its long-run average.
        That counts as a "hit".

    (b) Tracking error & information ratio: the annualized volatility of
        the return gap between the two stocks, and the expected payoff
        per unit of that risk.

    (c) A regime-split "current Z-score" reading: how far today's ratio
        sits from ITS OWN regime's average, rather than from one pooled
        average spanning both regimes. Step 2's Chow test says the
        relationship broke in mid-2024 - so a Z-score computed against
        the whole since-2022 sample is anchored to a mean the series no
        longer trades around. This is the number quoted in the report's
        Executive Summary, and it is deliberately a separate calculation
        from the pooled Z-score used for the backtest below.
    """
    ratio = np.exp(df["log_ratio"])  # back to the plain P911/RACE ratio

    # Rolling Z-score (pooled, whole sample): how many standard deviations
    # is today's log ratio from its own trailing 252-trading-day (~1 year)
    # average? This is the version the backtest below actually trades on.
    rolling_mean = df["log_ratio"].rolling(z_window).mean()
    rolling_std = df["log_ratio"].rolling(z_window).std()
    z_score = (df["log_ratio"] - rolling_mean) / rolling_std

    # Regime-split Z-score (post-break only, since "today" always falls
    # after break_date): rather than a fixed trailing window, we use an
    # EXPANDING mean/std over all data since the break. That way the
    # regime doesn't need a full 252-day lookback to start producing
    # readings right after the structural break - it just accumulates
    # more precision as more post-break history becomes available.
    post_break_ratio = df["log_ratio"][df.index >= break_date]
    post_break_mean = post_break_ratio.expanding(min_periods=20).mean()
    post_break_std = post_break_ratio.expanding(min_periods=20).std()
    regime_z_score = (post_break_ratio - post_break_mean) / post_break_std

    long_run_mean = ratio.mean()
    in_band = (z_score >= entry_low) & (z_score <= entry_high)
    entry_days = df.index[in_band & ~in_band.shift(1, fill_value=False)]

    hits_6m, hits_12m = [], []
    returns_6m, returns_12m = [], []

    for entry_date in entry_days:
        entry_level = ratio.loc[entry_date]
        gap_to_mean = long_run_mean - entry_level
        if gap_to_mean == 0:
            continue  # already sitting at the mean, nothing to test

        entry_position = ratio.index.get_loc(entry_date)
        for horizon_days, hit_list, return_list in [
            (126, hits_6m, returns_6m),    # ~6 trading months
            (252, hits_12m, returns_12m),  # ~12 trading months
        ]:
            future_position = entry_position + horizon_days
            if future_position >= len(ratio):
                continue  # not enough future data left for this horizon
            future_level = ratio.iloc[future_position]
            closed_gap = future_level - entry_level
            moved_the_right_way = np.sign(closed_gap) == np.sign(gap_to_mean)
            closed_half_the_gap = abs(closed_gap) >= 0.5 * abs(gap_to_mean)
            hit_list.append(moved_the_right_way and closed_half_the_gap)
            return_list.append(np.sign(gap_to_mean) * (future_level - entry_level) / entry_level)

    hit_rate_6m = np.mean(hits_6m) if hits_6m else np.nan
    hit_rate_12m = np.mean(hits_12m) if hits_12m else np.nan
    avg_return_6m = np.mean(returns_6m) if returns_6m else np.nan
    avg_return_12m = np.mean(returns_12m) if returns_12m else np.nan

    # Tracking error: annualized std dev of the daily return gap between
    # the two stocks. Information ratio: expected payoff per unit of that
    # risk. 0.384 is this thesis's illustrative target alpha - swap in a
    # freshly computed expected return before relying on this IR.
    p911_returns = np.log(df["P911"]).diff()
    race_returns = np.log(df["RACE"]).diff()
    tracking_error = (p911_returns - race_returns).std() * np.sqrt(252)
    expected_alpha = 0.384
    information_ratio = expected_alpha / tracking_error

    print("\n--- Step 5: Historical Backtest & Risk Metrics ---")
    print(f"Current Z-score (pooled, whole sample):        {z_score.iloc[-1]:.2f}")
    print(f"Current Z-score (regime-split, post-{break_date}): {regime_z_score.iloc[-1]:.2f}")
    print(f"Entry episodes (Z-score between {entry_low} and {entry_high}): {len(entry_days)}")
    print(f"  6-month hit rate:  {hit_rate_6m:.1%} | average return: {avg_return_6m:.2%}")
    print(f"  12-month hit rate: {hit_rate_12m:.1%} | average return: {avg_return_12m:.2%}")
    print(f"Annualized tracking error: {tracking_error:.2%}")
    print(f"Information ratio (on a {expected_alpha:.1%} target alpha): {information_ratio:.2f}")

    return {
        "current_z_score_pooled": z_score.iloc[-1],
        "current_z_score_regime_split": regime_z_score.iloc[-1],
        "n_episodes": len(entry_days),
        "hit_rate_6m": hit_rate_6m,
        "hit_rate_12m": hit_rate_12m,
        "avg_return_6m": avg_return_6m,
        "avg_return_12m": avg_return_12m,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
    }


# ----------------------------------------------------------------------
# Run everything, top to bottom
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("PORSCHE (P911) vs. FERRARI (RACE) - RELATIVE VALUE ANALYSIS")
    print("=" * 70)

    price_df = fetch_price_data()
    test_results = run_statistical_tests(price_df)
    ou_results = calculate_half_life(price_df)
    scenario_results = run_scenarios()
    backtest_results = backtest_spread(price_df)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"ADF p-value:           {test_results['p_value_adf']:.4f}  "
          f"({'mean-reverting' if test_results['p_value_adf'] < 0.05 else 'random walk'})")
    print(f"Chow p-value:          {test_results['chow_p_value']:.4f}  "
          f"({'break confirmed' if test_results['chow_p_value'] < 0.05 else 'no break'})")
    print(f"Half-life:             {ou_results['spread_half_life']:,.0f} trading days  |  "
          f"95% CI {tuple(round(v) for v in ou_results['half_life_ci95'])}")
    print(f"Non-reverting share:   {ou_results['frac_bootstrap_non_reverting']:.1%} of bootstrap draws")
    print(f"12m backtest hit rate: {backtest_results['hit_rate_12m']:.1%}")
    print(f"Information ratio:     {backtest_results['information_ratio']:.2f}")
