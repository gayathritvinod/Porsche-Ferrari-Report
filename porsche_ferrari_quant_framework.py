"""
porsche_ferrari_quant_framework.py
====================================================================

Production research framework for the Porsche AG (P911.DE) vs. Ferrari
N.V. (RACE.MI) relative-value pairs thesis, implementing the five-module
architecture specified in the "Porsche vs Ferrari: Quant Re-Architecture
Memo":

    1. DataIngestionEngine        - price ingestion + co-movement/regime tests
    2. RollingZScoreDetector      - rolling Z-score + persistence-filtered
                                     anomaly flagging
    3. OUMeanReversionEngine      - Ornstein-Uhlenbeck fit + bootstrap CIs
    4. ScenarioSensitivityEngine  - 3-state probability-weighted scenario
                                     tree + 2-way sensitivity grid
    5. RiskBacktestEngine         - tracking error, information ratio,
                                     historical Z-score backtest

Dependencies
------------
    pip install numpy pandas scipy statsmodels yfinance

`statsmodels` and `yfinance` are optional: the script degrades gracefully
(an approximate ADF statistic, and a calibrated offline synthetic price
series) if either is missing, so it always runs end to end.

Run directly for a full demo:

    python porsche_ferrari_quant_framework.py

Author: Framework generated per the Quant Re-Architecture Memo for
        Gayathri Vinod's Porsche vs Ferrari equity research.
"""

from __future__ import annotations

import datetime as dt
import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:
    import yfinance as yf

    _HAS_YFINANCE = True
except ImportError:  # pragma: no cover - exercised only when yfinance absent
    _HAS_YFINANCE = False

try:
    from statsmodels.tsa.stattools import adfuller

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover - exercised only when statsmodels absent
    _HAS_STATSMODELS = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# Module 1: Data Ingestion & Co-Movement Engine
# ======================================================================


@dataclass
class CoMovementTestResult:
    """Bundled output of the stationarity and structural-break tests."""

    adf_statistic: float
    adf_pvalue: float
    adf_is_stationary: bool
    chow_f_statistic: float
    chow_p_value: float
    chow_break_significant: bool
    break_date: pd.Timestamp


class DataIngestionEngine:
    """
    Ingests daily closing prices for P911.DE and RACE.MI, builds the log
    price ratio ln(P911 / RACE), and runs the two tests that decide
    whether that ratio behaves like a mean-reverting spread or a
    structurally broken series (Memo Section 2.1 / 2.2, and gap #2 in
    Section 1: the original report's 0.25 "fair value" anchor pools two
    regimes that its own narrative says are different).
    """

    P911_TICKER = "P911.DE"
    RACE_TICKER = "RACE.MI"

    def __init__(self, start: str = "2022-09-01", end: Optional[str] = None) -> None:
        self.start = start
        self.end = end or dt.date.today().isoformat()
        self.prices: Optional[pd.DataFrame] = None
        self.log_ratio: Optional[pd.Series] = None

    def fetch_prices(self, use_offline_fallback: bool = True) -> pd.DataFrame:
        """
        Fetch daily closes via yfinance. Falls back to a bundled synthetic
        dataset (calibrated to the original report's own price anchors and
        this memo's updated Sept-2026 levels) if yfinance is unavailable,
        the network call fails, or the sandbox has no internet access.
        """
        if _HAS_YFINANCE:
            try:
                logger.info("Fetching %s and %s via yfinance ...", self.P911_TICKER, self.RACE_TICKER)
                raw = yf.download(
                    [self.P911_TICKER, self.RACE_TICKER],
                    start=self.start,
                    end=self.end,
                    progress=False,
                    auto_adjust=True,
                )["Close"]
                raw = raw.rename(columns={self.P911_TICKER: "P911", self.RACE_TICKER: "RACE"})
                raw = raw.dropna(how="any")
                if raw.empty:
                    raise ValueError("yfinance returned an empty frame.")
                self.prices = raw
                logger.info("Fetched %d daily observations from yfinance.", len(raw))
                return self.prices
            except Exception as exc:  # noqa: BLE001 - network/library failures are broad
                logger.warning("yfinance ingestion failed (%s); falling back to offline dataset.", exc)

        if not use_offline_fallback:
            raise RuntimeError("Live data unavailable and offline fallback disabled.")

        self.prices = self._build_offline_fallback()
        logger.info("Using offline fallback dataset (%d observations).", len(self.prices))
        return self.prices

    @staticmethod
    def _build_offline_fallback(seed: int = 42) -> pd.DataFrame:
        """
        Synthetic daily price series calibrated to the original report's
        own anchor points (Appx A-1: IPO Sept-2022 through Feb-2026) and
        this memo's Section 3 update (Sept-2026 levels), so the rest of
        the pipeline is fully runnable with no network access. This is a
        stand-in for a live data pull, NOT a substitute for one in an
        actual research or trading workflow.
        """
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2022-09-29", dt.date.today().isoformat())
        n = len(dates)

        # Price anchors taken directly from the original report's Appx
        # A-1 table, extended with this memo's Section 3.1 current levels.
        anchors: dict[str, tuple[float, float]] = {
            "2022-09-29": (82.50, 191.70),   # IPO baseline
            "2022-11-15": (107.05, 208.20),  # Post-IPO peak optimism
            "2024-05-15": (82.40, 388.20),   # "Jaws of Death" decoupling
            "2025-09-15": (44.35, 420.00),   # DAX-removal trough
            "2026-01-15": (39.80, 315.00),   # Leiters-pivot low
            "2026-02-10": (41.61, 311.70),   # Original report's "current"
        }
        if dates[-1] > pd.Timestamp("2026-02-10"):
            anchors[dates[-1].date().isoformat()] = (45.12, 360.50)  # This memo's Sept-2026 update

        anchor_positions = [dates.get_indexer([pd.Timestamp(d)], method="nearest")[0] for d in anchors]
        anchor_p911 = [v[0] for v in anchors.values()]
        anchor_race = [v[1] for v in anchors.values()]

        log_p911 = np.interp(np.arange(n), anchor_positions, np.log(anchor_p911))
        log_race = np.interp(np.arange(n), anchor_positions, np.log(anchor_race))

        # Layer realistic daily noise on top of the interpolated trend, so
        # downstream statistics (ADF, OU fit, rolling Z-score, tracking
        # error) see a genuine stochastic series rather than a
        # piecewise-linear line with near-zero residual variance. A shared
        # market factor plus idiosyncratic shocks gives the two names
        # partial co-movement, with Ferrari carrying a lower beta to the
        # common factor - consistent with the "Jaws of Death" decoupling
        # from auto-sector beta the original report describes qualitatively.
        market_factor = rng.normal(0.0, 0.012, n)
        p911_idiosyncratic = rng.normal(0.0, 0.012, n)
        race_idiosyncratic = rng.normal(0.0, 0.010, n)
        p911_daily_shocks = market_factor + p911_idiosyncratic
        race_daily_shocks = 0.7 * market_factor + race_idiosyncratic

        p911_noise_path = np.cumsum(p911_daily_shocks - p911_daily_shocks.mean())
        race_noise_path = np.cumsum(race_daily_shocks - race_daily_shocks.mean())
        p911 = np.exp(log_p911 + p911_noise_path)
        race = np.exp(log_race + race_noise_path)

        return pd.DataFrame({"P911": p911, "RACE": race}, index=dates)

    def compute_log_ratio(self) -> pd.Series:
        """Compute ln(P911 / RACE) - the spread the entire framework operates on."""
        if self.prices is None:
            raise RuntimeError("Call fetch_prices() before compute_log_ratio().")
        ratio = self.prices["P911"] / self.prices["RACE"]
        self.log_ratio = np.log(ratio).rename("log_ratio")
        return self.log_ratio

    def run_adf_test(self) -> tuple[float, float]:
        """
        Augmented Dickey-Fuller test on the log ratio.

        H0: the series has a unit root (non-stationary, not mean-reverting).
        Rejecting H0 (p < 0.05) supports treating the ratio as a
        mean-reverting spread; failing to reject supports the "structural
        decoupling" / value-trap reading the memo flags as the load-bearing
        open question in the original report.
        """
        if self.log_ratio is None:
            self.compute_log_ratio()
        series = self.log_ratio.dropna()

        if _HAS_STATSMODELS:
            try:
                # Pin the classic tuple return explicitly: newer
                # statsmodels versions warn that the default will switch
                # to an ADFullerResult object in a future release.
                result = adfuller(series, autolag="AIC", result_object=False)
            except TypeError:
                # Older statsmodels versions do not accept result_object.
                result = adfuller(series, autolag="AIC")
            statistic, p_value = result[0], result[1]
        else:
            statistic, p_value = self._adf_fallback(series)
        return float(statistic), float(p_value)

    @staticmethod
    def _adf_fallback(series: pd.Series) -> tuple[float, float]:
        """
        Minimal ADF statistic used only if statsmodels is unavailable:
        regress delta(x_t) on x_{t-1} (no lag augmentation) and return the
        t-statistic on the x_{t-1} coefficient, with a normal-reference
        p-value. This is a conservative approximation - install
        statsmodels for the fully specified test with correct MacKinnon
        critical values before relying on this for a real decision.
        """
        warnings.warn(
            "statsmodels not installed - using an approximate ADF statistic. "
            "Install statsmodels for MacKinnon-correct critical values.",
            stacklevel=2,
        )
        x = series.to_numpy()
        dx = np.diff(x)
        x_lag = x[:-1]
        design = np.column_stack([np.ones_like(x_lag), x_lag])
        beta, *_ = np.linalg.lstsq(design, dx, rcond=None)
        resid = dx - design @ beta
        se = np.sqrt(np.sum(resid**2) / (len(dx) - 2) / np.sum((x_lag - x_lag.mean()) ** 2))
        t_stat = beta[1] / se
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        return float(t_stat), float(p_value)

    def run_chow_test(self, break_date: str = "2024-06-01") -> tuple[float, float]:
        """
        Chow structural-break test on the log ratio around the "Jaws of
        Death" decoupling the original report describes only qualitatively.

        Fits one OLS trend line on the pooled sample and two separate
        trend lines either side of `break_date`, then F-tests whether the
        two-line (broken) model fits significantly better than the
        one-line (pooled) model. A significant break (p < 0.05) means the
        pooled since-IPO mean used for the original report's Z-score
        (Appx A-2) is not a valid anchor - see gap #2 in Section 1 of the
        memo.
        """
        if self.log_ratio is None:
            self.compute_log_ratio()
        y = self.log_ratio.dropna()
        break_ts = pd.Timestamp(break_date)
        before = y[y.index < break_ts]
        after = y[y.index >= break_ts]
        if len(before) < 10 or len(after) < 10:
            raise ValueError("Not enough observations on one side of the break date to test.")

        def _residual_ssr(sub: pd.Series) -> tuple[float, int]:
            t = np.arange(len(sub), dtype=float)
            design = np.column_stack([np.ones_like(t), t])
            beta, *_ = np.linalg.lstsq(design, sub.to_numpy(), rcond=None)
            resid = sub.to_numpy() - design @ beta
            return float(np.sum(resid**2)), len(sub)

        ssr_pooled, n_pooled = _residual_ssr(y)
        ssr_before, _ = _residual_ssr(before)
        ssr_after, _ = _residual_ssr(after)
        ssr_unrestricted = ssr_before + ssr_after
        k = 2  # parameters per line: intercept + slope

        numerator = (ssr_pooled - ssr_unrestricted) / k
        denominator = ssr_unrestricted / (n_pooled - 2 * k)
        f_statistic = numerator / denominator
        p_value = float(1.0 - stats.f.cdf(f_statistic, k, n_pooled - 2 * k))
        return float(f_statistic), p_value

    def run_comovement_tests(self, break_date: str = "2024-06-01") -> CoMovementTestResult:
        """Convenience wrapper bundling the ADF and Chow tests (Memo Section 2.1/2.2)."""
        adf_stat, adf_p = self.run_adf_test()
        chow_f, chow_p = self.run_chow_test(break_date)
        return CoMovementTestResult(
            adf_statistic=adf_stat,
            adf_pvalue=adf_p,
            adf_is_stationary=adf_p < 0.05,
            chow_f_statistic=chow_f,
            chow_p_value=chow_p,
            chow_break_significant=chow_p < 0.05,
            break_date=pd.Timestamp(break_date),
        )


# ======================================================================
# Module 2: Dynamic Rolling Z-Score & Anomaly Detector
# ======================================================================


@dataclass
class AnomalyPeriod:
    """One episode where |Z| breached the threshold and persisted."""

    start: pd.Timestamp
    end: pd.Timestamp
    duration_days: int
    direction: str  # "cheap" (Z <= -threshold) or "rich" (Z >= +threshold)
    mean_z: float


class RollingZScoreDetector:
    """
    Rolling Z-score engine for the price ratio (or any other valuation
    multiple series passed in). Replaces the original report's single,
    six-point Z-score (Appx A-2) with a full rolling calculation, with an
    optional regime split around the structural break (Memo Section 2.1).
    """

    def __init__(self, window: int = 252, z_threshold: float = 2.0, min_persistence_days: int = 20) -> None:
        self.window = window
        self.z_threshold = z_threshold
        self.min_persistence_days = min_persistence_days

    def rolling_zscore(self, series: pd.Series) -> pd.DataFrame:
        """
        Z_t = (R_t - mu_{t-w:t}) / sigma_{t-w:t} for rolling window w.
        Returns the rolling mean, rolling std, and Z-score alongside the
        raw series.
        """
        rolling_mean = series.rolling(self.window, min_periods=self.window).mean()
        rolling_std = series.rolling(self.window, min_periods=self.window).std(ddof=1)
        z_score = (series - rolling_mean) / rolling_std
        return pd.DataFrame(
            {"value": series, "rolling_mean": rolling_mean, "rolling_std": rolling_std, "z_score": z_score}
        )

    def regime_split_zscore(self, series: pd.Series, break_date: str) -> pd.DataFrame:
        """
        Regime-adjusted Z-score: compute mu/sigma separately pre- and
        post-break_date rather than pooling the full since-IPO sample.

        Use this once DataIngestionEngine.run_chow_test() confirms a
        significant break - it is what Memo Section 2.1 prescribes in
        place of the naive pooled-sample Z-score that produced the
        original report's inconsistent 0.13 / 0.14 readings.
        """
        break_ts = pd.Timestamp(break_date)
        pre = series[series.index < break_ts]
        post = series[series.index >= break_ts]

        pre_mean, pre_std = pre.mean(), pre.std(ddof=1)
        pre_z = (pre - pre_mean) / pre_std

        # The post-break regime uses an EXPANDING window so early
        # post-break observations are not scored against a mean built
        # from too few points; it stabilizes as more history accumulates.
        post_expanding_mean = post.expanding(min_periods=20).mean()
        post_expanding_std = post.expanding(min_periods=20).std(ddof=1)
        post_z = (post - post_expanding_mean) / post_expanding_std

        combined_z = pd.concat([pre_z, post_z]).rename("regime_z_score")
        regime_label = pd.Series(
            np.where(series.index < break_ts, "pre_decoupling", "post_decoupling"),
            index=series.index,
            name="regime",
        )
        return pd.concat([series.rename("value"), regime_label, combined_z], axis=1)

    def flag_anomalies(self, z_scores: pd.Series) -> list[AnomalyPeriod]:
        """
        Anomaly filter (Memo Section 2.1 / 4.3): flag a mispricing only
        where |Z| >= z_threshold AND the breach persists for at least
        min_persistence_days consecutive trading days. This two-part gate
        is what prevents the framework from reacting to single-day noise.
        """
        z = z_scores.dropna()
        breach = z.abs() >= self.z_threshold
        episodes: list[AnomalyPeriod] = []

        # Group consecutive True runs: incrementing a counter every time
        # `breach` is False partitions the True runs into distinct groups.
        run_id = (~breach).cumsum()
        for _, group in z[breach].groupby(run_id[breach]):
            if len(group) >= self.min_persistence_days:
                episodes.append(
                    AnomalyPeriod(
                        start=group.index[0],
                        end=group.index[-1],
                        duration_days=len(group),
                        direction="cheap" if group.mean() < 0 else "rich",
                        mean_z=float(group.mean()),
                    )
                )
        return episodes


# ======================================================================
# Module 3: Ornstein-Uhlenbeck Mean-Reversion & Bootstrap Engine
# ======================================================================


@dataclass
class OUFitResult:
    """Point estimates and bootstrap confidence intervals for the OU fit."""

    kappa: float
    theta: float
    sigma: float
    half_life_days: float
    kappa_ci95: tuple[float, float]
    half_life_ci95_days: tuple[float, float]
    frac_bootstrap_non_reverting: float  # share of bootstrap resamples with kappa <= 0
    n_bootstrap: int


class OUMeanReversionEngine:
    """
    Fits the discretized Ornstein-Uhlenbeck process to the log price
    ratio and bootstraps confidence intervals for the reversion speed
    (kappa) and half-life (Memo Section 2.2 / 4.1). This is what gives
    the report's asserted "12-18 month horizon" an empirically derived
    speed, rather than a narrative assumption - see gap #8 in Section 1.
    """

    def __init__(self, dt_years: float = 1 / 252, n_bootstrap: int = 1000, random_state: int = 7) -> None:
        self.dt_years = dt_years
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    def _fit_ar1(series: np.ndarray) -> tuple[float, float, np.ndarray]:
        """
        OLS fit of X_t = a + b * X_{t-1} + eps_t, the discretized form of
        dX_t = kappa(theta - X_t) dt + sigma dW_t, where:
            b = 1 - kappa * dt   =>  kappa = (1 - b) / dt
            a = kappa * theta * dt  =>  theta = a / (kappa * dt)
        Returns (a, b, residuals).
        """
        x_t = series[1:]
        x_lag = series[:-1]
        design = np.column_stack([np.ones_like(x_lag), x_lag])
        beta, *_ = np.linalg.lstsq(design, x_t, rcond=None)
        a, b = beta
        resid = x_t - design @ beta
        return float(a), float(b), resid

    def _solve_ou_params(self, a: float, b: float, fallback_mean: float) -> tuple[float, float, float]:
        """
        Map AR(1) coefficients (a, b) to OU parameters: kappa (ANNUALIZED
        speed of reversion, per year), theta (long-run mean, same units as
        the input series), and half-life expressed in TRADING DAYS.

        kappa = (1 - b) / dt_years is annualized because dt_years = 1/252.
        ln(2) / kappa is therefore in YEARS; multiplying by 1 / dt_years
        (= 252) converts it to trading days. Skipping that conversion is
        an easy off-by-a-factor-of-252 bug - exactly the kind of unit
        error this framework exists to catch elsewhere in the report.

        Note: when kappa is small (the series is close to a unit root),
        theta's denominator (kappa * dt_years) is small, so this simple
        OLS-based theta estimate becomes numerically unstable. A theta far
        outside the series' observed range is itself a diagnostic that the
        series should not be treated as safely mean-reverting - corroborate
        with the ADF test before trusting it.
        """
        kappa = (1.0 - b) / self.dt_years
        theta = a / (kappa * self.dt_years) if kappa != 0 else fallback_mean
        if kappa > 0:
            half_life_years = np.log(2) / kappa
            half_life_days = half_life_years / self.dt_years
        else:
            half_life_days = np.inf
        return float(kappa), float(theta), float(half_life_days)

    def fit(self, series: pd.Series) -> OUFitResult:
        """Point estimate of kappa/theta/sigma/half-life, with bootstrapped 95% CIs."""
        x = series.dropna().to_numpy()
        a, b, resid = self._fit_ar1(x)
        kappa, theta, half_life_days = self._solve_ou_params(a, b, fallback_mean=float(x.mean()))

        # Diffusion coefficient sigma: approximation valid for small
        # kappa * dt (i.e. reversion that is slow relative to one day),
        # which holds for any half-life materially longer than a few
        # trading days. Use the exact OU variance formula instead if
        # kappa * dt is not small in your calibration.
        sigma = float(np.std(resid, ddof=2) / np.sqrt(self.dt_years))

        boot_kappas, boot_half_lives = self._bootstrap(x)
        kappa_ci = (float(np.percentile(boot_kappas, 2.5)), float(np.percentile(boot_kappas, 97.5)))

        # A negative or zero bootstrapped kappa implies no reversion at all
        # (half-life is undefined / infinite for that resample). Mixing
        # +inf into a percentile calculation can silently produce NaN
        # (inf - inf during interpolation), so the half-life CI is
        # computed only over the finite subset, and the excluded fraction
        # is reported explicitly - it is itself a key diagnostic: a large
        # frac_bootstrap_non_reverting means "no reversion" is a
        # statistically live possibility, not a rounding artefact.
        frac_non_reverting = float(np.mean(boot_kappas <= 0))
        finite_half_lives = boot_half_lives[np.isfinite(boot_half_lives)]
        min_finite_draws = max(20, int(0.05 * self.n_bootstrap))
        if len(finite_half_lives) >= min_finite_draws:
            half_life_ci = (
                float(np.percentile(finite_half_lives, 2.5)),
                float(np.percentile(finite_half_lives, 97.5)),
            )
        else:
            half_life_ci = (float("nan"), float("nan"))
            warnings.warn(
                f"Only {len(finite_half_lives)}/{self.n_bootstrap} bootstrap resamples produced a "
                "finite half-life; the reversion-speed CI is not meaningful. This itself signals "
                "the series may not be safely mean-reverting - see frac_bootstrap_non_reverting.",
                stacklevel=2,
            )

        return OUFitResult(
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            half_life_days=half_life_days,
            kappa_ci95=kappa_ci,
            half_life_ci95_days=half_life_ci,
            frac_bootstrap_non_reverting=frac_non_reverting,
            n_bootstrap=self.n_bootstrap,
        )

    def _bootstrap(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Case-resampling (pairs) bootstrap of the AR(1) regression:
        resample (x_{t-1}, x_t) pairs with replacement and refit
        kappa/half-life each time. This is fast, standard for regression
        confidence intervals, and adequate for the memo's baseline spec;
        prefer a block bootstrap in production if the residuals show
        strong autocorrelation the pairs bootstrap would understate.
        """
        x_lag = x[:-1]
        x_t = x[1:]
        n = len(x_t)

        boot_kappas = np.empty(self.n_bootstrap)
        boot_half_lives = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            idx = self.rng.integers(0, n, size=n)
            design = np.column_stack([np.ones(n), x_lag[idx]])
            beta, *_ = np.linalg.lstsq(design, x_t[idx], rcond=None)
            a_i, b_i = beta
            kappa_i, _, half_life_i = self._solve_ou_params(a_i, b_i, fallback_mean=float(x.mean()))
            boot_kappas[i] = kappa_i
            boot_half_lives[i] = half_life_i
        return boot_kappas, boot_half_lives


# ======================================================================
# Module 4: 3-State Probability-Weighted Scenario & Sensitivity Engine
# ======================================================================


@dataclass
class ScenarioAssumptions:
    """One state of the bear/base/bull tree (Memo Section 2.3)."""

    name: str
    probability: float
    china_revenue_growth: float  # e.g. -0.20 for a 20% decline in China revenue
    ev_margin_delta_bps: float  # basis points added/subtracted to blended EBITDA margin
    exit_ev_ebitda_p911: float
    exit_ev_ebitda_race: float


@dataclass
class CompanyFinancials:
    """Minimal balance-sheet/earnings inputs needed to bridge EBITDA to a share price."""

    name: str
    base_ebitda: float  # currency units consistent with net_debt (EUR millions here)
    base_ebitda_margin: float
    net_debt: float  # EUR millions - same unit as base_ebitda
    shares_outstanding: float  # MILLIONS of shares - same scale as base_ebitda/net_debt
    china_revenue_share: float  # fraction of revenue sourced from China


class ScenarioSensitivityEngine:
    """
    Implements Memo Section 2.3: a 3-state (bear/base/bull) probability-
    weighted scenario tree replacing the original report's single 50%
    stress test (Appx D-2) and single-point target-price bridge
    (Appx C-1), plus a 2-way EV/EBITDA-multiple x EBITDA-growth
    sensitivity grid ready for a tornado chart or heatmap.
    """

    def __init__(self, p911: CompanyFinancials, race: CompanyFinancials) -> None:
        self.p911 = p911
        self.race = race

    @staticmethod
    def _implied_price(ebitda: float, multiple: float, net_debt: float, shares: float) -> float:
        """
        Implied share price = (EBITDA * multiple - net debt) / shares
        outstanding. `ebitda`, `net_debt` and `shares` must all be on the
        SAME scale (this module uses EUR millions and millions of shares
        throughout) - mixing a raw share count with EBITDA in millions is
        exactly the unit-labelling error flagged as gap #5 in the memo,
        and silently collapses every implied price toward zero.
        """
        implied_ev = ebitda * multiple
        implied_equity = implied_ev - net_debt
        return implied_equity / shares

    @staticmethod
    def _scenario_ebitda(company: CompanyFinancials, scenario: ScenarioAssumptions) -> float:
        """
        Flex base EBITDA for (a) China revenue growth, applied only to
        the China-exposed revenue share, and (b) an EV-margin effect,
        applied as a basis-point shift to the blended EBITDA margin on
        total revenue.
        """
        implied_revenue = company.base_ebitda / company.base_ebitda_margin
        china_revenue = implied_revenue * company.china_revenue_share
        non_china_revenue = implied_revenue - china_revenue
        flexed_china_revenue = china_revenue * (1 + scenario.china_revenue_growth)
        flexed_revenue = non_china_revenue + flexed_china_revenue
        flexed_margin = company.base_ebitda_margin + scenario.ev_margin_delta_bps / 10_000
        return flexed_revenue * flexed_margin

    def run_scenarios(self, scenarios: Sequence[ScenarioAssumptions]) -> pd.DataFrame:
        """Run every scenario for both names; return implied prices, ratio, and probabilities."""
        rows = []
        for sc in scenarios:
            p911_ebitda = self._scenario_ebitda(self.p911, sc)
            race_ebitda = self._scenario_ebitda(self.race, sc)
            p911_price = self._implied_price(
                p911_ebitda, sc.exit_ev_ebitda_p911, self.p911.net_debt, self.p911.shares_outstanding
            )
            race_price = self._implied_price(
                race_ebitda, sc.exit_ev_ebitda_race, self.race.net_debt, self.race.shares_outstanding
            )
            rows.append(
                {
                    "scenario": sc.name,
                    "probability": sc.probability,
                    "p911_implied_ebitda": p911_ebitda,
                    "race_implied_ebitda": race_ebitda,
                    "p911_implied_price": p911_price,
                    "race_implied_price": race_price,
                    "implied_pr_ratio": p911_price / race_price,
                }
            )
        scenario_df = pd.DataFrame(rows)
        total_probability = scenario_df["probability"].sum()
        if not np.isclose(total_probability, 1.0, atol=1e-6):
            warnings.warn(f"Scenario probabilities sum to {total_probability:.4f}, not 1.0.", stacklevel=2)
        return scenario_df

    @staticmethod
    def expected_value(scenario_df: pd.DataFrame) -> pd.Series:
        """Probability-weighted expected target prices and ratio (Memo Section 2.3 output)."""
        weighted = scenario_df[["p911_implied_price", "race_implied_price", "implied_pr_ratio"]].multiply(
            scenario_df["probability"], axis=0
        )
        return weighted.sum().rename("expected_value")

    def sensitivity_grid(
        self,
        company: CompanyFinancials,
        multiple_range: Sequence[float],
        ebitda_growth_range: Sequence[float],
    ) -> pd.DataFrame:
        """
        2-way sensitivity of implied price to (exit EV/EBITDA multiple x
        EBITDA growth rate) for one company. Returns a DataFrame shaped
        (growth rows x multiple columns), ready to export directly into a
        heatmap or tornado chart.
        """
        grid = pd.DataFrame(
            index=pd.Index(ebitda_growth_range, name="ebitda_growth"),
            columns=pd.Index(multiple_range, name="exit_ev_ebitda"),
            dtype=float,
        )
        for growth in ebitda_growth_range:
            flexed_ebitda = company.base_ebitda * (1 + growth)
            for multiple in multiple_range:
                grid.loc[growth, multiple] = self._implied_price(
                    flexed_ebitda, multiple, company.net_debt, company.shares_outstanding
                )
        return grid


# ======================================================================
# Module 5: Risk Metrics & Historical Backtest
# ======================================================================


@dataclass
class BacktestSummary:
    """Forward hit rates and average returns following a Z-score signal."""

    n_episodes: int
    hit_rate_6m: float
    hit_rate_12m: float
    avg_return_6m: float
    avg_return_12m: float


class RiskBacktestEngine:
    """
    Memo Section 4.2 / 4.4: annualized tracking error, information ratio,
    and a historical backtest of what happened after the rolling Z-score
    breached the -1.5 to -2.0 "cheap Porsche" band - the "alternative
    framework" / "modelling cross-check" Altana's feedback asked for.
    """

    TRADING_DAYS_PER_YEAR = 252

    @staticmethod
    def tracking_error(p911_prices: pd.Series, race_prices: pd.Series) -> float:
        """Annualized TE = std(daily log-return differential) * sqrt(252)."""
        p911_returns = np.log(p911_prices).diff()
        race_returns = np.log(race_prices).diff()
        diff = (p911_returns - race_returns).dropna()
        return float(diff.std(ddof=1) * np.sqrt(RiskBacktestEngine.TRADING_DAYS_PER_YEAR))

    @staticmethod
    def information_ratio(expected_excess_return: float, tracking_error: float) -> float:
        """IR = E[excess return] / tracking error."""
        if tracking_error == 0:
            return float("nan")
        return expected_excess_return / tracking_error

    def backtest_zscore_signal(
        self,
        ratio: pd.Series,
        z_score: pd.Series,
        lower_threshold: float = -2.0,
        upper_threshold: float = -1.5,
        long_term_mean: Optional[float] = None,
    ) -> BacktestSummary:
        """
        Find every day the rolling Z-score first enters the
        [lower_threshold, upper_threshold] "cheap Porsche" band, then
        measure 6 and 12 months later: (a) whether the ratio closed at
        least half the gap back to the long-run mean (a "hit"), and
        (b) the realized return of the implied pair position, signed so
        that a move toward the mean is positive.
        """
        z = z_score.dropna()
        in_band = (z >= lower_threshold) & (z <= upper_threshold)
        entries = in_band & ~in_band.shift(1, fill_value=False)
        entry_dates = z.index[entries]

        if long_term_mean is None:
            long_term_mean = ratio.mean()

        hits_6m: list[bool] = []
        hits_12m: list[bool] = []
        returns_6m: list[float] = []
        returns_12m: list[float] = []

        for entry_date in entry_dates:
            level_at_entry = ratio.loc[entry_date]
            gap_to_mean = long_term_mean - level_at_entry
            if gap_to_mean == 0:
                continue  # already at the mean; no reversion to score

            for horizon_days, hit_bucket, return_bucket in (
                (126, hits_6m, returns_6m),  # ~6 trading months
                (252, hits_12m, returns_12m),  # ~12 trading months
            ):
                future_position = ratio.index.searchsorted(entry_date) + horizon_days
                if future_position >= len(ratio):
                    continue  # not enough forward data for this episode/horizon
                level_future = ratio.iloc[future_position]
                closed_gap = level_future - level_at_entry
                same_direction = np.sign(closed_gap) == np.sign(gap_to_mean)
                closed_half_the_gap = abs(closed_gap) >= 0.5 * abs(gap_to_mean)
                hit = same_direction and closed_half_the_gap
                pair_return = np.sign(gap_to_mean) * (level_future - level_at_entry) / level_at_entry
                hit_bucket.append(bool(hit))
                return_bucket.append(float(pair_return))

        def _safe_mean(values: list) -> float:
            return float(np.mean(values)) if values else float("nan")

        return BacktestSummary(
            n_episodes=len(entry_dates),
            hit_rate_6m=_safe_mean(hits_6m),
            hit_rate_12m=_safe_mean(hits_12m),
            avg_return_6m=_safe_mean(returns_6m),
            avg_return_12m=_safe_mean(returns_12m),
        )


# ======================================================================
# Demo assumptions and orchestration
# ======================================================================


def build_demo_financials() -> tuple[CompanyFinancials, CompanyFinancials]:
    """
    Illustrative current financials for the scenario engine, built from
    the memo's Section 3 research: Porsche's TTM revenue/margin (per
    Sept-2026 market data) and Ferrari's FY2026 guidance (per its Q2-2026
    release). All figures are in EUR millions throughout, to avoid the
    unit-labelling error flagged as gap #5 in Section 1 of the memo.
    Replace with a live data pull before using this for real position
    sizing.
    """
    p911 = CompanyFinancials(
        name="Porsche AG (P911)",
        base_ebitda=6_860.0,  # EUR millions; ~19.4% margin x ~EUR 35.3bn TTM revenue
        base_ebitda_margin=0.194,
        net_debt=4_100.0,  # EUR millions, illustrative
        shares_outstanding=911.0,  # MILLIONS of shares (911m shares outstanding)
        china_revenue_share=0.15,  # per the original report's Appx B-4
    )
    race = CompanyFinancials(
        name="Ferrari N.V. (RACE)",
        base_ebitda=2_970.0,  # EUR millions; FY2026 guidance: >=39% margin on ~EUR7.6bn revenue
        base_ebitda_margin=0.39,
        net_debt=2_000.0,  # EUR millions, illustrative
        shares_outstanding=180.0,  # MILLIONS of shares (~180m shares outstanding)
        china_revenue_share=0.09,  # per the original report's Appx B-4
    )
    return p911, race


def build_demo_scenarios() -> list[ScenarioAssumptions]:
    """3-state tree per Memo Section 2.3 (bear 25% / base 50% / bull 25%)."""
    return [
        ScenarioAssumptions(
            name="Bear",
            probability=0.25,
            china_revenue_growth=-0.20,
            ev_margin_delta_bps=-150,
            exit_ev_ebitda_p911=3.5,
            exit_ev_ebitda_race=22.0,
        ),
        ScenarioAssumptions(
            name="Base",
            probability=0.50,
            china_revenue_growth=-0.05,
            ev_margin_delta_bps=0,
            exit_ev_ebitda_p911=5.0,
            exit_ev_ebitda_race=27.0,
        ),
        ScenarioAssumptions(
            name="Bull",
            probability=0.25,
            china_revenue_growth=0.05,
            ev_margin_delta_bps=100,
            exit_ev_ebitda_p911=7.0,
            exit_ev_ebitda_race=32.0,
        ),
    ]


def print_interpretation_guide(
    comovement: CoMovementTestResult,
    ou_fit: OUFitResult,
    information_ratio: float,
) -> None:
    """
    Prints the decision thresholds that separate a statistical-arbitrage
    read from a structural value-trap read (Memo Section 4), then prints
    where this specific run landed against each one.
    """
    print("\n" + "=" * 88)
    print("OUTPUT INTERPRETATION GUIDE - STATISTICAL ARBITRAGE vs. STRUCTURAL VALUE TRAP")
    print("=" * 88)
    print(
        """
  1. STATIONARITY (ADF test, Module 1)
     - p-value < 0.05      -> reject the unit root -> the ratio IS mean-reverting.
     - p-value >= 0.05     -> cannot reject the unit root -> treat the ratio as a
                               trending / non-stationary series; a reversion
                               thesis is NOT statistically supported.

  2. STRUCTURAL BREAK (Chow test, Module 1)
     - p-value < 0.05 at the mid-2024 break -> the pooled since-IPO mean (0.25 in
       the original report) is not a valid reversion target; re-anchor to the
       POST-BREAK regime mean only (regime_split_zscore()).
     - p-value >= 0.05     -> no evidence of a regime shift; the full-sample mean
                               is defensible as used.

  3. HALF-LIFE (OU fit, Module 3)
     - t_half < ~180 trading days (~9 months)   -> reversion plausible within the
       report's 12-18 month horizon; a position can be sized with some confidence.
     - t_half ~180-500 trading days              -> reversion is slow relative to
       the horizon; lower conviction, smaller size.
     - t_half > ~500 trading days, OR kappa's 95% CI includes zero or is negative,
       OR frac_bootstrap_non_reverting exceeds ~20-30% -> no statistically
       reliable reversion speed. This is the quantitative signature of a
       VALUE TRAP, not a stat-arb opportunity.

  4. INFORMATION RATIO (Module 5)
     - IR > 0.5            -> risk-adjusted conviction supports a "high conviction" label.
     - IR 0.2 - 0.5        -> moderate conviction; size the position accordingly.
     - IR < 0.2            -> the expected alpha is not well compensated for spread
                               volatility; do not label this "high conviction."

  5. BACKTEST HIT RATE (Module 5)
     - 12-month hit rate > 60%, with a positive average pair return across
       multiple historical episodes -> the anomaly-and-reversion pattern has
       genuinely recurred; strengthens the stat-arb case.
     - Hit rate <= 50%, or too few episodes (n < 3) to be meaningful -> treat the
       current dislocation as unprecedented, not a repeat of a known pattern;
       lower conviction accordingly.

  BOTTOM LINE: only support a "mean-reversion trade" recommendation when (1) and
  (3) both point to reversion AND (4) clears a reasonable IR bar. If (1) fails,
  or (2) confirms a break with no re-based reversion evidence, or (3)'s
  half-life / CI is not economically meaningful, the correct call is the one
  the original report's own title worried about: a value trap, not a
  statistical arbitrage.
        """
    )
    print(
        f"THIS RUN -> ADF p={comovement.adf_pvalue:.3f} | "
        f"Chow p={comovement.chow_p_value:.3f} | "
        f"half-life={ou_fit.half_life_days:,.0f}d | "
        f"non-reverting bootstrap share={ou_fit.frac_bootstrap_non_reverting:.1%} | "
        f"IR={information_ratio:.2f}"
    )


def main() -> None:
    """End-to-end demo wiring all five modules together and printing a readable report."""
    print("=" * 88)
    print("PORSCHE (P911) vs FERRARI (RACE) - QUANT RE-ARCHITECTURE FRAMEWORK")
    print("=" * 88)

    # ---- Module 1: Data ingestion & co-movement tests ---------------------
    ingestion = DataIngestionEngine(start="2022-09-29")
    prices = ingestion.fetch_prices()
    log_ratio = ingestion.compute_log_ratio()
    comovement = ingestion.run_comovement_tests(break_date="2024-06-01")

    print("\n--- Module 1: Data Ingestion & Co-Movement Engine ---")
    print(f"Observations: {len(prices)} trading days, {prices.index[0].date()} to {prices.index[-1].date()}")
    print(
        f"ADF statistic: {comovement.adf_statistic:.4f} | p-value: {comovement.adf_pvalue:.4f} "
        f"| Stationary (mean-reverting)? {comovement.adf_is_stationary}"
    )
    print(
        f"Chow F-statistic: {comovement.chow_f_statistic:.4f} | p-value: {comovement.chow_p_value:.4f} "
        f"| Structural break at {comovement.break_date.date()}? {comovement.chow_break_significant}"
    )

    # ---- Module 2: Rolling Z-score & anomaly detector ----------------------
    detector = RollingZScoreDetector(window=252, z_threshold=2.0, min_persistence_days=20)
    z_table = detector.rolling_zscore(log_ratio)
    regime_table = detector.regime_split_zscore(log_ratio, break_date="2024-06-01")
    anomalies = detector.flag_anomalies(z_table["z_score"])

    print("\n--- Module 2: Rolling Z-Score & Anomaly Detector ---")
    print(f"Current pooled 252d Z-score: {z_table['z_score'].iloc[-1]:.2f}")
    print(
        f"Current regime-adjusted Z-score: {regime_table['regime_z_score'].iloc[-1]:.2f} "
        f"(regime: {regime_table['regime'].iloc[-1]})"
    )
    print(f"Anomaly episodes flagged (|Z| >= 2, persisting >= 20 trading days): {len(anomalies)}")
    for episode in anomalies[-5:]:
        print(
            f"  {episode.start.date()} -> {episode.end.date()} | {episode.duration_days}d "
            f"| {episode.direction} | mean Z = {episode.mean_z:.2f}"
        )

    # ---- Module 3: OU mean-reversion & bootstrap ---------------------------
    ou_engine = OUMeanReversionEngine(n_bootstrap=1000)
    ou_fit = ou_engine.fit(log_ratio)

    print("\n--- Module 3: Ornstein-Uhlenbeck Mean-Reversion & Bootstrap ---")
    kappa_ci_rounded = tuple(round(v, 4) for v in ou_fit.kappa_ci95)
    print(f"kappa (speed of reversion): {ou_fit.kappa:.4f} /yr | 95% CI: {kappa_ci_rounded}")
    print(
        f"theta (long-run equilibrium, log-ratio): {ou_fit.theta:.4f} "
        f"(ratio level: {np.exp(ou_fit.theta):.4f})"
    )
    print(
        f"Half-life: {ou_fit.half_life_days:,.0f} trading days (~{ou_fit.half_life_days / 21:.1f} months) "
        f"| 95% CI: {tuple(round(v, 0) for v in ou_fit.half_life_ci95_days)} trading days"
    )
    print(
        f"Bootstrap resamples implying NO reversion (kappa <= 0): "
        f"{ou_fit.frac_bootstrap_non_reverting:.1%} of {ou_fit.n_bootstrap}"
    )
    if ou_fit.kappa_ci95[0] <= 0 <= ou_fit.kappa_ci95[1] or ou_fit.frac_bootstrap_non_reverting > 0.20:
        print(
            "CAVEAT: kappa is not reliably distinguished from zero (or non-reverting bootstrap "
            "share > 20%) - treat theta and the half-life point estimate above as unstable, not "
            "as a trading level. See Section 3 of the interpretation guide below."
        )

    # ---- Module 4: Scenario & sensitivity engine ---------------------------
    p911_financials, race_financials = build_demo_financials()
    scenario_engine = ScenarioSensitivityEngine(p911_financials, race_financials)
    scenario_df = scenario_engine.run_scenarios(build_demo_scenarios())
    expected = scenario_engine.expected_value(scenario_df)
    sensitivity = scenario_engine.sensitivity_grid(
        p911_financials,
        multiple_range=[3.5, 4.5, 5.0, 5.5, 6.5, 7.0],
        ebitda_growth_range=[-0.15, -0.05, 0.0, 0.05, 0.15],
    )

    print("\n--- Module 4: Scenario & Sensitivity Engine ---")
    print(scenario_df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\nProbability-weighted expected values:")
    print(expected.to_string(float_format=lambda v: f"{v:,.3f}"))
    print("\nPorsche sensitivity grid (rows = EBITDA growth, cols = exit EV/EBITDA), implied price EUR:")
    print(sensitivity.to_string(float_format=lambda v: f"{v:,.1f}"))

    # ---- Module 5: Risk metrics & backtest ---------------------------------
    risk_engine = RiskBacktestEngine()
    tracking_error = risk_engine.tracking_error(prices["P911"], prices["RACE"])
    illustrative_expected_alpha = 0.384  # (0.18 / 0.13) - 1, per the original report's Appx A-2
    information_ratio = risk_engine.information_ratio(illustrative_expected_alpha, tracking_error)
    backtest = risk_engine.backtest_zscore_signal(np.exp(log_ratio), z_table["z_score"])

    print("\n--- Module 5: Risk Metrics & Historical Backtest ---")
    print(f"Annualized tracking error: {tracking_error:.2%}")
    print(f"Information ratio (on the memo's illustrative 38.4% target alpha): {information_ratio:.2f}")
    print(f"Backtest episodes (Z entering the -2.0 to -1.5 band): {backtest.n_episodes}")
    print(f"  6-month hit rate:  {backtest.hit_rate_6m:.1%} | avg return: {backtest.avg_return_6m:.2%}")
    print(f"  12-month hit rate: {backtest.hit_rate_12m:.1%} | avg return: {backtest.avg_return_12m:.2%}")

    print_interpretation_guide(comovement, ou_fit, information_ratio)


if __name__ == "__main__":
    main()
