# Porsche-Ferrari-Report
# Porsche (P911) vs. Ferrari (RACE): A Quantitative Relative-Value Analysis

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Reproducibility](https://img.shields.io/badge/Reproducibility-Locked%20Snapshot-brightgreen)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-lightgrey)

**An institutional-grade econometric and fundamental test battery that stress-tests — and rejects — a naive mean-reversion pairs trade between Porsche AG and Ferrari N.V.**

📄 **[Read the full 11-page research report →](./Porsche_Ferrari_Flagship_Report.pdf)**
📋 **[One-page proof-of-work tear sheet →](./Gayathri_Vinod_Executive_Tearsheet.pdf)**

---

## Objective

Porsche AG (ETR: P911) and Ferrari N.V. (BIT/NYSE: RACE) traded within a stable price-ratio band for roughly two years after Porsche's 2022 IPO — the kind of pattern that invites a straightforward relative-value pairs trade: buy the "cheap" name, sell the "expensive" one, and wait for the ratio to revert.

This project asks a sharper question: **is that reversion thesis actually supported by the data, or is it a statistical artifact of a regime that has already ended?**

The answer, built from four independent statistical tests plus a bottom-up fundamental cross-check, is that the relationship structurally broke in mid-2024 — this is a **value trap**, not a statistical arbitrage opportunity.

## Key Findings

| Test | Result | Verdict |
|---|---|---|
| Stationarity (Augmented Dickey-Fuller) | p = 0.6717 | Fails to reject a random walk — not mean-reverting |
| Structural Break (Chow Test) | F = 1,081.01, p < 0.0001 (break: June 2024) | Confirmed — the pre-2024 anchor is statistically invalid |
| Reversion Speed (Ornstein-Uhlenbeck) | κ = 0.40/yr; half-life ≈ 435 days; 95% CI spans zero | Point estimate plausible, not statistically resolved |
| Historical Backtest (29 episodes) | 13.8% 12-month hit rate | Worse than a coin flip — the pattern hasn't recurred |
| Fundamental Cross-Check (EV/EBITDA scenario engine) | Model ratio 0.076 vs. market ≈0.125 | Corroborates the statistical read, independently |

Full derivations, live-data terminal output, and a line-by-line interview-defense walkthrough of every statistic above are in the [full report](./Porsche_Ferrari_Flagship_Report.pdf).

## Repository Structure

```
.
├── porsche_ferrari_quant_analysis.py   # Main analysis script — 5 flat functions, no classes
├── data/
│   └── p911_race_prices_snapshot.csv   # Frozen price snapshot (see Reproducibility below)
├── requirements.txt
└── README.md
```

## Tech Stack

- **Python 3.11+**
- **pandas / numpy** — data handling and vectorized computation
- **statsmodels** — Augmented Dickey-Fuller test
- **scipy.stats** — F-distribution critical values for the Chow test
- **yfinance** — historical daily price data (Yahoo Finance)

No proprietary data or paid data licenses — everything here runs on free, publicly available market data.

## Methodology

Five steps, run top to bottom. No classes, no custom logging, no abstract wrappers — plain functions with inline comments explaining every statistical step, written to be read and defended line by line:

1. **`fetch_price_data()`** — pulls daily closes for P911.DE and RACE.MI, builds the log price ratio
2. **`run_statistical_tests()`** — Augmented Dickey-Fuller (stationarity) + a hand-built Chow test (structural break)
3. **`calculate_half_life()`** — Ornstein-Uhlenbeck speed of reversion via AR(1) OLS, with a 1,000-draw bootstrap confidence interval
4. **`run_scenarios()`** — bear/base/bull EV/EBITDA valuation bridge, plus a 2-way sensitivity grid
5. **`backtest_spread()`** — historical backtest of the Z-score entry signal, plus tracking error / information ratio

## How to Run

**1. Install dependencies**

```bash
git clone https://github.com/gayathritvinod/Porsche-Ferrari-Report.git
cd Porsche-Ferrari-Report
pip install -r requirements.txt
```

**2. Run it**

```bash
python porsche_ferrari_quant_analysis.py
```

That's it. By default, the script runs in **reproducible mode** (see below) and prints the full five-step analysis to your terminal — no API key, no network access, and no waiting on Yahoo Finance required.

## Reproducibility

Live market data isn't static: data vendors revise historical adjusted closing prices over time (dividend and split adjustments get recalculated retroactively), so re-pulling "the same" date range from Yahoo Finance on two different days is not guaranteed to return identical numbers. To make this project's findings genuinely reproducible — not just "close enough" — the script runs in one of two explicit modes, set by a single constant at the top of the file:

```python
ANALYSIS_MODE = "reproducible"   # or "live"
```

| Mode | Behavior |
|---|---|
| `"reproducible"` (default) | Loads the frozen CSV snapshot in `data/`. No network call at all — returns bit-for-bit identical results on any machine, in any year. **This is what the linked report's numbers are locked to.** |
| `"live"` | Pulls a fresh window from Yahoo Finance, up to and including `AS_OF_DATE`. Useful for checking how the thesis holds up against newer data — but the numbers will legitimately differ from the report above, and that's expected. |

To rebuild the snapshot (e.g. after changing `AS_OF_DATE` or `START_DATE`):

```python
from porsche_ferrari_quant_analysis import fetch_price_data, AS_OF_DATE, DATA_SNAPSHOT_PATH
df = fetch_price_data(mode="live", end_date=AS_OF_DATE)
df[["P911", "RACE"]].to_csv(DATA_SNAPSHOT_PATH)
```

## Disclaimer

This is a self-directed portfolio research project prepared for educational and job-application purposes. It is not investment research produced by a regulated broker-dealer, is not investment advice, and should not be relied upon for any investment decision. All company financial data is sourced from public disclosures; scenario assumptions in the valuation engine are explicitly illustrative (see the full report's Methodological Appendix).

## Author

**Gayathri Vinod** — [LinkedIn](#) · [gayathritvinod@gmail.com](mailto:gayathritvinod@gmail.com)