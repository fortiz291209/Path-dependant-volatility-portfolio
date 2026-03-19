# PDV Options — Volatility Trading Without Options Data

Implementation of **Guyon & Lekeufack (2022)**: *"Volatility is (Mostly) Path-Dependent"*.

## Key Insight

~90% of VIX variation is explained by SPX price history alone using a
path-dependent volatility model. The remaining ~10% (exogenous residual)
is a tradeable mean-reverting signal. **No options data subscription needed.**

## Quick Start

```bash
pip install -r requirements.txt
python run_all.py
```

## Data Sources (all free)

| Source | Install | Rate Limits |
|--------|---------|-------------|
| **FRED** (primary) | `pip install pandas-datareader` | None |
| yfinance (fallback) | `pip install yfinance` | Occasional blocks |
| Manual CSV | Download from Yahoo Finance | N/A |

## Architecture

```
data_fetcher.py   → FRED/yfinance data loader with caching
pdv_model.py      → Core PDV model (TSPL + 4-factor Markov)
option_pricer.py  → Black-Scholes + Monte Carlo + synthetic chains
trading_signals.py→ Residual z-score, VRP, regime detection
backtest.py       → Strategy backtester with performance metrics
run_all.py        → Full pipeline orchestrator
```

## Model

```
σ_t = β₀ + β₁·R₁(t) + β₂·Σ(t)

R₁ = Σ K₁(τᵢ)·rᵢ        trend feature (leverage effect)
Σ  = √(Σ K₂(τᵢ)·rᵢ²)    activity feature (vol clustering)
K(τ) = Z⁻¹·(τ+δ)^(-α)    time-shifted power law kernel
```

## Trading Strategies

1. **VIX Mean-Reversion**: Trade when actual VIX deviates from PDV prediction
2. **VRP Harvesting**: Sell vol when predicted IV >> predicted RV
3. **Combined**: Diversified blend of both signals

## Output

The pipeline generates 6 charts + 2 CSV files in `./output/`:

- `01_vix_prediction.png` — VIX vs PDV model prediction
- `02_scatter_plots.png` — Feature relationship visualization
- `03_synthetic_smile.png` — Model-implied vol smile surface
- `04_mc_smile.png` — Monte Carlo vs analytical smile
- `05_signals_backtest.png` — Signal dashboard + equity curves
- `06_residual_analysis.png` — Residual distribution + predictive power
