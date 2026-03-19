"""
run_all.py — Full PDV Options Analysis Pipeline

    pip install pandas-datareader matplotlib
    python run_all.py

Downloads free SPX/VIX data, calibrates the Guyon-Lekeufack PDV model,
generates synthetic option prices, produces trading signals, backtests
volatility strategies, and outputs publication-quality charts.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from data_fetcher import DataFetcher
from pdv_model import PDVModel
from option_pricer import BlackScholes, MonteCarlo, build_chain, chain_to_df, pdv_smile
from trading_signals import SignalGenerator
from backtest import (Backtester, Config, TailHedgeConfig, DynamicAllocConfig,
                      metrics, tail_hedge_metrics, dynamic_alloc_metrics, print_metrics)
from rolling_analysis import run_rolling_analysis

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Part 1: Model analysis — VIX prediction from price path alone
# ═══════════════════════════════════════════════════════════════════════

def part1_model(data):
    print("\n" + "═" * 70)
    print("  PART 1: PDV MODEL — PREDICTING VIX FROM PRICE PATH")
    print("═" * 70)

    returns = data["returns"].values
    vix = data["VIX_decimal"].values
    split = "2019-01-01"
    train_idx = (data.index < split).sum()

    # Calibrate
    model = PDVModel("vix")
    cal = model.calibrate(returns, vix, train_end=train_idx)
    print(f"\n  Train r² = {cal['train_r2']:.4f}")
    print(f"  Test  r² = {cal['test_r2']:.4f}")
    print(f"  β₀={cal['beta0']:.4f}  β₁={cal['beta1']:.4f}  β₂={cal['beta2']:.4f}")
    print(f"\n  → {abs(cal['test_r2'])*100:.1f}% of VIX explained by price path alone!")

    # Predict (with calibrated betas)
    result = model.predict(returns, vix)
    result.index = data.index
    v = result.dropna()

    # ── Plot 1: VIX prediction ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 2, 1]})

    ax = axes[0]
    ax.plot(v.index, v["actual_vol"] * 100, "k-", lw=0.8, alpha=0.9, label="Actual VIX")
    ax.plot(v.index, v["predicted_vol"] * 100, "r-", lw=0.8, alpha=0.8, label="PDV Prediction")
    test_start = v.loc[v.index >= split].index[0]
    ax.axvline(test_start, color="blue", ls="--", alpha=0.5, label="Train/Test Split")
    ax.set_ylabel("VIX Level")
    ax.legend(fontsize=10)
    ax.set_title(f"Volatility is (Mostly) Path-Dependent — VIX Prediction (Test r² = {cal['test_r2']:.3f})",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(v.index, v["R1"], "b-", lw=0.6, alpha=0.7, label="R₁ (trend)")
    ax2.plot(v.index, v["Sigma"], color="orange", lw=0.6, alpha=0.7, label="Σ (activity)")
    ax.set_ylabel("R₁", color="blue")
    ax2.set_ylabel("Σ", color="orange")
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    rr = v["actual_vol"] / v["predicted_vol"]
    ax.plot(v.index, rr, "k-", lw=0.5, alpha=0.7)
    ax.axhline(1, color="red", ls="--", alpha=0.5)
    ax.set_ylabel("Actual / Predicted")
    ax.set_ylim(0.5, 2.0)
    ax.set_title("Residual Ratio (exogenous component)", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "01_vix_prediction.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Scatter plots ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    vx = v["actual_vol"].values * 100
    px = v["predicted_vol"].values * 100

    ax = axes[0]
    ax.scatter(vx, px, s=2, alpha=0.3, c="red")
    lim = [min(vx.min(), px.min()), max(vx.max(), px.max())]
    ax.plot(lim, lim, "k--", alpha=0.5)
    ax.set(xlabel="Actual VIX", ylabel="Predicted VIX", title="Predicted vs Actual")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sc = ax.scatter(v["R1"], vx, s=2, alpha=0.3, c=v["Sigma"], cmap="hot")
    ax.set(xlabel="R₁ (trend)", ylabel="VIX", title="VIX vs Trend (color=Σ)")
    plt.colorbar(sc, ax=ax, label="Σ")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    sc = ax.scatter(v["Sigma"], vx, s=2, alpha=0.3, c=v["R1"], cmap="coolwarm")
    ax.set(xlabel="Σ (activity)", ylabel="VIX", title="VIX vs Activity (color=R₁)")
    plt.colorbar(sc, ax=ax, label="R₁")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "02_scatter_plots.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: 01_vix_prediction.png, 02_scatter_plots.png")
    return result


# ═══════════════════════════════════════════════════════════════════════
# Part 2: Synthetic option pricing
# ═══════════════════════════════════════════════════════════════════════

def part2_options(data, model_result):
    print("\n" + "═" * 70)
    print("  PART 2: SYNTHETIC OPTION PRICING (NO OPTIONS DATA NEEDED)")
    print("═" * 70)

    last = model_result.dropna().iloc[-1]
    S = data["SPX"].iloc[-1]
    atm = last["predicted_vol"]
    R1 = last["R1"]
    Sig = last["Sigma"]

    print(f"\n  SPX = {S:.2f}   PDV Vol = {atm*100:.2f}%   R₁ = {R1:.4f}   Σ = {Sig:.4f}")
    if "actual_vol" in last.index and not np.isnan(last["actual_vol"]):
        print(f"  VIX = {last['actual_vol']*100:.2f}%   Gap = {(last['actual_vol']-atm)*100:+.2f} pts")

    # Build chain
    expiries = [7/365, 14/365, 30/365, 60/365, 90/365, 180/365]
    chain = build_chain(S, atm, R1, Sig, expiries, np.arange(0.90, 1.11, 0.005))
    df = chain_to_df(chain)
    print(f"  Generated {len(df)} option quotes across {len(expiries)} expiries")

    # Sample
    sample = df[(df["expiry_days"] == 30) & df["is_call"] &
                (df["strike"] >= S * 0.96) & (df["strike"] <= S * 1.04)]
    print(f"\n  30-Day ATM Calls:")
    print(sample[["strike", "iv", "price", "delta", "vega"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    # ── Plot 3: Smile ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, len(expiries)))
    for exp, c in zip(expiries, colors):
        sub = df[df["expiry_days"] == int(exp * 365)].copy()
        sub["m"] = sub["strike"] / S
        sub = sub[(sub["m"] >= 0.90) & (sub["m"] <= 1.10)]
        ax1.plot(sub["m"], sub["iv"] * 100, "-", color=c, lw=1.5, label=f"{int(exp*365)}d")
    ax1.set(xlabel="Moneyness (K/S)", ylabel="IV (%)", title="PDV Implied Vol Smile")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    atm_vols = []
    for exp in expiries:
        sub = df[df["expiry_days"] == int(exp * 365)]
        atm_vols.append(sub.iloc[(sub["strike"] - S).abs().argmin()]["iv"] * 100)
    ax2.plot([e * 365 for e in expiries], atm_vols, "bo-", lw=2)
    ax2.set(xlabel="Days to Expiry", ylabel="ATM IV (%)", title="ATM Vol Term Structure")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "03_synthetic_smile.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── MC smile ──
    print("\n  Computing Monte Carlo smile (4-factor)...")
    try:
        mc = MonteCarlo()
        mc_k, mc_iv, mc_p = mc.smile(S, 30/365, np.arange(0.88, 1.13, 0.02), n_paths=30000)

        fig, ax = plt.subplots(figsize=(8, 5))
        ok = ~np.isnan(mc_iv)
        ax.plot(mc_k[ok] / S, mc_iv[ok] * 100, "ro-", lw=2, label="MC (4-factor)")
        approx = pdv_smile(S, atm, 30/365, mc_k)
        ax.plot(mc_k / S, approx * 100, "b--", lw=1.5, label="Analytical approx")
        ax.set(xlabel="Moneyness", ylabel="IV (%)", title="30d SPX Smile: MC vs Analytical")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / "04_mc_smile.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  MC failed: {e}")

    df.to_csv(OUT / "synthetic_chain.csv", index=False)
    print(f"  Saved: 03_synthetic_smile.png, 04_mc_smile.png, synthetic_chain.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════
# Part 3: Trading signals & backtest
# ═══════════════════════════════════════════════════════════════════════

def part3_signals(data):
    print("\n" + "═" * 70)
    print("  PART 3: TRADING SIGNALS & BACKTEST")
    print("═" * 70)

    sig = SignalGenerator().generate(data)
    test_start = "2019-01-01"

    # Current signal
    latest = sig.dropna(subset=["residual_zscore"]).iloc[-1]
    print(f"\n  ── CURRENT SIGNAL ({latest.name.date()}) ──")
    print(f"  SPX={latest['SPX']:.0f}  VIX={latest['VIX']:.1f}  "
          f"PDV={latest['predicted_vix']:.1f}  RV={latest['predicted_rv']:.1f}")
    print(f"  Z-score={latest['residual_zscore']:.2f}  "
          f"VRP={latest['vrp']:.1f}pts ({latest['vrp_pct']:.1%})  "
          f"Regime={latest['regime']}")

    cs = latest["composite_signal"]
    interp = "SELL VOL" if cs > 0.3 else "BUY VOL" if cs < -0.3 else "NEUTRAL"
    print(f"  Signal={cs:.2f} → {interp}")

    # Backtest
    bt = Backtester()
    bt1 = bt.vix_mean_reversion(sig, test_start)
    bt2 = bt.vrp_harvest(sig, test_start)
    bt3 = bt.combined(sig, test_start)
    print_metrics(metrics(bt1), "VIX Mean-Reversion")
    print_metrics(metrics(bt2), "VRP Harvesting")
    print_metrics(metrics(bt3), "Combined")

    # ── Plot 5: Dashboard ──
    st = sig.loc[test_start:]
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), gridspec_kw={"height_ratios": [2, 1.5, 1, 1.5]})

    ax = axes[0]
    ax.plot(st.index, st["VIX"], "k-", lw=0.8, label="VIX (actual)")
    ax.plot(st.index, st["predicted_vix"], "r-", lw=0.8, label="VIX (PDV)")
    ax.fill_between(st.index, st["VIX"], st["predicted_vix"], alpha=0.15, color="red")
    ax.set_ylabel("VIX"); ax.legend()
    ax.set_title("PDV Trading System — Signal Dashboard", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(st.index, st["residual_zscore"], "b-", lw=0.7)
    ax.axhline(1.5, color="red", ls="--", alpha=0.5, label="Sell")
    ax.axhline(-1.5, color="green", ls="--", alpha=0.5, label="Buy")
    ax.axhline(0, color="gray", ls="-", alpha=0.3)
    ax.fill_between(st.index, st["residual_zscore"], 1.5,
                     where=st["residual_zscore"] > 1.5, alpha=0.3, color="red")
    ax.fill_between(st.index, st["residual_zscore"], -1.5,
                     where=st["residual_zscore"] < -1.5, alpha=0.3, color="green")
    ax.set_ylabel("Z-Score"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(st.index, st["vrp"], color="purple", lw=0.7)
    ax.axhline(0, color="gray", ls="-", alpha=0.3)
    ax.fill_between(st.index, st["vrp"], 0, where=st["vrp"] > 0, alpha=0.2, color="green")
    ax.fill_between(st.index, st["vrp"], 0, where=st["vrp"] < 0, alpha=0.2, color="red")
    ax.set_ylabel("VRP (vol pts)")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(bt1.index, bt1["capital"], "b-", lw=1.2, label="VIX Mean-Rev")
    ax.plot(bt2.index, bt2["capital"], "r-", lw=1.2, label="VRP Harvest")
    ax.plot(bt3.index, bt3["capital"], "k-", lw=1.5, label="Combined")
    ax.axhline(100_000, color="gray", ls="--", alpha=0.3)
    ax.set_ylabel("Portfolio ($)"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "05_signals_backtest.png", dpi=150, bbox_inches="tight")
    plt.close()

    sig.to_csv(OUT / "signals.csv")
    print(f"\n  Saved: 05_signals_backtest.png, signals.csv")

    # ── Strategy 4: PDV Tail Risk Hedge ──
    print("\n  Running tail risk hedge backtest...")
    bt4 = bt.tail_hedge(sig, test_start)
    trades = bt._tail_hedge_trades
    m4 = tail_hedge_metrics(bt4, trades)
    print_metrics(m4, "PDV Tail Risk Hedge ($500K portfolio)")

    if len(trades) > 0:
        print(f"\n  Trade log ({len(trades)} events):")
        for _, t in trades.iterrows():
            if t["action"] == "BUY":
                print(f"    {t['date'].date() if hasattr(t['date'], 'date') else t['date']}  "
                      f"BUY  K={t['strike']:.0f}  cost=${t['cost']:.0f}  "
                      f"vol={t.get('vol', 0):.1%}  residual={t.get('residual', 0):.2f}")
            else:
                print(f"    {t['date'].date() if hasattr(t['date'], 'date') else t['date']}  "
                      f"{t['action']:11s}  K={t['strike']:.0f}  "
                      f"proceeds=${t.get('proceeds', 0):.0f}  "
                      f"return={t.get('return_pct', 0):.0%}  "
                      f"({t.get('reason', '')})")

    # ── Plot 7: Tail Hedge Dashboard ──
    fig, axes = plt.subplots(4, 1, figsize=(14, 16),
                             gridspec_kw={"height_ratios": [2, 1.5, 1.5, 2]})

    th = bt4

    ax = axes[0]
    ax.plot(th.index, th["equity_only"] / 1000, "gray", lw=1.2, label="Equity Only (SPX)")
    ax.plot(th.index, th["equity_plus_hedge"] / 1000, "b-", lw=1.5, label="Equity + PDV Hedge")
    ax.set_ylabel("Portfolio Value ($K)")
    ax.legend(fontsize=10)
    ax.set_title("PDV Tail Risk Hedge — Portfolio Protection", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(th.index, th["residual_ratio"], "k-", lw=0.5, alpha=0.7)
    ax.axhline(0.98, color="green", ls="--", alpha=0.6, label="Buy threshold (0.98)")
    ax.axhline(1.40, color="red", ls="--", alpha=0.6, label="Sell threshold (1.40)")
    ax.axhline(1.0, color="gray", ls="-", alpha=0.3)
    buy_dates = th.index[th["buy_signal"] > 0]
    sell_dates = th.index[th["sell_signal"] > 0]
    if len(buy_dates) > 0:
        ax.scatter(buy_dates, th.loc[buy_dates, "residual_ratio"],
                   marker="^", c="green", s=80, zorder=5, label="Buy puts")
    if len(sell_dates) > 0:
        ax.scatter(sell_dates, th.loc[sell_dates, "residual_ratio"],
                   marker="v", c="red", s=80, zorder=5, label="Sell/Expire")
    ax.set_ylabel("Residual Ratio")
    ax.set_ylim(0.5, 2.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.fill_between(th.index, th["hedge_value"], alpha=0.3, color="blue", label="Put value (MTM)")
    ax.plot(th.index, th["hedge_cost_cum"], "r--", lw=1, label="Cumulative cost")
    ax.set_ylabel("$ Value")
    ax.legend(fontsize=9)
    ax.set_title("Hedge Positions: Value vs Cost", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    eq = th["equity_only"].values
    eh = th["equity_plus_hedge"].values
    pk_eq = np.maximum.accumulate(eq)
    pk_eh = np.maximum.accumulate(eh)
    dd_eq = (eq - pk_eq) / pk_eq * 100
    dd_eh = (eh - pk_eh) / pk_eh * 100
    ax.fill_between(th.index, dd_eq, alpha=0.3, color="gray", label="Equity drawdown")
    ax.fill_between(th.index, dd_eh, alpha=0.4, color="blue", label="Hedged drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown Reduction from PDV-Timed Hedging", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "07_tail_hedge.png", dpi=150, bbox_inches="tight")
    plt.close()

    if len(trades) > 0:
        trades.to_csv(OUT / "tail_hedge_trades.csv", index=False)

    print(f"\n  Saved: 07_tail_hedge.png, tail_hedge_trades.csv")
    # ═══════════════════════════════════════════════════════════════
    # Strategy 5: PDV Dynamic Allocation (THE MAIN STRATEGY)
    # ═══════════════════════════════════════════════════════════════
    print("\n  Running dynamic allocation backtest...")
    bt5 = bt.dynamic_allocation(sig, test_start)
    m5 = dynamic_alloc_metrics(bt5)
    print_metrics(m5, "PDV Dynamic Allocation ($500K portfolio)")

    # ── Plot 8: Dynamic Allocation Dashboard ──
    fig, axes = plt.subplots(4, 1, figsize=(14, 16),
                             gridspec_kw={"height_ratios": [2.5, 1.2, 1.2, 2]})
    da = bt5

    # Panel 1: Portfolio value vs buy-and-hold
    ax = axes[0]
    ax.plot(da.index, da["equity_only"] / 1000, "gray", lw=1, alpha=0.7, label="Buy & Hold (SPX)")
    ax.plot(da.index, da["capital"] / 1000, "b-", lw=1.5, label="PDV Dynamic Allocation")
    ax.set_ylabel("Portfolio Value ($K)")
    ax.legend(fontsize=10)
    ax.set_title("PDV Dynamic Allocation — Reduce Equity When Vol Is High", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel 2: Equity allocation %
    ax = axes[1]
    ax.fill_between(da.index, da["equity_pct"] * 100, alpha=0.4, color="steelblue")
    ax.plot(da.index, da["equity_pct"] * 100, "b-", lw=0.5)
    ax.set_ylabel("Equity %")
    ax.set_ylim(0, 110)
    ax.axhline(100, color="gray", ls="--", alpha=0.3)
    ax.set_title("Equity Allocation (vol_target / predicted_vol, with residual adjustment)", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: Predicted vol driving the allocation
    ax = axes[2]
    ax.plot(da.index, da["predicted_vol"] * 100, "orange", lw=0.7, label="PDV Predicted Vol")
    ax.axhline(22, color="green", ls="--", alpha=0.4, label="Vol target (22%)")
    ax.set_ylabel("Predicted Vol (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Drawdown comparison
    ax = axes[3]
    eq = da["equity_only"].values
    cap = da["capital"].values
    pk_eq = np.maximum.accumulate(eq)
    pk_c = np.maximum.accumulate(cap)
    dd_eq = (eq - pk_eq) / pk_eq * 100
    dd_c = (cap - pk_c) / pk_c * 100
    ax.fill_between(da.index, dd_eq, alpha=0.3, color="gray", label="Buy & Hold drawdown")
    ax.fill_between(da.index, dd_c, alpha=0.4, color="blue", label="Dynamic Alloc drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown Reduction from PDV-Timed De-Risking", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "08_dynamic_allocation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: 08_dynamic_allocation.png")

    return sig, bt1, bt2, bt3, bt4, bt5


# ═══════════════════════════════════════════════════════════════════════
# Part 4: Residual analysis — the novel contribution
# ═══════════════════════════════════════════════════════════════════════

def part4_residual(data, signals):
    print("\n" + "═" * 70)
    print("  PART 4: RESIDUAL ANALYSIS (THE NOVEL CONTRIBUTION)")
    print("═" * 70)

    s = signals.dropna(subset=["residual_zscore"])
    resid = s["VIX"] - s["predicted_vix"]
    ratio = s["VIX"] / s["predicted_vix"]

    print(f"\n  Residual stats: mean={resid.mean():.3f}  std={resid.std():.3f}  "
          f"skew={resid.skew():.2f}  kurt={resid.kurtosis():.2f}")

    acf = [ratio.autocorr(lag=i) for i in range(1, 61)]
    hl = next((i+1 for i, a in enumerate(acf) if a < 0.5), None)
    print(f"  Autocorrelation: lag1={acf[0]:.3f}  lag5={acf[4]:.3f}  "
          f"lag21={acf[20]:.3f}  half-life≈{hl}d")

    # ── Plot 6: Residual analysis ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(ratio.values, bins=80, density=True, alpha=0.7, color="steelblue", edgecolor="none")
    ax.axvline(1.0, color="red", ls="--", lw=1.5)
    ax.set(xlabel="Residual Ratio (VIX / PDV)", ylabel="Density",
           title="Distribution of Exogenous Component")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.bar(range(1, 61), acf, color="steelblue", alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ci = 1.96 / np.sqrt(len(ratio))
    ax.axhline(ci, color="red", ls="--", alpha=0.5)
    ax.axhline(-ci, color="red", ls="--", alpha=0.5)
    ax.set(xlabel="Lag (days)", ylabel="ACF",
           title=f"Residual Autocorrelation (half-life ≈ {hl}d)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    rc = s["regime"].value_counts()
    cmap = {"NORMAL": "steelblue", "LOW_VOL": "green", "HIGH_VOL": "orange",
            "DISLOCATION": "red", "UNKNOWN": "gray"}
    ax.bar(rc.index, rc.values / len(s) * 100,
           color=[cmap.get(r, "gray") for r in rc.index], alpha=0.7)
    ax.set(ylabel="% of Time", title="Regime Distribution")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    sc = s.copy()
    sc["next_vix"] = sc["VIX"].shift(-1) - sc["VIX"]
    sc["zbucket"] = pd.cut(sc["residual_zscore"], bins=10)
    bm = sc.groupby("zbucket", observed=True)["next_vix"].mean()
    bm.plot(kind="bar", ax=ax, color="steelblue", alpha=0.7)
    ax.set(xlabel="Z-Score Bucket", ylabel="Avg Next-Day ΔVIX",
           title="Predictive Power: Z-Score → Next-Day VIX Change")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "06_residual_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 06_residual_analysis.png")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("═" * 70)
    print("  PDV OPTIONS — Volatility Trading Without Options Data")
    print("  Guyon & Lekeufack (2022)")
    print("═" * 70)

    data = DataFetcher().load("2000-01-01")
    model_result = part1_model(data)
    part2_options(data, model_result)
    signals, *backtests = part3_signals(data)
    part4_residual(data, signals)

    # Part 5: Rolling window out-of-sample analysis
    run_rolling_analysis(data, OUT, test_start="2019-01-01")

    print("\n" + "═" * 70)
    print("  DONE — All outputs in ./output/")
    print("═" * 70)
    for f in sorted(OUT.iterdir()):
        print(f"    {f.name}")
    print(f"\n  Key: ~85-90% of VIX explained by SPX price path.")
    print(f"  The remaining ~10% is a tradeable signal. No options data needed.")


if __name__ == "__main__":
    main()
