"""
backtest.py — Backtesting framework for PDV volatility strategies.

Strategies:
  1. VIX mean-reversion
  2. Variance Risk Premium harvesting
  3. Combined blend (1+2)
  4. PDV Tail Hedge (put options)
  5. PDV Dynamic Allocation (equity/cash, the main strategy)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from option_pricer import BlackScholes


# ═══════════════════════════════════════════════════════════════════
# Configuration dataclasses
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    capital: float = 100_000
    max_pos_pct: float = 0.10
    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    zscore_entry: float = 1.5
    zscore_exit: float = 0.5


@dataclass
class TailHedgeConfig:
    """Put-based tail hedge parameters."""
    portfolio_value: float = 500_000
    annual_budget_pct: float = 0.03
    trades_per_year: int = 6
    put_otm_pct: float = 0.05
    put_expiry_days: int = 42
    residual_buy_threshold: float = 0.98
    residual_sell_threshold: float = 1.40
    min_profit_to_sell: float = 0.50
    max_open_positions: int = 4
    rebalance_days: int = 5


@dataclass
class DynamicAllocConfig:
    """
    Dynamic vol-targeted allocation — the main strategy.

    Core idea: equity_pct = vol_target / predicted_vol
    - PDV predicts 30% vol → hold 50% equity (15/30), 50% cash
    - PDV predicts 10% vol → hold 100% equity (capped)
    - Cash earns the risk-free rate

    The residual adds a second signal:
    - When market prices MORE fear than model (rr > 1.2), reduce further
    - This catches exogenous risk the model misses

    Why this works better than put hedging:
    - No options needed, just equity/cash rebalancing
    - Works for SLOW drawdowns (2022 bear) and FAST crashes (COVID)
    - PDV prediction rises BEFORE crash deepens (leverage effect)
    - Cost = opportunity cost of holding cash, not premium decay
    """
    portfolio_value: float = 500_000
    vol_target: float = 0.22          # ~median predicted vol → 100% equity in calm markets
    min_equity_pct: float = 0.25      # never below 25% equity
    max_equity_pct: float = 1.00      # never above 100% equity
    residual_penalty: float = 0.30    # reduce equity 30% when rr > threshold
    residual_threshold: float = 1.20  # rr level triggering extra reduction
    smooth_days: int = 5              # EMA smoothing to avoid whipsaw
    rebalance_cost_bps: float = 3.0   # transaction cost per rebalance
    risk_free_rate: float = 0.04      # 4% annual on cash portion


# ═══════════════════════════════════════════════════════════════════
# Backtester class
# ═══════════════════════════════════════════════════════════════════

class Backtester:
    """Backtest PDV-based vol strategies."""

    def __init__(self, config: Config = None):
        self.cfg = config or Config()
        self._tail_hedge_trades = pd.DataFrame()

    # ──────────────────────────────────────────────────────────────
    # Strategy 1: VIX Mean-Reversion
    # ──────────────────────────────────────────────────────────────
    def vix_mean_reversion(self, signals, start="2019-01-01"):
        df = signals.loc[start:].copy()
        n, c = len(df), self.cfg
        capital, pos = c.capital, 0.0
        positions, pnl = np.zeros(n), np.zeros(n)
        vix, z, regime = df["VIX"].values, df["residual_zscore"].values, df["regime"].values
        for i in range(1, n):
            zz = z[i-1]
            if np.isnan(zz): continue
            if pos == 0:
                if zz > c.zscore_entry and regime[i-1] != "DISLOCATION": pos = -1.0
                elif zz < -c.zscore_entry: pos = 1.0
            elif pos < 0 and (zz < c.zscore_exit or zz > 4.0): pos = 0.0
            elif pos > 0 and zz > -c.zscore_exit: pos = 0.0
            positions[i] = pos
            notional = capital * c.max_pos_pct
            daily = -pos * (vix[i] - vix[i-1]) * notional / max(vix[i-1], 1)
            if positions[i] != positions[max(i-1, 0)]:
                daily -= notional * (c.cost_bps + c.slippage_bps) / 1e4
            pnl[i] = daily; capital += daily
        return self._add_columns(df, positions, pnl)

    # ──────────────────────────────────────────────────────────────
    # Strategy 2: VRP Harvesting
    # ──────────────────────────────────────────────────────────────
    def vrp_harvest(self, signals, start="2019-01-01"):
        df = signals.loc[start:].copy()
        n, c = len(df), self.cfg
        capital = c.capital
        positions, pnl = np.zeros(n), np.zeros(n)
        returns, iv, rv = df["returns"].values, df["predicted_vix"].values/100, df["predicted_rv"].values/100
        regime = df["regime"].values
        for i in range(1, n):
            vrp_pct = (iv[i-1] - rv[i-1]) / max(iv[i-1], 0.01)
            if regime[i-1] == "DISLOCATION": size = 0.0
            elif vrp_pct > 0.20: size = 1.0
            elif vrp_pct > 0.10: size = 0.6
            elif vrp_pct > 0.0: size = 0.3
            else: size = 0.0
            positions[i] = -size
            notional = capital * c.max_pos_pct * size
            daily = (iv[i-1]**2/252 - returns[i]**2) * notional
            if abs(positions[i] - positions[max(i-1, 0)]) > 0.01:
                daily -= notional * (c.cost_bps + c.slippage_bps) / 1e4
            pnl[i] = daily; capital += daily
        return self._add_columns(df, positions, pnl)

    # ──────────────────────────────────────────────────────────────
    # Strategy 3: Combined
    # ──────────────────────────────────────────────────────────────
    def combined(self, signals, start="2019-01-01", w1=0.5, w2=0.5):
        bt1, bt2 = self.vix_mean_reversion(signals, start), self.vrp_harvest(signals, start)
        df = signals.loc[start:].copy()
        df["daily_pnl"] = w1*bt1["daily_pnl"].values + w2*bt2["daily_pnl"].values
        df["cumulative_pnl"] = df["daily_pnl"].cumsum()
        df["capital"] = self.cfg.capital + df["cumulative_pnl"]
        df["position"] = w1*bt1["position"].values + w2*bt2["position"].values
        return df

    # ──────────────────────────────────────────────────────────────
    # Strategy 4: PDV Tail Hedge (put options)
    # ──────────────────────────────────────────────────────────────
    def tail_hedge(self, signals, start="2019-01-01", cfg=None):
        """Buy OTM puts when residual is low, hold to expiry or take profit."""
        h = cfg or TailHedgeConfig()
        df = signals.loc[start:].copy()
        n = len(df)
        per_trade = h.portfolio_value * h.annual_budget_pct / h.trades_per_year
        annual_cap = h.portfolio_value * h.annual_budget_pct

        open_puts, trade_log = [], []
        pnl, hedge_value, hedge_cost_cum = np.zeros(n), np.zeros(n), np.zeros(n)
        n_pos, buy_sig, sell_sig = np.zeros(n), np.zeros(n), np.zeros(n)
        spx, vix_dec = df["SPX"].values, df["VIX"].values / 100
        pred_vix = df["predicted_vix"].values / 100
        rr = np.where(pred_vix > 0.01, vix_dec / pred_vix, 1.0)
        regime = df["regime"].values
        cum_cost, cum_proceeds, year_cost = 0.0, 0.0, {}

        for i in range(1, n):
            S, sigma = spx[i], max(vix_dec[i], 0.05)
            rr_prev = rr[i-1] if not np.isnan(rr[i-1]) else 1.0
            daily_mtm, still_open = 0.0, []

            for p in open_puts:
                p["days_held"] += 1
                T_rem = max((p["expiry_days"] - p["days_held"]) / 252, 0.001)
                curr_val = BlackScholes.put(S, p["strike"], T_rem, sigma) * p["contracts"]
                current_return = curr_val / max(p["cost"], 1) - 1

                if T_rem <= 1/252:
                    payoff = max(p["strike"] - S, 0) * p["contracts"]
                    daily_mtm += payoff - p["prev_value"]; cum_proceeds += payoff
                    trade_log.append({"date": df.index[i], "action": "EXPIRE",
                        "strike": p["strike"], "cost": p["cost"], "proceeds": payoff,
                        "return_pct": payoff/max(p["cost"],1)-1, "days_held": p["days_held"],
                        "reason": "ITM" if payoff > 0 else "WORTHLESS"})
                    sell_sig[i] = 1; continue
                if current_return > h.min_profit_to_sell and rr_prev > h.residual_sell_threshold:
                    sell_px = curr_val * 0.98; daily_mtm += sell_px - p["prev_value"]; cum_proceeds += sell_px
                    trade_log.append({"date": df.index[i], "action": "SELL_PROFIT",
                        "strike": p["strike"], "cost": p["cost"], "proceeds": sell_px,
                        "return_pct": sell_px/max(p["cost"],1)-1, "days_held": p["days_held"],
                        "reason": f"PROFIT_rr={rr_prev:.2f}"})
                    sell_sig[i] = 1; continue
                if current_return > 4.0:
                    sell_px = curr_val * 0.98; daily_mtm += sell_px - p["prev_value"]; cum_proceeds += sell_px
                    trade_log.append({"date": df.index[i], "action": "SELL_BIG",
                        "strike": p["strike"], "cost": p["cost"], "proceeds": sell_px,
                        "return_pct": sell_px/max(p["cost"],1)-1, "days_held": p["days_held"],
                        "reason": "5X_PROFIT"})
                    sell_sig[i] = 1; continue
                daily_mtm += curr_val - p["prev_value"]; p["prev_value"] = curr_val
                still_open.append(p)
            open_puts = still_open

            yr = df.index[i].year; yr_spent = year_cost.get(yr, 0.0)
            if (rr_prev < h.residual_buy_threshold and len(open_puts) < h.max_open_positions
                    and i % h.rebalance_days == 0 and regime[i-1] != "DISLOCATION"
                    and yr_spent < annual_cap):
                K, T = S * (1 - h.put_otm_pct), h.put_expiry_days / 252
                px = BlackScholes.put(S, K, T, sigma)
                if px > 0.50:
                    budget = min(per_trade, annual_cap - yr_spent)
                    contracts, cost = budget / px, budget
                    if cost > 100:
                        open_puts.append({"strike": K, "expiry_days": h.put_expiry_days,
                            "days_held": 0, "contracts": contracts, "cost": cost,
                            "prev_value": cost, "entry_S": S, "entry_vol": sigma, "entry_rr": rr_prev})
                        cum_cost += cost; year_cost[yr] = yr_spent + cost; buy_sig[i] = 1
                        trade_log.append({"date": df.index[i], "action": "BUY",
                            "strike": K, "cost": cost, "S": S, "vol": sigma, "residual": rr_prev})

            total_val = sum(BlackScholes.put(S, p["strike"],
                max((p["expiry_days"]-p["days_held"])/252, 0.001), sigma)*p["contracts"]
                for p in open_puts)
            pnl[i] = daily_mtm; hedge_value[i] = total_val
            hedge_cost_cum[i] = cum_cost; n_pos[i] = len(open_puts)

        hpc = np.cumsum(pnl)
        df["daily_pnl"], df["cumulative_pnl"] = pnl, hpc
        df["capital"] = h.portfolio_value + hpc
        df["position"], df["hedge_value"] = n_pos, hedge_value
        df["hedge_cost_cum"] = hedge_cost_cum
        df["buy_signal"], df["sell_signal"] = buy_sig, sell_sig
        df["residual_ratio"] = rr
        spx_ret = df["returns"].values
        eq = h.portfolio_value * np.cumprod(1 + np.nan_to_num(spx_ret))
        df["equity_only"], df["equity_plus_hedge"] = eq, eq + hpc
        self._tail_hedge_trades = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
        return df

    # ──────────────────────────────────────────────────────────────
    # Strategy 5: PDV Dynamic Allocation ★ THE MAIN STRATEGY ★
    # ──────────────────────────────────────────────────────────────
    def dynamic_allocation(self, signals, start="2019-01-01", cfg=None):
        """
        Dynamically shift between equity and cash based on PDV vol prediction.

        Why this is the natural application of PDV:
        - The model predicts TOMORROW'S vol from today's price path
        - High predicted vol → reduce equity → avoid drawdown
        - Low predicted vol → full equity → capture gains
        - No options, no timing problem, no premium decay
        - Works for slow grinds (2022) AND fast crashes (2020)

        Allocation formula:
            raw_equity = vol_target / predicted_vol
            residual_adj = -penalty if rr > threshold else 0
            equity_pct = clip(smooth(raw_equity + residual_adj), min, max)
            cash_pct = 1 - equity_pct (earns risk-free rate)

        Daily P&L:
            equity_pct × SPX_return + cash_pct × rf/252 - rebalance_cost
        """
        d = cfg or DynamicAllocConfig()
        df = signals.loc[start:].copy()
        n = len(df)

        spx_ret = df["returns"].values
        pred_vol = df["predicted_vix"].values / 100  # PDV predicted vol
        vix_dec = df["VIX"].values / 100
        pred_vix_safe = np.where(pred_vol > 0.01, pred_vol, 0.15)
        rr = np.where(pred_vix_safe > 0.01, vix_dec / pred_vix_safe, 1.0)

        # ── Compute raw allocation ──
        raw_alloc = np.clip(d.vol_target / pred_vix_safe, d.min_equity_pct, d.max_equity_pct)

        # ── Residual penalty: reduce when market prices extra fear ──
        residual_adj = np.where(rr > d.residual_threshold, -d.residual_penalty, 0.0)
        target_alloc = np.clip(raw_alloc + residual_adj, d.min_equity_pct, d.max_equity_pct)

        # ── Smooth with EMA to avoid daily whipsaw ──
        smooth_alloc = np.copy(target_alloc)
        alpha = 2.0 / (d.smooth_days + 1)
        for i in range(1, n):
            smooth_alloc[i] = alpha * target_alloc[i] + (1 - alpha) * smooth_alloc[i-1]
        smooth_alloc = np.clip(smooth_alloc, d.min_equity_pct, d.max_equity_pct)

        # ── Simulate daily P&L ──
        capital = d.portfolio_value
        pnl = np.zeros(n)
        capital_arr = np.zeros(n)
        capital_arr[0] = capital

        rf_daily = d.risk_free_rate / 252

        for i in range(1, n):
            eq_pct = smooth_alloc[i-1]  # yesterday's allocation
            cash_pct = 1.0 - eq_pct
            r = spx_ret[i] if not np.isnan(spx_ret[i]) else 0.0

            # P&L = equity portion × market return + cash portion × rf
            daily = capital * (eq_pct * r + cash_pct * rf_daily)

            # Rebalance cost if allocation changed significantly
            alloc_change = abs(smooth_alloc[i] - smooth_alloc[i-1]) if i > 0 else 0
            if alloc_change > 0.02:  # only charge if >2% change
                daily -= capital * alloc_change * d.rebalance_cost_bps / 1e4

            pnl[i] = daily
            capital += daily
            capital_arr[i] = capital

        # ── Buy-and-hold comparison ──
        bh = d.portfolio_value * np.cumprod(1 + np.nan_to_num(spx_ret))

        # ── Output ──
        df["daily_pnl"] = pnl
        df["cumulative_pnl"] = np.cumsum(pnl)
        df["capital"] = capital_arr
        df["equity_pct"] = smooth_alloc
        df["target_alloc"] = target_alloc
        df["predicted_vol"] = pred_vol
        df["residual_ratio"] = rr
        df["equity_only"] = bh
        df["position"] = smooth_alloc  # for metrics compatibility

        return df

    @staticmethod
    def _add_columns(df, positions, pnl):
        df["position"] = positions
        df["daily_pnl"] = pnl
        df["cumulative_pnl"] = pnl.cumsum()
        df["capital"] = pnl.cumsum() + Config().capital
        return df


# ═══════════════════════════════════════════════════════════════════
# Performance metrics
# ═══════════════════════════════════════════════════════════════════

def metrics(bt):
    pnl = bt["daily_pnl"].dropna().values
    if len(pnl) == 0: return {}
    capital = bt["capital"].values
    total_ret = capital[-1] / capital[0] - 1
    n_yr = max(len(pnl) / 252, 0.01)
    ann_ret = (1 + total_ret) ** (1/n_yr) - 1
    ann_vol = np.std(pnl) * 252**0.5 / max(capital[0], 1)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    peak = np.maximum.accumulate(capital)
    max_dd = ((capital - peak) / peak).min()
    trading = pnl[pnl != 0]
    win = (trading > 0).mean() if len(trading) else 0
    gp = trading[trading > 0].sum() if (trading > 0).any() else 0
    gl = abs(trading[trading < 0].sum()) if (trading < 0).any() else 1
    return {
        "total_return": f"{total_ret:.2%}", "ann_return": f"{ann_ret:.2%}",
        "ann_vol": f"{ann_vol:.2%}", "sharpe": f"{sharpe:.2f}",
        "max_drawdown": f"{max_dd:.2%}",
        "calmar": f"{ann_ret / abs(max_dd):.2f}" if max_dd != 0 else "n/a",
        "win_rate": f"{win:.2%}", "profit_factor": f"{gp / max(gl, 1):.2f}",
        "n_trades": len(trading),
        "period": f"{bt.index[0].date()} → {bt.index[-1].date()}",
    }


def dynamic_alloc_metrics(bt):
    """Extended metrics for dynamic allocation."""
    m = metrics(bt)
    if "equity_only" in bt.columns:
        eq, cap = bt["equity_only"].values, bt["capital"].values
        dd_eq = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()
        dd_cap = ((cap - np.maximum.accumulate(cap)) / np.maximum.accumulate(cap)).min()
        m["buyhold_max_dd"] = f"{dd_eq:.2%}"
        m["strategy_max_dd"] = f"{dd_cap:.2%}"
        m["dd_reduction"] = f"{(1 - dd_cap/dd_eq):.1%}" if dd_eq != 0 else "n/a"

        bh_ret = eq[-1]/eq[0] - 1
        n_yr = max(len(eq)/252, 0.01)
        bh_ann = (1 + bh_ret)**(1/n_yr) - 1
        m["buyhold_ann_return"] = f"{bh_ann:.2%}"
        m["return_captured"] = f"{float(m['ann_return'].strip('%')) / max(bh_ann*100, 0.01):.0%}" if bh_ann > 0 else "n/a"

    if "equity_pct" in bt.columns:
        ep = bt["equity_pct"].values
        m["avg_equity"] = f"{np.mean(ep):.0%}"
        m["min_equity"] = f"{np.min(ep):.0%}"
        m["avg_cash"] = f"{1 - np.mean(ep):.0%}"
    return m


def tail_hedge_metrics(bt, trades):
    m = metrics(bt)
    if "equity_only" in bt.columns:
        eq, eh = bt["equity_only"].values, bt["equity_plus_hedge"].values
        dd_eq = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()
        dd_eh = ((eh - np.maximum.accumulate(eh)) / np.maximum.accumulate(eh)).min()
        m["equity_max_dd"] = f"{dd_eq:.2%}"
        m["hedged_max_dd"] = f"{dd_eh:.2%}"
        m["dd_reduction"] = f"{(1 - dd_eh/dd_eq):.1%}" if dd_eq != 0 else "n/a"
    if len(trades) > 0 and "action" in trades.columns:
        buys = trades[trades["action"] == "BUY"]
        exits = trades[trades["action"] != "BUY"]
        m["puts_bought"] = len(buys)
        m["total_spent"] = f"${buys['cost'].sum():,.0f}"
        if "return_pct" in exits.columns and len(exits) > 0:
            winners = exits[exits["return_pct"] > 0]
            m["winners"] = f"{len(winners)}/{len(exits)}"
            if len(winners) > 0:
                m["best_trade"] = f"{exits['return_pct'].max():.0%}"
            m["net_pnl"] = f"${exits['proceeds'].sum() - buys['cost'].sum():,.0f}"
    return m


def print_metrics(m, title="Performance"):
    print(f"\n{'═' * 55}")
    print(f"  {title}")
    print(f"{'═' * 55}")
    for k, v in m.items():
        print(f"  {k:25s}: {v}")
    print(f"{'═' * 55}")
