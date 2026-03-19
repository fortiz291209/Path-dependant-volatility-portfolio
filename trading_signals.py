"""
trading_signals.py — Generate trading signals from the PDV model.

Three signal types:
  1. VIX residual mean-reversion (z-score of actual-vs-predicted VIX)
  2. Variance Risk Premium harvesting (predicted IV vs predicted RV)
  3. Regime detection (vol level + residual dislocation)

All signals use ONLY free equity price data (SPX closes).
"""

import numpy as np
import pandas as pd
from pdv_model import PDVModel


class SignalGenerator:
    """
    Generate composite trading signals from PDV model.
    Input: SPX daily closes only. No options data needed.
    """

    def __init__(self, zscore_lookback: int = 63, vrp_threshold: float = 0.02):
        self.zscore_lookback = zscore_lookback
        self.vrp_threshold = vrp_threshold

    def generate(self, data: pd.DataFrame, train_end: str = "2018-12-31") -> pd.DataFrame:
        """
        Full signal pipeline.

        Args:
            data: DataFrame from DataFetcher.load() with [returns, VIX_decimal, RV_21d]
            train_end: calibration cutoff date

        Returns:
            DataFrame with all signals, features, and position suggestions
        """
        returns = data["returns"].values
        actual_vix = data["VIX_decimal"].values
        train_idx = (data.index < train_end).sum()

        # ── IV model (predicts VIX) ──
        iv_model = PDVModel("vix")
        iv_model.calibrate(returns, actual_vix, train_end=train_idx)
        iv_df = iv_model.predict(returns, actual_vix)

        # ── RV model (predicts realized vol) ──
        rv_model = PDVModel("rv_spx")
        rv_target = data.get("RV_21d")
        if rv_target is not None and rv_target.notna().sum() > 500:
            rv_model.calibrate(returns, rv_target.values, train_end=train_idx)
        rv_df = rv_model.predict(returns)

        pred_iv = iv_df["predicted_vol"].values
        pred_rv = rv_df["predicted_vol"].values

        # ── Signal 1: Residual z-score ──
        residual_ratio = actual_vix / np.maximum(pred_iv, 0.01)
        rr = pd.Series(residual_ratio)
        roll_mu = rr.rolling(self.zscore_lookback, min_periods=21).mean()
        roll_std = rr.rolling(self.zscore_lookback, min_periods=21).std().replace(0, np.nan)
        zscore = ((rr - roll_mu) / roll_std).values

        # ── Signal 2: Variance Risk Premium ──
        vrp = pred_iv - pred_rv
        vrp_pct = vrp / np.maximum(pred_iv, 0.01)

        # ── Signal 3: Regime ──
        regime = self._detect_regime(pred_iv, residual_ratio)

        # ── Composite: score from -1 (buy vol) to +1 (sell vol) ──
        comp = np.zeros(len(data))
        comp += np.where(zscore > 1.5, 0.4, 0)
        comp += np.where(zscore < -1.5, -0.4, 0)
        comp += np.where(vrp_pct > 0.15, 0.4, 0)
        comp += np.where(vrp_pct < 0.0, -0.4, 0)
        comp[regime == "DISLOCATION"] *= 0.25

        # ── Build output ──
        out = pd.DataFrame(index=data.index)
        out["SPX"] = data["SPX"].values
        out["VIX"] = data["VIX"].values
        out["returns"] = returns
        out["predicted_vix"] = pred_iv * 100
        out["predicted_rv"] = pred_rv * 100
        out["R1"] = iv_df["R1"].values
        out["Sigma"] = iv_df["Sigma"].values
        out["residual_zscore"] = zscore
        out["vrp"] = vrp * 100
        out["vrp_pct"] = vrp_pct
        out["regime"] = regime
        out["composite_signal"] = comp
        out["position"] = np.clip(comp, -1, 1)
        return out

    @staticmethod
    def _detect_regime(pred_vol: np.ndarray, residual_ratio: np.ndarray) -> np.ndarray:
        """Classify each day into LOW_VOL / NORMAL / HIGH_VOL / DISLOCATION."""
        regime = np.full(len(pred_vol), "NORMAL", dtype=object)
        for i in range(len(pred_vol)):
            pv = pred_vol[i]
            rr = residual_ratio[i] if not np.isnan(residual_ratio[i]) else 1.0
            if np.isnan(pv):
                regime[i] = "UNKNOWN"
            elif rr > 1.3 or rr < 0.7:
                regime[i] = "DISLOCATION" if pv > 0.25 else "DISLOCATION"
            elif pv > 0.25:
                regime[i] = "HIGH_VOL"
            elif pv < 0.12:
                regime[i] = "LOW_VOL"
        return regime


if __name__ == "__main__":
    np.random.seed(42)
    n = 2000
    r = np.random.randn(n) * 0.01
    vix = 0.15 + 0.05 * np.random.randn(n)
    model = PDVModel("vix")
    result = model.predict(r, vix)
    rr = vix / np.maximum(result["predicted_vol"].values, 0.01)
    s = pd.Series(rr)
    z = (s - s.rolling(63, min_periods=21).mean()) / s.rolling(63, min_periods=21).std()
    print(f"Z-score range: {z.min():.2f} → {z.max():.2f}")
