"""
pdv_model.py — Guyon-Lekeufack Path-Dependent Volatility model.

    σ_t = β₀ + β₁·R₁(t) + β₂·Σ(t)

    R₁ = Σ K₁(τᵢ)·rᵢ    (trend — captures leverage effect)
    Σ  = √(Σ K₂(τᵢ)·rᵢ²) (activity — captures vol clustering)

Reference: Guyon & Lekeufack, "Volatility is (Mostly) Path-Dependent", 2022
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.optimize import minimize
from typing import Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════
# Kernels
# ═══════════════════════════════════════════════════════════════════════

def tspl_kernel(tau: np.ndarray, alpha: float, delta: float) -> np.ndarray:
    """Time-Shifted Power Law: K(τ) = Z⁻¹·(τ+δ)^(-α), normalized."""
    raw = (tau + delta) ** (-alpha)
    dt = tau[1] - tau[0] if len(tau) > 1 else 1.0
    try:
        Z = np.trapezoid(raw, dx=dt)
    except AttributeError:
        Z = np.trapz(raw, dx=dt)
    return raw / Z if Z > 0 else raw


# ═══════════════════════════════════════════════════════════════════════
# Parameter dataclasses
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class KernelParams:
    """TSPL kernel params: K(τ) = Z⁻¹·(τ+δ)^(-α)"""
    alpha: float
    delta: float

@dataclass
class TwoExpParams:
    """Two-exponential kernel for 4-factor model."""
    lam0: float    # short memory (fast decay)
    lam1: float    # long memory  (slow decay)
    theta: float   # mixing weight for long-memory component

@dataclass
class PDVParams:
    """Full model: σ = β₀ + β₁·R₁ + β₂·Σ"""
    beta0: float
    beta1: float
    beta2: float
    K1: KernelParams
    K2: KernelParams

    def __str__(self):
        return (f"σ = {self.beta0:.4f} + ({self.beta1:.3f})·R₁ + ({self.beta2:.3f})·Σ  |  "
                f"K₁(α={self.K1.alpha:.3f}, δ={self.K1.delta:.4f})  "
                f"K₂(α={self.K2.alpha:.3f}, δ={self.K2.delta:.4f})")

@dataclass
class FourFactorParams:
    """4-factor Markovian approximation."""
    beta0: float
    beta1: float
    beta2: float
    K1: TwoExpParams
    K2: TwoExpParams


# ═══════════════════════════════════════════════════════════════════════
# Published calibrations (paper Table, slides 34 & 61)
# ═══════════════════════════════════════════════════════════════════════

PUBLISHED = {
    "vix": PDVParams(0.057, -23.829, 0.819, KernelParams(1.057, 0.020), KernelParams(1.597, 0.052)),
    "vix9d": PDVParams(0.045, -30.655, 0.884, KernelParams(0.993, 0.011), KernelParams(1.252, 0.011)),
    "rv_spx": PDVParams(0.018, -10.490, 0.708, KernelParams(2.821, 0.044), KernelParams(1.860, 0.025)),
    "4f_hist": FourFactorParams(0.04, -0.11, 0.65, TwoExpParams(55, 10, 0.25), TwoExpParams(20, 3, 0.5)),
    "4f_impl": FourFactorParams(0.048, -0.125, 0.46, TwoExpParams(80, 50, 0.25), TwoExpParams(89, 13, 0.7)),
}


# ═══════════════════════════════════════════════════════════════════════
# Feature computation
# ═══════════════════════════════════════════════════════════════════════

def compute_features_tspl(returns: np.ndarray, K1: KernelParams, K2: KernelParams,
                          dt: float = 1/252, lookback: int = 504) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute R₁ (trend) and Σ (activity) via TSPL kernels.

    Discrete sum: R₁ = Σ K(τᵢ)·rᵢ — NO dt scaling factor (paper slide 26).
    The β coefficients absorb the resulting scale.
    """
    n = len(returns)
    lags = np.arange(lookback) * dt
    w1 = tspl_kernel(lags, K1.alpha, K1.delta)
    w2 = tspl_kernel(lags, K2.alpha, K2.delta)

    R1 = np.full(n, np.nan)
    R2 = np.full(n, np.nan)
    for t in range(lookback, n):
        r = returns[t - lookback:t][::-1]
        R1[t] = w1[:len(r)] @ r
        R2[t] = w2[:len(r)] @ (r ** 2)

    return R1, np.sqrt(np.maximum(R2, 0))


def compute_features_4factor(returns: np.ndarray, K1: TwoExpParams, K2: TwoExpParams,
                             dt: float = 1/252) -> Tuple[np.ndarray, np.ndarray]:
    """
    4-factor Markovian model: two exponential components per feature.
    Euler: R_new = decay·R_old + λ·r  (NO dt on return term).
    """
    n = len(returns)
    R10, R11, R20, R21 = (np.zeros(n) for _ in range(4))
    d10, d11 = np.exp(-K1.lam0 * dt), np.exp(-K1.lam1 * dt)
    d20, d21 = np.exp(-K2.lam0 * dt), np.exp(-K2.lam1 * dt)

    for t in range(1, n):
        r = returns[t]
        R10[t] = d10 * R10[t-1] + K1.lam0 * r
        R11[t] = d11 * R11[t-1] + K1.lam1 * r
        R20[t] = d20 * R20[t-1] + K2.lam0 * r**2
        R21[t] = d21 * R21[t-1] + K2.lam1 * r**2

    R1 = (1 - K1.theta) * R10 + K1.theta * R11
    R2 = (1 - K2.theta) * R20 + K2.theta * R21
    return R1, np.sqrt(np.maximum(R2, 0))


# ═══════════════════════════════════════════════════════════════════════
# OLS calibration
# ═══════════════════════════════════════════════════════════════════════

def calibrate_ols(R1: np.ndarray, Sigma: np.ndarray,
                  target: np.ndarray, mask: np.ndarray = None) -> Tuple[float, float, float, float]:
    """OLS for β₀ + β₁·R₁ + β₂·Σ = target. Returns (β₀, β₁, β₂, r²)."""
    if mask is None:
        mask = ~(np.isnan(R1) | np.isnan(Sigma) | np.isnan(target))
    X = np.column_stack([np.ones(mask.sum()), R1[mask], Sigma[mask]])
    y = target[mask]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return beta[0], beta[1], beta[2], 1 - ss_res / ss_tot


# ═══════════════════════════════════════════════════════════════════════
# Main model class
# ═══════════════════════════════════════════════════════════════════════

class PDVModel:
    """
    Complete PDV pipeline: features → calibration → prediction → residuals.

    Usage:
        model = PDVModel("vix")
        model.calibrate(returns, vix_decimal, train_end=4000)
        result = model.predict(returns, vix_decimal)
    """

    def __init__(self, preset: str = "vix", params=None):
        self.params = params or PUBLISHED[preset]
        self.R1: Optional[np.ndarray] = None
        self.Sigma: Optional[np.ndarray] = None

    @property
    def is_4factor(self) -> bool:
        return isinstance(self.params, FourFactorParams)

    def _compute_features(self, returns: np.ndarray, lookback: int = 504):
        if self.is_4factor:
            self.R1, self.Sigma = compute_features_4factor(returns, self.params.K1, self.params.K2)
        else:
            self.R1, self.Sigma = compute_features_tspl(returns, self.params.K1, self.params.K2,
                                                        lookback=lookback)

    def _predict_vol(self) -> np.ndarray:
        p = self.params
        vol = p.beta0 + p.beta1 * self.R1 + p.beta2 * self.Sigma
        return np.maximum(vol, 0.01)

    def calibrate(self, returns: np.ndarray, target: np.ndarray,
                  train_end: int = None, lookback: int = 504) -> dict:
        """Calibrate β₀,β₁,β₂ via OLS on train, evaluate on test."""
        if train_end is None:
            train_end = int(len(returns) * 0.75)

        self._compute_features(returns, lookback)
        valid = ~(np.isnan(self.R1) | np.isnan(self.Sigma) | np.isnan(target))

        # Train
        train = valid.copy()
        train[train_end:] = False
        b0, b1, b2, train_r2 = calibrate_ols(self.R1, self.Sigma, target, train)

        # Test
        test = valid.copy()
        test[:train_end] = False
        pred = b0 + b1 * self.R1[test] + b2 * self.Sigma[test]
        actual = target[test]
        ss_res = ((actual - pred) ** 2).sum()
        ss_tot = ((actual - actual.mean()) ** 2).sum()
        test_r2 = 1 - ss_res / ss_tot

        # Store calibrated betas
        self.params.beta0, self.params.beta1, self.params.beta2 = b0, b1, b2

        return {"train_r2": train_r2, "test_r2": test_r2,
                "beta0": b0, "beta1": b1, "beta2": b2,
                "train_n": train.sum(), "test_n": test.sum()}

    def predict(self, returns: np.ndarray, target: np.ndarray = None,
                lookback: int = 504) -> pd.DataFrame:
        """Compute features and predict vol. Optionally compare to target."""
        self._compute_features(returns, lookback)
        pred = self._predict_vol()

        df = pd.DataFrame({"R1": self.R1, "Sigma": self.Sigma, "predicted_vol": pred})
        if target is not None:
            df["actual_vol"] = target
            df["residual"] = target - pred
            df["residual_ratio"] = target / np.maximum(pred, 0.01)
        return df

    def calibrate_kernels(self, returns: np.ndarray, target: np.ndarray,
                          lookback: int = 504) -> Tuple['PDVParams', float]:
        """Joint optimization of kernel + linear params via Nelder-Mead."""
        dt = 1 / 252

        def objective(x):
            a1, d1, a2, d2 = x
            if a1 <= 1 or a2 <= 1 or d1 <= 0 or d2 <= 0 or d1 > 1 or d2 > 1:
                return 1e10
            R1, Sig = compute_features_tspl(returns, KernelParams(a1, d1),
                                            KernelParams(a2, d2), dt, lookback)
            mask = ~(np.isnan(R1) | np.isnan(Sig) | np.isnan(target))
            if mask.sum() < 100:
                return 1e10
            _, _, _, r2 = calibrate_ols(R1, Sig, target, mask)
            return -r2

        print("  Calibrating kernel parameters...")
        x0 = [1.057, 0.020, 1.597, 0.052]
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 500, "xatol": 1e-3, "fatol": 1e-6})
        a1, d1, a2, d2 = res.x

        R1, Sig = compute_features_tspl(returns, KernelParams(a1, d1),
                                        KernelParams(a2, d2), dt, lookback)
        mask = ~(np.isnan(R1) | np.isnan(Sig) | np.isnan(target))
        b0, b1, b2, r2 = calibrate_ols(R1, Sig, target, mask)

        params = PDVParams(b0, b1, b2, KernelParams(a1, d1), KernelParams(a2, d2))
        self.params = params
        print(f"  Done: r² = {r2:.4f}  |  {params}")
        return params, r2


if __name__ == "__main__":
    np.random.seed(42)
    r = np.random.randn(5000) * 0.01
    model = PDVModel("vix")
    result = model.predict(r)
    print(f"Predicted vol range: {np.nanmin(result['predicted_vol']):.4f} "
          f"→ {np.nanmax(result['predicted_vol']):.4f}")
