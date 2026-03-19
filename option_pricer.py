"""
option_pricer.py — Option pricing using PDV-predicted volatility.

Key insight: PDV models are COMPLETE — derivatives have a unique price.
We can price options using only equity price history, no options data needed.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pdv_model import FourFactorParams, PUBLISHED


# ═══════════════════════════════════════════════════════════════════════
# Black-Scholes
# ═══════════════════════════════════════════════════════════════════════

class BlackScholes:
    """Vectorized Black-Scholes formulas."""

    @staticmethod
    def d1d2(S, K, T, σ, r=0.0):
        sqT = np.sqrt(np.maximum(T, 1e-10))
        d1 = (np.log(S / K) + (r + 0.5 * σ**2) * T) / (σ * sqT)
        return d1, d1 - σ * sqT

    @classmethod
    def call(cls, S, K, T, σ, r=0.0):
        if T <= 0 or σ <= 0:
            return max(S - K, 0.0)
        d1, d2 = cls.d1d2(S, K, T, σ, r)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @classmethod
    def put(cls, S, K, T, σ, r=0.0):
        if T <= 0 or σ <= 0:
            return max(K - S, 0.0)
        d1, d2 = cls.d1d2(S, K, T, σ, r)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @classmethod
    def price(cls, S, K, T, σ, r=0.0, is_call=True):
        return cls.call(S, K, T, σ, r) if is_call else cls.put(S, K, T, σ, r)

    @classmethod
    def delta(cls, S, K, T, σ, r=0.0, is_call=True):
        if T <= 0 or σ <= 0:
            return (1.0 if S > K else 0.0) if is_call else (-1.0 if S < K else 0.0)
        d1, _ = cls.d1d2(S, K, T, σ, r)
        return norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0

    @classmethod
    def vega(cls, S, K, T, σ, r=0.0):
        if T <= 0 or σ <= 0: return 0.0
        d1, _ = cls.d1d2(S, K, T, σ, r)
        return S * np.sqrt(T) * norm.pdf(d1)

    @classmethod
    def gamma(cls, S, K, T, σ, r=0.0):
        if T <= 0 or σ <= 0: return 0.0
        d1, _ = cls.d1d2(S, K, T, σ, r)
        return norm.pdf(d1) / (S * σ * np.sqrt(T))

    @classmethod
    def theta(cls, S, K, T, σ, r=0.0, is_call=True):
        if T <= 0 or σ <= 0: return 0.0
        d1, d2 = cls.d1d2(S, K, T, σ, r)
        t1 = -S * norm.pdf(d1) * σ / (2 * np.sqrt(T))
        t2 = -r * K * np.exp(-r * T) * norm.cdf(d2) if is_call \
            else r * K * np.exp(-r * T) * norm.cdf(-d2)
        return (t1 + t2) / 252

    @classmethod
    def implied_vol(cls, price, S, K, T, r=0.0, is_call=True):
        fn = cls.call if is_call else cls.put
        intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
        if price <= intrinsic + 1e-10:
            return 0.001
        try:
            return brentq(lambda s: fn(S, K, T, s, r) - price, 0.001, 5.0, xtol=1e-8)
        except ValueError:
            return np.nan


# ═══════════════════════════════════════════════════════════════════════
# PDV smile approximation
# ═══════════════════════════════════════════════════════════════════════

def pdv_smile(S: float, atm_vol: float, T: float, strikes: np.ndarray,
              convexity: float = 0.1) -> np.ndarray:
    """
    Approximate IV smile using PDV insights.
    Captures negative skew (leverage) + convexity (vol-of-vol).
    """
    log_m = np.log(strikes / S)
    base_skew = -1.5 * (30/365 / max(T, 1e-6)) ** 0.5
    skew = np.clip(base_skew, -5.0, -0.3)
    return np.maximum(atm_vol + skew * log_m + convexity * log_m**2, 0.01)


# ═══════════════════════════════════════════════════════════════════════
# Monte Carlo pricing (4-factor PDV)
# ═══════════════════════════════════════════════════════════════════════

class MonteCarlo:
    """Monte Carlo simulation of the 4-factor Markovian PDV model."""

    def __init__(self, params: FourFactorParams = None):
        self.params = params or PUBLISHED["4f_impl"]

    def simulate(self, S0: float, T: float, n_paths: int = 50000,
                 n_steps: int = None, seed: int = 42,
                 R10=0.0, R11=0.0, R20=0.02, R21=0.02):
        """Simulate price paths. Returns (S_paths, vol_paths)."""
        p = self.params
        if n_steps is None:
            n_steps = max(int(T * 252), 1)
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        np.random.seed(seed)

        S = np.full(n_paths, S0)
        r10, r11, r20, r21 = (np.full(n_paths, v) for v in [R10, R11, R20, R21])
        d10, d11 = np.exp(-p.K1.lam0 * dt), np.exp(-p.K1.lam1 * dt)
        d20, d21 = np.exp(-p.K2.lam0 * dt), np.exp(-p.K2.lam1 * dt)

        S_paths = np.zeros((n_paths, n_steps + 1))
        vol_paths = np.zeros((n_paths, n_steps + 1))
        S_paths[:, 0] = S0

        for i in range(n_steps):
            R1 = (1 - p.K1.theta) * r10 + p.K1.theta * r11
            R2 = (1 - p.K2.theta) * r20 + p.K2.theta * r21
            sig = np.maximum(p.beta0 + p.beta1 * R1 + p.beta2 * np.sqrt(np.maximum(R2, 1e-10)), 0.01)
            vol_paths[:, i] = sig

            dW = np.random.randn(n_paths) * sqrt_dt
            dS = sig * dW
            S = S * np.exp(dS - 0.5 * sig**2 * dt)
            S_paths[:, i + 1] = S

            # Update factors — NO dt on return term
            r10 = d10 * r10 + p.K1.lam0 * dS
            r11 = d11 * r11 + p.K1.lam1 * dS
            r20 = d20 * r20 + p.K2.lam0 * dS**2
            r21 = d21 * r21 + p.K2.lam1 * dS**2

        # Final vol
        R1 = (1 - p.K1.theta) * r10 + p.K1.theta * r11
        R2 = (1 - p.K2.theta) * r20 + p.K2.theta * r21
        vol_paths[:, -1] = np.maximum(p.beta0 + p.beta1 * R1 + p.beta2 * np.sqrt(np.maximum(R2, 1e-10)), 0.01)
        return S_paths, vol_paths

    def price_option(self, S0, K, T, is_call=True, n_paths=50000, r=0.0, seed=42):
        """Price a single option → (price, std_error, implied_vol)."""
        S_paths, _ = self.simulate(S0, T, n_paths=n_paths, seed=seed)
        S_T = S_paths[:, -1]
        payoffs = np.maximum(S_T - K, 0) if is_call else np.maximum(K - S_T, 0)
        price = np.exp(-r * T) * payoffs.mean()
        se = np.exp(-r * T) * payoffs.std() / np.sqrt(n_paths)
        iv = BlackScholes.implied_vol(price, S0, K, T, r, is_call)
        return price, se, iv

    def smile(self, S0, T, moneyness=None, n_paths=50000, r=0.0):
        """Full smile → (strikes, ivs, prices)."""
        if moneyness is None:
            moneyness = np.arange(0.80, 1.21, 0.025)
        strikes = S0 * moneyness
        S_paths, _ = self.simulate(S0, T, n_paths=n_paths, seed=42)
        S_T = S_paths[:, -1]

        ivs, prices = np.zeros(len(strikes)), np.zeros(len(strikes))
        for i, K in enumerate(strikes):
            is_call = K >= S0
            pay = np.maximum(S_T - K, 0) if is_call else np.maximum(K - S_T, 0)
            prices[i] = np.exp(-r * T) * pay.mean()
            ivs[i] = BlackScholes.implied_vol(prices[i], S0, K, T, r, is_call)
        return strikes, ivs, prices


# ═══════════════════════════════════════════════════════════════════════
# Synthetic option chain builder
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OptionQuote:
    strike: float
    expiry_days: int
    is_call: bool
    iv: float
    price: float
    delta: float = 0.0
    vega: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0


def build_chain(S: float, atm_vol: float, R1: float = 0.0, Sigma: float = 0.15,
                expiries: List[float] = None,
                moneyness: np.ndarray = None, r: float = 0.0) -> List[OptionQuote]:
    """Build synthetic option chain — the 'replace options data' function."""
    if expiries is None:
        expiries = [7/365, 14/365, 30/365, 60/365, 90/365, 180/365, 1.0]
    if moneyness is None:
        moneyness = np.arange(0.85, 1.16, 0.01)

    BS = BlackScholes
    chain = []
    for T in expiries:
        strikes = S * moneyness
        ivs = pdv_smile(S, atm_vol, T, strikes)
        for K, iv in zip(strikes, ivs):
            is_call = K >= S
            chain.append(OptionQuote(
                strike=K, expiry_days=int(T * 365), is_call=is_call, iv=iv,
                price=BS.price(S, K, T, iv, r, is_call),
                delta=BS.delta(S, K, T, iv, r, is_call),
                vega=BS.vega(S, K, T, iv, r),
                gamma=BS.gamma(S, K, T, iv, r),
                theta=BS.theta(S, K, T, iv, r, is_call),
            ))
    return chain


def chain_to_df(chain: List[OptionQuote]) -> 'pd.DataFrame':
    import pandas as pd
    return pd.DataFrame([vars(q) for q in chain])


if __name__ == "__main__":
    S = 5800.0
    print("=== BS with PDV vol ===")
    for K in [5500, 5700, 5800, 5900, 6100]:
        print(f"  K={K}: Call={BlackScholes.call(S, K, 30/365, 0.16):.2f}")

    print("\n=== MC Smile (30d) ===")
    mc = MonteCarlo()
    strikes, ivs, _ = mc.smile(S, 30/365, n_paths=20000)
    for k, iv in zip(strikes, ivs):
        if not np.isnan(iv):
            print(f"  K={k:.0f}  IV={iv:.4f}")
