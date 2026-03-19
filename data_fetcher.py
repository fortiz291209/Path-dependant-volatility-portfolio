"""
data_fetcher.py — Market data loader for PDV model.
Primary: FRED (free, no rate limits). Fallback: yfinance.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


class DataFetcher:
    """Fetch and cache SPX / VIX data from free sources."""

    FRED_MAP = {"SPX": "SP500", "VIX": "VIXCLS"}
    YF_MAP = {"SPX": "^GSPC", "VIX": "^VIX"}

    def __init__(self, cache_dir: Path = DATA_DIR):
        self.cache_dir = cache_dir

    def load(self, start: str = "2000-01-01", end: str = None) -> pd.DataFrame:
        """Load aligned SPX + VIX data → DataFrame[SPX, VIX, returns, RV_21d, VIX_decimal]."""
        spx = self._get("SPX", start, end)
        vix = self._get("VIX", start, end)

        df = pd.DataFrame({"SPX": spx, "VIX": vix}).dropna()
        if df.empty:
            raise RuntimeError("No overlapping SPX/VIX data found.")

        df["returns"] = df["SPX"].pct_change()
        df["RV_21d"] = df["returns"].rolling(21).std() * 252 ** 0.5
        df["VIX_decimal"] = df["VIX"] / 100.0
        df.dropna(subset=["returns"], inplace=True)

        print(f"  Data: {len(df)} days  [{df.index[0].date()} → {df.index[-1].date()}]")
        return df

    def _get(self, name: str, start: str, end: str) -> pd.Series:
        # 1. Cache
        s = self._read_cache(name)
        if s is not None:
            return s
        # 2. FRED
        s = self._try_fred(name, start, end)
        if s is not None:
            self._write_cache(name, s)
            return s
        # 3. yfinance
        s = self._try_yfinance(name, start, end)
        if s is not None:
            self._write_cache(name, s)
            return s
        raise RuntimeError(
            f"Cannot download {name}. Install pandas-datareader:\n"
            f"  pip install pandas-datareader\n"
            f"Or download CSV manually from Yahoo Finance → save to data/{name}.csv"
        )

    def _read_cache(self, name: str) -> Optional[pd.Series]:
        for fname in [f"{name}.csv", f"{name.lower()}.csv"]:
            path = self.cache_dir / fname
            if path.exists():
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                if len(df) > 100:
                    col = "Close" if "Close" in df.columns else df.columns[0]
                    print(f"  [cache] {name}: {len(df)} rows")
                    return df[col].squeeze().rename(name)
                path.unlink()
        return None

    def _write_cache(self, name: str, s: pd.Series):
        s.to_frame(name).to_csv(self.cache_dir / f"{name}.csv")

    def _try_fred(self, name: str, start: str, end: str) -> Optional[pd.Series]:
        code = self.FRED_MAP.get(name)
        if code is None:
            return None
        try:
            import pandas_datareader as pdr
            print(f"  [FRED] downloading {name}...")
            s = pdr.get_data_fred(code, start=start, end=end).squeeze().dropna()
            s.name = name
            print(f"  [FRED] {name}: {len(s)} rows")
            return s if len(s) > 100 else None
        except ImportError:
            print("  [FRED] pandas-datareader not installed — skipping")
            return None
        except Exception as e:
            print(f"  [FRED] {name} failed: {e}")
            return None

    def _try_yfinance(self, name: str, start: str, end: str) -> Optional[pd.Series]:
        ticker = self.YF_MAP.get(name)
        if ticker is None:
            return None
        try:
            import yfinance as yf
            import time
            for attempt in range(1, 4):
                print(f"  [yfinance] downloading {name} (attempt {attempt}/3)...")
                try:
                    df = yf.download(ticker, start=start, end=end,
                                     auto_adjust=True, progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if len(df) > 100:
                        s = df["Close"].squeeze().rename(name)
                        print(f"  [yfinance] {name}: {len(s)} rows")
                        return s
                except Exception as e:
                    print(f"  [yfinance] error: {e}")
                time.sleep(10 * attempt)
        except ImportError:
            print("  [yfinance] not installed — skipping")
        return None


def load_spx_vix(start="2000-01-01", end=None) -> pd.DataFrame:
    """Convenience function."""
    return DataFetcher().load(start, end)


if __name__ == "__main__":
    print(load_spx_vix().tail())
