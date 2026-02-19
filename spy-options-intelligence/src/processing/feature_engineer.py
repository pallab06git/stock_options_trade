# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Lagged percentage-change feature engineering for minute-level data.

Produces three types of feature files:
  1. Equity features (SPY / VIX):  lagged % change for price and volume.
  2. Options features:             lagged % change for price and implied volatility.

All features use windows defined in config["pipeline_v2"]["feature_engineering"]["lag_windows"].
Features are zeroed out for bars where minutes_since_open < window (not enough history).

Output paths:
  data/processed/features/spy/{date}.parquet
  data/processed/features/vix/{date}.parquet
  data/processed/features/options/{safe_ticker}/{date}.parquet
    where safe_ticker = ticker.replace(":", "_")
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytz

from src.utils.logger import get_logger

logger = get_logger()

_ET_TZ = pytz.timezone("America/New_York")

# Guarded import — tests can run without py_vollib installed
try:
    from py_vollib.black_scholes.implied_volatility import (
        implied_volatility as bs_iv,
    )
    _VOLLIB_AVAILABLE = True
except ImportError:
    _VOLLIB_AVAILABLE = False


class FeatureEngineer:
    """Compute lagged % change features for equity and options minute data.

    Reads raw minute Parquet files from data/raw/ and writes feature
    Parquet files to data/processed/features/.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full merged config dict.
        """
        v2 = config.get("pipeline_v2", {})
        fe_cfg = v2.get("feature_engineering", {})
        self.lag_windows: List[int] = fe_cfg.get("lag_windows", [1, 5, 10, 15])
        self.market_open_et: str = fe_cfg.get("market_open_et", "09:30")

        self.risk_free_rate: float = v2.get("risk_free_rate", 0.045)
        self.dividend_yield: float = v2.get("dividend_yield", 0.015)
        self.fallback_iv: float = 0.20

        parquet_cfg = config.get("sinks", {}).get("parquet", {})
        self.raw_path = Path(parquet_cfg.get("base_path", "data/raw"))
        self.features_path = Path("data/processed/features")
        self.compression = parquet_cfg.get("compression", "snappy")

        # Parse market open hour/minute
        h, m = map(int, self.market_open_et.split(":"))
        self._open_hour = h
        self._open_minute = m

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def engineer_equity(self, date: str, ticker: str) -> pd.DataFrame:
        """Compute lagged % change features for an equity or index ticker.

        Reads data/raw/{source}/{date}.parquet (source = "spy" or "vix").
        Writes data/processed/features/{source}/{date}.parquet.

        Args:
            date: Trading date (YYYY-MM-DD).
            ticker: Ticker string ("SPY" or "I:VIX" etc.).

        Returns:
            Feature DataFrame (empty if no raw data).
        """
        source = self._source_name(ticker)
        in_path = self.raw_path / source / f"{date}.parquet"
        out_path = self.features_path / source / f"{date}.parquet"

        if not in_path.exists():
            logger.warning(f"Raw data not found for {ticker}/{date}: {in_path}")
            return pd.DataFrame()

        df = pd.read_parquet(in_path).sort_values("timestamp").reset_index(drop=True)
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df["minutes_since_open"] = df["timestamp"].apply(
            lambda ts: self._minutes_since_open(ts)
        )

        for w in self.lag_windows:
            # Price % change
            shifted_close = df["close"].shift(w).replace(0, np.nan)
            raw_pc = (df["close"] - shifted_close) / shifted_close * 100
            # Zero out bars with insufficient history
            df[f"price_change_{w}m"] = raw_pc.where(df["minutes_since_open"] >= w, 0.0)

            # Volume % change
            if "volume" in df.columns:
                shifted_vol = df["volume"].shift(w).replace(0, np.nan)
                raw_vc = (df["volume"] - shifted_vol) / shifted_vol * 100
                df[f"volume_change_{w}m"] = raw_vc.where(df["minutes_since_open"] >= w, 0.0)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, engine="pyarrow", compression=self.compression, index=False)
        logger.info(f"Wrote equity features for {ticker}/{date} → {out_path}")
        return df

    def engineer_options(self, date: str) -> Dict[str, pd.DataFrame]:
        """Compute lagged % change + IV features for all options on a date.

        For each contract in data/raw/options/minute/*/{date}.parquet:
          - Merges with SPY minute data (nearest timestamp backward-fill)
          - Computes implied volatility per bar
          - Adds price_change_{w}m and iv_change_{w}m columns
          - Adds price_change_open and iv_change_open columns

        Writes data/processed/features/options/{safe_ticker}/{date}.parquet.

        Args:
            date: Trading date (YYYY-MM-DD).

        Returns:
            Dict of {ticker: DataFrame} for all processed contracts.
        """
        options_dir = self.raw_path / "options" / "minute"
        spy_path = self.raw_path / "spy" / f"{date}.parquet"

        if not options_dir.exists():
            return {}

        # Load SPY for IV computation (may be empty — IV will fallback)
        spy_df = pd.DataFrame()
        if spy_path.exists():
            spy_df = pd.read_parquet(spy_path).sort_values("timestamp").reset_index(drop=True)

        results: Dict[str, pd.DataFrame] = []
        results = {}

        for ticker_dir in sorted(options_dir.iterdir()):
            if not ticker_dir.is_dir():
                continue
            opt_path = ticker_dir / f"{date}.parquet"
            if not opt_path.exists():
                continue

            safe_ticker = ticker_dir.name  # already safe (colon replaced)
            original_ticker = safe_ticker.replace("_", ":", 1)  # e.g. O_SPY... → O:SPY...

            try:
                df = self._engineer_single_option(
                    opt_path, safe_ticker, original_ticker, spy_df, date
                )
                if df.empty:
                    continue
                results[safe_ticker] = df
            except Exception as exc:
                logger.warning(f"Feature engineering failed for {safe_ticker}/{date}: {exc}")

        return results

    def run(
        self,
        start_date: str,
        end_date: str,
        source: str = "all",
    ) -> Dict[str, Any]:
        """Run feature engineering for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            source: "spy", "vix", "options", or "all".

        Returns:
            Aggregated stats dict.
        """
        from datetime import timedelta

        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        stats: Dict[str, Any] = {
            "dates_processed": 0,
            "dates_skipped": 0,
            "equity_files": 0,
            "options_files": 0,
        }

        while current <= end:
            date = current.strftime("%Y-%m-%d")

            if source in ("spy", "all"):
                df = self.engineer_equity(date, "SPY")
                if not df.empty:
                    stats["equity_files"] += 1

            if source in ("vix", "all"):
                df = self.engineer_equity(date, "I:VIX")
                if not df.empty:
                    stats["equity_files"] += 1

            if source in ("options", "all"):
                opts = self.engineer_options(date)
                stats["options_files"] += len(opts)

            stats["dates_processed"] += 1
            current += timedelta(days=1)

        logger.info(
            f"FeatureEngineer run: {stats['dates_processed']} dates, "
            f"{stats['equity_files']} equity files, {stats['options_files']} options files"
        )
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _engineer_single_option(
        self,
        opt_path: Path,
        safe_ticker: str,
        original_ticker: str,
        spy_df: pd.DataFrame,
        date: str,
    ) -> pd.DataFrame:
        """Process one options contract for a date."""
        df = pd.read_parquet(opt_path).sort_values("timestamp").reset_index(drop=True)
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df["minutes_since_open"] = df["timestamp"].apply(self._minutes_since_open)

        # Merge SPY close for IV computation
        if not spy_df.empty:
            spy_close = spy_df[["timestamp", "close"]].rename(
                columns={"close": "spy_close"}
            )
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                spy_close.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
        else:
            df["spy_close"] = np.nan

        # Parse contract metadata from ticker
        strike, tte_days, flag = self._parse_contract_meta(original_ticker, date)

        # Compute IV per bar
        df["implied_volatility"] = df.apply(
            lambda row: self._calc_iv(
                float(row["close"] or 0),
                float(row.get("spy_close") or 0),
                strike,
                tte_days,
                flag,
            ),
            axis=1,
        )

        # Lagged price and IV features
        for w in self.lag_windows:
            shifted_close = df["close"].shift(w).replace(0, np.nan)
            raw_pc = (df["close"] - shifted_close) / shifted_close * 100
            df[f"price_change_{w}m"] = raw_pc.where(df["minutes_since_open"] >= w, 0.0)

            shifted_iv = df["implied_volatility"].shift(w).replace(0, np.nan)
            raw_ivc = (df["implied_volatility"] - shifted_iv) / shifted_iv * 100
            df[f"iv_change_{w}m"] = raw_ivc.where(df["minutes_since_open"] >= w, 0.0)

        # Change from first bar of day
        first_close = df["close"].iloc[0] if len(df) > 0 else np.nan
        first_iv = df["implied_volatility"].iloc[0] if len(df) > 0 else np.nan

        if first_close and first_close > 0:
            df["price_change_open"] = (df["close"] - first_close) / first_close * 100
        else:
            df["price_change_open"] = np.nan

        if first_iv and first_iv > 0:
            df["iv_change_open"] = (df["implied_volatility"] - first_iv) / first_iv * 100
        else:
            df["iv_change_open"] = np.nan

        out_path = self.features_path / "options" / safe_ticker / f"{date}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, engine="pyarrow", compression=self.compression, index=False)

        return df

    def _parse_contract_meta(
        self, ticker: str, trade_date: str
    ) -> tuple:
        """Extract (strike, tte_days, flag) from a standard options ticker.

        Format: O:SPY250307C00625000
          - YYMMDD = expiration date
          - C/P = contract type
          - 8 digits = strike * 1000

        Returns (strike, tte_days, flag) where flag is "c" or "p".
        Falls back to (400.0, 1.0, "c") on parse failure.
        """
        try:
            # Remove "O:" or "O_" prefix
            body = ticker.lstrip("O:").lstrip("O_")
            # Remove underlying (SPY) — variable length — then YYMMDD (6) + type (1) + strike (8)
            # Pattern: {UNDERLYING}{YYMMDD}{C/P}{8-digit-strike}
            # Find where the 6-digit date starts (first digit after letters)
            import re
            m = re.search(r"(\d{6})([CP])(\d{8})$", body, re.IGNORECASE)
            if not m:
                return 400.0, 1.0, "c"

            date_str, ctype, strike_str = m.groups()
            year = 2000 + int(date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            exp_dt = datetime(year, month, day, tzinfo=timezone.utc)

            trade_dt = datetime.strptime(trade_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            tte_days = max((exp_dt - trade_dt).total_seconds() / 86400, 0.01)

            strike = int(strike_str) / 1000.0
            flag = "c" if ctype.upper() == "C" else "p"
            return strike, tte_days, flag
        except Exception:
            return 400.0, 1.0, "c"

    def _calc_iv(
        self,
        price: float,
        S: float,
        K: float,
        tte_days: float,
        flag: str,
    ) -> float:
        """Calculate implied volatility; returns fallback on failure."""
        if not _VOLLIB_AVAILABLE:
            return self.fallback_iv
        if price <= 0 or S <= 0 or K <= 0 or tte_days <= 0:
            return self.fallback_iv
        t = tte_days / 365.0
        try:
            iv = bs_iv(price, S, K, t, self.risk_free_rate, flag)
            if iv is None or np.isnan(iv) or iv < 0.01 or iv > 5.0:
                return self.fallback_iv
            return float(iv)
        except Exception:
            return self.fallback_iv

    def _minutes_since_open(self, ts_ms: int) -> int:
        """Compute minutes since market open from a Unix millisecond timestamp."""
        try:
            dt_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            dt_et = dt_utc.astimezone(_ET_TZ)
            return (
                (dt_et.hour - self._open_hour) * 60
                + (dt_et.minute - self._open_minute)
            )
        except Exception:
            return 0

    def _source_name(self, ticker: str) -> str:
        """Map ticker to directory name (same logic as MinuteDownloader)."""
        if ticker.upper() == "I:VIX":
            return "vix"
        return ticker.lower().replace(":", "_")
