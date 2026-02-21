# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Comprehensive ML feature engineering for SPY options spike prediction.

Produces one row per minute per option contract with 61 engineered features
and a binary target label: did the option price rise ≥20% in the next 120 min?

Feature groups
--------------
  Time (8):           hour_et, minute_et, minute_of_day, minutes_since_open,
                      is_morning, pct_day_elapsed, is_last_hour, spy_bar_count
  SPY momentum (5):   spy_return_{1,5,15,30,60}m
  SPY volume (6):     spy_volume, spy_vol_ma5, spy_vol_ma30,
                      spy_vol_ratio_5m, spy_vol_ratio_30m, spy_vol_zscore
  SPY volatility (4): spy_vol_std_5m, spy_vol_std_30m,
                      spy_hl_range_5m, spy_hl_range_30m
  SPY technicals (7): spy_rsi_14, spy_ema_9, spy_ema_21, spy_ema_diff,
                      spy_macd, spy_macd_signal, spy_macd_hist
  SPY Bollinger (3):  spy_bb_upper, spy_bb_lower, spy_bb_pct_b
  SPY VWAP (2):       spy_vwap, spy_vwap_dist_pct
  Opt momentum (5):   opt_return_{1,5,15,30,60}m
  Opt intraday (5):   opt_price_change_open, opt_vol_ma5,
                      opt_vol_ratio_5m, opt_rsi_14, opt_hl_range_5m
  Contract (6):       strike, moneyness, log_moneyness, time_to_expiry_days,
                      contract_type, is_0dte
  IV (4):             implied_volatility, iv_change_1m, iv_change_5m, iv_change_open
  Cross-asset (5):    opt_vs_spy_return_1m, opt_vwap_dist_pct,
                      spy_vol_regime, opt_vol_pct_cumday, transactions_ratio
  Position (1):       opt_bar_count

Target columns
--------------
  target         : 1 if option price rose ≥threshold% within next N minutes, else 0
  max_gain_120m  : maximum % gain achieved within the forward window
  time_to_max_min: minutes from entry bar to the bar with maximum gain

Output
------
  data/processed/features/{date}_features.csv
    one row per minute bar per option contract on that date

Configuration (read from config dict)
--------------------------------------
  feature_engineering.lookback_windows_minutes : list[int]  default [1,5,15,30,60]
  feature_engineering.volatility_windows_minutes: list[int]  default [5,30]
  feature_engineering.rsi_period               : int         default 14
  feature_engineering.ema_periods              : list[int]  default [9,21]
  feature_engineering.target_threshold_pct     : float       default 20.0
  feature_engineering.target_lookforward_minutes: int        default 120
"""

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytz

from src.utils.logger import get_logger

logger = get_logger()

_ET_TZ = pytz.timezone("America/New_York")

# Guarded import: ta library (already in requirements.txt)
try:
    from ta.momentum import RSIIndicator
    from ta.trend import EMAIndicator, MACD
    from ta.volatility import BollingerBands

    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False
    logger.warning(
        "ta library not available — RSI/EMA/MACD/BB features will be NaN. "
        "Install with: pip install ta"
    )

# Guarded import: Black-Scholes IV library
try:
    from py_vollib.black_scholes.implied_volatility import implied_volatility as _bs_iv

    _VOLLIB_AVAILABLE = True
except ImportError:
    _VOLLIB_AVAILABLE = False

# Minutes in a full NYSE trading day (9:30 AM–4:00 PM ET)
_TRADING_MINUTES = 390


class MLFeatureEngineer:
    """Comprehensive ML feature engineering for SPY options spike prediction.

    Reads raw minute Parquet files (SPY + options) and writes one CSV per
    trading date to data/processed/features/{date}_features.csv.

    Each row represents one minute bar of one option contract and contains:
      - ~61 engineered features (time, SPY technicals, options metrics, IV)
      - Binary target: did the option rise ≥20% within the next 120 minutes?
      - Label metadata: max_gain_120m, time_to_max_min

    Usage::

        eng = MLFeatureEngineer(config)
        df = eng.engineer_date("2025-03-03")
        stats = eng.run("2025-03-03", "2025-03-31")
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Merged config dict.  Feature-engineering params are
                    read from config["feature_engineering"]; other params
                    fall back to sensible defaults so the module can be
                    used standalone without a full config stack.
        """
        fe_cfg = config.get("feature_engineering", {})

        self.lookback_windows: List[int] = fe_cfg.get(
            "lookback_windows_minutes", [1, 5, 15, 30, 60]
        )
        self.volatility_windows: List[int] = fe_cfg.get(
            "volatility_windows_minutes", [5, 30]
        )
        self.rsi_period: int = fe_cfg.get("rsi_period", 14)
        self.ema_periods: List[int] = fe_cfg.get("ema_periods", [9, 21])
        self.target_threshold_pct: float = fe_cfg.get("target_threshold_pct", 20.0)
        self.target_lookforward_minutes: int = fe_cfg.get(
            "target_lookforward_minutes", 120
        )

        v2 = config.get("pipeline_v2", {})
        self.risk_free_rate: float = v2.get("risk_free_rate", 0.045)
        self.dividend_yield: float = v2.get("dividend_yield", 0.015)
        self._fallback_iv: float = 0.20

        parquet_cfg = config.get("sinks", {}).get("parquet", {})
        self._raw_path = Path(parquet_cfg.get("base_path", "data/raw"))
        self._output_path = Path("data/processed/features")

        # Market open for minutes_since_open computation
        open_str = v2.get("feature_engineering", {}).get("market_open_et", "09:30")
        h, m = map(int, open_str.split(":"))
        self._open_hour = h
        self._open_minute = m

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def engineer_date(self, date: str) -> pd.DataFrame:
        """Generate ML features for all options contracts on a given date.

        Steps:
          1. Load raw SPY minute data for the date.
          2. Compute comprehensive SPY-side features once (shared across all
             options contracts).
          3. For each options contract, load its raw data, merge with SPY
             features, compute option-specific features, and generate target
             labels.
          4. Concatenate all contracts, save to CSV, return the DataFrame.

        Args:
            date: Trading date string (YYYY-MM-DD).

        Returns:
            Combined feature DataFrame.  Empty if no data is available.
        """
        spy_df = self._load_spy(date)
        if spy_df.empty:
            logger.warning(f"No SPY data for {date} — skipping")
            return pd.DataFrame()

        spy_features = self._compute_spy_features(spy_df, date)

        opt_base = self._raw_path / "options" / "minute"
        if not opt_base.exists():
            logger.warning(f"Options directory not found: {opt_base}")
            return pd.DataFrame()

        result_dfs: List[pd.DataFrame] = []
        for ticker_dir in sorted(opt_base.iterdir()):
            if not ticker_dir.is_dir():
                continue
            opt_file = ticker_dir / f"{date}.parquet"
            if not opt_file.exists():
                continue

            safe_ticker = ticker_dir.name
            original_ticker = safe_ticker.replace("_", ":", 1)

            try:
                df = self._process_option(
                    opt_file, original_ticker, spy_features, date
                )
                if not df.empty:
                    result_dfs.append(df)
            except Exception as exc:
                logger.warning(
                    f"Feature engineering failed for {safe_ticker}/{date}: {exc}"
                )

        if not result_dfs:
            logger.warning(f"No options data for {date}")
            return pd.DataFrame()

        combined = pd.concat(result_dfs, ignore_index=True)
        self._save(combined, date)

        n_contracts = combined["ticker"].nunique() if "ticker" in combined.columns else 0
        pos_rate = combined["target"].mean() if "target" in combined.columns else float("nan")
        logger.info(
            f"MLFeatureEngineer {date}: {len(combined)} rows, "
            f"{n_contracts} contracts, target rate {pos_rate:.4f}"
        )
        return combined

    def run(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Run feature engineering for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date:   End date   (YYYY-MM-DD).

        Returns:
            Stats dict: dates_processed, dates_skipped, total_rows,
            total_contracts, positive_rate.
        """
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        stats: Dict[str, Any] = {
            "dates_processed": 0,
            "dates_skipped": 0,
            "total_rows": 0,
            "total_contracts": 0,
            "positive_rate": 0.0,
        }
        total_positive = 0

        while current <= end:
            date = current.strftime("%Y-%m-%d")
            df = self.engineer_date(date)

            if df.empty:
                stats["dates_skipped"] += 1
            else:
                stats["dates_processed"] += 1
                stats["total_rows"] += len(df)
                if "ticker" in df.columns:
                    stats["total_contracts"] += df["ticker"].nunique()
                if "target" in df.columns:
                    total_positive += int(df["target"].sum())

            current += timedelta(days=1)

        if stats["total_rows"] > 0:
            stats["positive_rate"] = total_positive / stats["total_rows"]

        logger.info(
            f"MLFeatureEngineer run complete: {stats['dates_processed']} dates, "
            f"{stats['total_rows']} rows, {stats['positive_rate']:.4f} positive rate"
        )
        return stats

    # ------------------------------------------------------------------
    # SPY feature computation
    # ------------------------------------------------------------------

    def _load_spy(self, date: str) -> pd.DataFrame:
        """Load raw SPY minute Parquet for a date; return sorted DataFrame."""
        path = self._raw_path / "spy" / f"{date}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)

    def _compute_spy_features(self, spy_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """Compute all SPY-side features for a full trading day.

        Returns a DataFrame with columns:
          timestamp (merge key) + spy_* features + time features
        No raw OHLCV columns are included — only prefixed / computed columns.
        This prevents column name conflicts when merging with options data.

        Args:
            spy_df: Raw SPY minute data sorted by timestamp.
            date:   Trading date (YYYY-MM-DD) — not used for computation
                    but kept for context.

        Returns:
            spy_features DataFrame keyed on timestamp.
        """
        df = spy_df.copy()
        close = df["close"]
        n = len(df)

        # --- Prefix raw columns so they survive the merge without conflicts ---
        df["spy_close"] = close
        df["spy_open"] = df["open"]
        df["spy_high"] = df["high"]
        df["spy_low"] = df["low"]
        df["spy_volume"] = df["volume"]
        df["spy_vwap"] = df.get("vwap", pd.Series(np.nan, index=df.index))
        df["spy_transactions"] = df.get(
            "transactions", pd.Series(np.nan, index=df.index)
        )

        # --- Time features ---
        et_times = df["timestamp"].apply(self._ts_to_et)
        df["hour_et"] = et_times.apply(lambda dt: dt.hour)
        df["minute_et"] = et_times.apply(lambda dt: dt.minute)
        df["minute_of_day"] = et_times.apply(lambda dt: dt.hour * 60 + dt.minute)
        df["minutes_since_open"] = df["timestamp"].apply(self._minutes_since_open)
        df["is_morning"] = (df["minutes_since_open"] < 120).astype(int)
        df["pct_day_elapsed"] = (
            df["minutes_since_open"] / _TRADING_MINUTES
        ).clip(0.0, 1.0)
        df["is_last_hour"] = (df["minutes_since_open"] >= 330).astype(int)
        df["spy_bar_count"] = range(1, n + 1)

        # --- SPY momentum (log returns for numerical stability) ---
        for w in self.lookback_windows:
            log_ret = np.log(close / close.shift(w)) * 100
            df[f"spy_return_{w}m"] = log_ret.where(
                df["minutes_since_open"] >= w, 0.0
            )

        # --- SPY volume features ---
        vol = df["spy_volume"]
        df["spy_vol_ma5"] = vol.rolling(5, min_periods=1).mean()
        df["spy_vol_ma30"] = vol.rolling(30, min_periods=1).mean()

        ma5_safe = df["spy_vol_ma5"].replace(0, np.nan)
        ma30_safe = df["spy_vol_ma30"].replace(0, np.nan)
        df["spy_vol_ratio_5m"] = vol / ma5_safe
        df["spy_vol_ratio_30m"] = vol / ma30_safe

        vol_std30 = vol.rolling(30, min_periods=5).std().replace(0, np.nan)
        df["spy_vol_zscore"] = (vol - df["spy_vol_ma30"]) / vol_std30

        # --- SPY volatility ---
        ret_1m = np.log(close / close.shift(1))
        for w in self.volatility_windows:
            df[f"spy_vol_std_{w}m"] = (
                ret_1m.rolling(w, min_periods=2).std() * 100
            )

        for w in self.volatility_windows:
            hl_range = (
                df["spy_high"].rolling(w, min_periods=1).max()
                - df["spy_low"].rolling(w, min_periods=1).min()
            )
            df[f"spy_hl_range_{w}m"] = hl_range / close.replace(0, np.nan) * 100

        # --- SPY technical indicators ---
        if _TA_AVAILABLE and n >= max(26, self.rsi_period):
            rsi_ind = RSIIndicator(close=close, window=self.rsi_period)
            df["spy_rsi_14"] = rsi_ind.rsi()

            for period in self.ema_periods:
                ema_ind = EMAIndicator(close=close, window=period)
                df[f"spy_ema_{period}"] = ema_ind.ema_indicator()

            if len(self.ema_periods) >= 2:
                e1, e2 = self.ema_periods[0], self.ema_periods[1]
                df["spy_ema_diff"] = df[f"spy_ema_{e1}"] - df[f"spy_ema_{e2}"]
            else:
                df["spy_ema_diff"] = np.nan

            macd_ind = MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
            df["spy_macd"] = macd_ind.macd()
            df["spy_macd_signal"] = macd_ind.macd_signal()
            df["spy_macd_hist"] = macd_ind.macd_diff()

            bb_ind = BollingerBands(close=close, window=20, window_dev=2)
            df["spy_bb_upper"] = bb_ind.bollinger_hband()
            df["spy_bb_lower"] = bb_ind.bollinger_lband()
            df["spy_bb_pct_b"] = bb_ind.bollinger_pband()
        else:
            for col in [
                "spy_rsi_14", "spy_ema_diff",
                "spy_macd", "spy_macd_signal", "spy_macd_hist",
                "spy_bb_upper", "spy_bb_lower", "spy_bb_pct_b",
            ]:
                df[col] = np.nan
            for period in self.ema_periods:
                df[f"spy_ema_{period}"] = np.nan

        # --- SPY VWAP distance ---
        vwap_safe = df["spy_vwap"].replace(0, np.nan)
        df["spy_vwap_dist_pct"] = (close - df["spy_vwap"]) / vwap_safe * 100

        # --- Fill NaN from insufficient warm-up history ---
        # Technical indicator NaNs at start of day are filled with the
        # first computable value (bfill). This avoids zero-padding which
        # would distort indicator distributions.
        ta_cols = [
            c for c in df.columns
            if c.startswith("spy_rsi") or c.startswith("spy_ema")
            or c.startswith("spy_macd") or c.startswith("spy_bb")
            or c.startswith("spy_vol_std") or c.startswith("spy_hl")
        ]
        df[ta_cols] = df[ta_cols].ffill().bfill()

        # --- Return only timestamp + computed/prefixed columns ---
        # Note: spy_bar_count starts with "spy_" so it is already included
        # in the spy_* list; do not list it again to avoid duplicate columns.
        keep = (
            ["timestamp"]
            + [c for c in df.columns if c.startswith("spy_")]
            + [
                "hour_et", "minute_et", "minute_of_day",
                "minutes_since_open", "is_morning",
                "pct_day_elapsed", "is_last_hour",
            ]
        )
        return df[keep].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Options processing
    # ------------------------------------------------------------------

    def _process_option(
        self,
        opt_file: Path,
        original_ticker: str,
        spy_features: pd.DataFrame,
        date: str,
    ) -> pd.DataFrame:
        """Process one options contract for a trading date.

        Steps:
          1. Load raw options Parquet.
          2. Merge SPY features onto each option bar (nearest preceding bar).
          3. Compute option-specific features.
          4. Compute target labels (forward-looking 120-min spike detection).
          5. Add metadata columns (date, ticker, opt_close).

        Args:
            opt_file:       Path to the raw options Parquet file.
            original_ticker: Ticker in O:SPY... format.
            spy_features:   Pre-computed SPY feature DataFrame.
            date:           Trading date string (YYYY-MM-DD).

        Returns:
            Feature DataFrame for this contract.  Empty if the file is empty.
        """
        opt_df = pd.read_parquet(opt_file).sort_values("timestamp").reset_index(drop=True)
        if opt_df.empty:
            return pd.DataFrame()

        # merge_asof matches each options bar to the nearest preceding SPY bar.
        # Since spy_features has only spy_* and time columns, there are no
        # column name conflicts with opt_df's raw OHLCV columns.
        merged = pd.merge_asof(
            opt_df.sort_values("timestamp"),
            spy_features.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

        # Compute option-specific features
        merged = self._compute_option_features(merged, original_ticker, date)

        # Compute forward-looking targets
        merged = self._compute_targets(merged)

        # Standardize metadata columns
        if "ticker" not in merged.columns:
            merged["ticker"] = original_ticker
        merged["date"] = date
        merged["opt_close"] = merged["close"]  # alias for clarity

        return merged.reset_index(drop=True)

    def _compute_option_features(
        self, df: pd.DataFrame, original_ticker: str, date: str
    ) -> pd.DataFrame:
        """Add option-specific feature columns to the merged DataFrame.

        Computes:
          - Option momentum (log returns at multiple windows)
          - Intraday change from open
          - Volume MA and ratio
          - Option RSI-14
          - High-low range
          - Contract metadata (strike, moneyness, TTE, type)
          - Implied volatility and its changes
          - Cross-asset and relative features
          - Option bar counter

        Args:
            df:              Merged DataFrame (opt_df + spy_features).
            original_ticker: Option ticker (O:SPY... format).
            date:            Trading date (YYYY-MM-DD).

        Returns:
            df with option features added in-place (copy returned).
        """
        df = df.copy()
        close = df["close"]  # option close price (no suffix after merge)

        # --- Option momentum ---
        for w in self.lookback_windows:
            shifted = close.shift(w).replace(0, np.nan)
            log_ret = np.log(close / shifted) * 100
            df[f"opt_return_{w}m"] = log_ret.where(
                df["minutes_since_open"] >= w, 0.0
            )

        # --- Intraday change from first bar ---
        first_close = close.iloc[0] if len(close) > 0 and close.iloc[0] > 0 else np.nan
        if first_close and not np.isnan(first_close):
            df["opt_price_change_open"] = (close - first_close) / first_close * 100
        else:
            df["opt_price_change_open"] = np.nan

        # --- Option volume features ---
        opt_vol = df.get("volume", pd.Series(np.nan, index=df.index))
        df["opt_vol_ma5"] = opt_vol.rolling(5, min_periods=1).mean()
        vol_ma5_safe = df["opt_vol_ma5"].replace(0, np.nan)
        df["opt_vol_ratio_5m"] = opt_vol / vol_ma5_safe

        # --- Option RSI-14 ---
        if _TA_AVAILABLE and len(close) >= self.rsi_period:
            opt_rsi = RSIIndicator(close=close, window=self.rsi_period)
            df["opt_rsi_14"] = opt_rsi.rsi().ffill().bfill()
        else:
            df["opt_rsi_14"] = np.nan

        # --- Option high-low range over 5 bars ---
        if "high" in df.columns and "low" in df.columns:
            hi_5 = df["high"].rolling(5, min_periods=1).max()
            lo_5 = df["low"].rolling(5, min_periods=1).min()
            close_safe = close.replace(0, np.nan)
            df["opt_hl_range_5m"] = (hi_5 - lo_5) / close_safe * 100
        else:
            df["opt_hl_range_5m"] = np.nan

        # --- Contract metadata (parsed from ticker) ---
        strike, tte_days, flag = self._parse_contract_meta(original_ticker, date)
        df["strike"] = strike
        df["contract_type"] = 1 if flag == "c" else 0
        df["time_to_expiry_days"] = tte_days
        df["is_0dte"] = int(tte_days < 1)

        # Moneyness: spot / strike (>1 = call ITM, <1 = put ITM)
        spy_close = df.get("spy_close", pd.Series(np.nan, index=df.index))
        strike_safe = strike if strike > 0 else np.nan
        df["moneyness"] = spy_close / strike_safe
        with np.errstate(divide="ignore", invalid="ignore"):
            df["log_moneyness"] = np.log(spy_close / strike_safe).replace(
                [np.inf, -np.inf], np.nan
            )

        # --- Implied volatility ---
        df["implied_volatility"] = df.apply(
            lambda row: self._calc_iv(
                float(row.get("close") or 0),
                float(row.get("spy_close") or 0),
                strike,
                tte_days,
                flag,
            ),
            axis=1,
        )
        iv = df["implied_volatility"]
        iv_s1 = iv.shift(1).replace(0, np.nan)
        iv_s5 = iv.shift(5).replace(0, np.nan)
        df["iv_change_1m"] = (iv - iv_s1) / iv_s1 * 100
        df["iv_change_5m"] = (iv - iv_s5) / iv_s5 * 100
        first_iv = iv.iloc[0] if len(iv) > 0 and iv.iloc[0] > 0 else np.nan
        if first_iv and not np.isnan(first_iv):
            df["iv_change_open"] = (iv - first_iv) / first_iv * 100
        else:
            df["iv_change_open"] = np.nan

        # --- Cross-asset features ---
        # Option return minus SPY return (excess return)
        if "spy_return_1m" in df.columns:
            df["opt_vs_spy_return_1m"] = df.get("opt_return_1m", np.nan) - df["spy_return_1m"]
        else:
            df["opt_vs_spy_return_1m"] = np.nan

        # Option VWAP distance
        opt_vwap = df.get("vwap", pd.Series(np.nan, index=df.index))
        opt_vwap_safe = opt_vwap.replace(0, np.nan)
        df["opt_vwap_dist_pct"] = (close - opt_vwap) / opt_vwap_safe * 100

        # SPY volume regime: current volume relative to rolling MA
        if "spy_volume" in df.columns and "spy_vol_ma30" in df.columns:
            ma30_safe = df["spy_vol_ma30"].replace(0, np.nan)
            df["spy_vol_regime"] = df["spy_volume"] / ma30_safe
        else:
            df["spy_vol_regime"] = np.nan

        # Option cumulative volume as fraction of total day volume
        day_vol_total = opt_vol.sum()
        if day_vol_total > 0:
            df["opt_vol_pct_cumday"] = opt_vol.cumsum() / day_vol_total
        else:
            df["opt_vol_pct_cumday"] = np.nan

        # Transactions ratio: option transactions / SPY transactions
        opt_tx = df.get("transactions", pd.Series(np.nan, index=df.index))
        spy_tx = df.get("spy_transactions", pd.Series(np.nan, index=df.index))
        spy_tx_safe = spy_tx.replace(0, np.nan)
        df["transactions_ratio"] = opt_tx / spy_tx_safe

        # --- Option bar counter (sequential within this contract's day) ---
        df["opt_bar_count"] = range(1, len(df) + 1)

        return df

    def _compute_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate forward-looking binary target labels.

        For each bar at time T:
          - Look forward up to target_lookforward_minutes bars
          - If any future bar's close >= entry_close * (1 + threshold/100)
            → target = 1, else target = 0

        Also stores:
          - max_gain_120m   : maximum % gain achievable in the forward window
          - min_loss_120m   : minimum % change (worst drawdown) in the forward window
          - time_to_max_min : minutes from T to the bar with the highest gain

        Uses numpy binary search (searchsorted) for O(n log n) performance.

        Args:
            df: Merged DataFrame with 'close' and 'timestamp' columns,
                sorted ascending by timestamp.

        Returns:
            df with target, max_gain_120m, min_loss_120m, time_to_max_min columns added.
        """
        df = df.copy()
        timestamps = df["timestamp"].values
        closes = df["close"].values
        lookforward_ms = self.target_lookforward_minutes * 60_000

        targets = np.zeros(len(df), dtype=np.int8)
        max_gains = np.full(len(df), np.nan)
        min_losses = np.full(len(df), np.nan)
        times_to_max = np.full(len(df), np.nan)

        for i in range(len(df)):
            entry_price = closes[i]
            if entry_price <= 0 or np.isnan(entry_price):
                continue

            entry_ts = timestamps[i]
            cutoff_ts = entry_ts + lookforward_ms

            # Binary search: find the first bar after entry_ts and the
            # last bar at/before cutoff_ts.
            start_idx = i + 1
            end_idx = int(np.searchsorted(timestamps, cutoff_ts, side="right"))

            if start_idx >= end_idx:
                # No future bars within the window
                continue

            future_closes = closes[start_idx:end_idx]
            future_ts = timestamps[start_idx:end_idx]

            gains = (future_closes - entry_price) / entry_price * 100

            # Index of the bar with the highest gain
            best_idx = int(np.nanargmax(gains))
            max_gain = gains[best_idx]
            max_gains[i] = max_gain
            times_to_max[i] = (future_ts[best_idx] - entry_ts) / 60_000.0

            # Worst drawdown in the forward window (for risk / stop-loss analysis)
            min_losses[i] = float(np.nanmin(gains))

            if max_gain >= self.target_threshold_pct:
                targets[i] = 1

        df["target"] = targets
        df["max_gain_120m"] = max_gains
        df["min_loss_120m"] = min_losses
        df["time_to_max_min"] = times_to_max

        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ts_to_et(self, ts_ms: int) -> datetime:
        """Convert Unix millisecond timestamp to ET-localized datetime."""
        dt_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        return dt_utc.astimezone(_ET_TZ)

    def _minutes_since_open(self, ts_ms: int) -> int:
        """Compute minutes elapsed since 9:30 AM ET for a Unix ms timestamp."""
        try:
            dt = self._ts_to_et(ts_ms)
            return (dt.hour - self._open_hour) * 60 + (dt.minute - self._open_minute)
        except Exception:
            return 0

    def _parse_contract_meta(
        self, ticker: str, trade_date: str
    ) -> Tuple[float, float, str]:
        """Extract (strike, tte_days, flag) from a standard options ticker.

        Format: O:SPY250307C00625000
          YYMMDD = expiration | C/P = type | 8 digits = strike × 1000

        Returns:
            (strike, tte_days, flag) where flag is "c" or "p".
            Falls back to (400.0, 1.0, "c") on any parse failure.
        """
        try:
            body = re.sub(r"^O[_:]", "", ticker)
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
        self, price: float, S: float, K: float, tte_days: float, flag: str
    ) -> float:
        """Compute Black-Scholes implied volatility; returns fallback on error."""
        if not _VOLLIB_AVAILABLE:
            return self._fallback_iv
        if price <= 0 or S <= 0 or K <= 0 or tte_days <= 0:
            return self._fallback_iv
        t = tte_days / 365.0
        try:
            iv = _bs_iv(price, S, K, t, self.risk_free_rate, flag)
            if iv is None or np.isnan(iv) or iv < 0.01 or iv > 5.0:
                return self._fallback_iv
            return float(iv)
        except Exception:
            return self._fallback_iv

    def _save(self, df: pd.DataFrame, date: str) -> Path:
        """Save the feature DataFrame to a date-partitioned CSV file."""
        self._output_path.mkdir(parents=True, exist_ok=True)
        path = self._output_path / f"{date}_features.csv"
        df.to_csv(path, index=False)
        logger.info(
            f"Saved ML features → {path} "
            f"({len(df)} rows × {len(df.columns)} cols)"
        )
        return path
