# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Data consolidation engine for multi-source SPY options intelligence.

Produces a per-option-per-minute flat dataset by:
  1. Aggregating raw per-second SPY/VIX/Options data to 1-minute bars
  2. Aligning VIX onto SPY minute grid (merge_asof forward-fill)
  3. Computing technical indicators (RSI, MACD, Bollinger) on 1-min SPY
  4. Computing momentum (price change, ROC) on 1-min SPY
  5. Attaching latest news sentiment
  6. Flattening to one row per option per minute (inner join)
  7. Computing Black-Scholes Greeks per row (flat scalars)

Output is written to date-partitioned Parquet files in
data/processed/consolidated/.

Note: target_future_prices is NOT computed here. That is handled by
the TrainingDataPrep module which reads consolidated output and adds
historical lookahead targets for ML training.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

from src.utils.logger import get_logger

logger = get_logger()

# Greeks imports — guarded so tests can mock if needed
try:
    from py_vollib.black_scholes.greeks.analytical import (
        delta as bs_delta,
        gamma as bs_gamma,
        theta as bs_theta,
        vega as bs_vega,
        rho as bs_rho,
    )
    from py_vollib.black_scholes.implied_volatility import (
        implied_volatility as bs_iv,
    )

    _VOLLIB_AVAILABLE = True
except ImportError:
    _VOLLIB_AVAILABLE = False


class Consolidator:
    """Consolidate SPY, VIX, options, and news data into an enriched dataset.

    Output schema: one row per option per minute with flat scalar columns
    for Greeks, indicators, momentum, and news sentiment.
    """

    def __init__(self, config: Dict[str, Any]):
        cons = config.get("consolidation", {})
        self.risk_free_rate = cons.get("risk_free_rate", 0.045)
        self.dividend_yield = cons.get("dividend_yield", 0.015)

        ind = cons.get("indicators", {})
        self.rsi_period = ind.get("rsi", {}).get("period", 14)
        self.macd_fast = ind.get("macd", {}).get("fast", 12)
        self.macd_slow = ind.get("macd", {}).get("slow", 26)
        self.macd_signal = ind.get("macd", {}).get("signal", 9)
        self.bb_period = ind.get("bollinger", {}).get("period", 20)
        self.bb_std = ind.get("bollinger", {}).get("std_dev", 2.0)

        self.momentum_windows = cons.get("momentum", {}).get("windows", [5, 30, 60])

        greeks_cfg = cons.get("greeks", {})
        self.min_tte_days = greeks_cfg.get("min_time_to_expiry_days", 0.01)
        self.min_iv = greeks_cfg.get("min_iv", 0.01)
        self.max_iv = greeks_cfg.get("max_iv", 5.0)
        self.fallback_iv = greeks_cfg.get("fallback_iv", 0.20)

        self.news_lookback_hours = cons.get("news", {}).get("max_lookback_hours", 24)

        parquet_cfg = config.get("sinks", {}).get("parquet", {})
        self.base_path = Path(parquet_cfg.get("base_path", "data/raw"))
        self.consolidated_path = Path(
            parquet_cfg.get("consolidated_path", "data/processed/consolidated")
        )
        self.compression = parquet_cfg.get("compression", "snappy")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consolidate(self, date: str) -> Dict[str, Any]:
        """Run the full consolidation pipeline for a single trading day.

        Pipeline:
          Load raw per-second data
          → Aggregate SPY/VIX/Options to 1-min bars
          → Align VIX onto SPY minute grid
          → Compute indicators on 1-min SPY
          → Compute momentum on 1-min SPY
          → Attach news sentiment
          → Flatten: one row per option per minute
          → Compute Greeks per-row (flat scalars)
          → Write Parquet

        Returns:
            Stats dict with status, row counts, and availability flags.
        """
        # Load raw per-second data
        spy_raw = self._load_spy(date)
        if spy_raw.empty:
            logger.error(f"No SPY data for {date} — consolidation failed")
            return {"status": "failed", "date": date, "reason": "missing_spy_data"}

        vix_raw = self._load_vix(date)
        options_raw = self._load_options(date)
        contracts = self._load_contracts(date)
        news_df = self._load_news(date)

        vix_available = not vix_raw.empty
        news_available = not news_df.empty

        # Aggregate to 1-minute bars
        spy_1m = self._aggregate_spy_per_minute(spy_raw)
        vix_1m = self._aggregate_vix_per_minute(vix_raw) if vix_available else pd.DataFrame()
        options_1m = self._aggregate_options_per_minute(options_raw) if not options_raw.empty else pd.DataFrame()

        # Rename SPY columns for consolidated namespace
        spy_1m = spy_1m.rename(columns={
            "open": "spy_open",
            "high": "spy_high",
            "low": "spy_low",
            "close": "spy_close",
            "volume": "spy_volume",
            "vwap": "spy_vwap",
            "transactions": "spy_transactions",
        })

        # Align VIX onto SPY minute grid
        spy_enriched = self._align_vix(spy_1m, vix_1m)

        # Technical indicators on 1-min SPY bars
        spy_enriched = self._compute_indicators(spy_enriched)

        # Momentum on 1-min SPY bars
        spy_enriched = self._compute_momentum(spy_enriched)

        # Attach news sentiment
        spy_enriched = self._attach_news(spy_enriched, news_df)

        # Flatten: one row per option per minute
        df = self._flatten_to_per_option(spy_enriched, options_1m, contracts)

        # Compute Greeks per row (flat scalars)
        df = self._compute_greeks_flat(df)

        # Source tag
        df["source"] = "consolidated"

        # Drop any stray source columns from merges
        for col in ["source_x", "source_y"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Compute stats
        unique_options = df["ticker"].dropna().nunique() if "ticker" in df.columns else 0
        minutes = df["timestamp"].nunique()

        # Write output
        self._write_output(df, date)

        logger.info(
            f"Consolidation complete for {date}: "
            f"{len(df)} rows, {minutes} minutes, {unique_options} options, "
            f"VIX={'yes' if vix_available else 'no'}, "
            f"news={'yes' if news_available else 'no'}"
        )

        return {
            "status": "success",
            "date": date,
            "total_rows": len(df),
            "minutes": minutes,
            "unique_options": unique_options,
            "options_contracts_processed": len(contracts),
            "vix_available": vix_available,
            "news_available": news_available,
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_spy(self, date: str) -> pd.DataFrame:
        path = self.base_path / "spy" / f"{date}.parquet"
        if not path.exists():
            logger.warning(f"SPY data not found: {path}")
            return pd.DataFrame()
        df = pd.read_parquet(path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _load_vix(self, date: str) -> pd.DataFrame:
        path = self.base_path / "vix" / f"{date}.parquet"
        if not path.exists():
            logger.debug(f"VIX data not found: {path}")
            return pd.DataFrame()
        df = pd.read_parquet(path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _load_options(self, date: str) -> pd.DataFrame:
        path = self.base_path / "options" / f"{date}.parquet"
        if not path.exists():
            logger.debug(f"Options data not found: {path}")
            return pd.DataFrame()
        df = pd.read_parquet(path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _load_contracts(self, date: str) -> List[Dict]:
        path = self.base_path / "options" / "contracts" / f"{date}_contracts.json"
        if not path.exists():
            logger.debug(f"Contracts metadata not found: {path}")
            return []
        with open(path, "r") as f:
            return json.load(f)

    def _load_news(self, date: str) -> pd.DataFrame:
        path = self.base_path / "news" / f"{date}.parquet"
        if not path.exists():
            logger.debug(f"News data not found: {path}")
            return pd.DataFrame()
        df = pd.read_parquet(path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Per-minute aggregation
    # ------------------------------------------------------------------

    def _aggregate_spy_per_minute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-second SPY data to 1-minute OHLCV bars."""
        df = df.copy()
        df["minute"] = (df["timestamp"] // 60000) * 60000

        agg = df.groupby("minute").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            transactions=("transactions", "sum"),
        ).reset_index()

        # Volume-weighted average price
        vol_sum = df.groupby("minute")["volume"].sum()
        vwap_num = df.groupby("minute").apply(
            lambda g: (g["vwap"] * g["volume"]).sum(), include_groups=False
        )
        vwap = (vwap_num / vol_sum).replace([np.inf, -np.inf], np.nan)
        agg["vwap"] = agg["minute"].map(vwap)

        agg = agg.rename(columns={"minute": "timestamp"})
        return agg.sort_values("timestamp").reset_index(drop=True)

    def _aggregate_vix_per_minute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-second VIX data to 1-minute OHLC bars."""
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["minute"] = (df["timestamp"] // 60000) * 60000

        agg = df.groupby("minute").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        ).reset_index()

        agg = agg.rename(columns={"minute": "timestamp"})
        return agg.sort_values("timestamp").reset_index(drop=True)

    def _aggregate_options_per_minute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-second options data to 1-minute bars per ticker."""
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["minute"] = (df["timestamp"] // 60000) * 60000

        agg = df.groupby(["minute", "ticker"]).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).reset_index()

        # option_avg_price = average of close prices within each minute per ticker
        avg_price = df.groupby(["minute", "ticker"])["close"].mean().reset_index()
        avg_price = avg_price.rename(columns={"close": "option_avg_price"})
        agg = agg.merge(avg_price, on=["minute", "ticker"], how="left")

        agg = agg.rename(columns={"minute": "timestamp"})
        return agg.sort_values(["timestamp", "ticker"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # VIX alignment
    # ------------------------------------------------------------------

    def _align_vix(self, spy_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
        """Merge VIX data onto SPY timestamps via forward-fill merge_asof."""
        if vix_df.empty:
            spy_df["vix_open"] = np.nan
            spy_df["vix_high"] = np.nan
            spy_df["vix_low"] = np.nan
            spy_df["vix_close"] = np.nan
            return spy_df

        vix_renamed = vix_df.rename(
            columns={
                "open": "vix_open",
                "high": "vix_high",
                "low": "vix_low",
                "close": "vix_close",
            }
        )
        vix_cols = vix_renamed[["timestamp", "vix_open", "vix_high", "vix_low", "vix_close"]]

        merged = pd.merge_asof(
            spy_df.sort_values("timestamp"),
            vix_cols.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        return merged

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI, MACD, and Bollinger Band columns to the DataFrame."""
        min_bars = max(self.macd_slow, self.bb_period) + 10
        if len(df) < min_bars:
            logger.warning(
                f"Insufficient data ({len(df)} bars < {min_bars}) — skipping indicators"
            )
            for col in [
                "spy_rsi_14",
                "spy_macd",
                "spy_macd_signal",
                "spy_macd_histogram",
                "spy_bb_upper",
                "spy_bb_middle",
                "spy_bb_lower",
                "spy_bb_width",
            ]:
                df[col] = np.nan
            return df

        close = df["spy_close"]

        # RSI
        rsi = RSIIndicator(close, window=self.rsi_period)
        df["spy_rsi_14"] = rsi.rsi()

        # MACD
        macd = MACD(
            close,
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal,
        )
        df["spy_macd"] = macd.macd()
        df["spy_macd_signal"] = macd.macd_signal()
        df["spy_macd_histogram"] = macd.macd_diff()

        # Bollinger Bands
        bb = BollingerBands(close, window=self.bb_period, window_dev=self.bb_std)
        df["spy_bb_upper"] = bb.bollinger_hband()
        df["spy_bb_middle"] = bb.bollinger_mavg()
        df["spy_bb_lower"] = bb.bollinger_lband()
        df["spy_bb_width"] = bb.bollinger_wband()

        return df

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------

    def _compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price change and rate-of-change columns for each window."""
        close = df["spy_close"]
        for w in self.momentum_windows:
            df[f"spy_price_change_{w}"] = close.diff(w)
            df[f"spy_roc_{w}"] = close.pct_change(w)
        return df

    # ------------------------------------------------------------------
    # News attachment
    # ------------------------------------------------------------------

    def _attach_news(self, df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """Attach the latest news sentiment at-or-before each timestamp."""
        if news_df.empty:
            df["news_sentiment"] = np.nan
            df["news_sentiment_reasoning"] = None
            df["news_article_id"] = None
            return df

        lookback_ms = self.news_lookback_hours * 3600 * 1000
        news_cols = news_df[["timestamp", "sentiment", "sentiment_reasoning", "article_id"]].copy()
        news_cols = news_cols.rename(
            columns={
                "sentiment": "news_sentiment",
                "sentiment_reasoning": "news_sentiment_reasoning",
                "article_id": "news_article_id",
            }
        )

        merged = pd.merge_asof(
            df.sort_values("timestamp"),
            news_cols.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=lookback_ms,
        )
        return merged

    # ------------------------------------------------------------------
    # Flatten to per-option-per-minute
    # ------------------------------------------------------------------

    def _flatten_to_per_option(
        self,
        spy_enriched: pd.DataFrame,
        options_1m: pd.DataFrame,
        contracts: List[Dict],
    ) -> pd.DataFrame:
        """Flatten enriched SPY data × options to one row per option per minute.

        If no options or contracts are available, returns SPY-only rows
        with null option columns (one row per minute).
        """
        # Build contract metadata map: ticker → {strike, expiration, type}
        contract_map = {}
        for c in contracts:
            ticker = c.get("ticker", "")
            contract_map[ticker] = {
                "strike_price": float(c.get("strike_price", 0)),
                "expiration_date": c.get("expiration_date", ""),
                "contract_type": c.get("contract_type", "").lower(),
            }

        if options_1m.empty or not contracts:
            # SPY-only: one row per minute, null option columns
            df = spy_enriched.copy()
            df["ticker"] = None
            df["contract_type"] = None
            df["strike_price"] = np.nan
            df["time_to_expiry_days"] = np.nan
            df["option_avg_price"] = np.nan
            return df

        # Add contract metadata to options
        opts = options_1m.copy()

        # Filter to only tickers present in contracts
        opts = opts[opts["ticker"].isin(contract_map)].copy()
        if opts.empty:
            df = spy_enriched.copy()
            df["ticker"] = None
            df["contract_type"] = None
            df["strike_price"] = np.nan
            df["time_to_expiry_days"] = np.nan
            df["option_avg_price"] = np.nan
            return df

        opts["strike_price"] = opts["ticker"].map(lambda t: contract_map[t]["strike_price"])
        opts["contract_type"] = opts["ticker"].map(lambda t: contract_map[t]["contract_type"])
        opts["expiration_date"] = opts["ticker"].map(lambda t: contract_map[t]["expiration_date"])

        # Compute time to expiry for each row
        def _compute_tte(row):
            try:
                ts_dt = datetime.fromtimestamp(row["timestamp"] / 1000, tz=timezone.utc)
                exp_dt = datetime.strptime(row["expiration_date"], "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                return (exp_dt - ts_dt).total_seconds() / 86400
            except (ValueError, TypeError):
                return np.nan

        opts["time_to_expiry_days"] = opts.apply(_compute_tte, axis=1)

        # Select option columns for the join
        opt_cols = opts[["timestamp", "ticker", "contract_type", "strike_price",
                         "time_to_expiry_days", "option_avg_price"]]

        # Inner join on timestamp
        df = spy_enriched.merge(opt_cols, on="timestamp", how="inner")

        return df.sort_values(["timestamp", "ticker"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Greeks (flat per-row)
    # ------------------------------------------------------------------

    def _compute_greeks_flat(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Black-Scholes Greeks as flat scalar columns per row.

        Rows without a ticker (SPY-only) or with missing data get NaN.
        """
        greek_cols = ["delta", "gamma", "theta", "vega", "rho", "implied_volatility"]

        if not _VOLLIB_AVAILABLE:
            logger.warning("py_vollib not installed — skipping Greeks")
            for col in greek_cols:
                df[col] = np.nan
            return df

        deltas = []
        gammas = []
        thetas = []
        vegas = []
        rhos = []
        ivs = []

        for _, row in df.iterrows():
            ticker = row.get("ticker")
            if ticker is None or pd.isna(row.get("strike_price")) or pd.isna(row.get("spy_close")):
                deltas.append(np.nan)
                gammas.append(np.nan)
                thetas.append(np.nan)
                vegas.append(np.nan)
                rhos.append(np.nan)
                ivs.append(np.nan)
                continue

            tte_days = row.get("time_to_expiry_days", 0)
            if pd.isna(tte_days) or tte_days < self.min_tte_days:
                deltas.append(np.nan)
                gammas.append(np.nan)
                thetas.append(np.nan)
                vegas.append(np.nan)
                rhos.append(np.nan)
                ivs.append(np.nan)
                continue

            spy_price = float(row["spy_close"])
            strike = float(row["strike_price"])
            tte_years = tte_days / 365.0
            ctype = row.get("contract_type", "")
            flag = "c" if ctype in ("call", "c") else "p"
            opt_price = float(row.get("option_avg_price", 0))

            # Implied volatility
            iv = self._calc_iv(opt_price, spy_price, strike, tte_years, flag)

            # Greeks
            try:
                d = bs_delta(flag, spy_price, strike, tte_years, self.risk_free_rate, iv)
                g = bs_gamma(flag, spy_price, strike, tte_years, self.risk_free_rate, iv)
                t = bs_theta(flag, spy_price, strike, tte_years, self.risk_free_rate, iv)
                v = bs_vega(flag, spy_price, strike, tte_years, self.risk_free_rate, iv)
                r = bs_rho(flag, spy_price, strike, tte_years, self.risk_free_rate, iv)
            except Exception:
                d = g = t = v = r = np.nan

            deltas.append(d)
            gammas.append(g)
            thetas.append(t)
            vegas.append(v)
            rhos.append(r)
            ivs.append(iv)

        df["delta"] = deltas
        df["gamma"] = gammas
        df["theta"] = thetas
        df["vega"] = vegas
        df["rho"] = rhos
        df["implied_volatility"] = ivs
        return df

    def _calc_iv(
        self, price: float, S: float, K: float, t: float, flag: str
    ) -> float:
        """Calculate implied volatility with fallback."""
        if price <= 0 or S <= 0 or K <= 0 or t <= 0:
            return self.fallback_iv
        try:
            iv = bs_iv(price, S, K, t, self.risk_free_rate, flag)
            if iv < self.min_iv or iv > self.max_iv:
                return self.fallback_iv
            return iv
        except Exception:
            return self.fallback_iv

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _write_output(self, df: pd.DataFrame, date: str) -> Path:
        """Write the consolidated DataFrame to Parquet."""
        self.consolidated_path.mkdir(parents=True, exist_ok=True)
        path = self.consolidated_path / f"{date}.parquet"
        df.to_parquet(
            path,
            engine="pyarrow",
            compression=self.compression,
            index=False,
        )
        logger.info(f"Wrote consolidated data to {path} ({len(df)} rows)")
        return path
