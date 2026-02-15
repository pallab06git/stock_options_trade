# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Data consolidation engine for multi-source SPY options intelligence.

Merges SPY aggregates, VIX, options (with Greeks), and news sentiment
into a single enriched dataset per trading day. Output is written to
date-partitioned Parquet files in data/processed/consolidated/.
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
    """Consolidate SPY, VIX, options, and news data into an enriched dataset."""

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

        Args:
            date: Trading date string (YYYY-MM-DD).

        Returns:
            Stats dict with status, row counts, and availability flags.
        """
        spy_df = self._load_spy(date)
        if spy_df.empty:
            logger.error(f"No SPY data for {date} — consolidation failed")
            return {"status": "failed", "date": date, "reason": "missing_spy_data"}

        # Rename SPY columns
        rename_map = {
            "open": "spy_open",
            "high": "spy_high",
            "low": "spy_low",
            "close": "spy_close",
            "volume": "spy_volume",
            "vwap": "spy_vwap",
            "transactions": "spy_transactions",
        }
        spy_df = spy_df.rename(columns=rename_map)

        # Load optional sources
        vix_df = self._load_vix(date)
        options_df = self._load_options(date)
        contracts = self._load_contracts(date)
        news_df = self._load_news(date)

        vix_available = not vix_df.empty
        news_available = not news_df.empty

        # Align VIX
        df = self._align_vix(spy_df, vix_df)

        # Technical indicators
        df = self._compute_indicators(df)

        # Momentum
        df = self._compute_momentum(df)

        # Greeks
        greeks_df = self._compute_greeks(options_df, df, contracts)
        if not greeks_df.empty:
            df = df.merge(greeks_df, on="timestamp", how="left")
        else:
            # Add empty list columns
            for col in [
                "option_tickers",
                "option_strikes",
                "option_types",
                "option_close_prices",
                "option_deltas",
                "option_gammas",
                "option_thetas",
                "option_vegas",
                "option_rhos",
                "option_ivs",
            ]:
                df[col] = [[] for _ in range(len(df))]

        # News
        df = self._attach_news(df, news_df)

        # Source tag
        df["source"] = "consolidated"

        # Drop original source column if carried over
        for col in ["source_x", "source_y"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        contracts_processed = len(contracts)

        # Write output
        self._write_output(df, date)

        logger.info(
            f"Consolidation complete for {date}: "
            f"{len(df)} rows, {contracts_processed} contracts, "
            f"VIX={'yes' if vix_available else 'no'}, "
            f"news={'yes' if news_available else 'no'}"
        )

        return {
            "status": "success",
            "date": date,
            "total_rows": len(df),
            "options_contracts_processed": contracts_processed,
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
    # Greeks
    # ------------------------------------------------------------------

    def _compute_greeks(
        self,
        options_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        contracts: List[Dict],
    ) -> pd.DataFrame:
        """Compute Black-Scholes Greeks for each SPY timestamp.

        Returns a DataFrame with timestamp + list columns for Greeks.
        """
        if options_df.empty or not contracts:
            return pd.DataFrame()

        if not _VOLLIB_AVAILABLE:
            logger.warning("py_vollib not installed — skipping Greeks")
            return pd.DataFrame()

        # Build contract lookup: ticker → metadata
        contract_map = {}
        for c in contracts:
            ticker = c.get("ticker", "")
            contract_map[ticker] = {
                "strike_price": float(c.get("strike_price", 0)),
                "expiration_date": c.get("expiration_date", ""),
                "contract_type": c.get("contract_type", "").lower(),
            }

        timestamps = spy_df["timestamp"].values
        spy_closes = spy_df["spy_close"].values

        results = []
        for ts, spy_price in zip(timestamps, spy_closes):
            row = self._greeks_at_timestamp(
                ts, spy_price, options_df, contract_map
            )
            results.append(row)

        return pd.DataFrame(results)

    def _greeks_at_timestamp(
        self,
        timestamp: int,
        spy_price: float,
        options_df: pd.DataFrame,
        contract_map: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """Compute Greeks for all option contracts at a single timestamp."""
        row: Dict[str, Any] = {"timestamp": timestamp}
        tickers = []
        strikes = []
        types = []
        close_prices = []
        deltas = []
        gammas = []
        thetas = []
        vegas = []
        rhos = []
        ivs = []

        # Find closest options data at or before this timestamp
        mask = options_df["timestamp"] <= timestamp
        if not mask.any():
            row.update(self._empty_greeks_lists())
            return row

        ts_options = options_df[mask].drop_duplicates(subset=["ticker"], keep="last")

        for _, opt in ts_options.iterrows():
            ticker = opt.get("ticker", "")
            if ticker not in contract_map:
                continue

            meta = contract_map[ticker]
            strike = meta["strike_price"]
            exp_str = meta["expiration_date"]
            ctype = meta["contract_type"]

            # Calculate time to expiry
            try:
                ts_dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
                exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                tte_days = (exp_dt - ts_dt).total_seconds() / 86400
            except (ValueError, TypeError):
                continue

            if tte_days < self.min_tte_days:
                continue

            tte_years = tte_days / 365.0
            flag = "c" if ctype in ("call", "c") else "p"
            opt_close = float(opt.get("close", 0))

            # Implied volatility
            iv = self._calc_iv(opt_close, spy_price, strike, tte_years, flag)

            # Greeks
            try:
                d = bs_delta(flag, spy_price, strike, tte_years, self.risk_free_rate, iv)
                g = bs_gamma(flag, spy_price, strike, tte_years, self.risk_free_rate, iv)
                t = bs_theta(flag, spy_price, strike, tte_years, self.risk_free_rate, iv)
                v = bs_vega(flag, spy_price, strike, tte_years, self.risk_free_rate, iv)
                r = bs_rho(flag, spy_price, strike, tte_years, self.risk_free_rate, iv)
            except Exception:
                d = g = t = v = r = np.nan

            tickers.append(ticker)
            strikes.append(strike)
            types.append(flag)
            close_prices.append(opt_close)
            deltas.append(d)
            gammas.append(g)
            thetas.append(t)
            vegas.append(v)
            rhos.append(r)
            ivs.append(iv)

        row["option_tickers"] = tickers
        row["option_strikes"] = strikes
        row["option_types"] = types
        row["option_close_prices"] = close_prices
        row["option_deltas"] = deltas
        row["option_gammas"] = gammas
        row["option_thetas"] = thetas
        row["option_vegas"] = vegas
        row["option_rhos"] = rhos
        row["option_ivs"] = ivs
        return row

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

    def _empty_greeks_lists(self) -> Dict[str, List]:
        return {
            "option_tickers": [],
            "option_strikes": [],
            "option_types": [],
            "option_close_prices": [],
            "option_deltas": [],
            "option_gammas": [],
            "option_thetas": [],
            "option_vegas": [],
            "option_rhos": [],
            "option_ivs": [],
        }

    # ------------------------------------------------------------------
    # News attachment
    # ------------------------------------------------------------------

    def _attach_news(self, df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """Attach the latest news sentiment at-or-before each timestamp."""
        if news_df.empty:
            df["news_sentiment"] = np.nan
            df["news_sentiment_reasoning"] = None
            df["news_article_id"] = None
            df["news_timestamp"] = np.nan
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
        news_cols["news_timestamp"] = news_cols["timestamp"]

        merged = pd.merge_asof(
            df.sort_values("timestamp"),
            news_cols.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=lookback_ms,
        )
        return merged

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
