# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Microstructure feature engineering for SPY and options data.

Computes transaction-intensity, volume-per-transaction, and intrabar
volatility features that capture market microstructure dynamics.  These
features help the model distinguish between genuine momentum moves and
low-participation noise.

Feature groups
--------------
  SPY microstructure (6):
    spy_tx_intensity_5m     — transactions / 5-bar rolling mean(transactions)
    spy_vol_per_tx          — volume / transactions
    spy_vol_per_tx_zscore   — z-score of vol_per_tx vs 30-bar MA
    spy_intrabar_vol        — (high - low) / close * 100  (percent)
    spy_intrabar_vol_ratio  — intrabar_vol / 5-bar rolling mean
    spy_tick_direction       — sign(close - open)  (+1 up, -1 down, 0 doji)

  Option microstructure (6):
    opt_tx_intensity_5m, opt_vol_per_tx, opt_vol_per_tx_zscore,
    opt_intrabar_vol, opt_intrabar_vol_ratio, opt_tick_direction

Usage
-----
    from src.processing.microstructure_features import MicrostructureFeatureEngineer

    micro = MicrostructureFeatureEngineer()
    spy_df = micro.compute_spy_microstructure(spy_df)
    merged_df = micro.compute_option_microstructure(merged_df)
"""

import numpy as np
import pandas as pd


class MicrostructureFeatureEngineer:
    """Compute microstructure features for SPY and option bar data."""

    def compute_spy_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 6 SPY microstructure columns to the DataFrame.

        Expects columns: spy_transactions, spy_volume, spy_high, spy_low,
        spy_close, spy_open.

        Args:
            df: SPY feature DataFrame (from _compute_spy_features).

        Returns:
            DataFrame with 6 additional spy_* columns.
        """
        df = df.copy()

        tx = df.get("spy_transactions", pd.Series(np.nan, index=df.index))
        vol = df.get("spy_volume", pd.Series(np.nan, index=df.index))
        high = df.get("spy_high", pd.Series(np.nan, index=df.index))
        low = df.get("spy_low", pd.Series(np.nan, index=df.index))
        close = df.get("spy_close", pd.Series(np.nan, index=df.index))
        opn = df.get("spy_open", pd.Series(np.nan, index=df.index))

        # Transaction intensity: current transactions / 5-bar rolling mean
        tx_ma5 = tx.rolling(5, min_periods=1).mean().replace(0, np.nan)
        df["spy_tx_intensity_5m"] = tx / tx_ma5

        # Volume per transaction
        tx_safe = tx.replace(0, np.nan)
        vol_per_tx = vol / tx_safe
        df["spy_vol_per_tx"] = vol_per_tx

        # Z-score of volume-per-transaction vs 30-bar moving average
        vpt_ma30 = vol_per_tx.rolling(30, min_periods=5).mean()
        vpt_std30 = vol_per_tx.rolling(30, min_periods=5).std().replace(0, np.nan)
        df["spy_vol_per_tx_zscore"] = (vol_per_tx - vpt_ma30) / vpt_std30

        # Intrabar volatility: (high - low) / close as percentage
        close_safe = close.replace(0, np.nan)
        intrabar = (high - low) / close_safe * 100
        df["spy_intrabar_vol"] = intrabar

        # Intrabar volatility ratio vs 5-bar rolling mean
        intrabar_ma5 = intrabar.rolling(5, min_periods=1).mean().replace(0, np.nan)
        df["spy_intrabar_vol_ratio"] = intrabar / intrabar_ma5

        # Tick direction: sign of (close - open)
        df["spy_tick_direction"] = np.sign(close - opn).fillna(0)

        return df

    def compute_option_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 6 option microstructure columns to the merged DataFrame.

        Expects columns: transactions, volume, high, low, close, open
        (raw option OHLCV from the merge step, before spy_ prefix).

        Args:
            df: Merged option + SPY feature DataFrame.

        Returns:
            DataFrame with 6 additional opt_* columns.
        """
        df = df.copy()

        tx = df.get("transactions", pd.Series(np.nan, index=df.index))
        vol = df.get("volume", pd.Series(np.nan, index=df.index))
        high = df.get("high", pd.Series(np.nan, index=df.index))
        low = df.get("low", pd.Series(np.nan, index=df.index))
        close = df.get("close", pd.Series(np.nan, index=df.index))
        opn = df.get("open", pd.Series(np.nan, index=df.index))

        # Transaction intensity: current transactions / 5-bar rolling mean
        tx_ma5 = tx.rolling(5, min_periods=1).mean().replace(0, np.nan)
        df["opt_tx_intensity_5m"] = tx / tx_ma5

        # Volume per transaction
        tx_safe = tx.replace(0, np.nan)
        vol_per_tx = vol / tx_safe
        df["opt_vol_per_tx"] = vol_per_tx

        # Z-score of volume-per-transaction vs 30-bar moving average
        vpt_ma30 = vol_per_tx.rolling(30, min_periods=5).mean()
        vpt_std30 = vol_per_tx.rolling(30, min_periods=5).std().replace(0, np.nan)
        df["opt_vol_per_tx_zscore"] = (vol_per_tx - vpt_ma30) / vpt_std30

        # Intrabar volatility: (high - low) / close as percentage
        close_safe = close.replace(0, np.nan)
        intrabar = (high - low) / close_safe * 100
        df["opt_intrabar_vol"] = intrabar

        # Intrabar volatility ratio vs 5-bar rolling mean
        intrabar_ma5 = intrabar.rolling(5, min_periods=1).mean().replace(0, np.nan)
        df["opt_intrabar_vol_ratio"] = intrabar / intrabar_ma5

        # Tick direction: sign of (close - open)
        df["opt_tick_direction"] = np.sign(close - opn).fillna(0)

        return df
