# © 2026 Pallab Basu Roy. All rights reserved.

"""Unit tests for MicrostructureFeatureEngineer."""

import numpy as np
import pandas as pd
import pytest

from src.processing.microstructure_features import MicrostructureFeatureEngineer


@pytest.fixture
def micro():
    return MicrostructureFeatureEngineer()


def _make_spy_df(n=30):
    """Synthetic SPY DataFrame with required columns."""
    rng = np.random.RandomState(42)
    base = 580.0
    prices = base + np.cumsum(rng.randn(n) * 0.1)
    return pd.DataFrame({
        "spy_open": prices - 0.05,
        "spy_high": prices + rng.uniform(0.1, 0.3, n),
        "spy_low": prices - rng.uniform(0.1, 0.3, n),
        "spy_close": prices,
        "spy_volume": rng.randint(5000, 50000, n).astype(float),
        "spy_transactions": rng.randint(50, 500, n).astype(float),
    })


def _make_option_df(n=30):
    """Synthetic options DataFrame with required columns."""
    rng = np.random.RandomState(42)
    base = 4.0
    prices = base + np.cumsum(rng.randn(n) * 0.05)
    return pd.DataFrame({
        "open": prices - 0.02,
        "high": prices + rng.uniform(0.05, 0.15, n),
        "low": prices - rng.uniform(0.05, 0.15, n),
        "close": prices,
        "volume": rng.randint(10, 500, n).astype(float),
        "transactions": rng.randint(1, 50, n).astype(float),
    })


class TestComputeSpyMicrostructure:
    """Tests for compute_spy_microstructure."""

    def test_adds_six_columns(self, micro):
        df = _make_spy_df()
        result = micro.compute_spy_microstructure(df)
        expected = {
            "spy_tx_intensity_5m", "spy_vol_per_tx", "spy_vol_per_tx_zscore",
            "spy_intrabar_vol", "spy_intrabar_vol_ratio", "spy_tick_direction",
        }
        assert expected.issubset(set(result.columns))

    def test_does_not_modify_input(self, micro):
        df = _make_spy_df()
        orig_cols = set(df.columns)
        micro.compute_spy_microstructure(df)
        assert set(df.columns) == orig_cols

    def test_tx_intensity_around_one_mean(self, micro):
        df = _make_spy_df(50)
        result = micro.compute_spy_microstructure(df)
        # Mean intensity should be near 1.0 for random data
        mean_intensity = result["spy_tx_intensity_5m"].dropna().mean()
        assert 0.5 < mean_intensity < 2.0

    def test_intrabar_vol_positive(self, micro):
        df = _make_spy_df()
        result = micro.compute_spy_microstructure(df)
        # Intrabar vol should be positive where high > low
        valid = result["spy_intrabar_vol"].dropna()
        assert (valid >= 0).all()

    def test_tick_direction_values(self, micro):
        df = _make_spy_df()
        result = micro.compute_spy_microstructure(df)
        vals = set(result["spy_tick_direction"].unique())
        assert vals.issubset({-1.0, 0.0, 1.0})

    def test_vol_per_tx_zscore_centered(self, micro):
        df = _make_spy_df(60)
        result = micro.compute_spy_microstructure(df)
        z = result["spy_vol_per_tx_zscore"].dropna()
        # Z-scores should be roughly centered on 0
        assert abs(z.mean()) < 2.0

    def test_handles_zero_transactions(self, micro):
        df = _make_spy_df()
        df.loc[5, "spy_transactions"] = 0
        result = micro.compute_spy_microstructure(df)
        # Should produce NaN for zero-transaction rows, not error
        assert np.isnan(result.loc[5, "spy_vol_per_tx"])

    def test_handles_missing_columns(self, micro):
        df = pd.DataFrame({"spy_close": [580, 581, 582]})
        result = micro.compute_spy_microstructure(df)
        assert "spy_tx_intensity_5m" in result.columns
        assert result["spy_tx_intensity_5m"].isna().all()


class TestComputeOptionMicrostructure:
    """Tests for compute_option_microstructure."""

    def test_adds_six_columns(self, micro):
        df = _make_option_df()
        result = micro.compute_option_microstructure(df)
        expected = {
            "opt_tx_intensity_5m", "opt_vol_per_tx", "opt_vol_per_tx_zscore",
            "opt_intrabar_vol", "opt_intrabar_vol_ratio", "opt_tick_direction",
        }
        assert expected.issubset(set(result.columns))

    def test_does_not_modify_input(self, micro):
        df = _make_option_df()
        orig_cols = set(df.columns)
        micro.compute_option_microstructure(df)
        assert set(df.columns) == orig_cols

    def test_intrabar_vol_positive(self, micro):
        df = _make_option_df()
        result = micro.compute_option_microstructure(df)
        valid = result["opt_intrabar_vol"].dropna()
        assert (valid >= 0).all()

    def test_tick_direction_values(self, micro):
        df = _make_option_df()
        result = micro.compute_option_microstructure(df)
        vals = set(result["opt_tick_direction"].unique())
        assert vals.issubset({-1.0, 0.0, 1.0})

    def test_handles_zero_transactions(self, micro):
        df = _make_option_df()
        df.loc[5, "transactions"] = 0
        result = micro.compute_option_microstructure(df)
        assert np.isnan(result.loc[5, "opt_vol_per_tx"])

    def test_short_dataframe(self, micro):
        df = _make_option_df(3)
        result = micro.compute_option_microstructure(df)
        assert len(result) == 3
        assert "opt_tx_intensity_5m" in result.columns

    def test_single_row(self, micro):
        df = _make_option_df(1)
        result = micro.compute_option_microstructure(df)
        assert len(result) == 1
