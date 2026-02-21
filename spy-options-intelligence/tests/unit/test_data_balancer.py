# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for data_balancer module."""

import numpy as np
import pandas as pd
import pytest

from src.ml.data_balancer import (
    DataBalancer,
    _check_target_col,
    calculate_class_weights,
    undersample_majority,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_imbalanced_df(
    n_positive: int = 100,
    n_negative: int = 9900,
    random_state: int = 0,
) -> pd.DataFrame:
    """Return a DataFrame with `n_positive` rows of class 1 and `n_negative` of class 0."""
    rng = np.random.default_rng(random_state)
    labels = np.array([1] * n_positive + [0] * n_negative, dtype=np.int8)
    values = rng.standard_normal(len(labels))
    df = pd.DataFrame({"feature": values, "target": labels})
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def _make_config(
    balance_method: str = "undersample",
    random_state: int = 42,
    target_col: str = "target",
) -> dict:
    return {
        "data_preparation": {
            "balance_method": balance_method,
            "random_state": random_state,
            "target_col": target_col,
        }
    }


# ---------------------------------------------------------------------------
# TestCheckTargetCol
# ---------------------------------------------------------------------------


class TestCheckTargetCol:
    def test_passes_when_column_present(self):
        df = pd.DataFrame({"target": [0, 1]})
        _check_target_col(df, "target")  # should not raise

    def test_raises_when_column_missing(self):
        df = pd.DataFrame({"feature": [0, 1]})
        with pytest.raises(ValueError, match="target"):
            _check_target_col(df, "target")


# ---------------------------------------------------------------------------
# TestUndersampleMajority
# ---------------------------------------------------------------------------


class TestUndersampleMajority:
    def test_output_size_is_twice_minority(self):
        df = _make_imbalanced_df(n_positive=100, n_negative=9900)
        result = undersample_majority(df)
        # minority = 100, so balanced = 200
        assert len(result) == 200

    def test_output_classes_are_equal(self):
        df = _make_imbalanced_df(n_positive=100, n_negative=9900)
        result = undersample_majority(df)
        counts = result["target"].value_counts()
        assert counts[0] == counts[1]

    def test_minority_class_fully_preserved(self):
        df = _make_imbalanced_df(n_positive=50, n_negative=5000)
        result = undersample_majority(df)
        # All 50 positive rows should be in the output
        assert (result["target"] == 1).sum() == 50

    def test_random_state_reproducible(self):
        df = _make_imbalanced_df(n_positive=100, n_negative=9900)
        r1 = undersample_majority(df, random_state=7)
        r2 = undersample_majority(df, random_state=7)
        pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))

    def test_different_seeds_produce_different_results(self):
        df = _make_imbalanced_df(n_positive=100, n_negative=9900)
        r1 = undersample_majority(df, random_state=1)
        r2 = undersample_majority(df, random_state=2)
        # Not exactly equal (different majority samples drawn)
        assert not r1["feature"].tolist() == r2["feature"].tolist()

    def test_already_balanced_returns_shuffled(self):
        df = _make_imbalanced_df(n_positive=200, n_negative=200)
        result = undersample_majority(df)
        assert len(result) == 400
        counts = result["target"].value_counts()
        assert counts[0] == counts[1] == 200

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["feature", "target"])
        result = undersample_majority(df)
        assert result.empty

    def test_raises_on_missing_target_col(self):
        df = pd.DataFrame({"feature": [1, 2, 3]})
        with pytest.raises(ValueError, match="target"):
            undersample_majority(df)

    def test_single_class_returns_unchanged_shuffled(self):
        df = pd.DataFrame({"feature": [1.0, 2.0, 3.0], "target": [0, 0, 0]})
        result = undersample_majority(df)
        assert len(result) == 3
        assert set(result["target"]) == {0}

    def test_custom_target_col_name(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4], "label": [1, 1, 0, 0]})
        result = undersample_majority(df, target_col="label")
        assert len(result) == 4

    def test_original_df_not_mutated(self):
        df = _make_imbalanced_df(n_positive=50, n_negative=500)
        original_len = len(df)
        _ = undersample_majority(df)
        assert len(df) == original_len

    def test_index_reset_to_zero_based(self):
        df = _make_imbalanced_df(n_positive=50, n_negative=500)
        result = undersample_majority(df)
        assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# TestCalculateClassWeights
# ---------------------------------------------------------------------------


class TestCalculateClassWeights:
    def test_returns_dict_with_both_classes(self):
        df = _make_imbalanced_df(n_positive=100, n_negative=900)
        weights = calculate_class_weights(df)
        assert 0 in weights
        assert 1 in weights

    def test_positive_class_weight_greater_than_negative(self):
        """Minority class (positive) must always get a higher weight."""
        df = _make_imbalanced_df(n_positive=100, n_negative=900)
        weights = calculate_class_weights(df)
        assert weights[1] > weights[0]

    def test_weights_multiply_to_n_samples(self):
        """For balanced formula: sum(count_i * weight_i) == n_total."""
        df = _make_imbalanced_df(n_positive=100, n_negative=900)
        weights = calculate_class_weights(df)
        counts = df["target"].value_counts()
        total = sum(counts[cls] * weights[cls] for cls in counts.index)
        # Should equal n_total (1000)
        assert abs(total - len(df)) < 1e-6

    def test_equal_classes_get_equal_weights(self):
        df = pd.DataFrame({"target": [0] * 500 + [1] * 500})
        weights = calculate_class_weights(df)
        assert abs(weights[0] - weights[1]) < 1e-9

    def test_known_imbalance_values(self):
        """Manual verification of the balanced weight formula.

        n_total=1000, n_neg=990, n_pos=10
        weight_1 = 1000 / (2 * 10)  = 50.0
        weight_0 = 1000 / (2 * 990) ≈ 0.5051
        """
        df = pd.DataFrame({"target": [0] * 990 + [1] * 10})
        weights = calculate_class_weights(df)
        assert abs(weights[1] - 50.0) < 1e-6
        assert abs(weights[0] - (1000 / (2 * 990))) < 1e-6

    def test_raises_on_empty_df(self):
        df = pd.DataFrame(columns=["target"])
        with pytest.raises(ValueError, match="empty"):
            calculate_class_weights(df)

    def test_raises_on_missing_target_col(self):
        df = pd.DataFrame({"feature": [1, 2]})
        with pytest.raises(ValueError, match="target"):
            calculate_class_weights(df)

    def test_custom_target_col_name(self):
        df = pd.DataFrame({"label": [0, 0, 0, 1]})
        weights = calculate_class_weights(df, target_col="label")
        assert 0 in weights and 1 in weights


# ---------------------------------------------------------------------------
# TestDataBalancer
# ---------------------------------------------------------------------------


class TestDataBalancer:
    def test_default_config_undersample(self):
        balancer = DataBalancer({})
        assert balancer.balance_method == "undersample"
        assert balancer.target_col == "target"
        assert balancer.random_state == 42

    def test_config_overrides_applied(self):
        balancer = DataBalancer(_make_config("class_weights", random_state=99))
        assert balancer.balance_method == "class_weights"
        assert balancer.random_state == 99

    def test_invalid_balance_method_raises(self):
        with pytest.raises(ValueError, match="unknown balance_method"):
            DataBalancer(_make_config("smote"))

    def test_balance_undersample_strategy(self):
        df = _make_imbalanced_df(n_positive=100, n_negative=9900)
        balancer = DataBalancer(_make_config("undersample"))
        result = balancer.balance(df)
        assert len(result) == 200
        counts = result["target"].value_counts()
        assert counts[0] == counts[1]

    def test_balance_class_weights_strategy_returns_unchanged(self):
        df = _make_imbalanced_df(n_positive=100, n_negative=900)
        balancer = DataBalancer(_make_config("class_weights"))
        result = balancer.balance(df)
        assert len(result) == len(df)

    def test_get_class_weights(self):
        df = _make_imbalanced_df(n_positive=100, n_negative=900)
        balancer = DataBalancer(_make_config())
        weights = balancer.get_class_weights(df)
        assert weights[1] > weights[0]

    def test_get_summary_keys(self):
        df = _make_imbalanced_df(n_positive=100, n_negative=9900)
        balancer = DataBalancer(_make_config())
        stats = balancer.get_summary(df)
        for key in ["n_total", "n_positive", "n_negative",
                    "positive_rate", "imbalance_ratio",
                    "class_weights", "balance_method"]:
            assert key in stats, f"Missing key: {key}"

    def test_get_summary_values_correct(self):
        df = _make_imbalanced_df(n_positive=100, n_negative=900)
        balancer = DataBalancer(_make_config())
        stats = balancer.get_summary(df)
        assert stats["n_total"] == 1000
        assert stats["n_positive"] == 100
        assert stats["n_negative"] == 900
        assert abs(stats["positive_rate"] - 0.10) < 1e-9
        assert abs(stats["imbalance_ratio"] - 9.0) < 1e-9

    def test_get_summary_raises_on_missing_target_col(self):
        df = pd.DataFrame({"feature": [1, 2, 3]})
        balancer = DataBalancer(_make_config())
        with pytest.raises(ValueError, match="target"):
            balancer.get_summary(df)
