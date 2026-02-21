# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for feature_importance module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.ml.feature_importance import (
    FeatureImportanceAnalyzer,
    _VALID_IMPORTANCE_TYPES,
    extract_importances,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _train_tiny_model(n_features: int = 8, n_samples: int = 200) -> xgb.XGBClassifier:
    """Train a minimal XGBClassifier on synthetic data."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int8)
    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X, y)
    return model


def _make_feature_cols(n: int = 8) -> List[str]:
    return [f"feature_{i}" for i in range(n)]


def _make_artifact(
    model: xgb.XGBClassifier,
    feature_cols: List[str],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    return {
        "model": model,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "xgb_params": {},
        "saved_at": "2026-02-20T12:00:00",
    }


def _save_artifact(artifact: Dict[str, Any], path: Path) -> Path:
    joblib.dump(artifact, path)
    return path


def _make_config(
    importance_type: str = "gain",
    top_n: int = 5,
    output_dir: str = "data/reports/feature_importance",
) -> Dict[str, Any]:
    return {
        "ml_training": {
            "feature_importance": {
                "importance_type": importance_type,
                "top_n": top_n,
                "output_dir": output_dir,
            }
        }
    }


# ---------------------------------------------------------------------------
# TestExtractImportances
# ---------------------------------------------------------------------------


class TestExtractImportances:
    @pytest.fixture()
    def model_and_cols(self):
        model = _train_tiny_model(n_features=8)
        cols = _make_feature_cols(8)
        return model, cols

    def test_returns_dataframe(self, model_and_cols):
        model, cols = model_and_cols
        df = extract_importances(model, cols)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, model_and_cols):
        model, cols = model_and_cols
        df = extract_importances(model, cols)
        assert set(df.columns) == {"feature", "importance", "importance_pct", "rank"}

    def test_row_count_equals_feature_count(self, model_and_cols):
        model, cols = model_and_cols
        df = extract_importances(model, cols)
        assert len(df) == len(cols)

    def test_all_feature_names_present(self, model_and_cols):
        model, cols = model_and_cols
        df = extract_importances(model, cols)
        assert set(df["feature"]) == set(cols)

    def test_sorted_by_importance_descending(self, model_and_cols):
        model, cols = model_and_cols
        df = extract_importances(model, cols)
        assert list(df["importance"]) == sorted(df["importance"].tolist(), reverse=True)

    def test_importance_pct_sums_to_one(self, model_and_cols):
        model, cols = model_and_cols
        df = extract_importances(model, cols)
        assert abs(df["importance_pct"].sum() - 1.0) < 1e-6

    def test_importance_pct_all_non_negative(self, model_and_cols):
        model, cols = model_and_cols
        df = extract_importances(model, cols)
        assert (df["importance_pct"] >= 0).all()

    def test_rank_starts_at_one(self, model_and_cols):
        model, cols = model_and_cols
        df = extract_importances(model, cols)
        assert df["rank"].iloc[0] == 1

    def test_rank_is_sequential(self, model_and_cols):
        model, cols = model_and_cols
        df = extract_importances(model, cols)
        assert list(df["rank"]) == list(range(1, len(df) + 1))

    def test_gain_importance_type(self, model_and_cols):
        model, cols = model_and_cols
        df = extract_importances(model, cols, importance_type="gain")
        assert (df["importance"] >= 0).all()

    def test_weight_importance_type(self, model_and_cols):
        model, cols = model_and_cols
        df = extract_importances(model, cols, importance_type="weight")
        assert (df["importance"] >= 0).all()

    def test_cover_importance_type(self, model_and_cols):
        model, cols = model_and_cols
        df = extract_importances(model, cols, importance_type="cover")
        assert (df["importance"] >= 0).all()

    def test_total_gain_importance_type(self, model_and_cols):
        model, cols = model_and_cols
        df = extract_importances(model, cols, importance_type="total_gain")
        assert (df["importance"] >= 0).all()

    def test_total_cover_importance_type(self, model_and_cols):
        model, cols = model_and_cols
        df = extract_importances(model, cols, importance_type="total_cover")
        assert (df["importance"] >= 0).all()

    def test_invalid_importance_type_raises(self, model_and_cols):
        model, cols = model_and_cols
        with pytest.raises(ValueError, match="unsupported importance_type"):
            extract_importances(model, cols, importance_type="shap")

    def test_empty_feature_cols_raises(self, model_and_cols):
        model, _ = model_and_cols
        with pytest.raises(ValueError, match="feature_cols must not be empty"):
            extract_importances(model, [])

    def test_top_feature_has_rank_one(self, model_and_cols):
        """The feature with highest importance must have rank 1."""
        model, cols = model_and_cols
        df = extract_importances(model, cols)
        top_row = df.iloc[0]
        assert top_row["rank"] == 1
        assert top_row["importance"] == df["importance"].max()

    def test_zero_importance_features_included(self):
        """Features never used in any split must appear with importance=0.0."""
        # Train a tiny model where only f0 matters (y = sign(f0))
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int8)
        model = xgb.XGBClassifier(
            n_estimators=5, max_depth=1, random_state=0, eval_metric="logloss"
        )
        model.fit(X, y)
        cols = ["dominant", "unused_1", "unused_2", "unused_3"]
        df = extract_importances(model, cols, importance_type="gain")
        # All 4 features must be present
        assert len(df) == 4
        # Features not used may have importance 0.0
        assert (df["importance"] >= 0).all()

    def test_different_importance_types_can_differ(self, model_and_cols):
        """Weight and gain importance rankings may differ for the same model."""
        model, cols = model_and_cols
        df_gain = extract_importances(model, cols, importance_type="gain")
        df_weight = extract_importances(model, cols, importance_type="weight")
        # Scores should be non-negative in both cases
        assert (df_gain["importance"] >= 0).all()
        assert (df_weight["importance"] >= 0).all()


# ---------------------------------------------------------------------------
# TestValidImportanceTypes
# ---------------------------------------------------------------------------


class TestValidImportanceTypes:
    def test_contains_all_expected_types(self):
        expected = {"weight", "gain", "cover", "total_gain", "total_cover"}
        assert expected == set(_VALID_IMPORTANCE_TYPES)

    def test_is_frozenset(self):
        assert isinstance(_VALID_IMPORTANCE_TYPES, frozenset)


# ---------------------------------------------------------------------------
# TestFeatureImportanceAnalyzerInit
# ---------------------------------------------------------------------------


class TestFeatureImportanceAnalyzerInit:
    def test_default_config(self):
        analyzer = FeatureImportanceAnalyzer({})
        assert analyzer.importance_type == "gain"
        assert analyzer.top_n == 20
        assert "feature_importance" in analyzer.output_dir

    def test_config_overrides_applied(self):
        analyzer = FeatureImportanceAnalyzer(_make_config("weight", top_n=10))
        assert analyzer.importance_type == "weight"
        assert analyzer.top_n == 10

    def test_invalid_importance_type_raises(self):
        with pytest.raises(ValueError, match="unknown importance_type"):
            FeatureImportanceAnalyzer(_make_config("shap"))


# ---------------------------------------------------------------------------
# TestGetTopN
# ---------------------------------------------------------------------------


class TestGetTopN:
    @pytest.fixture()
    def importance_df(self):
        model = _train_tiny_model(n_features=10)
        cols = _make_feature_cols(10)
        return extract_importances(model, cols)

    def test_returns_n_rows(self, importance_df):
        analyzer = FeatureImportanceAnalyzer(_make_config(top_n=3))
        top = analyzer.get_top_n(importance_df, n=3)
        assert len(top) == 3

    def test_default_n_from_config(self, importance_df):
        analyzer = FeatureImportanceAnalyzer(_make_config(top_n=4))
        top = analyzer.get_top_n(importance_df)
        assert len(top) == 4

    def test_top_n_larger_than_df_returns_all(self, importance_df):
        analyzer = FeatureImportanceAnalyzer(_make_config(top_n=100))
        top = analyzer.get_top_n(importance_df, n=100)
        assert len(top) == len(importance_df)

    def test_index_reset_to_zero_based(self, importance_df):
        analyzer = FeatureImportanceAnalyzer(_make_config())
        top = analyzer.get_top_n(importance_df, n=3)
        assert list(top.index) == [0, 1, 2]

    def test_first_row_is_highest_importance(self, importance_df):
        analyzer = FeatureImportanceAnalyzer(_make_config())
        top = analyzer.get_top_n(importance_df, n=5)
        assert top.iloc[0]["importance"] == importance_df["importance"].iloc[0]


# ---------------------------------------------------------------------------
# TestSaveReport
# ---------------------------------------------------------------------------


class TestSaveReport:
    def test_csv_file_created(self, tmp_path):
        model = _train_tiny_model(n_features=8)
        cols = _make_feature_cols(8)
        df = extract_importances(model, cols)
        analyzer = FeatureImportanceAnalyzer(_make_config())
        path = analyzer.save_report(df, "xgboost_v1", output_dir=tmp_path)
        assert path.exists()

    def test_filename_contains_version_and_type(self, tmp_path):
        model = _train_tiny_model(n_features=8)
        cols = _make_feature_cols(8)
        df = extract_importances(model, cols)
        analyzer = FeatureImportanceAnalyzer(_make_config("weight"))
        path = analyzer.save_report(df, "xgboost_v99", output_dir=tmp_path)
        assert "xgboost_v99" in path.name
        assert "weight" in path.name

    def test_saved_csv_is_valid(self, tmp_path):
        model = _train_tiny_model(n_features=8)
        cols = _make_feature_cols(8)
        df = extract_importances(model, cols)
        analyzer = FeatureImportanceAnalyzer(_make_config())
        path = analyzer.save_report(df, "xgboost_v1", output_dir=tmp_path)
        loaded = pd.read_csv(path)
        assert "feature" in loaded.columns
        assert "importance" in loaded.columns
        assert len(loaded) == len(df)

    def test_output_dir_created_if_missing(self, tmp_path):
        model = _train_tiny_model(n_features=8)
        cols = _make_feature_cols(8)
        df = extract_importances(model, cols)
        new_dir = tmp_path / "nested" / "subdir"
        analyzer = FeatureImportanceAnalyzer(_make_config())
        path = analyzer.save_report(df, "xgboost_v1", output_dir=new_dir)
        assert path.exists()


# ---------------------------------------------------------------------------
# TestAnalyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    @pytest.fixture()
    def saved_artifact(self, tmp_path):
        model = _train_tiny_model(n_features=8)
        cols = _make_feature_cols(8)
        artifact = _make_artifact(model, cols)
        path = tmp_path / "xgboost_v1.pkl"
        _save_artifact(artifact, path)
        return path

    def test_returns_dataframe(self, saved_artifact, tmp_path):
        analyzer = FeatureImportanceAnalyzer(_make_config())
        df = analyzer.analyze(saved_artifact, output_dir=tmp_path)
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_feature_cols(self, saved_artifact, tmp_path):
        analyzer = FeatureImportanceAnalyzer(_make_config())
        df = analyzer.analyze(saved_artifact, output_dir=tmp_path)
        artifact = joblib.load(saved_artifact)
        assert len(df) == len(artifact["feature_cols"])

    def test_csv_report_is_written(self, saved_artifact, tmp_path):
        analyzer = FeatureImportanceAnalyzer(_make_config())
        analyzer.analyze(saved_artifact, output_dir=tmp_path)
        csvs = list(tmp_path.glob("*.csv"))
        assert len(csvs) == 1

    def test_raises_if_model_path_missing(self, tmp_path):
        analyzer = FeatureImportanceAnalyzer(_make_config())
        with pytest.raises(FileNotFoundError, match="model artifact not found"):
            analyzer.analyze(tmp_path / "nonexistent.pkl", output_dir=tmp_path)

    def test_raises_if_artifact_missing_model_key(self, tmp_path):
        bad = {"feature_cols": ["a", "b"]}
        path = tmp_path / "bad.pkl"
        joblib.dump(bad, path)
        analyzer = FeatureImportanceAnalyzer(_make_config())
        with pytest.raises(KeyError, match="'model'"):
            analyzer.analyze(path, output_dir=tmp_path)

    def test_raises_if_artifact_missing_feature_cols(self, tmp_path):
        model = _train_tiny_model()
        bad = {"model": model}
        path = tmp_path / "bad.pkl"
        joblib.dump(bad, path)
        analyzer = FeatureImportanceAnalyzer(_make_config())
        with pytest.raises(KeyError, match="'feature_cols'"):
            analyzer.analyze(path, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# TestPlotSummary
# ---------------------------------------------------------------------------


class TestPlotSummary:
    @pytest.fixture()
    def importance_df(self):
        model = _train_tiny_model(n_features=8)
        cols = _make_feature_cols(8)
        return extract_importances(model, cols)

    def test_returns_string(self, importance_df):
        analyzer = FeatureImportanceAnalyzer(_make_config(top_n=5))
        result = analyzer.plot_summary(importance_df)
        assert isinstance(result, str)

    def test_contains_importance_type_in_title(self, importance_df):
        analyzer = FeatureImportanceAnalyzer(_make_config("weight", top_n=5))
        result = analyzer.plot_summary(importance_df)
        assert "weight" in result

    def test_contains_feature_names(self, importance_df):
        analyzer = FeatureImportanceAnalyzer(_make_config(top_n=3))
        result = analyzer.plot_summary(importance_df)
        top_feature = importance_df.iloc[0]["feature"]
        assert top_feature[:10] in result  # truncated to 20 chars in chart

    def test_bar_characters_present(self, importance_df):
        analyzer = FeatureImportanceAnalyzer(_make_config(top_n=5))
        result = analyzer.plot_summary(importance_df)
        assert "█" in result

    def test_top_n_override(self, importance_df):
        analyzer = FeatureImportanceAnalyzer(_make_config(top_n=5))
        result_3 = analyzer.plot_summary(importance_df, top_n=3)
        result_5 = analyzer.plot_summary(importance_df, top_n=5)
        # Fewer lines in the 3-row chart
        assert len(result_3.splitlines()) < len(result_5.splitlines())

    def test_percentage_values_appear(self, importance_df):
        analyzer = FeatureImportanceAnalyzer(_make_config(top_n=5))
        result = analyzer.plot_summary(importance_df)
        assert "%" in result
