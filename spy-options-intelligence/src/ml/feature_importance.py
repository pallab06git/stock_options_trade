# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Feature importance analysis for trained XGBoost models.

This module loads a saved model artifact (from ``train_xgboost.py``),
extracts per-feature importance scores from the XGBoost booster, and
produces ranked importance reports.

Importance types
----------------
XGBoost computes importance in five ways.  All are supported here:

  ``weight``       Number of times a feature appears in a split across all
                   trees.  Fast but biased toward high-cardinality features.

  ``gain``         Average *gain* (reduction in loss) contributed by each
                   split that uses the feature.  Best overall quality signal.
                   **Default.**

  ``cover``        Average number of training samples affected by splits on
                   the feature.

  ``total_gain``   Sum of gain across all splits using the feature.

  ``total_cover``  Sum of cover across all splits using the feature.

Feature naming
--------------
When ``XGBClassifier`` is trained on a NumPy array (rather than a named
DataFrame) XGBoost internally labels features ``f0``, ``f1``, …, ``fN-1``.
The saved artifact stores the ordered ``feature_cols`` list so that the
mapping ``f{i}`` → ``feature_cols[i]`` can be reconstructed at analysis time.
Features that never appear in any split receive a score of 0.0.

Output schema
-------------
``extract_importances`` returns a DataFrame with columns:

  ``feature``         Feature name (string).
  ``importance``      Raw importance score (non-negative float).
  ``importance_pct``  Fraction of total importance (sums to 1.0 when
                      at least one feature has non-zero importance).
  ``rank``            Integer rank (1 = most important).

Module-level function
---------------------
  ``extract_importances(model, feature_cols, importance_type)``
    → DataFrame (all features, sorted by importance DESC)

``FeatureImportanceAnalyzer`` class (config-driven)
----------------------------------------------------
  ``analyze(model_path, output_dir)``  → DataFrame (load + extract + save)
  ``get_top_n(df, n)``                 → DataFrame (top-N slice)
  ``save_report(df, model_version)``   → Path to saved CSV
  ``plot_summary(df, top_n)``          → ASCII bar chart string

Configuration keys (all under ``ml_training.feature_importance``, optional)
---------------------------------------------------------------------------
  importance_type  str  "gain"
  top_n            int  20
  output_dir       str  "data/reports/feature_importance"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import xgboost as xgb

from src.utils.logger import get_logger

logger = get_logger()

# Importance types accepted by XGBoost's get_score()
_VALID_IMPORTANCE_TYPES: frozenset = frozenset(
    {"weight", "gain", "cover", "total_gain", "total_cover"}
)


# ---------------------------------------------------------------------------
# Module-level function (primary API)
# ---------------------------------------------------------------------------


def extract_importances(
    model: xgb.XGBClassifier,
    feature_cols: List[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Extract per-feature importance scores from a trained XGBClassifier.

    Features that do not appear in any split receive a score of 0.0 and
    are included in the returned DataFrame so that the result always has
    exactly ``len(feature_cols)`` rows.

    Args:
        model:           Trained ``XGBClassifier`` instance.
        feature_cols:    Ordered list of feature names used during training.
                         Index ``i`` maps to internal name ``f{i}``.
        importance_type: One of ``{"weight", "gain", "cover",
                         "total_gain", "total_cover"}``.  Default ``"gain"``.

    Returns:
        DataFrame with columns ``["feature", "importance",
        "importance_pct", "rank"]``, sorted by ``importance`` descending.

    Raises:
        ValueError: If ``importance_type`` is not one of the supported values.
        ValueError: If ``feature_cols`` is empty.
    """
    if importance_type not in _VALID_IMPORTANCE_TYPES:
        raise ValueError(
            f"extract_importances: unsupported importance_type {importance_type!r}. "
            f"Choose one of {sorted(_VALID_IMPORTANCE_TYPES)}"
        )
    if not feature_cols:
        raise ValueError("extract_importances: feature_cols must not be empty")

    booster = model.get_booster()
    # get_score returns {f0: score, f1: score, ...} — only features used in splits
    raw_scores: Dict[str, float] = booster.get_score(importance_type=importance_type)

    # Map internal names (f0, f1, …) → actual feature names
    named_scores: Dict[str, float] = {}
    for key, val in raw_scores.items():
        if key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
            if idx < len(feature_cols):
                named_scores[feature_cols[idx]] = float(val)
        else:
            # Feature names already readable (model trained on DataFrame)
            named_scores[key] = float(val)

    # Build full score dict — unseen features get 0.0
    full_scores = {col: named_scores.get(col, 0.0) for col in feature_cols}

    rows = [{"feature": col, "importance": score} for col, score in full_scores.items()]
    df = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)

    total = df["importance"].sum()
    df["importance_pct"] = (df["importance"] / total) if total > 0 else 0.0
    df["rank"] = df.index + 1

    logger.info(
        f"extract_importances ({importance_type}): "
        f"{len(df)} features | "
        f"top feature: {df.iloc[0]['feature']!r} ({df.iloc[0]['importance']:.4f})"
    )
    return df[["feature", "importance", "importance_pct", "rank"]]


# ---------------------------------------------------------------------------
# FeatureImportanceAnalyzer class (config-driven)
# ---------------------------------------------------------------------------


class FeatureImportanceAnalyzer:
    """Config-driven feature importance analysis for saved XGBoost artifacts.

    Loads a model artifact produced by ``XGBoostTrainer``, extracts
    feature importances, saves a CSV report, and can render an ASCII
    bar chart for quick terminal inspection.

    Usage::

        analyzer = FeatureImportanceAnalyzer(config)
        df = analyzer.analyze("models/xgboost_v1.pkl")
        print(analyzer.plot_summary(df, top_n=15))
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Merged application config dict.  Settings are read from
                    ``config["ml_training"]["feature_importance"]``.
        """
        fi_cfg = config.get("ml_training", {}).get("feature_importance", {})
        self.importance_type: str = fi_cfg.get("importance_type", "gain")
        self.top_n: int = int(fi_cfg.get("top_n", 20))
        self.output_dir: str = fi_cfg.get(
            "output_dir", "data/reports/feature_importance"
        )

        if self.importance_type not in _VALID_IMPORTANCE_TYPES:
            raise ValueError(
                f"FeatureImportanceAnalyzer: unknown importance_type "
                f"{self.importance_type!r}. "
                f"Choose one of {sorted(_VALID_IMPORTANCE_TYPES)}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        model_path: str | Path,
        output_dir: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """Load model artifact, extract importances, save report.

        Args:
            model_path: Path to a ``.pkl`` artifact produced by
                        ``XGBoostTrainer._save_model``.
            output_dir: Override the configured output directory.

        Returns:
            Full importance DataFrame (all features, sorted by importance
            DESC) as returned by ``extract_importances``.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
            KeyError: If the artifact is missing ``model`` or
                      ``feature_cols`` keys.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"FeatureImportanceAnalyzer: model artifact not found: {model_path}"
            )

        artifact = joblib.load(model_path)

        if "model" not in artifact:
            raise KeyError(
                f"FeatureImportanceAnalyzer: artifact at {model_path} is missing "
                f"'model' key.  Got keys: {list(artifact.keys())}"
            )
        if "feature_cols" not in artifact:
            raise KeyError(
                f"FeatureImportanceAnalyzer: artifact at {model_path} is missing "
                f"'feature_cols' key.  Got keys: {list(artifact.keys())}"
            )

        model: xgb.XGBClassifier = artifact["model"]
        feature_cols: List[str] = artifact["feature_cols"]

        # Infer model version from file stem, e.g. "xgboost_v1"
        model_version = model_path.stem  # "xgboost_v1"

        df = extract_importances(model, feature_cols, self.importance_type)

        out_dir = Path(output_dir) if output_dir is not None else Path(self.output_dir)
        self.save_report(df, model_version, out_dir)

        return df

    def get_top_n(
        self,
        df: pd.DataFrame,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return the top-N most important features.

        Args:
            df: Importance DataFrame as returned by ``analyze`` or
                ``extract_importances`` (sorted by importance DESC).
            n:  Number of features to return.  Defaults to ``self.top_n``.

        Returns:
            DataFrame slice with the ``n`` highest-importance features.
            If ``n`` exceeds the number of rows the full DataFrame is returned.
        """
        n = n if n is not None else self.top_n
        return df.head(n).reset_index(drop=True)

    def save_report(
        self,
        df: pd.DataFrame,
        model_version: str,
        output_dir: Optional[str | Path] = None,
    ) -> Path:
        """Save the importance DataFrame as a CSV file.

        File name: ``{model_version}_{importance_type}_importance.csv``

        Args:
            df:            Full importance DataFrame.
            model_version: Version string embedded in the file name
                           (e.g. ``"xgboost_v1"``).
            output_dir:    Directory to write the CSV.  Defaults to
                           ``self.output_dir``.  Created if absent.

        Returns:
            Path to the written CSV file.
        """
        out_dir = Path(output_dir) if output_dir is not None else Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{model_version}_{self.importance_type}_importance.csv"
        df.to_csv(path, index=False)
        logger.info(f"Feature importance report saved → {path}")
        return path

    def plot_summary(
        self,
        df: pd.DataFrame,
        top_n: Optional[int] = None,
    ) -> str:
        """Render an ASCII horizontal bar chart of the top-N features.

        Bars are proportional to ``importance_pct``.  The chart is
        returned as a plain string (no matplotlib / external dependencies).

        Args:
            df:    Importance DataFrame sorted by importance DESC.
            top_n: Number of features to show.  Defaults to ``self.top_n``.

        Returns:
            Multi-line string ready to be printed to the terminal.

        Example output::

            Feature Importance (gain) — top 5
            ──────────────────────────────────
            opt_return_1m    ████████████████░░░░  32.15%
            spy_rsi_14       ███████░░░░░░░░░░░░░  14.23%
            moneyness        █████░░░░░░░░░░░░░░░  10.01%
            implied_vol      ████░░░░░░░░░░░░░░░░   8.88%
            minutes_since_op ████░░░░░░░░░░░░░░░░   8.12%
        """
        top_n = top_n if top_n is not None else self.top_n
        top = self.get_top_n(df, top_n)

        bar_width = 20  # total bar characters
        max_name = max((len(r["feature"]) for _, r in top.iterrows()), default=10)
        max_name = min(max_name, 20)  # cap display width

        title = f"Feature Importance ({self.importance_type}) — top {len(top)}"
        divider = "─" * (max_name + bar_width + 12)

        lines = [title, divider]
        for _, row in top.iterrows():
            name = str(row["feature"])[:max_name].ljust(max_name)
            pct = float(row["importance_pct"])
            filled = int(round(pct * bar_width))
            empty = bar_width - filled
            bar = "█" * filled + "░" * empty
            lines.append(f"{name}  {bar}  {pct * 100:5.2f}%")

        return "\n".join(lines)
