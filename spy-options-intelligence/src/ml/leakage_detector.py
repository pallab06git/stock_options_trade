# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Comprehensive data leakage detection for ML trading models.

Tests
-----
1. **Random-data test** — strongest indicator. Feed pure Gaussian noise through
   the model. A clean model must be uncertain (p < 0.80) on noise; if it fires
   ≥5 high-confidence signals per 1 000 samples, something is wrong.

2. **Source-code audit** — scan ml_feature_engineer.py for backward-shift
   abuse (.shift(-n)), total-day aggregates used per-bar, and target columns
   referenced in feature code.

3. **Known lookahead features** — check whether any column in the deployed
   feature set is a known future-leaking feature (e.g. opt_vol_pct_cumday
   divides by total-day volume, which is only known after market close).

4. **Target-in-features** — explicit check that no target/label column
   (target, max_gain_120m, min_loss_120m, time_to_max_min) appears in the
   feature list used by the model.

5. **Temporal ordering** — verify the feature DataFrame is sorted strictly
   ascending by timestamp (no future rows mixed in).

6. **Train / test contamination** — confirm zero date overlap between the
   training window and the test window, and that the gap ≥ 0 days.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Known features that use future data
# ---------------------------------------------------------------------------

#: Features that require data beyond bar-time T, making them unavailable
#: for live inference at T.
_KNOWN_LOOKAHEAD_FEATURES: Dict[str, str] = {
    "opt_vol_pct_cumday": (
        "Divides running cumulative volume by total-day volume "
        "(opt_vol.sum()), which is only known after market close. "
        "At bar T this denominator includes all future bars on the day."
    ),
}

#: Label/target columns that must never appear as model inputs.
_TARGET_COLS: frozenset = frozenset(
    {"target", "max_gain_120m", "min_loss_120m", "max_gain_pct", "time_to_max_min"}
)

#: Regex to flag features whose names imply end-of-day or cumulative-day quantities.
_CUMDAY_PATTERN: re.Pattern = re.compile(
    r"cumday|cumulative|day_total|pct_day", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_non_code(source: str) -> str:
    """Return only executable lines from Python source.

    Removes:
    * Module/function/class docstrings (triple-quoted strings)
    * Inline comments (# …)

    This prevents the regex-based audit from matching pattern names that
    appear only in documentation rather than in actual computation logic.
    """
    # Remove triple-quoted strings (docstrings) first
    source = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    source = re.sub(r"'''.*?'''", "", source, flags=re.DOTALL)
    # Remove inline comments
    lines = []
    for line in source.splitlines():
        stripped = line.split("#")[0]  # drop everything after first #
        lines.append(stripped)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LeakageDetector
# ---------------------------------------------------------------------------


class LeakageDetector:
    """Detect various forms of data leakage in the ML training pipeline."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results: Dict[str, Any] = {}
        np.random.seed(random_state)

    # ------------------------------------------------------------------
    # Test 1: Random data (strongest test)
    # ------------------------------------------------------------------

    def test_on_random_data(
        self,
        model,
        feature_cols: Optional[List[str]] = None,
        n_samples: int = 1_000,
        high_confidence_threshold: float = 0.80,
        max_acceptable_signals: int = 5,
    ) -> Dict[str, Any]:
        """Feed pure Gaussian noise through the model and inspect confidence.

        A properly trained model should be *uncertain* on random data — it
        learned real patterns, not memorised noise. If the model fires many
        high-confidence signals on noise, the features carry implicit future
        information (the model essentially memorised the label distribution).

        Args:
            model: Trained sklearn-compatible classifier (must have
                   ``predict_proba``).
            feature_cols: Column names (for display only). If ``None`` the
                          number of features is read from
                          ``model.n_features_in_``.
            n_samples: Number of random rows to generate.
            high_confidence_threshold: Probability cutoff that defines
                                        "high confidence" (default 0.80).
            max_acceptable_signals: How many high-confidence predictions on
                                     noise are still considered acceptable
                                     (default 5 per 1 000 samples).

        Returns:
            Dict with keys: ``leakage_detected``, ``high_confidence_count``,
            ``medium_confidence_count``, ``low_confidence_count``,
            ``avg_confidence``, ``max_confidence``, ``percentile_95``,
            ``verdict``, ``explanation``.
        """
        if feature_cols is not None:
            n_features = len(feature_cols)
        elif hasattr(model, "n_features_in_"):
            n_features = model.n_features_in_
        else:
            n_features = 66  # fallback to our known feature count

        X_random = np.random.randn(n_samples, n_features).astype(np.float32)

        try:
            proba = model.predict_proba(X_random)[:, 1]
        except Exception as exc:
            result: Dict[str, Any] = {
                "leakage_detected": None,
                "error": str(exc),
                "verdict": "ERROR",
                "explanation": f"Could not run predict_proba: {exc}",
            }
            self.results["random_data_test"] = result
            return result

        high   = int((proba >= high_confidence_threshold).sum())
        medium = int(((proba >= 0.60) & (proba < high_confidence_threshold)).sum())
        low    = int((proba < 0.60).sum())

        avg_conf   = float(proba.mean())
        max_conf   = float(proba.max())
        p95        = float(np.percentile(proba, 95))

        leakage_detected = high > max_acceptable_signals

        if leakage_detected:
            verdict = "LEAKAGE DETECTED"
            explanation = (
                f"Model generated {high} signals ≥{high_confidence_threshold:.0%} "
                f"confidence on {n_samples} RANDOM samples "
                f"(threshold: >{max_acceptable_signals}). "
                "A clean model must be uncertain on Gaussian noise."
            )
        else:
            verdict = "PASS"
            explanation = (
                f"Model generated only {high} high-confidence signal(s) on "
                f"{n_samples} random samples (threshold ≤{max_acceptable_signals}). "
                "Model is uncertain on noise — consistent with no leakage."
            )

        result = {
            "leakage_detected": leakage_detected,
            "n_samples": n_samples,
            "n_features": n_features,
            "high_confidence_count": high,
            "medium_confidence_count": medium,
            "low_confidence_count": low,
            "avg_confidence": avg_conf,
            "max_confidence": max_conf,
            "percentile_95": p95,
            "high_confidence_threshold": high_confidence_threshold,
            "max_acceptable_signals": max_acceptable_signals,
            "verdict": verdict,
            "explanation": explanation,
        }
        self.results["random_data_test"] = result
        return result

    # ------------------------------------------------------------------
    # Test 2: Source-code audit
    # ------------------------------------------------------------------

    def audit_feature_definitions(
        self,
        feature_engineer_path: str = "src/processing/ml_feature_engineer.py",
    ) -> Dict[str, Any]:
        """Scan the feature engineering source for lookahead patterns.

        Checks
        ------
        * ``.shift(-n)`` — negative shift reads future rows
        * ``pct_change(periods=-n)`` — negative period reads future rows
        * ``opt_vol.sum()`` used as per-bar denominator (total-day lookahead)
        * ``max_gain_120m`` / ``min_loss_120m`` referenced outside
          ``_compute_targets`` (would be target leakage)

        Returns dict with ``suspicious_patterns`` (list), ``leakage_likely``
        (bool), ``pattern_count`` (int).
        """
        try:
            code = Path(feature_engineer_path).read_text()
        except FileNotFoundError:
            result: Dict[str, Any] = {
                "error": f"File not found: {feature_engineer_path}",
                "leakage_likely": False,
                "suspicious_patterns": [],
                "pattern_count": 0,
            }
            self.results["feature_audit"] = result
            return result

        # Strip docstrings and comments before pattern matching so we only
        # scan actual executable code lines, not documentation.
        executable_lines = _strip_non_code(code)

        patterns: List[Dict[str, str]] = []

        # ── Negative shift ────────────────────────────────────────────
        neg_shifts = re.findall(r"\.shift\(-\d+\)", executable_lines)
        if neg_shifts:
            patterns.append({
                "pattern": f".shift(-n) [{', '.join(set(neg_shifts))}]",
                "severity": "HIGH",
                "description": "Negative shift reads data from future rows.",
                "recommendation": "Replace with positive shift for lookback.",
            })

        # ── Negative pct_change ───────────────────────────────────────
        if re.search(r"pct_change\s*\(\s*(?:periods\s*=\s*)?-\d+", executable_lines):
            patterns.append({
                "pattern": "pct_change(periods=-n)",
                "severity": "HIGH",
                "description": "Negative periods reads future rows.",
                "recommendation": "Use positive periods for backward change.",
            })

        # ── Total-day aggregate used as per-bar denominator ───────────
        if re.search(r"day_vol_total\s*=.*\.sum\(\)", executable_lines):
            patterns.append({
                "pattern": "day_vol_total = opt_vol.sum() → opt_vol_pct_cumday",
                "severity": "MEDIUM",
                "description": (
                    "Total day volume (opt_vol.sum()) is used to normalise the "
                    "per-bar cumulative volume feature. At bar T, total day "
                    "volume includes all future bars — unavailable in live trading."
                ),
                "recommendation": (
                    "Replace with a rolling or fixed estimate of daily volume, "
                    "or drop opt_vol_pct_cumday from the feature set."
                ),
            })

        # ── Target columns assigned OUTSIDE _compute_targets ─────────
        # Only flag executable assignments, not docstring mentions.
        in_compute_targets = False
        for line in executable_lines.splitlines():
            stripped = line.strip()
            if "def _compute_targets" in stripped:
                in_compute_targets = True
            elif stripped.startswith("def ") and in_compute_targets:
                in_compute_targets = False

            if in_compute_targets:
                continue  # target columns are expected here

            for tc in _TARGET_COLS:
                # Match assignment patterns: df["target"] = or df['target'] =
                if re.search(rf"""df\[['\"]{tc}['\"]]\s*=""", stripped):
                    patterns.append({
                        "pattern": f"df['{tc}'] assigned outside _compute_targets",
                        "severity": "CRITICAL",
                        "description": (
                            f"Target column '{tc}' is assigned in executable code "
                            "outside the dedicated target computation block."
                        ),
                        "recommendation": "Ensure target columns are only set in _compute_targets.",
                    })

        leakage_likely = any(
            p["severity"] in ("HIGH", "CRITICAL") for p in patterns
        )

        result = {
            "suspicious_patterns": patterns,
            "leakage_likely": leakage_likely,
            "pattern_count": len(patterns),
        }
        self.results["feature_audit"] = result
        return result

    # ------------------------------------------------------------------
    # Test 3: Known lookahead features
    # ------------------------------------------------------------------

    def check_known_lookahead_features(
        self, feature_cols: List[str]
    ) -> Dict[str, Any]:
        """Check if any deployed feature is a known future-leaking column.

        Args:
            feature_cols: The feature list stored in the model artifact.

        Returns:
            Dict with ``lookahead_features`` (list of {name, reason}) and
            ``leakage_detected`` (bool).
        """
        found = [
            {"feature": f, "reason": _KNOWN_LOOKAHEAD_FEATURES[f]}
            for f in feature_cols
            if f in _KNOWN_LOOKAHEAD_FEATURES
        ]

        leakage_detected = len(found) > 0
        result = {
            "leakage_detected": leakage_detected,
            "lookahead_features": found,
            "features_checked": len(feature_cols),
        }
        self.results["known_lookahead_check"] = result
        return result

    # ------------------------------------------------------------------
    # Test 4: Target columns in feature set
    # ------------------------------------------------------------------

    def verify_target_not_in_features(
        self, feature_cols: List[str]
    ) -> Dict[str, Any]:
        """Verify no target/label column is included in the model's feature set.

        Args:
            feature_cols: List of feature column names used by the model.

        Returns:
            Dict with ``leakage_detected`` (bool) and ``contaminated_cols``
            (list of offending column names).
        """
        contaminated = [c for c in feature_cols if c in _TARGET_COLS]
        leakage_detected = len(contaminated) > 0

        result = {
            "leakage_detected": leakage_detected,
            "contaminated_cols": contaminated,
            "target_cols_checked": sorted(_TARGET_COLS),
            "features_inspected": len(feature_cols),
        }
        self.results["target_in_features"] = result
        return result

    # ------------------------------------------------------------------
    # Test 5: Temporal ordering
    # ------------------------------------------------------------------

    def verify_temporal_ordering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Confirm the feature DataFrame is sorted strictly ascending by timestamp.

        A violation (timestamp at row i > timestamp at row i+1) would mean
        future bars are interleaved with past bars, which could corrupt
        rolling / lag features computed on the combined frame.

        Args:
            df: Feature DataFrame (must contain a ``timestamp`` column).

        Returns:
            Dict with ``ordering_valid`` (bool), ``violations`` (int), and
            ``total_rows`` (int).
        """
        if "timestamp" not in df.columns:
            result: Dict[str, Any] = {
                "ordering_valid": None,
                "error": "No 'timestamp' column in DataFrame",
                "violations": 0,
                "total_rows": len(df),
            }
            self.results["temporal_ordering"] = result
            return result

        ts = df["timestamp"].values
        diffs = np.diff(ts.astype(np.int64))
        violations = int((diffs < 0).sum())

        result = {
            "ordering_valid": violations == 0,
            "violations": violations,
            "total_rows": len(df),
        }
        self.results["temporal_ordering"] = result
        return result

    # ------------------------------------------------------------------
    # Test 6: Train / test contamination
    # ------------------------------------------------------------------

    def detect_train_test_contamination(
        self,
        train_dates: List[str],
        test_dates: List[str],
    ) -> Dict[str, Any]:
        """Check for date overlap between train and test sets.

        Args:
            train_dates: List of YYYY-MM-DD strings used for training.
            test_dates:  List of YYYY-MM-DD strings used for testing.

        Returns:
            Dict with ``contamination_detected`` (bool), ``overlap_dates``
            (list), ``gap_days`` (int), train/test start-end dates.
        """
        train_set = set(train_dates)
        test_set  = set(test_dates)
        overlap   = sorted(train_set & test_set)

        train_max = max(train_dates)
        test_min  = min(test_dates)
        gap_days  = (
            pd.to_datetime(test_min) - pd.to_datetime(train_max)
        ).days

        contamination_detected = len(overlap) > 0 or gap_days < 0

        result = {
            "contamination_detected": contamination_detected,
            "overlap_dates": overlap,
            "overlap_count": len(overlap),
            "gap_days": gap_days,
            "train_start": min(train_dates),
            "train_end": train_max,
            "test_start": test_min,
            "test_end": max(test_dates),
        }
        self.results["contamination_check"] = result
        return result

    # ------------------------------------------------------------------
    # Test 7: 120-minute specific leakage (correlation + name patterns)
    # ------------------------------------------------------------------

    def check_120min_specific_leaks(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        high_corr_threshold: float = 0.40,
        mod_corr_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """Detect features suspicious for 120-minute forward leakage.

        Two complementary checks:

        1. **Name pattern** — flags features whose names imply end-of-day or
           cumulative-day quantities (``cumday``, ``pct_day``, etc.).

        2. **Pearson correlation** — flags features with |corr| above thresholds
           against ``max_gain_120m`` (or ``target_sustained`` / ``target`` if
           the primary outcome column is absent).

        High correlation alone does not prove leakage (momentum features
        legitimately correlate with future moves), but values ≥ 0.40 are
        unusual enough to warrant manual review.

        Args:
            df: DataFrame containing both feature columns and outcome columns.
            feature_cols: Feature column names to audit.
            high_corr_threshold: |corr| ≥ this → mark as HIGH suspicion.
            mod_corr_threshold:  |corr| ≥ this → mark as MODERATE suspicion.

        Returns:
            Dict with ``suspicious_by_name``, ``suspicious_by_correlation``,
            ``correlation_table_top20``, ``outcome_column_used``, and
            ``leakage_suspected`` (bool).
        """
        # ── Name-pattern check ─────────────────────────────────────────
        suspicious_by_name = [
            {
                "feature": f,
                "reason": "Name implies cumulative / end-of-day aggregate",
            }
            for f in feature_cols
            if _CUMDAY_PATTERN.search(f)
        ]

        # ── Correlation check ──────────────────────────────────────────
        outcome_col = None
        for candidate in ("max_gain_120m", "target_sustained", "target"):
            if candidate in df.columns:
                outcome_col = candidate
                break

        corr_records: List[Dict[str, Any]] = []
        suspicious_by_corr: List[Dict[str, Any]] = []

        if outcome_col is not None:
            outcome_series = df[outcome_col].astype(float)
            for f in feature_cols:
                if f not in df.columns:
                    continue
                try:
                    corr_val = float(df[f].astype(float).corr(outcome_series))
                except Exception:
                    corr_val = float("nan")
                abs_corr = abs(corr_val) if not np.isnan(corr_val) else 0.0
                entry: Dict[str, Any] = {
                    "feature": f,
                    "correlation": round(corr_val, 4),
                    "abs_corr": round(abs_corr, 4),
                }
                corr_records.append(entry)

                if abs_corr >= high_corr_threshold:
                    suspicious_by_corr.append({**entry, "severity": "HIGH"})
                elif abs_corr >= mod_corr_threshold:
                    suspicious_by_corr.append({**entry, "severity": "MODERATE"})

            corr_records.sort(key=lambda x: x["abs_corr"], reverse=True)

        leakage_suspected = len(suspicious_by_name) > 0 or any(
            r.get("severity") == "HIGH" for r in suspicious_by_corr
        )

        result: Dict[str, Any] = {
            "leakage_suspected": leakage_suspected,
            "outcome_column_used": outcome_col,
            "suspicious_by_name": suspicious_by_name,
            "suspicious_by_correlation": suspicious_by_corr,
            "correlation_table_top20": corr_records[:20],
            "high_corr_threshold": high_corr_threshold,
            "mod_corr_threshold": mod_corr_threshold,
        }
        self.results["correlation_leakage_check"] = result
        return result

    # ------------------------------------------------------------------
    # Test 8: Feature importance analysis with red-flag detection
    # ------------------------------------------------------------------

    def analyze_feature_importance(
        self,
        model,
        feature_cols: List[str],
        top_n: int = 20,
        dominance_threshold: float = 0.15,
    ) -> Dict[str, Any]:
        """Rank features by model importance and flag suspicious ones.

        Red flags:

        * A known lookahead feature appears in the ranked importances.
        * A feature with a cumulative-day name pattern appears in the top 10.
        * A single feature claims ≥ ``dominance_threshold`` of total importance
          (suggests the model over-relies on one signal — possible leakage).

        Args:
            model: Trained sklearn-compatible tree model (must expose
                   ``feature_importances_``).
            feature_cols: Feature names in the exact order used during training.
            top_n: How many top features to include in the ranked list.
            dominance_threshold: Importance share ≥ this triggers a WARNING.

        Returns:
            Dict with ``ranked_features`` (list), ``red_flags`` (list),
            ``leakage_suspected`` (bool), ``top_feature``, ``total_features``.
        """
        if not hasattr(model, "feature_importances_"):
            result: Dict[str, Any] = {
                "leakage_suspected": False,
                "error": "Model does not expose feature_importances_",
                "red_flags": [],
                "ranked_features": [],
                "top_feature": None,
                "total_features": len(feature_cols),
            }
            self.results["feature_importance_analysis"] = result
            return result

        importances = np.array(model.feature_importances_, dtype=float)
        total = importances.sum()
        if total > 0:
            importances = importances / total

        indexed = sorted(
            enumerate(importances), key=lambda x: x[1], reverse=True
        )

        ranked: List[Dict[str, Any]] = []
        for rank, (idx, imp) in enumerate(indexed[:top_n], start=1):
            fname = (
                feature_cols[idx] if idx < len(feature_cols) else f"feature_{idx}"
            )
            is_known_leak = fname in _KNOWN_LOOKAHEAD_FEATURES
            is_cumday = bool(_CUMDAY_PATTERN.search(fname))
            is_dominant = imp >= dominance_threshold

            if is_known_leak:
                flag = "CRITICAL - known lookahead"
            elif is_cumday and rank <= 10:
                flag = "WARNING - cumday pattern"
            elif is_dominant:
                flag = "WARNING - dominant feature"
            else:
                flag = "OK"

            ranked.append(
                {
                    "rank": rank,
                    "feature": fname,
                    "importance": round(float(imp), 5),
                    "importance_pct": f"{imp * 100:.2f}%",
                    "is_known_leak": is_known_leak,
                    "is_cumday": is_cumday,
                    "flag": flag,
                }
            )

        red_flags: List[str] = []
        for entry in ranked:
            if entry["is_known_leak"]:
                red_flags.append(
                    f"CRITICAL: '{entry['feature']}' is a known lookahead feature "
                    f"(rank #{entry['rank']}, {entry['importance_pct']} importance)"
                )
            elif "WARNING - cumday" in entry["flag"]:
                red_flags.append(
                    f"WARNING: '{entry['feature']}' has end-of-day name pattern "
                    f"in top 10 (rank #{entry['rank']}, {entry['importance_pct']})"
                )
            elif "WARNING - dominant" in entry["flag"]:
                red_flags.append(
                    f"WARNING: '{entry['feature']}' dominates with "
                    f"{entry['importance_pct']} of total importance "
                    f"(rank #{entry['rank']})"
                )

        leakage_suspected = any("CRITICAL" in f for f in red_flags)

        result = {
            "leakage_suspected": leakage_suspected,
            "red_flags": red_flags,
            "ranked_features": ranked,
            "top_feature": ranked[0]["feature"] if ranked else None,
            "total_features": len(feature_cols),
            "dominance_threshold": dominance_threshold,
        }
        self.results["feature_importance_analysis"] = result
        return result

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(
        self, output_path: str = "data/reports/leakage_detection/leakage_report.json"
    ) -> Dict[str, Any]:
        """Compile all test results into a single JSON report.

        Args:
            output_path: Path where the JSON report is written.

        Returns:
            Report dict with ``overall_verdict``, ``safe_to_proceed``,
            ``critical_issues``, and ``test_results``.
        """
        critical_issues: List[str] = []

        # Random data test
        rdt = self.results.get("random_data_test", {})
        if rdt.get("leakage_detected") is True:
            critical_issues.append(
                f"Random-data test: model fired {rdt.get('high_confidence_count', '?')} "
                f"high-confidence signals on noise "
                f"(max acceptable: {rdt.get('max_acceptable_signals', 5)})"
            )

        # Feature audit
        fa = self.results.get("feature_audit", {})
        if fa.get("leakage_likely"):
            for p in fa.get("suspicious_patterns", []):
                if p.get("severity") in ("HIGH", "CRITICAL"):
                    critical_issues.append(
                        f"Feature audit [{p['severity']}]: {p['pattern']}"
                    )

        # Known lookahead
        kl = self.results.get("known_lookahead_check", {})
        if kl.get("leakage_detected"):
            for f in kl.get("lookahead_features", []):
                critical_issues.append(
                    f"Lookahead feature in model: '{f['feature']}' — {f['reason']}"
                )

        # Target in features
        tf = self.results.get("target_in_features", {})
        if tf.get("leakage_detected"):
            critical_issues.append(
                f"Target columns in feature set: {tf.get('contaminated_cols')}"
            )

        # Contamination
        cc = self.results.get("contamination_check", {})
        if cc.get("contamination_detected"):
            critical_issues.append(
                f"Train/test contamination: {cc.get('overlap_count', 0)} overlapping dates, "
                f"gap={cc.get('gap_days', '?')} days"
            )

        # Correlation leakage check (test 7)
        cl = self.results.get("correlation_leakage_check", {})
        if cl.get("leakage_suspected"):
            for item in cl.get("suspicious_by_name", []):
                critical_issues.append(
                    f"Cumday name pattern in features: '{item['feature']}'"
                )
            for item in cl.get("suspicious_by_correlation", []):
                if item.get("severity") == "HIGH":
                    critical_issues.append(
                        f"High correlation [{item.get('abs_corr', '?'):.3f}] "
                        f"with outcome: '{item['feature']}'"
                    )

        # Feature importance analysis (test 8)
        fi = self.results.get("feature_importance_analysis", {})
        if fi.get("leakage_suspected"):
            for flag in fi.get("red_flags", []):
                if "CRITICAL" in flag:
                    critical_issues.append(flag)

        overall_leakage = len(critical_issues) > 0

        # Temporal ordering is a WARNING, not a blocker by itself
        to = self.results.get("temporal_ordering", {})
        warnings: List[str] = []
        if to.get("ordering_valid") is False:
            warnings.append(
                f"Temporal ordering: {to.get('violations', '?')} ordering violations"
            )

        report: Dict[str, Any] = {
            "overall_verdict": "LEAKAGE DETECTED" if overall_leakage else "NO LEAKAGE",
            "safe_to_proceed": not overall_leakage,
            "critical_issues": critical_issues,
            "warnings": warnings,
            "test_results": self.results,
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(report, fh, indent=2, default=str)

        return report
