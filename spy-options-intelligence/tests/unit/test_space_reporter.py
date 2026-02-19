# © 2026 Pallab Basu Roy. All rights reserved.
"""Unit tests for SpaceReporter."""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from src.utils.space_reporter import SpaceReporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path):
    return {
        "pipeline_v2": {
            "reporting": {
                "reports_dir": str(tmp_path / "reports"),
            }
        }
    }


def _create_data_tree(tmp_path):
    """Create a minimal data/ directory structure with files."""
    # Patch the data root to use tmp_path
    (tmp_path / "data" / "raw" / "spy").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "vix").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "features").mkdir(parents=True)

    (tmp_path / "data" / "raw" / "spy" / "2025-03-03.parquet").write_bytes(b"x" * 1000)
    (tmp_path / "data" / "raw" / "vix" / "2025-03-03.parquet").write_bytes(b"x" * 500)
    (tmp_path / "data" / "processed" / "features" / "test.parquet").write_bytes(b"x" * 200)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCollect:
    def test_empty_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()

        reporter = SpaceReporter(_make_config(tmp_path))
        reporter.data_root = tmp_path / "data"
        result = reporter.collect()

        assert result["total_bytes"] == 0
        assert result["total_files"] == 0

    def test_counts_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _create_data_tree(tmp_path)

        reporter = SpaceReporter(_make_config(tmp_path))
        reporter.data_root = tmp_path / "data"
        result = reporter.collect()

        assert result["total_files"] == 3
        assert result["total_bytes"] == 1700

    def test_tree_keys_present(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _create_data_tree(tmp_path)

        reporter = SpaceReporter(_make_config(tmp_path))
        reporter.data_root = tmp_path / "data"
        result = reporter.collect()

        assert "tree" in result
        # At least one key should be in the tree
        assert len(result["tree"]) > 0


class TestEstimateCompression:
    def test_raises_on_missing_file(self, tmp_path):
        reporter = SpaceReporter(_make_config(tmp_path))
        with pytest.raises(FileNotFoundError):
            reporter.estimate_compression(str(tmp_path / "nonexistent.parquet"))

    def test_returns_codec_sizes(self, tmp_path):
        # Write a small Parquet file
        df = pd.DataFrame({"a": range(100), "b": [f"val_{i}" for i in range(100)]})
        p = tmp_path / "test.parquet"
        df.to_parquet(p, engine="pyarrow", compression="snappy", index=False)

        reporter = SpaceReporter(_make_config(tmp_path))
        result = reporter.estimate_compression(str(p))

        assert "snappy_mb" in result
        assert "gzip_mb" in result
        assert "zstd_mb" in result
        assert result["row_sample"] == 100
        # All sizes should be positive
        assert result["snappy_mb"] > 0


class TestGenerateReport:
    def test_writes_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()

        reporter = SpaceReporter(_make_config(tmp_path))
        reporter.data_root = tmp_path / "data"
        reporter.reports_dir = tmp_path / "reports"

        path = reporter.generate_report()
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
        assert "total_mb" in data
        assert "total_files" in data

    def test_report_filename_contains_date(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()

        reporter = SpaceReporter(_make_config(tmp_path))
        reporter.data_root = tmp_path / "data"
        reporter.reports_dir = tmp_path / "reports"

        path = reporter.generate_report()
        # Filename should be YYYY-MM-DD_space.json
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2}_space\.json$", path.name)
