# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for PurgeManager."""

import os
import time

import pytest

from src.utils.purge_manager import PurgeManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(retention=None):
    """Build a minimal config with retention settings."""
    return {"retention": retention or {}}


def _create_file(path, age_days=0, content=b"data"):
    """Create a file and backdate its mtime by age_days."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    if age_days > 0:
        old_time = time.time() - (age_days * 86400)
        os.utime(path, (old_time, old_time))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDryRun:

    def test_dry_run_reports_without_deleting(self, tmp_path, monkeypatch):
        """Dry run reports files that would be purged without deleting."""
        monkeypatch.chdir(tmp_path)

        # Create old file in data/raw/spy
        old_file = tmp_path / "data" / "raw" / "spy" / "2025-01-01.parquet"
        _create_file(old_file, age_days=10)

        pm = PurgeManager(_make_config({"raw_data": 3}))
        result = pm.purge_category("raw_data", dry_run=True)

        assert result["files_purged"] == 1
        assert result["bytes_freed"] > 0
        assert old_file.exists()  # NOT deleted


class TestPurgeOldFiles:

    def test_deletes_files_older_than_retention(self, tmp_path, monkeypatch):
        """Files older than retention are deleted."""
        monkeypatch.chdir(tmp_path)

        old_file = tmp_path / "data" / "raw" / "spy" / "2025-01-01.parquet"
        _create_file(old_file, age_days=10)

        pm = PurgeManager(_make_config({"raw_data": 3}))
        result = pm.purge_category("raw_data", dry_run=False)

        assert result["files_purged"] == 1
        assert not old_file.exists()

    def test_keeps_files_within_retention(self, tmp_path, monkeypatch):
        """Files within retention period are preserved."""
        monkeypatch.chdir(tmp_path)

        recent = tmp_path / "data" / "raw" / "spy" / "2025-02-14.parquet"
        _create_file(recent, age_days=1)

        pm = PurgeManager(_make_config({"raw_data": 3}))
        result = pm.purge_category("raw_data", dry_run=False)

        assert result["files_scanned"] == 1
        assert result["files_purged"] == 0
        assert recent.exists()


class TestPurgeAll:

    def test_aggregates_across_categories(self, tmp_path, monkeypatch):
        """purge_all sums results from all categories."""
        monkeypatch.chdir(tmp_path)

        # Old raw file
        raw = tmp_path / "data" / "raw" / "spy" / "old.parquet"
        _create_file(raw, age_days=10)

        # Old heartbeat file
        hb = tmp_path / "data" / "logs" / "heartbeat" / "status.json"
        _create_file(hb, age_days=5)

        pm = PurgeManager(_make_config({"raw_data": 3, "heartbeat": 1}))
        result = pm.purge_all(dry_run=False)

        assert result["files_purged"] == 2
        assert not raw.exists()
        assert not hb.exists()


class TestRetentionZero:

    def test_retention_zero_disables_purging(self, tmp_path, monkeypatch):
        """Retention of 0 skips the category entirely."""
        monkeypatch.chdir(tmp_path)

        old_file = tmp_path / "data" / "raw" / "spy" / "ancient.parquet"
        _create_file(old_file, age_days=100)

        pm = PurgeManager(_make_config({"raw_data": 0}))
        result = pm.purge_category("raw_data", dry_run=False)

        assert result["skipped"] is True
        assert result["files_scanned"] == 0
        assert old_file.exists()


class TestMissingDirectories:

    def test_handles_missing_directories(self, tmp_path, monkeypatch):
        """Missing data directories don't cause errors."""
        monkeypatch.chdir(tmp_path)
        # No directories created

        pm = PurgeManager(_make_config({"raw_data": 3}))
        result = pm.purge_category("raw_data", dry_run=False)

        assert result["files_scanned"] == 0
        assert result["files_purged"] == 0
        assert result["files_failed"] == 0


class TestPermissionErrors:

    def test_handles_permission_errors(self, tmp_path, monkeypatch):
        """Permission errors are counted as failed, not raised."""
        monkeypatch.chdir(tmp_path)

        old_file = tmp_path / "data" / "raw" / "spy" / "locked.parquet"
        _create_file(old_file, age_days=10)

        # Make file read-only and parent dir unwritable
        old_file.chmod(0o444)
        old_file.parent.chmod(0o555)

        pm = PurgeManager(_make_config({"raw_data": 3}))
        result = pm.purge_category("raw_data", dry_run=False)

        # Restore permissions for cleanup
        old_file.parent.chmod(0o755)
        old_file.chmod(0o644)

        assert result["files_failed"] == 1
        assert old_file.exists()


class TestCheckpointPatternFilter:

    def test_only_checkpoint_json_purged(self, tmp_path, monkeypatch):
        """Checkpoint category only matches checkpoint_*.json files."""
        monkeypatch.chdir(tmp_path)

        exec_dir = tmp_path / "data" / "logs" / "execution"
        checkpoint = exec_dir / "checkpoint_spy_2025-01-01.json"
        logfile = exec_dir / "run_2025-01-01.log"
        _create_file(checkpoint, age_days=10)
        _create_file(logfile, age_days=10)

        pm = PurgeManager(_make_config({"checkpoints": 3}))
        result = pm.purge_category("checkpoints", dry_run=False)

        assert result["files_scanned"] == 1  # only checkpoint matched
        assert result["files_purged"] == 1
        assert not checkpoint.exists()
        assert logfile.exists()  # log file untouched


class TestRetentionOverride:

    def test_cli_retention_override(self, tmp_path, monkeypatch):
        """retention_days_override takes precedence over config."""
        monkeypatch.chdir(tmp_path)

        # File is 5 days old
        f = tmp_path / "data" / "raw" / "spy" / "mid.parquet"
        _create_file(f, age_days=5)

        pm = PurgeManager(_make_config({"raw_data": 10}))  # config says 10 days

        # With config retention (10 days) — file should be kept
        result = pm.purge_category("raw_data", dry_run=True)
        assert result["files_purged"] == 0

        # With override (3 days) — file should be purged
        result = pm.purge_category("raw_data", retention_days_override=3, dry_run=True)
        assert result["files_purged"] == 1


class TestUnknownCategory:

    def test_unknown_category_returns_empty(self):
        """Unknown category returns zero counts without error."""
        pm = PurgeManager(_make_config())
        result = pm.purge_category("nonexistent")

        assert result["files_scanned"] == 0
        assert result["files_purged"] == 0


class TestMultipleDirectories:

    def test_purges_across_multiple_subdirs(self, tmp_path, monkeypatch):
        """raw_data spans spy, vix, options, news — all are scanned."""
        monkeypatch.chdir(tmp_path)

        for subdir in ["spy", "vix", "news"]:
            f = tmp_path / "data" / "raw" / subdir / "old.parquet"
            _create_file(f, age_days=10)

        pm = PurgeManager(_make_config({"raw_data": 3}))
        result = pm.purge_category("raw_data", dry_run=False)

        assert result["files_purged"] == 3


class TestDefaultRetention:

    def test_uses_defaults_when_config_empty(self, tmp_path, monkeypatch):
        """Missing retention config falls back to hardcoded defaults."""
        monkeypatch.chdir(tmp_path)

        # Heartbeat default is 1 day — a 5-day-old file should be purged
        f = tmp_path / "data" / "logs" / "heartbeat" / "old.json"
        _create_file(f, age_days=5)

        pm = PurgeManager(_make_config({}))
        result = pm.purge_category("heartbeat", dry_run=False)

        assert result["files_purged"] == 1
        assert result["retention_days"] == 1
