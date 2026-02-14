# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Integration tests for HistoricalRunner with live Polygon data.

Runs the full pipeline (fetch → validate → dedup → Parquet) for 1 day
of real SPY data, then verifies checkpoint/resume skips completed dates.
"""

import json
import pytest
from dotenv import load_dotenv

from src.utils.config_loader import ConfigLoader
from src.orchestrator.historical_runner import HistoricalRunner


load_dotenv()

TEST_DATE = "2025-01-27"


@pytest.fixture(scope="module")
def live_config(tmp_path_factory):
    """Load config and override output paths to temp directory."""
    loader = ConfigLoader(config_dir="config", env_file=".env")
    config = loader.load()

    output_dir = tmp_path_factory.mktemp("runner_integration")
    checkpoint_dir = tmp_path_factory.mktemp("runner_checkpoint")

    config["sinks"] = {
        "parquet": {
            "base_path": str(output_dir),
            "compression": "snappy",
            "row_group_size": 10000,
        }
    }
    config["historical"] = {
        "backfill": {
            "start_date": TEST_DATE,
            "end_date": TEST_DATE,
            "batch_size": 10000,
        }
    }
    config["logging"] = {
        "execution_log_path": str(checkpoint_dir),
        "error_log_path": str(checkpoint_dir / "errors"),
        "console_level": "INFO",
        "file_level": "DEBUG",
    }
    config["_output_dir"] = str(output_dir)
    config["_checkpoint_dir"] = str(checkpoint_dir)
    return config


class TestHistoricalRunnerLive:
    """End-to-end: HistoricalRunner fetches, validates, deduplicates, stores."""

    def test_full_pipeline_one_day(self, live_config):
        """Run pipeline for 1 day and verify stats and output file."""
        runner = HistoricalRunner(live_config)
        stats = runner.run()

        assert stats["dates_processed"] >= 1
        assert stats["total_written"] > 0
        assert stats["total_fetched"] > 0
        assert stats["total_invalid"] == 0

        # Verify Parquet file exists
        from pathlib import Path
        output_dir = Path(live_config["_output_dir"])
        parquet_path = output_dir / "spy" / f"{TEST_DATE}.parquet"
        assert parquet_path.exists(), f"Expected Parquet at {parquet_path}"
        print(f"\n  Pipeline: {stats['total_written']} records written to {parquet_path}")

    def test_checkpoint_created(self, live_config):
        """After a successful run, checkpoint file should exist."""
        from pathlib import Path
        checkpoint_dir = Path(live_config["_checkpoint_dir"])
        checkpoint_path = checkpoint_dir / f"checkpoint_{TEST_DATE}_{TEST_DATE}.json"
        assert checkpoint_path.exists(), f"Expected checkpoint at {checkpoint_path}"

        data = json.loads(checkpoint_path.read_text())
        assert TEST_DATE in data["completed_dates"]
        print(f"\n  Checkpoint: {data}")

    def test_resume_skips_completed(self, live_config):
        """Resume run should skip already-completed dates and fetch 0 records."""
        runner = HistoricalRunner(live_config)
        stats = runner.run(resume=True)

        # All dates should be skipped via checkpoint
        assert stats["total_fetched"] == 0
        assert stats["dates_skipped"] == 1
        print(f"\n  Resume: {stats['dates_skipped']} dates skipped, 0 fetched")
