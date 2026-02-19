# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Storage space utilization reporter.

Walks the data/ directory tree, computes size by sub-directory,
and optionally estimates compression efficiency for Parquet files.

Output: data/reports/space/{YYYY-MM-DD}_space.json
"""

import io
import json
import os
from datetime import date as date_cls
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()


class SpaceReporter:
    """Report storage space usage across the data/ directory.

    Configuration is read from config["pipeline_v2"]["reporting"]:
      - reports_dir: base path for report output (default "data/reports")
    """

    def __init__(self, config: Dict[str, Any]):
        v2 = config.get("pipeline_v2", {})
        reporting = v2.get("reporting", {})
        self.reports_dir = Path(reporting.get("reports_dir", "data/reports"))
        self.data_root = Path("data")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self) -> Dict[str, Any]:
        """Walk data/ and return a nested size tree.

        Returns:
            Dict with structure:
              {
                "total_bytes": int,
                "total_files": int,
                "tree": {
                  "raw/spy": {"bytes": int, "files": int},
                  ...
                }
              }
        """
        if not self.data_root.exists():
            return {"total_bytes": 0, "total_files": 0, "tree": {}}

        tree: Dict[str, Dict] = {}
        total_bytes = 0
        total_files = 0

        for root, dirs, files in os.walk(self.data_root):
            root_path = Path(root)
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for fname in files:
                fpath = root_path / fname
                try:
                    size = fpath.stat().st_size
                except OSError:
                    continue

                # Key is relative path from data root
                rel = str(root_path.relative_to(self.data_root))
                if rel == ".":
                    rel = "."

                if rel not in tree:
                    tree[rel] = {"bytes": 0, "files": 0}
                tree[rel]["bytes"] += size
                tree[rel]["files"] += 1
                total_bytes += size
                total_files += 1

        return {
            "total_bytes": total_bytes,
            "total_files": total_files,
            "tree": tree,
        }

    def estimate_compression(self, parquet_path: str) -> Dict[str, float]:
        """Compare compression algorithms for a Parquet file.

        Samples the first 10,000 rows and re-serializes with snappy,
        gzip, and zstd to estimate compressed size in MB.

        Args:
            parquet_path: Path to an existing Parquet file.

        Returns:
            Dict with keys "snappy_mb", "gzip_mb", "zstd_mb",
            and "row_sample" (number of rows actually used).
        """
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        df = pd.read_parquet(path)
        sample = df.head(10_000)
        n = len(sample)

        results: Dict[str, float] = {"row_sample": n}
        for codec in ("snappy", "gzip", "zstd"):
            buf = io.BytesIO()
            try:
                sample.to_parquet(buf, engine="pyarrow", compression=codec, index=False)
                results[f"{codec}_mb"] = round(buf.tell() / (1024 ** 2), 4)
            except Exception as exc:
                logger.warning(f"Compression estimate failed for {codec}: {exc}")
                results[f"{codec}_mb"] = None

        return results

    def generate_report(self) -> Path:
        """Collect storage stats and write JSON report.

        Returns:
            Path to the written JSON file.
        """
        today = date_cls.today().strftime("%Y-%m-%d")
        out_dir = self.reports_dir / "space"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{today}_space.json"

        data = self.collect()

        # Add human-readable MB values
        data["total_mb"] = round(data["total_bytes"] / (1024 ** 2), 3)

        tree_mb: Dict[str, Any] = {}
        for key, val in data["tree"].items():
            tree_mb[key] = {
                "bytes": val["bytes"],
                "mb": round(val["bytes"] / (1024 ** 2), 3),
                "files": val["files"],
            }
        data["tree_mb"] = tree_mb

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        # Console summary
        self._print_summary(data)
        logger.info(f"Space report written to {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _print_summary(self, data: Dict[str, Any]) -> None:
        """Print a formatted summary table to stdout."""
        print(f"\n--- Storage Utilization Report ---")
        print(f"Total:  {data['total_mb']:.1f} MB  ({data['total_files']} files)")
        print(f"\n{'Directory':<40} {'MB':>8}  {'Files':>6}")
        print("-" * 58)

        tree = data.get("tree_mb", data.get("tree", {}))
        rows = sorted(tree.items(), key=lambda kv: kv[1].get("bytes", 0), reverse=True)
        for key, val in rows[:20]:
            mb = val.get("mb", round(val.get("bytes", 0) / (1024 ** 2), 3))
            files = val.get("files", 0)
            print(f"{key:<40} {mb:>8.2f}  {files:>6}")
