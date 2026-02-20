# © 2026 Pallab Basu Roy. All rights reserved.

"""Unit tests for MassiveOptionsDownloader.

All massive.RESTClient calls and ContractSelector are mocked — no network.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Patch `massive` before importing the module under test
# ---------------------------------------------------------------------------
_mock_massive_cls = MagicMock()

with patch.dict(
    "sys.modules",
    {"massive": MagicMock(RESTClient=_mock_massive_cls)},
):
    from src.data_sources.massive_options_downloader import MassiveOptionsDownloader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_CONFIG = {
    "pipeline_v2": {
        "massive_options": {
            "limit_per_request": 500,
            "max_workers": 2,
        }
    },
    "sinks":   {"parquet": {"base_path": "data/raw", "compression": "snappy"}},
    "polygon": {"api_key": "test-polygon-key"},
    "massive": {"api_key": "test-massive-key"},
    "retry":   {"polygon": {"max_attempts": 1, "backoff_base": 0}},
}


def _make_downloader(
    tmp_path: Path,
    selector: MagicMock | None = None,
    config: dict | None = None,
    api_key: str = "test-api-key",
) -> MassiveOptionsDownloader:
    cfg = dict(config or _BASE_CONFIG)
    cfg = {**cfg, "sinks": {"parquet": {
        "base_path": str(tmp_path), "compression": "snappy"
    }}}
    sel = selector or _mock_selector()
    with patch.dict(
        "sys.modules",
        {"massive": MagicMock(RESTClient=_mock_massive_cls)},
    ):
        dl = MassiveOptionsDownloader(cfg, api_key, sel)
    dl.base_path = tmp_path
    # Give each downloader its own fresh client mock so tests don't share state
    dl._client = MagicMock()
    return dl


def _mock_selector(
    mode: str = "test",
    underlying: str = "SPY",
    needs_prompt: bool = False,
    contracts: list | None = None,
) -> MagicMock:
    sel = MagicMock()
    sel.mode = mode
    sel.needs_prompt = needs_prompt
    sel.underlying = underlying
    sel.get_contracts.return_value = contracts if contracts is not None else []
    return sel


def _make_contract(
    ticker: str = "O:SPY250304C00601000",
    strike: float = 601.0,
    contract_type: str = "call",
    expiry_date: str = "2025-03-04",
) -> dict:
    return {
        "ticker":        ticker,
        "strike":        strike,
        "contract_type": contract_type,
        "expiry_date":   expiry_date,
        "underlying":    "SPY",
    }


def _make_agg(ts: int = 1_000_000) -> MagicMock:
    agg = MagicMock()
    agg.timestamp    = ts
    agg.open         = 10.0
    agg.high         = 11.0
    agg.low          = 9.0
    agg.close        = 10.5
    agg.volume       = 100
    agg.vwap         = 10.25
    agg.transactions = 50
    return agg


def _write_spy(tmp_path: Path, date: str, open_price: float = 600.0) -> None:
    path = tmp_path / "spy" / f"{date}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"timestamp": 1000, "open": open_price}]).to_parquet(path, index=False)


# ===========================================================================
# Constructor
# ===========================================================================

class TestConstructor:
    def test_empty_api_key_raises(self, tmp_path):
        with pytest.raises(ValueError, match="api_key must not be empty"):
            _make_downloader(tmp_path, api_key="")

    def test_massive_available_in_class_context(self):
        # Confirms the class was imported with the mock massive in place.
        # Checks __globals__ directly — avoids triggering a fresh reimport.
        assert MassiveOptionsDownloader.__init__.__globals__["_MASSIVE_AVAILABLE"] is True

    def test_config_defaults_applied(self, tmp_path):
        dl = _make_downloader(tmp_path)
        assert dl._limit       == 500
        assert dl._max_workers == 2

    def test_custom_config_values(self, tmp_path):
        cfg = {**_BASE_CONFIG, "pipeline_v2": {"massive_options": {
            "limit_per_request": 200,
            "max_workers": 8,
        }}}
        dl = _make_downloader(tmp_path, config=cfg)
        assert dl._limit       == 200
        assert dl._max_workers == 8


# ===========================================================================
# get_opening_price
# ===========================================================================

class TestGetOpeningPrice:
    def test_reads_first_bar(self, tmp_path):
        _write_spy(tmp_path, "2025-03-03", 601.25)
        dl = _make_downloader(tmp_path)
        assert dl.get_opening_price("2025-03-03") == pytest.approx(601.25)

    def test_file_not_found_raises(self, tmp_path):
        dl = _make_downloader(tmp_path)
        with pytest.raises(FileNotFoundError, match="minute data not found"):
            dl.get_opening_price("2025-03-03")

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "spy" / "2025-03-03.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_parquet(path, index=False)
        dl = _make_downloader(tmp_path)
        with pytest.raises(ValueError, match="empty"):
            dl.get_opening_price("2025-03-03")

    def test_non_spy_underlying(self, tmp_path):
        path = tmp_path / "tsla" / "2025-03-03.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"timestamp": 1, "open": 250.5}]).to_parquet(path, index=False)
        dl = _make_downloader(tmp_path)
        assert dl.get_opening_price("2025-03-03", underlying="TSLA") == pytest.approx(250.5)

    def test_underlying_path_lowercased(self, tmp_path):
        path = tmp_path / "tsla" / "2025-03-03.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"timestamp": 1, "open": 250.0}]).to_parquet(path, index=False)
        dl = _make_downloader(tmp_path)
        # Passing uppercase — path must be lowercased
        assert dl.get_opening_price("2025-03-03", underlying="TSLA") == pytest.approx(250.0)


# ===========================================================================
# _fetch_bars
# ===========================================================================

class TestFetchBars:
    def test_returns_bars_on_success(self, tmp_path):
        dl = _make_downloader(tmp_path)
        dl._client.list_aggs.return_value = [_make_agg(ts=1_000_000)]
        bars = dl._fetch_bars("O:SPY250304C00601000", "2025-03-03")
        assert len(bars) == 1
        assert bars[0]["ticker"] == "O:SPY250304C00601000"
        assert bars[0]["source"] == "options_minute_massive"

    def test_returns_empty_on_no_data(self, tmp_path):
        dl = _make_downloader(tmp_path)
        dl._client.list_aggs.return_value = []
        assert dl._fetch_bars("O:SPY250304C00601000", "2025-03-03") == []

    def test_returns_empty_on_exception(self, tmp_path):
        dl = _make_downloader(tmp_path)
        dl._client.list_aggs.side_effect = Exception("network error")
        assert dl._fetch_bars("O:SPY250304C00601000", "2025-03-03") == []

    def test_all_fields_present(self, tmp_path):
        dl = _make_downloader(tmp_path)
        dl._client.list_aggs.return_value = [_make_agg()]
        bars = dl._fetch_bars("O:SPY250304C00601000", "2025-03-03")
        expected = {
            "timestamp", "open", "high", "low", "close",
            "volume", "vwap", "transactions", "ticker", "source",
        }
        assert set(bars[0].keys()) == expected

    def test_correct_list_aggs_call(self, tmp_path):
        dl = _make_downloader(tmp_path)
        dl._client.list_aggs.return_value = []
        dl._fetch_bars("O:SPY250304C00601000", "2025-03-03")
        dl._client.list_aggs.assert_called_once_with(
            "O:SPY250304C00601000",
            1,
            "minute",
            "2025-03-03",
            "2025-03-03",
            adjusted="true",
            sort="asc",
            limit=500,
        )


# ===========================================================================
# _download_single
# ===========================================================================

class TestDownloadSingle:
    def test_writes_parquet_on_success(self, tmp_path):
        dl = _make_downloader(tmp_path)
        dl._client.list_aggs.return_value = [_make_agg(ts=1_000_000)]
        contract = _make_contract()

        bars = dl._download_single(contract, "2025-03-03")

        out = (tmp_path / "options" / "minute"
               / "O_SPY250304C00601000" / "2025-03-03.parquet")
        assert out.exists()
        assert bars == 1

    def test_returns_zero_on_no_data(self, tmp_path):
        dl = _make_downloader(tmp_path)
        dl._client.list_aggs.return_value = []
        assert dl._download_single(_make_contract(), "2025-03-03") == 0

    def test_resume_skips_existing_file(self, tmp_path):
        # Pre-write the Parquet
        out_dir = tmp_path / "options" / "minute" / "O_SPY250304C00601000"
        out_dir.mkdir(parents=True)
        existing = pd.DataFrame([{"timestamp": 1, "open": 10.0, "high": 11.0,
                                   "low": 9.0, "close": 10.5, "volume": 100,
                                   "vwap": 10.25, "transactions": 50,
                                   "ticker": "O:SPY250304C00601000",
                                   "source": "options_minute_massive"}])
        existing.to_parquet(out_dir / "2025-03-03.parquet", index=False)

        dl = _make_downloader(tmp_path)
        bars = dl._download_single(_make_contract(), "2025-03-03", resume=True)

        dl._client.list_aggs.assert_not_called()
        assert bars == 1

    def test_resume_false_re_downloads(self, tmp_path):
        # Pre-write a file
        out_dir = tmp_path / "options" / "minute" / "O_SPY250304C00601000"
        out_dir.mkdir(parents=True)
        pd.DataFrame([{"timestamp": 1}]).to_parquet(
            out_dir / "2025-03-03.parquet", index=False
        )
        dl = _make_downloader(tmp_path)
        dl._client.list_aggs.return_value = [_make_agg(), _make_agg()]

        bars = dl._download_single(_make_contract(), "2025-03-03", resume=False)

        dl._client.list_aggs.assert_called_once()
        assert bars == 2

    def test_safe_ticker_path_colon_replaced(self, tmp_path):
        dl = _make_downloader(tmp_path)
        dl._client.list_aggs.return_value = [_make_agg()]
        dl._download_single(_make_contract(ticker="O:SPY250304P00600000"), "2025-03-03")
        out = (tmp_path / "options" / "minute"
               / "O_SPY250304P00600000" / "2025-03-03.parquet")
        assert out.exists()

    def test_parquet_schema_correct(self, tmp_path):
        dl = _make_downloader(tmp_path)
        dl._client.list_aggs.return_value = [_make_agg(ts=9_999_999)]
        dl._download_single(_make_contract(), "2025-03-03")
        out = (tmp_path / "options" / "minute"
               / "O_SPY250304C00601000" / "2025-03-03.parquet")
        df = pd.read_parquet(out)
        assert "ticker" in df.columns
        assert "source" in df.columns
        assert df.iloc[0]["ticker"] == "O:SPY250304C00601000"
        assert df.iloc[0]["source"] == "options_minute_massive"


# ===========================================================================
# download_tickers — parallel execution
# ===========================================================================

class TestDownloadTickers:
    def test_downloads_all_contracts(self, tmp_path):
        dl = _make_downloader(tmp_path)
        dl._client.list_aggs.return_value = [_make_agg()]
        contracts = [
            _make_contract("O:SPY250304C00601000", 601.0, "call"),
            _make_contract("O:SPY250304P00600000", 600.0, "put"),
        ]
        total = dl.download_tickers(contracts, "2025-03-03")
        assert total == 2

    def test_empty_contract_list_returns_zero(self, tmp_path):
        dl = _make_downloader(tmp_path)
        assert dl.download_tickers([], "2025-03-03") == 0

    def test_partial_success_counted_correctly(self, tmp_path):
        dl = _make_downloader(tmp_path)
        call_count = 0

        def _side_effect(ticker, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "C00601000" in ticker:
                return [_make_agg()]   # call → 1 bar
            return []                  # put → no data

        dl._client.list_aggs.side_effect = _side_effect

        contracts = [
            _make_contract("O:SPY250304C00601000", 601.0, "call"),
            _make_contract("O:SPY250304P00600000", 600.0, "put"),
        ]
        total = dl.download_tickers(contracts, "2025-03-03")
        assert total == 1

    def test_exception_in_one_worker_does_not_stop_others(self, tmp_path):
        dl = _make_downloader(tmp_path)
        call_count = 0

        def _side_effect(ticker, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "C00601000" in ticker:
                raise RuntimeError("simulated failure")
            return [_make_agg()]

        dl._client.list_aggs.side_effect = _side_effect

        contracts = [
            _make_contract("O:SPY250304C00601000", 601.0, "call"),
            _make_contract("O:SPY250304P00600000", 600.0, "put"),
        ]
        total = dl.download_tickers(contracts, "2025-03-03")
        # Put should still succeed
        assert total == 1


# ===========================================================================
# run — end-to-end loop
# ===========================================================================

class TestRun:
    def test_skips_dates_without_spy_data(self, tmp_path):
        dl = _make_downloader(tmp_path)
        stats = dl.run("2025-03-03", "2025-03-04")
        assert stats["dates_skipped"] == 2
        assert stats["dates_processed"] == 0

    def test_processes_dates_with_spy_data(self, tmp_path):
        _write_spy(tmp_path, "2025-03-03", 600.0)
        sel = _mock_selector(
            contracts=[
                _make_contract("O:SPY250304C00601000", 601.0, "call"),
                _make_contract("O:SPY250304P00600000", 600.0, "put"),
            ]
        )
        dl = _make_downloader(tmp_path, selector=sel)
        dl._client.list_aggs.return_value = [_make_agg()]

        stats = dl.run("2025-03-03", "2025-03-03")
        assert stats["dates_processed"] == 1
        assert stats["contracts_found"] == 2
        assert stats["total_bars"] == 2

    def test_prompt_once_called_before_loop(self, tmp_path):
        _write_spy(tmp_path, "2025-03-03", 600.0)
        sel = _mock_selector(needs_prompt=True, contracts=[])
        dl = _make_downloader(tmp_path, selector=sel)

        dl.run("2025-03-03", "2025-03-03")

        sel.prompt_once.assert_called_once()

    def test_prompt_not_called_when_already_set(self, tmp_path):
        _write_spy(tmp_path, "2025-03-03", 600.0)
        sel = _mock_selector(needs_prompt=False, contracts=[])
        dl = _make_downloader(tmp_path, selector=sel)

        dl.run("2025-03-03", "2025-03-03")

        sel.prompt_once.assert_not_called()

    def test_selector_get_contracts_called_with_opening_price(self, tmp_path):
        _write_spy(tmp_path, "2025-03-03", 601.25)
        sel = _mock_selector(contracts=[])
        dl = _make_downloader(tmp_path, selector=sel)

        dl.run("2025-03-03", "2025-03-03")

        sel.get_contracts.assert_called_once_with("2025-03-03", pytest.approx(601.25))

    def test_underlying_from_selector_used_for_parquet_path(self, tmp_path):
        # TSLA underlying — file must be read from tsla/ directory
        tsla_path = tmp_path / "tsla" / "2025-03-03.parquet"
        tsla_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"timestamp": 1, "open": 250.0}]).to_parquet(
            tsla_path, index=False
        )
        sel = _mock_selector(underlying="TSLA", contracts=[])
        dl = _make_downloader(tmp_path, selector=sel)

        stats = dl.run("2025-03-03", "2025-03-03")
        # Should process the date (TSLA file found), no bars because contracts=[]
        assert stats["dates_skipped"] == 1   # contracts empty → skip

    def test_date_range_inclusive(self, tmp_path):
        for date in ("2025-03-03", "2025-03-04", "2025-03-05"):
            _write_spy(tmp_path, date, 600.0)
        sel = _mock_selector(contracts=[])
        dl = _make_downloader(tmp_path, selector=sel)

        dl.run("2025-03-03", "2025-03-05")

        assert sel.get_contracts.call_count == 3

    def test_contract_selection_error_skips_date(self, tmp_path):
        _write_spy(tmp_path, "2025-03-03", 600.0)
        sel = _mock_selector()
        sel.get_contracts.side_effect = RuntimeError("API down")
        dl = _make_downloader(tmp_path, selector=sel)

        stats = dl.run("2025-03-03", "2025-03-03")
        assert stats["dates_skipped"] == 1


# ===========================================================================
# from_config factory
# ===========================================================================

class TestFromConfig:
    """Test api key resolution in from_config.

    We intercept __init__ with patch.object so we can inspect the resolved
    api_key without needing to re-import the module or fight mock state.
    """

    def test_resolves_massive_key(self, monkeypatch):
        monkeypatch.setenv("MASSIVE_API_KEY", "massive-123")
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        with patch.object(MassiveOptionsDownloader, "__init__", return_value=None) as m:
            MassiveOptionsDownloader.from_config(_BASE_CONFIG, _mock_selector())
        _, api_key_arg, _ = m.call_args[0]
        assert api_key_arg == "massive-123"

    def test_falls_back_to_polygon_key(self, monkeypatch):
        monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
        monkeypatch.setenv("POLYGON_API_KEY", "poly-456")
        cfg = {**_BASE_CONFIG, "massive": {"api_key": ""}}
        with patch.object(MassiveOptionsDownloader, "__init__", return_value=None) as m:
            MassiveOptionsDownloader.from_config(cfg, _mock_selector())
        _, api_key_arg, _ = m.call_args[0]
        assert api_key_arg == "poly-456"

    def test_raises_when_no_key(self, monkeypatch):
        monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        no_key = {**_BASE_CONFIG, "polygon": {"api_key": ""}, "massive": {"api_key": ""}}
        with pytest.raises(ValueError, match="No API key"):
            MassiveOptionsDownloader.from_config(no_key, _mock_selector())
