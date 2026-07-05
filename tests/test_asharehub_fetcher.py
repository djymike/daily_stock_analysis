# -*- coding: utf-8 -*-
"""Regression tests for the AShareHub chip distribution fetcher."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

if "fake_useragent" not in sys.modules:
    sys.modules["fake_useragent"] = MagicMock()

from data_provider.base import DataFetcherManager
from data_provider.asharehub_fetcher import AShareHubFetcher


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_asharehub_chip_distribution_maps_cost_percentiles():
    fetcher = AShareHubFetcher(api_key="test-key")
    payload = {
        "data": [
            {
                "trade_date": "20260703",
                "winner_rate": 62.5,
                "weight_avg": 10.2,
                "cost_5": 8.0,
                "cost_15": 9.0,
                "cost_85": 12.0,
                "cost_95": 13.0,
            }
        ]
    }

    with patch("data_provider.asharehub_fetcher.requests.get", return_value=_FakeResponse(payload)) as mock_get:
        chip = fetcher.get_chip_distribution("000001")

    assert chip is not None
    assert chip.source == "asharehub"
    assert chip.date == "2026-07-03"
    assert chip.profit_ratio == 0.625
    assert chip.avg_cost == 10.2
    assert chip.cost_90_low == 8.0
    assert chip.cost_90_high == 13.0
    assert chip.cost_70_low == 9.0
    assert chip.cost_70_high == 12.0
    assert chip.concentration_90 == round((13.0 - 8.0) / (13.0 + 8.0), 4)
    assert chip.concentration_70 == round((12.0 - 9.0) / (12.0 + 9.0), 4)

    _, kwargs = mock_get.call_args
    assert kwargs["params"] == {"symbol": "000001.SZ"}
    assert kwargs["headers"]["X-API-Key"] == "test-key"


def test_asharehub_only_available_for_chip_distribution():
    fetcher = AShareHubFetcher(api_key="test-key")

    assert fetcher.is_available_for_request("chip_distribution") is True
    assert fetcher.is_available_for_request("daily_data") is False
    assert fetcher.is_available_for_request("realtime_quote") is False


def test_manager_places_configured_asharehub_first_for_chip(monkeypatch):
    monkeypatch.setenv("ASHAREHUB_API_KEY", "test-key")
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    cfg = SimpleNamespace(
        tushare_token=None,
        tickflow_api_key=None,
        tickflow_kline_adjust="none",
        tickflow_batch_daily_enabled=True,
        tickflow_batch_size=100,
        tickflow_priority=2,
    )

    with patch("src.config.get_config", return_value=cfg):
        manager = DataFetcherManager()

    fetchers = manager._get_fetchers_snapshot()
    assert fetchers[0].name == "AShareHubFetcher"
    assert fetchers[0].is_available_for_request("chip_distribution") is True
    assert fetchers[0].is_available_for_request("daily_data") is False
