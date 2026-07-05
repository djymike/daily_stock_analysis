# -*- coding: utf-8 -*-
"""AShareHub fetcher for A-share chip distribution fallback."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import requests

from .base import BaseFetcher, DataFetchError, normalize_stock_code
from .realtime_types import ChipDistribution, safe_float

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.asharehub.com"
_DEFAULT_TIMEOUT_SECONDS = 10.0


def _parse_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("%s is not a valid number, using default %s", name, default)
        return default


def _parse_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("%s is not a valid integer, using default %s", name, default)
        return default


class AShareHubFetcher(BaseFetcher):
    """Only participates in chip distribution requests when API key is configured."""

    name = "AShareHubFetcher"
    priority = _parse_env_int("ASHAREHUB_PRIORITY", -2)

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("ASHAREHUB_API_KEY") or "").strip()
        self.base_url = (base_url or os.getenv("ASHAREHUB_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else _parse_env_float("ASHAREHUB_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS)
        )

    def is_available_for_request(self, capability: str = "") -> bool:
        return bool(self.api_key) and capability == "chip_distribution"

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise DataFetchError("AShareHubFetcher only supports chip_distribution")

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        raise DataFetchError("AShareHubFetcher only supports chip_distribution")

    def get_chip_distribution(self, stock_code: str) -> Optional[ChipDistribution]:
        code = normalize_stock_code(stock_code)
        symbol = _to_asharehub_symbol(code)
        if not symbol:
            logger.debug("[AShareHub] unsupported stock code for chip distribution: %s", stock_code)
            return None

        if not self.api_key:
            logger.debug("[AShareHub] ASHAREHUB_API_KEY is not configured")
            return None

        try:
            payload = self._get_json("/v2/chips/distribution", params={"symbol": symbol})
            item = _extract_latest_item(payload)
            if not item:
                logger.warning("[AShareHub] empty chip distribution result for %s", symbol)
                return None

            chip = _build_chip_distribution(code, item)
            if chip is None:
                logger.warning("[AShareHub] incomplete chip distribution result for %s", symbol)
                return None

            logger.info(
                "[AShareHub] 筹码分布 %s 日期=%s: 获利比例=%.1f%%, 平均成本=%s, "
                "90%%集中度=%.2f%%, 70%%集中度=%.2f%%",
                code,
                chip.date,
                chip.profit_ratio * 100,
                chip.avg_cost,
                chip.concentration_90 * 100,
                chip.concentration_70 * 100,
            )
            return chip
        except Exception as exc:
            logger.warning("[AShareHub] 获取筹码分布失败 %s: %s", stock_code, exc)
            return None

    def _get_json(self, path: str, params: dict[str, Any]) -> Any:
        response = requests.get(
            f"{self.base_url}{path}",
            params=params,
            headers={
                "Accept": "application/json",
                "User-Agent": "daily-stock-analysis/asharehub",
                "X-API-Key": self.api_key,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()


def _to_asharehub_symbol(stock_code: str) -> str:
    code = normalize_stock_code(stock_code)
    if not code or not code.isdigit() or len(code) != 6:
        return ""
    if code.startswith(("6", "5", "9")):
        return f"{code}.SH"
    if code.startswith(("0", "1", "2", "3")):
        return f"{code}.SZ"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return ""


def _extract_latest_item(payload: Any) -> Optional[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("data", "result", "items", "list"):
            value = payload.get(key)
            if isinstance(value, list):
                return _latest_from_list(value)
            if isinstance(value, dict):
                return value
        return payload
    if isinstance(payload, list):
        return _latest_from_list(payload)
    return None


def _latest_from_list(items: list[Any]) -> Optional[dict[str, Any]]:
    dict_items = [item for item in items if isinstance(item, dict)]
    if not dict_items:
        return None

    def sort_key(item: dict[str, Any]) -> str:
        return str(item.get("trade_date") or item.get("date") or item.get("dt") or "")

    return sorted(dict_items, key=sort_key)[-1]


def _build_chip_distribution(code: str, item: dict[str, Any]) -> Optional[ChipDistribution]:
    winner_rate = _ratio(item, "winner_rate", "profit_ratio")
    avg_cost = _number(item, "weight_avg", "avg_cost", "average_cost")
    cost_90_low = _number(item, "cost_5", "cost_90_low")
    cost_90_high = _number(item, "cost_95", "cost_90_high")
    cost_70_low = _number(item, "cost_15", "cost_70_low")
    cost_70_high = _number(item, "cost_85", "cost_70_high")

    if avg_cost <= 0 or cost_90_low <= 0 or cost_90_high <= 0:
        return None

    return ChipDistribution(
        code=code,
        date=_format_date(item.get("trade_date") or item.get("date") or item.get("dt")),
        source="asharehub",
        profit_ratio=winner_rate,
        avg_cost=avg_cost,
        cost_90_low=cost_90_low,
        cost_90_high=cost_90_high,
        concentration_90=_concentration(cost_90_low, cost_90_high),
        cost_70_low=cost_70_low,
        cost_70_high=cost_70_high,
        concentration_70=_concentration(cost_70_low, cost_70_high),
    )


def _number(item: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = item.get(key)
        if value is not None:
            return safe_float(value)
    return 0.0


def _ratio(item: dict[str, Any], *keys: str) -> float:
    value = _number(item, *keys)
    if value > 1:
        return value / 100
    return value


def _concentration(low: float, high: float) -> float:
    denominator = high + low
    if denominator <= 0:
        return 0.0
    return round((high - low) / denominator, 4)


def _format_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").strftime("%Y-%m-%d")
    return text[:10]
