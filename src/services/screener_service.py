# -*- coding: utf-8 -*-
"""
===================================
ScreenerService - 全市场条件选股服务
===================================

数据源仅使用 Pytdx（通达信直连），不调用新闻搜索或 AI 分析。

筛选条件：
  A. MACD ∈ (-3, 0)
  B. 收盘价 > MA60
  C. 最新金叉→死叉之间涨幅 ≥ 40%
"""
from __future__ import annotations

import logging
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal

import pandas as pd
import numpy as np

from data_provider.pytdx_fetcher import PytdxFetcher, _parse_hosts_from_env

logger = logging.getLogger(__name__)


@dataclass
class CrossEvent:
    """MACD 交叉事件"""
    index: int
    cross_type: Literal["golden", "death"]
    dif: float
    dea: float
    price: float


@dataclass
class ScreenerResult:
    """筛选结果"""
    code: str
    name: str
    score: int
    macd: float
    dif: float
    dea: float
    price: float
    ma60: float
    ma60_diff_pct: float
    rise_since_golden: float
    golden_index: int
    death_index: int


class ScreenerService:
    """全市场条件选股服务，只使用通达信数据源"""

    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    MIN_TRADING_DAYS = 60
    MIN_RISE_PCT = 0.40
    MACD_MIN = -3.0
    MACD_MAX = 0.0
    MAX_WORKERS = int(os.getenv("SCREENER_MAX_WORKERS", "10"))
    CONNECT_TIMEOUT = float(os.getenv("PYTDX_CONNECT_TIMEOUT", "3"))
    MIN_EXPECTED_POOL_SIZE = int(os.getenv("SCREENER_MIN_POOL_SIZE", "3000"))
    EXTRA_TDX_HOSTS: List[Tuple[str, int]] = [
        ("114.80.63.12", 7709),
        ("114.80.63.35", 7709),
        ("124.74.236.94", 7709),
        ("218.75.126.9", 7709),
        ("115.238.90.165", 7709),
        ("115.238.56.198", 7709),
        ("218.108.98.244", 7709),
        ("218.108.47.69", 7709),
        ("14.17.75.71", 7709),
        ("180.153.18.170", 7709),
        ("180.153.18.171", 7709),
        ("180.153.18.172", 7709),
        ("202.108.253.130", 7709),
        ("60.191.117.167", 7709),
        ("jstdx.gtjas.com", 7709),
        ("shtdx.gtjas.com", 7709),
        ("sztdx.gtjas.com", 7709),
    ]

    def __init__(self):
        """初始化筛选服务，直接使用 pytdx 直连通达信"""
        self._thread_local = threading.local()
        self._session_lock = threading.Lock()
        self._worker_sessions = []
        self._preferred_data_hosts: List[Tuple[str, int]] = []

    # ==================== 公开接口 ====================

    def run(self, top_n: int = 10) -> List[ScreenerResult]:
        """执行全市场条件选股"""
        pool = self._build_pool()
        total = len(pool)
        if total == 0:
            logger.error("[Screener] 股票池为空，无法执行筛选")
            return []

        logger.info("[Screener] 筛选池共 %s 只股票，开始并行筛选（%s 线程）", total, self.MAX_WORKERS)

        results: List[ScreenerResult] = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            future_map = {}
            for code, name in pool:
                future = executor.submit(self._process_stock, code, name)
                future_map[future] = (code, name)

            done = 0
            for future in as_completed(future_map):
                done += 1
                code, name = future_map[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.warning("[Screener] %s(%s) 分析异常: %s", name, code, e)

                if done % 200 == 0 or done == total:
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    logger.info(
                        "[Screener] 进度: %s/%s (%.1f%%), 已符合 %s 只, 速率 %.2f 只/秒",
                        done, total, done / total * 100, len(results), rate,
                    )

        self._disconnect_worker_sessions()

        elapsed = time.time() - start_time
        results.sort(key=lambda r: r.score, reverse=True)
        top = results[:top_n]

        logger.info(
            "[Screener] 筛选完成: 共检查 %s 只, 符合 %s 只, 耗时 %.1f 秒",
            total, len(results), elapsed,
        )
        return top

    def format_notification(self, results: List[ScreenerResult]) -> str:
        """格式化为飞书推送的 Markdown 文本"""
        if not results:
            return (
                "📊 强势股回踩筛选报告\n\n"
                "暂未发现符合条件的股票。\n\n"
                f"筛选时间: {time.strftime('%Y-%m-%d %H:%M')}\n数据源: 通达信"
            )

        lines = [
            "📊 **强势股回踩筛选报告**\n",
            "筛选项:",
            "• MACD: -3 ~ 0（回调整理阶段）",
            "• 股价 > MA60（中期趋势完好）",
            "• 金叉→死叉涨幅 ≥ 40%（有主力拉升痕迹）\n",
            f"符合条件: {len(results)} 只",
            "━━━━━━━━━━━━━━━━━━━━",
        ]

        medals = ["🥇", "🥈", "🥉"]
        for i, r in enumerate(results):
            medal = medals[i] if i < 3 else "▫"
            lines.append(
                f"{medal} **{r.name}({r.code})**  评分 {r.score}\n"
                f"  现价: {r.price:.2f} | MACD: {r.macd:.2f}\n"
                f"  60日线: {r.ma60:.2f} | 距MA60: {r.ma60_diff_pct:+.2f}%\n"
                f"  上波涨幅: {r.rise_since_golden * 100:.1f}%\n"
            )

        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"筛选时间: {time.strftime('%Y-%m-%d %H:%M')}")
        lines.append("数据源: 通达信")

        return "\n".join(lines)

    def format_console(self, results: List[ScreenerResult]) -> str:
        """格式化为控制台输出"""
        if not results:
            return "暂未发现符合条件的股票。"

        lines = [
            "=" * 60,
            "  强势股回踩筛选结果",
            "=" * 60,
            f"符合条件: {len(results)} 只",
            f"条件: MACD∈(-3,0) + 股价>MA60 + 金叉→死叉涨幅≥40%",
            "-" * 60,
        ]

        for i, r in enumerate(results):
            lines.append(f"  #{i+1} {r.name}({r.code})  评分: {r.score}")
            lines.append(f"      现价={r.price:.2f}  MACD={r.macd:.2f}  DIF={r.dif:.2f}  DEA={r.dea:.2f}")
            lines.append(f"      MA60={r.ma60:.2f}  距MA60={r.ma60_diff_pct:+.2f}%")
            lines.append(f"      金叉→死叉涨幅={r.rise_since_golden*100:.1f}%")

        lines.append("-" * 60)
        lines.append(f"筛选时间: {time.strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 60)

        return "\n".join(lines)

    # ==================== 股票池构建 ====================

    def _build_pool(self) -> List[Tuple[str, str]]:
        """
        构建筛选池：直连通达信获取全市场股票列表

        股票池仅保留沪深主板个股：
        - 深市：000、001、002、003
        - 沪市：600、601、603、605
        """
        hosts = self._tdx_hosts()
        best_pool: List[Tuple[str, str]] = []
        best_stats = ""
        best_host: Optional[Tuple[str, int]] = None
        failed_hosts = []

        for host, port in hosts:
            try:
                pool, stats = self._build_pool_from_host(host, port)
            except Exception as e:
                failed_hosts.append(f"{host}:{port}={e}")
                continue

            if not pool:
                failed_hosts.append(f"{host}:{port}=empty")
                continue

            logger.info(
                "[Screener] 通达信股票池候选: %s:%s 有效=%s, %s",
                host,
                port,
                len(pool),
                stats,
            )
            if len(pool) > len(best_pool):
                best_pool = pool
                best_stats = f"{host}:{port}, {stats}"
                best_host = (host, port)

            if len(best_pool) >= self.MIN_EXPECTED_POOL_SIZE:
                break

        if not best_pool:
            logger.error(
                "[Screener] 无法从任何通达信服务器构建股票池，已尝试 %s 个: %s",
                len(hosts),
                "; ".join(failed_hosts[:20]),
            )
            return []

        if len(best_pool) < self.MIN_EXPECTED_POOL_SIZE:
            logger.warning(
                "[Screener] 股票池数量 %s 低于预期阈值 %s，疑似通达信节点列表不完整；最佳节点: %s",
                len(best_pool),
                self.MIN_EXPECTED_POOL_SIZE,
                best_stats,
            )
            em_pool, em_stats = self._build_pool_from_eastmoney()
            if len(em_pool) > len(best_pool):
                logger.info("[Screener] 使用东方财富免费列表补全股票池: 有效=%s, %s", len(em_pool), em_stats)
                best_pool = em_pool
                best_stats = f"东方财富免费列表, {em_stats}"

        logger.info("[Screener] 股票池构建完成: 有效=%s, 来源=%s", len(best_pool), best_stats)
        if best_host is not None:
            self._preferred_data_hosts = [best_host] + [item for item in hosts if item != best_host]
        return best_pool

    def _build_pool_from_eastmoney(self) -> Tuple[List[Tuple[str, str]], str]:
        """使用东方财富免费行情列表补齐股票池，仅取代码和名称。"""
        if os.getenv("SCREENER_EASTMONEY_POOL_ENABLED", "true").lower() in ("0", "false", "no"):
            return [], "disabled"

        markets = [
            (0, "深市主板", "m:0+t:6"),
            (1, "沪市主板", "m:1+t:2"),
        ]
        pool_map = {}
        stats = []

        for market, label, fs in markets:
            rows = self._fetch_eastmoney_list(fs)
            accepted = 0
            for row in rows:
                code = str(row.get("f12", "")).strip()
                name = str(row.get("f14", "")).strip()
                if not code or not name:
                    continue
                if not self._is_main_board_stock(market, code):
                    continue
                pool_map[f"{market}:{code}"] = (code, name)
                accepted += 1
            stats.append(f"{label}={accepted}")

        return list(pool_map.values()), ", ".join(stats)

    def _fetch_eastmoney_list(self, fs: str) -> List[dict]:
        """分页读取东方财富列表接口。"""
        rows: List[dict] = []
        page = 1
        page_size = 100

        while True:
            params = {
                "pn": page,
                "pz": page_size,
                "po": 1,
                "np": 1,
                "fltt": 2,
                "invt": 2,
                "fid": "f3",
                "fs": fs,
                "fields": "f12,f14",
            }
            url = "https://push2.eastmoney.com/api/qt/clist/get?" + urllib.parse.urlencode(params)
            payload = self._fetch_json_with_retry(url, retries=3)
            if payload is None:
                logger.warning("[Screener] 东方财富股票列表获取失败 fs=%s page=%s", fs, page)
                break

            data = payload.get("data") or {}
            diff = data.get("diff") or []
            if not diff:
                break

            rows.extend(diff)
            total = int(data.get("total") or 0)
            if len(rows) >= total or len(diff) < page_size:
                break
            page += 1

        return rows

    @staticmethod
    def _fetch_json_with_retry(url: str, retries: int = 3) -> Optional[dict]:
        """带重试的 JSON 请求，供免费行情列表兜底使用。"""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://quote.eastmoney.com/",
            "Connection": "close",
        }
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=10) as response:
                    return json.loads(response.read().decode("utf-8"))
            except Exception as e:
                last_error = e
                time.sleep(min(2, attempt * 0.5))

        logger.warning("[Screener] JSON 请求失败: %s", last_error)
        return None

    def _build_pool_from_host(self, host: str, port: int) -> Tuple[List[Tuple[str, str]], str]:
        """从单个通达信服务器构建主板股票池。"""
        from pytdx.hq import TdxHq_API

        api = TdxHq_API()
        pool_map = {}
        raw_counts = {0: 0, 1: 0}

        try:
            if not api.connect(host, port, time_out=self.CONNECT_TIMEOUT):
                return [], "connect_false"

            for market in (0, 1):
                start = 0
                while True:
                    stocks = api.get_security_list(market, start) or []
                    raw_counts[market] += len(stocks)
                    for stock in stocks:
                        code = str(stock.get('code', ''))
                        name = str(stock.get('name', ''))
                        if not code or not name:
                            continue
                        if not self._is_main_board_stock(market, code):
                            continue
                        key = f"{market}:{code}"
                        pool_map[key] = (code, name)
                    if len(stocks) < PytdxFetcher.SECURITY_LIST_PAGE_SIZE:
                        break
                    start += PytdxFetcher.SECURITY_LIST_PAGE_SIZE
        finally:
            try:
                api.disconnect()
            except Exception:
                pass

        sz_main = sum(1 for key in pool_map if key.startswith("0:"))
        sh_main = sum(1 for key in pool_map if key.startswith("1:"))
        stats = f"深市主板={sz_main}, 沪市主板={sh_main}, 深市原始={raw_counts[0]}, 沪市原始={raw_counts[1]}"
        return list(pool_map.values()), stats

    # ==================== 单只股票处理 ====================

    def _process_stock(self, code: str, name: str) -> Optional[ScreenerResult]:
        """处理单只股票"""
        df = self._fetch_daily_data(code)
        if df is None or len(df) < self.MIN_TRADING_DAYS:
            return None

        close = df["close"].values
        latest_close = float(close[-1])

        # 条件 B: 股价在 60 日线上方
        ma60 = pd.Series(close).rolling(60).mean().iloc[-1]
        if np.isnan(ma60) or latest_close <= ma60:
            return None
        ma60_diff_pct = (latest_close - ma60) / ma60 * 100

        # 计算 MACD
        dif, dea, macd = self._compute_macd(close)
        current_macd = float(macd[-1])

        # 条件 A: MACD 在 -3 ~ 0
        if not (self.MACD_MIN < current_macd < self.MACD_MAX):
            return None

        # 条件 C: 金叉→死叉涨幅
        crosses = self._detect_crosses(dif, dea, df)
        cross_rise = self._check_cross_rise(close, crosses)
        if cross_rise is None:
            return None
        rise_pct, golden_index, death_index = cross_rise
        if rise_pct < self.MIN_RISE_PCT:
            return None

        score = self._calc_score(current_macd, ma60_diff_pct, rise_pct)

        return ScreenerResult(
            code=code, name=name, score=score,
            macd=current_macd, dif=float(dif[-1]), dea=float(dea[-1]),
            price=latest_close, ma60=float(ma60),
            ma60_diff_pct=float(ma60_diff_pct),
            rise_since_golden=rise_pct,
            golden_index=golden_index,
            death_index=death_index,
        )

    # ==================== 数据获取 ====================

    def _fetch_daily_data(self, code: str) -> Optional[pd.DataFrame]:
        """
        获取日线数据（直连通达信，绕过 DataFetcherManager）

        PytdxFetcher 只实现了 _fetch_raw_data + _normalize_data，
        没有公开 get_daily_data，所以此处直接调用通达信 API。
        """
        market = self._stock_market(code)
        last_error = None
        for _ in range(2):
            try:
                api, host, port = self._get_thread_api()
                data = api.get_security_bars(
                    category=9,  # 日线
                    market=market,
                    code=code,
                    start=0,
                    count=240,  # 约一年交易日，避免最新金叉/死叉窗口过短。
                )
                if data is None or len(data) == 0:
                    return None

                df = api.to_df(data)
                df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
                df = df.sort_values("date").reset_index(drop=True)

                # 标准化成交量列名，便于后续扩展复用。
                if "vol" in df.columns:
                    df = df.rename(columns={"vol": "volume"})

                return df
            except Exception as e:
                last_error = e
                logger.debug("[Screener] %s 日线获取失败，重置线程连接: %s", code, e)
                self._reset_thread_api()

        if last_error is not None:
            logger.debug("[Screener] %s 日线获取最终失败: %s", code, last_error)
        return None

    # ==================== MACD 计算 ====================

    def _compute_macd(self, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算 MACD 指标"""
        ema_fast = self._ema(close, self.MACD_FAST)
        ema_slow = self._ema(close, self.MACD_SLOW)
        dif = ema_fast - ema_slow
        dea = self._ema(dif, self.MACD_SIGNAL)
        macd = (dif - dea) * 2
        return dif, dea, macd

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """计算指数移动平均"""
        result = np.zeros_like(data, dtype=float)
        multiplier = 2.0 / (period + 1)
        valid_indexes = np.flatnonzero(~np.isnan(data))
        if len(valid_indexes) < period:
            result[:] = np.nan
            return result

        start = valid_indexes[period - 1]
        seed_indexes = valid_indexes[:period]
        result[:start] = np.nan
        result[start] = np.mean(data[seed_indexes])
        for i in range(start + 1, len(data)):
            if np.isnan(data[i]):
                result[i] = result[i - 1]
                continue
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result

    @staticmethod
    def _is_main_board_stock(market: int, code: str) -> bool:
        """仅保留沪深主板个股，排除科创板、创业板、北交所、指数和基金。"""
        if market == 0:
            return code.startswith(("000", "001", "002", "003"))
        if market == 1:
            return code.startswith(("600", "601", "603", "605"))
        return False

    @staticmethod
    def _stock_market(code: str) -> int:
        """根据 A 股代码判断通达信市场：0=深圳，1=上海。"""
        if code.startswith(("600", "601", "603", "605", "688")):
            return 1
        return 0

    def _tdx_hosts(self) -> List[Tuple[str, int]]:
        """合并环境变量、项目默认、pytdx 自带和补充通达信主站。"""
        hosts: List[Tuple[str, int]] = []
        env_hosts = _parse_hosts_from_env()
        if env_hosts:
            hosts.extend(env_hosts)

        hosts.extend(getattr(PytdxFetcher, "DEFAULT_HOSTS", []))
        hosts.extend(self._pytdx_config_hosts())
        hosts.extend(self.EXTRA_TDX_HOSTS)

        seen = set()
        unique_hosts = []
        for host, port in hosts:
            key = (str(host).strip(), int(port))
            if not key[0] or key in seen:
                continue
            seen.add(key)
            unique_hosts.append(key)
        return unique_hosts

    def _ordered_data_hosts(self) -> List[Tuple[str, int]]:
        """日线数据优先使用股票池构建成功且数量最完整的节点。"""
        hosts = self._preferred_data_hosts + self._tdx_hosts()
        seen = set()
        result = []
        for host, port in hosts:
            key = (str(host).strip(), int(port))
            if not key[0] or key in seen:
                continue
            seen.add(key)
            result.append(key)
        return result

    def _get_thread_api(self):
        """每个筛选线程复用自己的通达信连接，避免每只股票重复 TCP 连接。"""
        session = getattr(self._thread_local, "tdx_session", None)
        if session is not None:
            return session

        from pytdx.hq import TdxHq_API

        last_error = None
        for host, port in self._ordered_data_hosts():
            api = TdxHq_API()
            try:
                if api.connect(host, port, time_out=self.CONNECT_TIMEOUT):
                    session = (api, host, port)
                    self._thread_local.tdx_session = session
                    with self._session_lock:
                        self._worker_sessions.append(api)
                    logger.debug("[Screener] 线程通达信连接成功: %s:%s", host, port)
                    return session
                last_error = f"{host}:{port}=connect_false"
            except Exception as e:
                last_error = e
                try:
                    api.disconnect()
                except Exception:
                    pass

        raise RuntimeError(f"无法连接可用通达信日线服务器: {last_error}")

    def _reset_thread_api(self) -> None:
        """当前线程的通达信连接异常时断开并在下次请求重连。"""
        session = getattr(self._thread_local, "tdx_session", None)
        if session is None:
            return

        api, _, _ = session
        try:
            api.disconnect()
        except Exception:
            pass
        self._thread_local.tdx_session = None

    def _disconnect_worker_sessions(self) -> None:
        """筛选完成后关闭所有 worker 持有的通达信连接。"""
        with self._session_lock:
            sessions = list(self._worker_sessions)
            self._worker_sessions.clear()

        for api in sessions:
            try:
                api.disconnect()
            except Exception:
                pass

    @staticmethod
    def _pytdx_config_hosts() -> List[Tuple[str, int]]:
        """读取 pytdx 包内置 host 配置，兼容不同版本的数据结构。"""
        try:
            from pytdx.config.hosts import hq_hosts
        except Exception:
            return []

        result: List[Tuple[str, int]] = []
        for item in hq_hosts:
            if isinstance(item, dict):
                host = item.get("ip") or item.get("host")
                port = item.get("port", 7709)
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                host, port = item[0], item[1]
            else:
                continue
            try:
                result.append((str(host), int(port)))
            except Exception:
                continue
        return result

    # ==================== 金叉/死叉检测 ====================

    def _detect_crosses(self, dif: np.ndarray, dea: np.ndarray, df: pd.DataFrame) -> List[CrossEvent]:
        """检测 MACD 金叉和死叉"""
        closes = df["close"].values
        crosses = []
        for i in range(1, len(dif)):
            if np.isnan(dif[i]) or np.isnan(dea[i]) or np.isnan(dif[i - 1]) or np.isnan(dea[i - 1]):
                continue
            prev_diff = dif[i - 1] - dea[i - 1]
            curr_diff = dif[i] - dea[i]
            if prev_diff < 0 and curr_diff >= 0:
                crosses.append(CrossEvent(index=i, cross_type="golden",
                    dif=float(dif[i]), dea=float(dea[i]), price=float(closes[i])))
            elif prev_diff > 0 and curr_diff <= 0:
                crosses.append(CrossEvent(index=i, cross_type="death",
                    dif=float(dif[i]), dea=float(dea[i]), price=float(closes[i])))
        return crosses

    # ==================== 涨幅计算 ====================

    def _check_cross_rise(self, close: np.ndarray, crosses: List[CrossEvent]) -> Optional[Tuple[float, int, int]]:
        """计算最新完整金叉→死叉区间内的最大收盘涨幅"""
        if len(crosses) < 2:
            return None

        second_last = crosses[-2]
        last = crosses[-1]

        if second_last.cross_type == "golden" and last.cross_type == "death":
            golden, death = second_last, last
        elif second_last.cross_type == "death" and last.cross_type == "golden":
            if len(crosses) >= 3 and crosses[-3].cross_type == "golden":
                golden, death = crosses[-3], second_last
            else:
                return None
        else:
            return None

        if golden.price <= 0 or golden.index >= death.index:
            return None

        window = close[golden.index:death.index + 1]
        if len(window) == 0 or np.all(np.isnan(window)):
            return None

        max_close = float(np.nanmax(window))
        return (max_close - golden.price) / golden.price, golden.index, death.index

    # ==================== 综合评分 ====================

    def _calc_score(self, macd: float, ma60_diff_pct: float, rise_pct: float) -> int:
        """计算综合评分（0-100），仅用于排序"""
        score = 0

        # MACD 偏离度（0-30 分）：越接近 -3 分越高（回调越充分）
        macd_norm = max(0, min(1, (macd - self.MACD_MIN) / (self.MACD_MAX - self.MACD_MIN)))
        score += int((1 - macd_norm) * 30)

        # MA60 安全边际（0-25 分）：刚站上 MA60 分数最高
        if 0 < ma60_diff_pct <= 1:
            score += 25
        elif ma60_diff_pct <= 8:
            score += max(5, int((1 - (ma60_diff_pct - 1) / 7) * 25))
        else:
            score += 5

        # 涨幅质量（0-25 分）：40%-80% 为理想区间
        if 0.40 <= rise_pct <= 0.80:
            score += max(10, int((1 - (rise_pct - 0.40) / 0.40) * 25))
        elif rise_pct <= 1.50:
            score += 8
        else:
            score += 3

        # 回踩充分度（0-20 分）：越靠近 MA60 越好
        dist = abs(ma60_diff_pct)
        if dist <= 3:
            score += 20
        elif dist <= 8:
            score += 12
        elif dist <= 15:
            score += 6
        else:
            score += 2

        return min(100, max(0, score))
