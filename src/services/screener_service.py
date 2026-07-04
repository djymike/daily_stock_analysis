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
import time
from datetime import datetime, timedelta
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
    MAX_WORKERS = 10

   def __init__(self):
       """初始化筛选服务，直接使用 pytdx 直连通达信"""
       pass

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
                        "[Screener] 进度: %s/%s (%.1f%%), 已符合 %s 只, 速率 %.0f 只/秒",
                        done, total, done / total * 100, len(results), rate,
                    )

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

        排除规则：
        - 688（科创板）
        - 300（创业板）
        - 8（北交所）
        - 4（老三板）
        - 2（B股/B股转H股）
        """
        from pytdx.hq import TdxHq_API

        hosts = _parse_hosts_from_env()
        if hosts is None:
            hosts = PytdxFetcher.DEFAULT_HOSTS

        api = TdxHq_API()
        connected = False
        try:
            for host, port in hosts:
                try:
                    if api.connect(host, port, time_out=5):
                        connected = True
                        break
                except Exception:
                    continue

            if not connected:
                logger.error("[Screener] 无法连接任何通达信服务器")
                return []

            pool_map = {}
            for market in (0, 1):
                start = 0
                while True:
                    stocks = api.get_security_list(market, start) or []
                    for stock in stocks:
                        code = str(stock.get('code', ''))
                        name = str(stock.get('name', ''))
                        if not code or not name:
                            continue
                        if code.startswith(("688", "300", "8", "4", "2")):
                            continue
                        pool_map[code] = name
                    if len(stocks) < PytdxFetcher.SECURITY_LIST_PAGE_SIZE:
                        break
                    start += PytdxFetcher.SECURITY_LIST_PAGE_SIZE
        finally:
            try:
                api.disconnect()
            except Exception:
                pass

        pool = list(pool_map.items())
        logger.info("[Screener] 股票池构建完成: 有效=%s", len(pool))
        return pool

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
        rise_pct = self._check_cross_rise(close, crosses)
        if rise_pct is None or rise_pct < self.MIN_RISE_PCT:
            return None

        score = self._calc_score(current_macd, ma60_diff_pct, rise_pct)

        return ScreenerResult(
            code=code, name=name, score=score,
            macd=current_macd, dif=float(dif[-1]), dea=float(dea[-1]),
            price=latest_close, ma60=float(ma60),
            ma60_diff_pct=float(ma60_diff_pct),
            rise_since_golden=rise_pct,
            golden_index=crosses[-2].index if len(crosses) >= 2 else crosses[-1].index,
            death_index=crosses[-1].index,
        )

    # ==================== 数据获取 ====================

   def _fetch_daily_data(self, code: str) -> Optional[pd.DataFrame]:
       """
       获取日线数据（直连通达信，绕过 DataFetcherManager）

       PytdxFetcher 只实现了 _fetch_raw_data + _normalize_data，
       没有公开 get_daily_data，所以此处直接调用通达信 API。
       """
       from pytdx.hq import TdxHq_API

       # 计算日期范围（90 个自然日→约 60 个交易日，取 120 天确保足够）
       end = datetime.now()
       start = end - timedelta(days=120)
       start_str = start.strftime("%Y-%m-%d")
       end_str = end.strftime("%Y-%m-%d")

       # 判断市场代码
       if code.startswith(("60", "68")):
           market = 1  # 上海
       else:
           market = 0  # 深圳

       hosts = _parse_hosts_from_env()
       if hosts is None:
           hosts = PytdxFetcher.DEFAULT_HOSTS

       api = TdxHq_API()
       try:
           api.connect(hosts[0][0], hosts[0][1], time_out=5)
           data = api.get_security_bars(
               category=9,  # 日线
               market=market,
               code=code,
               start=0,
               count=80  # 80 条 ≈ 3-4 个月交易日
           )
           if data is None or len(data) == 0:
               return None

           df = api.to_df(data)
           df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
           df = df.sort_values("date").reset_index(drop=True)

           # 标准化列名
           if "vol" in df.columns:
               df = df.rename(columns={"vol": "volume"})

           return df
       except Exception as e:
           logger.debug("[Screener] %s 日线获取失败: %s", code, e)
           return None
       finally:
           try:
               api.disconnect()
           except Exception:
               pass

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
        result[:period] = np.nan
        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
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

    def _check_cross_rise(self, close: np.ndarray, crosses: List[CrossEvent]) -> Optional[float]:
        """计算最新金叉→死叉之间的涨幅"""
        if len(crosses) < 2:
            return None

        second_last = crosses[-2]
        last = crosses[-1]

        if second_last.cross_type == "golden" and last.cross_type == "death":
            golden_price, death_price = second_last.price, last.price
        elif second_last.cross_type == "death" and last.cross_type == "golden":
            if len(crosses) >= 3 and crosses[-3].cross_type == "golden":
                golden_price, death_price = crosses[-3].price, second_last.price
            else:
                return None
        else:
            return None

        if golden_price <= 0:
            return None
        return (death_price - golden_price) / golden_price

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
