"""
tools/backtest_live_config.py — 라이브 봇과 완전히 같은 설정으로 백테스트한다.

config/regime_config_*.json 을 그대로 읽어 RegimeSwitchingStrategy에 넘긴다.
매매 로직은 건드리지 않는 리포팅 전용 스크립트다.

    python tools/backtest_live_config.py
    python tools/backtest_live_config.py --config config/regime_config_BTC-USDT_futures_4h.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.paths import CONFIG
from core.strategies import RegimeSwitchingStrategy
from research.backtester import Backtester
from research.binance_env import build_funding_lookup
from research.strategy_lab import load_candles, slice_period


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(CONFIG / 'regime_config_BTC-USDT_futures_4h.json'))
    ap.add_argument('--capital', type=float, default=10000.0)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    symbol = cfg['symbol']
    timeframe = cfg['timeframe']
    is_futures = cfg.get('is_futures', True)
    confirm = cfg.get('regime_confirm_candles', 1)

    print(f"설정: {os.path.basename(args.config)}")
    print(f"  {symbol} {timeframe} {'선물' if is_futures else '현물'} | 국면확정 {confirm}봉")
    for regime, block in cfg['regime_strategies'].items():
        print(f"  {regime:<9} {block['strategy_name']}")
    print()

    df = load_candles(symbol, timeframe, is_futures)
    funding = build_funding_lookup(symbol)

    def build():
        # 설정 블록을 그대로 넘긴다. 예전에는 이 키가 조용히 무시돼서
        # "라이브 설정 백테스트"가 기본 파라미터로 돌고 있었다.
        return RegimeSwitchingStrategy(
            regime_strategies=cfg['regime_strategies'],
            regime_confirm_candles=confirm,
        )

    def run(start, end, label):
        d = slice_period(df, start, end)
        if len(d) < 100:
            print(f"{label}: 데이터 부족 ({len(d)}봉)")
            return None
        bt = Backtester(initial_capital=args.capital, symbol=symbol,
                        funding_lookup=funding, regime_confirm_candles=confirm)
        m, eq, trades = bt.run(d, build(), is_futures=is_futures)
        print(f"{label} ({d.datetime.iloc[0].date()} ~ {d.datetime.iloc[-1].date()})")
        print(f"  전략 {m['total_return']:+.2%}  |  Buy&Hold {m['buy_hold_return']:+.2%}")
        print(f"  MDD {m['max_drawdown']:.2%}  샤프 {m['sharpe_ratio']:.2f}  "
              f"거래 {m['total_trades']}건  승률 {m['win_rate']:.1%}  PF {m['profit_factor']:.2f}")
        print(f"  수수료 {m['total_fees']:,.0f}  펀딩 {m['total_funding']:+,.0f}  "
              f"회전율 {m['annual_turnover']:.0f}회/년  노출 {m['exposure_pct']:.0%}")
        return m

    print("=== 연도별 ===")
    for s, e in [('2023-09-18', '2024-09-18'), ('2024-09-18', '2025-09-18'),
                 ('2025-09-18', '2026-09-18')]:
        run(s, e, f"{s[:4]}~{e[:4]}")
        print()

    print("=== 전체 구간 ===")
    m_all = run('2023-09-18', '2026-09-18', "3년")
    print()

    print("=== 비용 스트레스 (전체 구간) ===")
    d = slice_period(df, '2023-09-18', '2026-09-18')
    for mult in (1.0, 1.5, 2.0, 3.0):
        bt = Backtester(initial_capital=args.capital, symbol=symbol,
                        funding_lookup=funding, cost_multiplier=mult,
                        regime_confirm_candles=confirm)
        m, _, _ = bt.run(d, build(), is_futures=is_futures)
        print(f"  비용 x{mult}: {m['total_return']:+.2%}  "
              f"샤프 {m['sharpe_ratio']:+.2f}  MDD {m['max_drawdown']:.2%}")

    print()
    print("=== 6개월 롤링 ===")
    for start in pd.date_range('2023-10-01', '2026-03-18', freq='3MS'):
        run(start, start + pd.DateOffset(months=6), f"  {start.date()}")


if __name__ == '__main__':
    main()
