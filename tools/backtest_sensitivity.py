"""
tools/backtest_sensitivity.py — 라이브 설정의 민감도/강건성 점검. 리포팅 전용.

비용 가정을 흔들었을 때 성과가 얼마나 버티는지 본다. 현재 전략처럼 회전율이
높으면 비용 가정이 조금만 틀려도 결론이 뒤집힌다.

    python tools/backtest_sensitivity.py
"""

import argparse
import json
import os
import sys
from collections import Counter

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
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    symbol, timeframe = cfg['symbol'], cfg['timeframe']
    confirm = cfg.get('regime_confirm_candles', 1)

    df = load_candles(symbol, timeframe, cfg.get('is_futures', True))
    funding = build_funding_lookup(symbol)

    def build():
        return RegimeSwitchingStrategy(regime_strategies=cfg['regime_strategies'],
                                       regime_confirm_candles=confirm)

    def run(start, end, **bt_kwargs):
        d = slice_period(df, start, end)
        kwargs = dict(initial_capital=10000.0, symbol=symbol,
                      funding_lookup=funding, regime_confirm_candles=confirm)
        kwargs.update(bt_kwargs)
        bt = Backtester(**kwargs)
        return bt.run(d, build(), is_futures=True)

    YEAR = ('2025-09-18', '2026-09-18')

    print("=== 최근 1년 체결 성격 ===")
    m, eq, tr = run(*YEAR)
    print(f"  거래 {m['total_trades']}건")
    print(f"  청산사유: " + ", ".join(f"{k} {v}" for k, v in
                                   Counter(t['exit_reason'] for t in tr).most_common()))
    longs = sum(1 for t in tr if t['direction'] == 'LONG')
    print(f"  평균 보유 {m['avg_hold_hours']:.1f}시간  |  롱 {longs} / 숏 {len(tr) - longs}")
    print(f"  거래대금 {m['total_volume']:,.0f} USDT "
          f"(연 회전율 {m['annual_turnover']:.0f}회)")
    print(f"  수수료 {m['total_fees']:,.1f}  펀딩 {m['total_funding']:+,.1f} USDT")

    print("\n=== 슬리피지 민감도 (최근 1년) ===")
    print("  ATR 비례 모델의 기본 계수를 바꿔 본다.")
    for base, coef, label in ((0.0001, 0.02, '기본 (1bp + 0.02×ATR%)'),
                              (0.0003, 0.02, '3bp + 0.02×ATR%'),
                              (0.0005, 0.04, '5bp + 0.04×ATR%'),
                              (0.0010, 0.08, '10bp + 0.08×ATR%')):
        m, _, _ = run(*YEAR, slippage_base=base, slippage_atr_coef=coef)
        print(f"  {label:<26} {m['total_return']:+.2%}  샤프 {m['sharpe_ratio']:+.2f}")

    print("\n=== 펀딩비 민감도 (최근 1년) ===")
    print("  실제 이력 대신 고정 요율을 가정했을 때. 숏이 많으면 부호가 유리할 수 있다.")
    m, _, _ = run(*YEAR, apply_funding=False)
    print(f"  펀딩 없음                  {m['total_return']:+.2%}")
    m, _, _ = run(*YEAR)
    print(f"  실제 이력 (기본)           {m['total_return']:+.2%}")
    from research.binance_env import build_funding_lookup as bfl
    for rate in (0.0001, 0.0002, 0.0003):
        m, _, _ = run(*YEAR, funding_lookup=bfl(symbol, flat_rate=rate))
        print(f"  8시간당 고정 {rate*100:.2f}%      {m['total_return']:+.2%}")

    print("\n=== 비용 일괄 배수 (최근 1년) ===")
    for mult in (1.0, 1.5, 2.0, 3.0):
        m, _, _ = run(*YEAR, cost_multiplier=mult)
        print(f"  x{mult}: {m['total_return']:+.2%}  샤프 {m['sharpe_ratio']:+.2f}  "
              f"MDD {m['max_drawdown']:.2%}")

    print("\n=== 진짜 아웃오브샘플 (파라미터 확정 2026-06-24 이후) ===")
    for s, e, label in (('2026-06-24', '2026-09-18', '최적화 이후'),
                        ('2026-06-05', '2026-09-18', '재최적화 이후')):
        m, _, tr2 = run(s, e)
        days = (pd.to_datetime(e) - pd.to_datetime(s)).days
        ann = (1 + m['total_return']) ** (365 / days) - 1
        print(f"  {label} ({s}~{e}, {days}일): {m['total_return']:+.2%}  "
              f"거래 {m['total_trades']}건  (연율 {ann:+.1%})")

    print("\n=== 봇이 실제로 돌던 구간만 (다운타임 제외) ===")
    for s, e in (('2026-06-24', '2026-07-29'), ('2026-09-14', '2026-09-18')):
        m, _, _ = run(s, e)
        print(f"  {s}~{e}: {m['total_return']:+.2%}, 거래 {m['total_trades']}건")


if __name__ == '__main__':
    main()
