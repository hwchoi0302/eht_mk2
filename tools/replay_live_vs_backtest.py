"""
tools/replay_live_vs_backtest.py — 라이브 판단 경로를 백테스트 신호와 날짜별로 대조한다.

모의 매매 기간에 확인할 것은 수익이 아니라 "라이브가 백테스트와 같은 판단을
하는가"다. 이 도구는 과거 각 날짜에 대해 라이브 봇이 거쳤을 경로
(최근 REGIME_LOOKBACK봉만 받아 지표 계산 → 국면 확정 → 해당 국면 전략 신호 →
국면 전환 봉 게이팅)를 그대로 재현하고, 전 이력으로 돌린 백테스트 신호와 비교한다.

2026-10 수정 전에는 최근 399일 중 17일이 어긋났다(국면 전환 당일 재진입).

    python tools/replay_live_vs_backtest.py
    python tools/replay_live_vs_backtest.py --config config/regime_config_BTC-USDT_futures_1d.json --days 400
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.indicators import add_all_indicators, confirm_regimes, REGIME_LOOKBACK
from core.strategies import RegimeSwitchingStrategy, get_strategy_by_name
from live.regime_bot import RegimeLiveTrader
from research.strategy_lab import load_candles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/regime_config_BTC-USDT_futures_1d.json')
    ap.add_argument('--days', type=int, default=400)
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    confirm = cfg.get('regime_confirm_candles', 1)
    raw = load_candles(cfg['symbol'], cfg['timeframe'], cfg.get('is_futures', True),
                       with_indicators=False)
    full = add_all_indicators(raw)
    bt = RegimeSwitchingStrategy(regime_strategies=cfg['regime_strategies'],
                                 regime_confirm_candles=confirm).generate_signals(full)

    mismatches, n = [], 0
    for t in range(max(1, len(raw) - a.days), len(raw) - 1):
        # 라이브는 진행 중인 봉을 포함해 받으므로, 확정 봉 t가 iloc[-2]에 오도록 자른다
        df = raw.iloc[max(0, t + 2 - REGIME_LOOKBACK):t + 2].reset_index(drop=True)
        di = add_all_indicators(df)
        conf = confirm_regimes(di['regime'], confirm)
        blk = cfg['regime_strategies'][conf.iloc[-2]]
        sub = get_strategy_by_name(blk['strategy_name'], **blk['strategy_params'])
        live = RegimeLiveTrader._gate_regime_change(None, sub.generate_signals(di).iloc[-2], conf)
        n += 1
        if live != bt.iloc[t]:
            mismatches.append((full['datetime'].iloc[t].date(), conf.iloc[-2],
                               int(live), int(bt.iloc[t])))

    print(f"{cfg['symbol']} {cfg['timeframe']} — 최근 {n}봉 재생, 불일치 {len(mismatches)}건")
    for d, reg, lv, bv in mismatches[:20]:
        print(f"  {d}  국면 {reg:<9} 라이브 {lv:+d}  백테스트 {bv:+d}")
    return 1 if mismatches else 0


if __name__ == '__main__':
    sys.exit(main())
