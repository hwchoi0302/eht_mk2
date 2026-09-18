"""
research/make_config.py — 워크포워드 결과에서 라이브 설정 파일을 만든다.

구간마다 고른 파라미터가 다르므로, 그중 하나를 손으로 베껴 쓰면 결국
"가장 좋아 보이는 구간"을 고르게 된다. 여기서는 전 구간에 걸쳐 가장 자주
선택된 값(수치형은 중앙값)을 취한다 — 고원 선택을 구간 축으로 한 번 더 하는 셈이다.

    python research/make_config.py --strategy "시장국면 동적결합" --timeframe 1d
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.paths import REPORTS, CONFIG


def consensus_params(param_dicts):
    """구간별로 고른 파라미터들을 하나로 합친다."""
    if not param_dicts:
        return {}
    keys = set().union(*(set(d) for d in param_dicts))
    out = {}
    for key in keys:
        values = [d[key] for d in param_dicts if key in d]
        if not values:
            continue
        numeric = [v for v in values
                   if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(numeric) == len(values):
            median = float(np.median(numeric))
            # 실제로 선택된 적 있는 값 중 중앙값에 가장 가까운 것으로 스냅
            out[key] = min(numeric, key=lambda v: abs(v - median))
        else:
            out[key] = Counter(values).most_common(1)[0][0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='BTC/USDT')
    ap.add_argument('--timeframe', default='1d')
    ap.add_argument('--strategy', required=True)
    ap.add_argument('--cost', type=float, default=1.0)
    ap.add_argument('--confirm-candles', type=int, default=2)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    tag = f"{args.symbol.replace('/', '-')}_{args.timeframe}"
    src = REPORTS / f"wfa_{tag}.json"
    if not src.exists():
        raise SystemExit(f"결과 파일 없음: {src}")

    data = json.loads(src.read_text())
    match = [r for r in data['results']
             if r['strategy'] == args.strategy and r['cost_multiplier'] == args.cost]
    if not match:
        names = sorted({r['strategy'] for r in data['results']})
        raise SystemExit(f"'{args.strategy}' 결과 없음.\n사용 가능: {names}")

    result = match[0]
    chosen = consensus_params(result.get('chosen_params', []))

    print(f"전략: {args.strategy} ({args.symbol} {args.timeframe})")
    print(f"아웃샘플: {result['oos_total_return']:+.2%} | "
          f"샤프 {result['oos_sharpe']:+.2f} | MDD {result['oos_mdd']:.1%} | "
          f"일관성 {result['oos_consistency']:.0%} | 회전 {result['oos_turnover']:.0f}")
    print(f"\n{len(result.get('chosen_params', []))}개 구간의 합의 파라미터:")
    for k, v in sorted(chosen.items()):
        print(f"  {k}: {v}")

    risk = {k: chosen.get(k) for k in
            ('leverage', 'stop_loss_pct', 'take_profit_pct', 'max_allocation_pct')
            if chosen.get(k) is not None}

    if args.strategy == '시장국면 동적결합':
        # 국면별 하위 전략 구조로 펼친다
        cfg = {
            'symbol': args.symbol,
            'is_futures': True,
            'timeframe': args.timeframe,
            'regime_confirm_candles': args.confirm_candles,
            'regime_strategies': {
                'BULL': {
                    'strategy_name': '듀얼 모멘텀',
                    'strategy_params': {
                        **risk,
                        'lookback_period': chosen.get('bull_lookback', 46),
                        'trend_period': chosen.get('bull_trend', 94),
                    },
                },
                'BEAR': {
                    'strategy_name': '삼중 EMA',
                    'strategy_params': {
                        **risk,
                        'fast_period': chosen.get('bear_fast', 20),
                        'mid_period': chosen.get('bear_mid', 49),
                        'slow_period': chosen.get('bear_slow', 96),
                    },
                },
                'SIDEWAYS': {
                    'strategy_name': 'Z-Score 평균회귀',
                    'strategy_params': {
                        **risk,
                        'period': chosen.get('side_period', 20),
                        'z_threshold': chosen.get('side_z_threshold', 2.0),
                    },
                },
            },
        }
    else:
        # 단일 전략은 세 국면 모두 같은 전략을 쓴다 (국면 전환 시 청산만 발생)
        params = {k: v for k, v in chosen.items()}
        cfg = {
            'symbol': args.symbol,
            'is_futures': True,
            'timeframe': args.timeframe,
            'regime_confirm_candles': args.confirm_candles,
            'regime_strategies': {
                regime: {'strategy_name': args.strategy, 'strategy_params': dict(params)}
                for regime in ('BULL', 'BEAR', 'SIDEWAYS')
            },
        }

    cfg['note'] = (
        f"워크포워드 재탐색 결과 (인샘플 6개월 → 아웃샘플 2개월 롤링, "
        f"{result['folds']}개 구간). 아웃샘플 {result['oos_total_return']:+.1%}, "
        f"샤프 {result['oos_sharpe']:+.2f}, 일관성 {result['oos_consistency']:.0%}, "
        f"연 회전율 {result['oos_turnover']:.0f}회. 파라미터는 구간별 선택값의 합의치."
    )
    cfg['generated_at'] = datetime.now().isoformat(timespec='seconds')

    out = args.out or str(CONFIG / f"regime_config_{args.symbol.replace('/', '-')}"
                                   f"_futures_{args.timeframe}.json")
    with open(out, 'w') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out}")


if __name__ == '__main__':
    main()
