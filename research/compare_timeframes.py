"""
research/compare_timeframes.py — 타임프레임별 워크포워드 결과를 나란히 놓고 본다.

회전율이 성과를 지배하는지 확인하기 위한 도구다. 4h와 1d를 같은 기준으로
비교하면 "비용을 넘는 알파가 있는가"와 "그냥 덜 거래하면 되는가"를 구분할 수 있다.

    python research/compare_timeframes.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.paths import REPORTS


def load(symbol, timeframe):
    path = REPORTS / f"wfa_{symbol.replace('/', '-')}_{timeframe}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def index_by(data, cost):
    return {r['strategy']: r for r in data['results'] if r['cost_multiplier'] == cost}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='BTC/USDT')
    ap.add_argument('--timeframes', default='4h,1d')
    args = ap.parse_args()

    tfs = args.timeframes.split(',')
    loaded = {tf: load(args.symbol, tf) for tf in tfs}
    missing = [tf for tf, d in loaded.items() if d is None]
    if missing:
        raise SystemExit(f"결과 없음: {missing}. research/run_wfa.py 를 먼저 돌리세요.")

    base = {tf: index_by(d, 1.0) for tf, d in loaded.items()}
    stress = {tf: index_by(d, 2.0) for tf, d in loaded.items()}

    names = sorted(set().union(*(set(b) for b in base.values())))

    print(f"=== {args.symbol} 타임프레임 비교 (아웃샘플) ===\n")
    header = f"{'전략':<22}"
    for tf in tfs:
        header += f"{tf+' 수익':>11}{tf+' 샤프':>9}{tf+' 회전':>9}{tf+' x2':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for name in names:
        line = f"{name:<22}"
        best_sharpe = -99
        for tf in tfs:
            r = base[tf].get(name)
            s = stress[tf].get(name)
            if not r:
                line += f"{'—':>11}{'—':>9}{'—':>9}{'—':>10}"
                continue
            line += (f"{r['oos_total_return']:>10.1%}"
                     f"{r['oos_sharpe']:>9.2f}"
                     f"{r['oos_turnover']:>9.0f}"
                     f"{s['oos_total_return'] if s else 0:>10.1%}")
            best_sharpe = max(best_sharpe, r['oos_sharpe'])
        rows.append((best_sharpe, line))

    for _, line in sorted(rows, reverse=True):
        print(line)

    print()
    for tf in tfs:
        d = loaded[tf]
        any_r = next(iter(base[tf].values()), None)
        if any_r:
            print(f"{tf} 구간 Buy & Hold 복리: {any_r['buy_hold_compounded']:+.2%}")

    # 합격선 통과자
    print("\n=== 합격선 통과 (수익>0, 샤프>0, 일관성>=50%, 비용2배에서도 수익>0) ===")
    found = False
    for tf in tfs:
        for name, r in base[tf].items():
            s = stress[tf].get(name)
            if (r['oos_total_return'] > 0 and r['oos_sharpe'] > 0
                    and r['oos_consistency'] >= 0.5
                    and s and s['oos_total_return'] > 0):
                found = True
                print(f"  [{tf}] {name}: {r['oos_total_return']:+.2%} "
                      f"(샤프 {r['oos_sharpe']:+.2f}, 일관성 {r['oos_consistency']:.0%}, "
                      f"회전 {r['oos_turnover']:.0f}, 비용2배 {s['oos_total_return']:+.2%})")
    if not found:
        print("  없음")


if __name__ == '__main__':
    main()
