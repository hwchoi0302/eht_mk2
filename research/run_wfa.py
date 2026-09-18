"""
research/run_wfa.py — 전 전략 워크포워드 스윕.

인샘플 6개월에서 파라미터를 고르고 뒤따르는 아웃샘플 2개월 성과만 집계한다.
비용 1배와 2배 두 번 돌려서, 비용 가정이 틀려도 살아남는 전략만 후보로 남긴다.

    python research/run_wfa.py
    python research/run_wfa.py --symbol ETH/USDT --timeframe 4h
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.paths import REPORTS
from core.strategies import ALL_STRATEGY_NAMES
from research.strategy_lab import load_candles, walk_forward


def _one(args):
    """워커 프로세스 하나가 담당하는 일: 전략 1개 × 비용배수 1개."""
    name, symbol, timeframe, cost_mult = args
    try:
        df = load_candles(symbol, timeframe, is_futures=True)
        result = walk_forward(df, name, symbol=symbol, cost_multiplier=cost_mult)
        if result is None:
            return None
        # fold_records는 직렬화가 무거우니 요약만 남긴다
        result['fold_returns'] = [f['oos_return'] for f in result['fold_records']]
        result['chosen_params'] = [f['params'] for f in result['fold_records']]
        del result['fold_records']
        return result
    except Exception as e:
        return {'strategy': name, 'cost_multiplier': cost_mult, 'error': f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='BTC/USDT')
    ap.add_argument('--timeframe', default='4h')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--costs', default='1.0,2.0',
                    help='쉼표로 구분한 비용 배수 목록')
    args = ap.parse_args()

    costs = [float(c) for c in args.costs.split(',')]
    jobs = [(name, args.symbol, args.timeframe, c)
            for c in costs for name in ALL_STRATEGY_NAMES]

    print(f"워크포워드 스윕: {args.symbol} {args.timeframe} | "
          f"전략 {len(ALL_STRATEGY_NAMES)}개 × 비용 {costs} = {len(jobs)}작업")
    print(f"워커 {args.workers}개\n", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            job = futures[fut]
            if r is None:
                print(f"[{i}/{len(jobs)}] {job[0]} (x{job[3]}): 폴드 없음", flush=True)
                continue
            if 'error' in r:
                print(f"[{i}/{len(jobs)}] {job[0]} (x{job[3]}): ❌ {r['error']}", flush=True)
                continue
            results.append(r)
            print(f"[{i}/{len(jobs)}] {r['strategy']} (비용 x{r['cost_multiplier']}): "
                  f"OOS {r['oos_total_return']:+.2%} | 샤프 {r['oos_sharpe']:+.2f} | "
                  f"MDD {r['oos_mdd']:.1%} | 일관성 {r['oos_consistency']:.0%} | "
                  f"거래 {r['oos_trades']} | 회전 {r['oos_turnover']:.0f}", flush=True)

    tag = f"{args.symbol.replace('/', '-')}_{args.timeframe}"
    out = REPORTS / f"wfa_{tag}.json"
    out.write_text(json.dumps({
        'generated_at': datetime.now().isoformat(),
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'costs': costs,
        'results': results,
    }, ensure_ascii=False, indent=2, default=str))
    print(f"\n저장: {out}")


if __name__ == '__main__':
    main()
