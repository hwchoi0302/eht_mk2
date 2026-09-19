"""
research/explore.py — 샤프 개선을 목표로 여러 알고리즘 계열을 탐색한다.

기존 결론(1d 국면전환, 샤프 0.73, 연환산 24.5%)이 Buy & Hold와 거의 같은
수익에 낙폭만 조금 낫다는 수준이라, 다른 계열을 폭넓게 본다.

탐색하는 축:

  1. 변동성 타겟팅 — 실현변동성에 반비례해 크기 조절. 업계에서 샤프를 올리는
     가장 확실한 손잡이다. 고정 배분은 급변동 구간에서 위험을 과하게 진다.
  2. ATR 손절 — 고정 퍼센트 대신 변동성 비례.
  3. 횡단면 로테이션 — BTC/ETH 중 상대적으로 강한 쪽에 붙는다. 둘 다 애매하면 쉰다.
  4. 펀딩 캐리 — 4년 평균 연 +6.97% 콘탱고. 숏이 캐리를 받는다.

평가는 모두 워크포워드 아웃샘플이다 (인샘플 6개월 → 아웃샘플 2개월 롤링).

    python research/explore.py single --workers 12
    python research/explore.py rotation --workers 12
"""

import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.paths import REPORTS
from core.strategies import get_strategy_by_name
from research.binance_env import build_funding_lookup, load_funding_history
from research.portfolio_backtester import PortfolioBacktester
from research.strategy_lab import (
    load_candles, make_folds, slice_with_warmup, restrict_metrics,
    plateau_pick, score, get_funding,
)
from research.backtester import Backtester

TIMEFRAMES = ['4h', '8h', '12h', '1d', '3d']


# ─────────────────────────────────────────────────────────────────────────────
# 1) 단일 종목 — 새 전략 계열
# ─────────────────────────────────────────────────────────────────────────────

GRIDS = {
    '변동성타겟 추세': {
        'fast_period': [10, 20, 50], 'slow_period': [100, 200],
        'target_vol': [0.20, 0.40], 'atr_sl_mult': [2.0, 4.0],
    },
    '도니안 변동성타겟': {
        'entry_period': [20, 55], 'exit_period': [10, 20],
        'target_vol': [0.20, 0.40], 'atr_sl_mult': [2.0, 4.0],
    },
    '펀딩 캐리': {
        'lookback': [21, 63], 'entry_z': [0.5, 1.0, 1.5],
        'target_vol': [0.20, 0.40],
    },
}


def expand(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def build(name, params, symbol):
    s = get_strategy_by_name(name, **params)
    if hasattr(s, 'attach_funding'):
        s.attach_funding(load_funding_history(symbol))
    return s


def eval_win(df, a, b, name, params, symbol, cost):
    w, i = slice_with_warmup(df, a, b)
    if len(w) - i < 10:
        return None
    bt = Backtester(symbol=symbol, cost_multiplier=cost,
                    funding_lookup=get_funding(symbol), regime_confirm_candles=2)
    m, e, t = bt.run(w, build(name, params, symbol), is_futures=True)
    return restrict_metrics(m, e, t, i)


def wfa_single(args):
    name, tf, symbol, cost = args
    try:
        df = load_candles(symbol, tf, is_futures=True)
        combos = [p for p in expand(GRIDS[name])
                  if p.get('fast_period', 0) < p.get('slow_period', 10**9)]
        rets, shs, turns = [], [], []
        worst = 0.0; nt = 0
        for (a, b, c) in make_folds(df):
            scored = []
            for p in combos:
                m = eval_win(df, a, b, name, p, symbol, cost)
                if m:
                    scored.append((p, score(m, min_trades=3)))
            if not scored:
                continue
            ch = plateau_pick(scored)
            m = eval_win(df, b, c, name, ch, symbol, cost)
            if not m:
                continue
            rets.append(m['total_return']); shs.append(m['sharpe_ratio'])
            turns.append(m.get('annual_turnover', 0)); nt += m['total_trades']
            worst = min(worst, m['max_drawdown'])
        if not rets:
            return None
        return {'kind': 'single', 'strategy': name, 'timeframe': tf,
                'symbol': symbol, 'cost_multiplier': cost,
                'oos_total_return': float(np.prod([1 + r for r in rets]) - 1),
                'oos_sharpe': float(np.mean(shs)), 'oos_mdd': float(worst),
                'oos_consistency': sum(1 for r in rets if r > 0) / len(rets),
                'oos_turnover': float(np.mean(turns)), 'oos_trades': nt,
                'folds': len(rets)}
    except Exception as e:
        return {'kind': 'single', 'strategy': name, 'timeframe': tf,
                'cost_multiplier': cost, 'error': f"{type(e).__name__}: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# 2) BTC/ETH 횡단면 로테이션
# ─────────────────────────────────────────────────────────────────────────────

def make_rotation_signal(frames, lookback, trend_period, allow_short, min_strength):
    """상대적으로 강한 쪽에 붙는 신호 생성기.

    각 종목의 모멘텀(lookback 수익률)을 비교해 더 강한 쪽을 고른다.
    단, 장기추세 필터를 통과해야 하고 모멘텀 크기가 min_strength를 넘어야 한다.
    둘 다 조건 미달이면 현금으로 쉰다 — 이게 단일 종목 전략과 가장 다른 점이다.
    """
    syms = list(frames)
    mom, trend = {}, {}
    for s in syms:
        c = frames[s]['close']
        mom[s] = (c / c.shift(lookback) - 1).to_numpy()
        trend[s] = (c > c.ewm(span=trend_period, adjust=False).mean()).to_numpy()

    def sig(i, A):
        if i < max(lookback, trend_period):
            return None, 0
        best, best_score = None, 0.0
        for s in syms:
            m = mom[s][i]
            if not np.isfinite(m):
                continue
            if m > 0 and trend[s][i]:
                cand, sc = 1, m
            elif allow_short and m < 0 and not trend[s][i]:
                cand, sc = -1, -m
            else:
                continue
            if sc > best_score and sc >= min_strength:
                best, best_score = (s, cand), sc
        return best if best else (None, 0)
    return sig


def wfa_rotation(args):
    tf, cost, symbols = args
    try:
        frames_full = {s: load_candles(s, tf, is_futures=True) for s in symbols}
        # 두 종목의 timestamp 축을 교집합으로 맞춘다
        common = None
        for d in frames_full.values():
            ts = set(d['timestamp'])
            common = ts if common is None else (common & ts)
        common = sorted(common)
        frames_full = {s: d[d['timestamp'].isin(common)].reset_index(drop=True)
                       for s, d in frames_full.items()}
        base = frames_full[symbols[0]]
        fl = {s: build_funding_lookup(s) for s in symbols}

        grid = [{'lookback': lb, 'trend_period': tp, 'allow_short': sh,
                 'min_strength': ms, 'target_vol': tv}
                for lb in (20, 60) for tp in (50, 100)
                for sh in (True, False) for ms in (0.0, 0.05) for tv in (0.20, 0.40)]

        folds = make_folds(base)
        rets, shs, turns = [], [], []
        worst = 0.0; nt = 0

        def run_window(a, b, params):
            mask = (base['datetime'] >= a) & (base['datetime'] < b)
            idx = np.where(mask.to_numpy())[0]
            if len(idx) < 10:
                return None
            lo = max(0, idx[0] - 260); hi = idx[-1] + 1
            fr = {s: d.iloc[lo:hi].reset_index(drop=True) for s, d in frames_full.items()}
            sigf = make_rotation_signal(fr, params['lookback'], params['trend_period'],
                                        params['allow_short'], params['min_strength'])
            bt = PortfolioBacktester(cost_multiplier=cost, target_vol=params['target_vol'],
                                     atr_sl_mult=3.0, funding_lookups=fl)
            m, eq, tr = bt.run(fr, sigf)
            ev = idx[0] - lo
            return restrict_metrics(m, eq, tr, ev)

        for (a, b, c) in folds:
            scored = []
            for p in grid:
                m = run_window(a, b, p)
                if m:
                    scored.append((p, score(m, min_trades=3)))
            if not scored:
                continue
            ch = plateau_pick(scored)
            m = run_window(b, c, ch)
            if not m:
                continue
            rets.append(m['total_return']); shs.append(m['sharpe_ratio'])
            turns.append(m.get('annual_turnover', 0)); nt += m['total_trades']
            worst = min(worst, m['max_drawdown'])

        if not rets:
            return None
        return {'kind': 'rotation', 'strategy': 'BTC/ETH 로테이션', 'timeframe': tf,
                'cost_multiplier': cost,
                'oos_total_return': float(np.prod([1 + r for r in rets]) - 1),
                'oos_sharpe': float(np.mean(shs)), 'oos_mdd': float(worst),
                'oos_consistency': sum(1 for r in rets if r > 0) / len(rets),
                'oos_turnover': float(np.mean(turns)), 'oos_trades': nt,
                'folds': len(rets)}
    except Exception as e:
        return {'kind': 'rotation', 'timeframe': tf, 'cost_multiplier': cost,
                'error': f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['single', 'rotation', 'all'])
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--timeframes', default=','.join(TIMEFRAMES))
    ap.add_argument('--costs', default='1.0,2.0')
    a = ap.parse_args()

    tfs = a.timeframes.split(',')
    costs = [float(c) for c in a.costs.split(',')]
    jobs, fn = [], {}

    if a.mode in ('single', 'all'):
        for name in GRIDS:
            for tf in tfs:
                for c in costs:
                    for sym in ('BTC/USDT', 'ETH/USDT'):
                        jobs.append(('single', (name, tf, sym, c)))
    if a.mode in ('rotation', 'all'):
        for tf in tfs:
            for c in costs:
                jobs.append(('rotation', (tf, c, ['BTC/USDT', 'ETH/USDT'])))

    print(f"탐색 {len(jobs)}작업 / 워커 {a.workers}", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(wfa_single if k == 'single' else wfa_rotation, arg): k
                for k, arg in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if not r:
                continue
            if 'error' in r:
                print(f"  ❌ {r.get('strategy','rotation')} {r['timeframe']}: {r['error']}", flush=True)
                continue
            out.append(r)
            print(f"  [{i}/{len(jobs)}] {r['strategy'][:16]:<16} {r.get('symbol','BTC+ETH')[:8]:<8} "
                  f"{r['timeframe']:<4} x{r['cost_multiplier']}: "
                  f"{r['oos_total_return']:+8.1%} 샤프 {r['oos_sharpe']:+5.2f} "
                  f"MDD {r['oos_mdd']:6.1%} 일관성 {r['oos_consistency']:.0%} "
                  f"회전 {r['oos_turnover']:.0f}", flush=True)

    p = REPORTS / f"explore_{a.mode}.json"
    p.write_text(json.dumps({'results': out}, ensure_ascii=False, indent=2, default=str))
    print(f"\n저장: {p}")


if __name__ == '__main__':
    main()
