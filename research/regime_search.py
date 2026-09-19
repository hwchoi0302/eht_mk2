"""
research/regime_search.py — 타임프레임 × 국면 × 전략 전수 탐색.

## 왜 2단계인가

전략 15종을 세 국면에 독립 배치하면 15³ = 3,375가지다. 여기에 타임프레임과
파라미터까지 곱하면 감당이 안 된다.

그런데 RegimeSwitchingStrategy에서 각 하위 전략은 **자기 국면에서만** 거래한다.
즉 국면끼리 거의 분리해서 평가할 수 있다. 그래서:

  1단계: 국면별로 따로 탐색한다 (15 × 3 = 45). 각 국면에서 최고를 고른다.
  2단계: 1단계 승자들을 조합해 전 구간 워크포워드로 재검증한다.

완전히 분리되지는 않는다 — 국면 전환이 포지션을 강제 청산시키는 상호작용이
남는다. 그래서 2단계 재검증이 필수다. 1단계 점수는 후보를 좁히는 용도일 뿐,
최종 판단 근거가 아니다.

## 실행

    # 1단계 (무겁다, 데스크탑에서)
    python research/regime_search.py stage1 --timeframes 1h,2h,4h,6h,8h,12h,1d,3d --workers 12

    # 2단계 (1단계 결과를 읽어 조합 검증)
    python research/regime_search.py stage2 --workers 12
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.paths import REPORTS
from core.strategies import (
    ALL_STRATEGY_NAMES, get_strategy_by_name, RegimeRestrictedStrategy,
    RegimeSwitchingStrategy,
)
from research.backtester import Backtester
from research.strategy_lab import (
    load_candles, make_folds, slice_with_warmup, restrict_metrics,
    expand_grid, valid_params, plateau_pick, score,
    STRATEGY_GRIDS, get_funding,
)

REGIMES = ('BULL', 'BEAR', 'SIDEWAYS')

# 국면 제한 평가에서는 자기 국면에서만 거래하므로 거래 수가 적다.
# 전 구간 기준과 같은 하한을 쓰면 전부 탈락한다.
MIN_TRADES_RESTRICTED = 3

# 래퍼 전략은 레지스트리에 없으므로 여기서 제외한다
SEARCHABLE = [n for n in ALL_STRATEGY_NAMES if n != '시장국면 동적결합']


# ─────────────────────────────────────────────────────────────────────────────
# 공통
# ─────────────────────────────────────────────────────────────────────────────

def _build_restricted(strategy_name, params, regime, confirm):
    base = get_strategy_by_name(strategy_name, **params)
    return RegimeRestrictedStrategy(
        base_strategy=base, target_regime=regime, regime_confirm_candles=confirm)


def _eval_window(df, start, end, strategy_factory, symbol, cost_multiplier,
                 confirm, capital=10000.0):
    """워밍업을 앞에 붙여 돌린 뒤 평가 구간 성과만 돌려준다."""
    window, eval_i = slice_with_warmup(df, start, end)
    if len(window) - eval_i < 10:
        return None
    bt = Backtester(initial_capital=capital, symbol=symbol,
                    cost_multiplier=cost_multiplier,
                    funding_lookup=get_funding(symbol),
                    regime_confirm_candles=confirm)
    metrics, equity, trades = bt.run(window, strategy_factory(), is_futures=True)
    return restrict_metrics(metrics, equity, trades, eval_i, capital)


# ─────────────────────────────────────────────────────────────────────────────
# 1단계 — 국면별 탐색
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_restricted(df, strategy_name, regime, symbol='BTC/USDT',
                            confirm=2, cost_multiplier=1.0,
                            is_months=6, oos_months=2):
    """전략 하나를 특정 국면에 제한해 워크포워드로 평가한다."""
    combos = [p for p in expand_grid(STRATEGY_GRIDS.get(strategy_name, {}),
                                     strategy_name=strategy_name)
              if valid_params(strategy_name, p)]
    folds = make_folds(df, is_months, oos_months)
    if not folds:
        return None

    oos_returns, oos_sharpes, oos_turnovers = [], [], []
    oos_trades = 0
    worst_dd = 0.0
    chosen_all = []
    # 폴드별 (인샘플 점수, 고른 파라미터). 3단계 중첩 워크포워드에서
    # "이 폴드의 인샘플만 보고 국면별 전략을 고르는" 데 쓴다.
    fold_picks = []

    for fold_idx, (is_start, is_end, oos_end) in enumerate(folds):
        scored = []
        for params in combos:
            try:
                m = _eval_window(df, is_start, is_end,
                                 lambda p=params: _build_restricted(strategy_name, p, regime, confirm),
                                 symbol, cost_multiplier, confirm)
                if m is None:
                    continue
                scored.append((params, score(m, min_trades=MIN_TRADES_RESTRICTED)))
            except Exception:
                continue
        if not scored:
            continue

        chosen = plateau_pick(scored)
        best_is_score = max(s for _p, s in scored)
        fold_picks.append({'fold': fold_idx, 'is_score': float(best_is_score),
                           'params': chosen})
        try:
            m_oos = _eval_window(df, is_end, oos_end,
                                 lambda: _build_restricted(strategy_name, chosen, regime, confirm),
                                 symbol, cost_multiplier, confirm)
        except Exception:
            continue
        if m_oos is None:
            continue

        oos_returns.append(m_oos['total_return'])
        oos_sharpes.append(m_oos['sharpe_ratio'])
        oos_turnovers.append(m_oos.get('annual_turnover', 0.0))
        oos_trades += m_oos['total_trades']
        worst_dd = min(worst_dd, m_oos['max_drawdown'])
        chosen_all.append(chosen)

    if not oos_returns:
        return None

    positive = sum(1 for r in oos_returns if r > 0)
    return {
        'strategy': strategy_name,
        'regime': regime,
        'folds': len(oos_returns),
        'oos_total_return': float(np.prod([1 + r for r in oos_returns]) - 1),
        'oos_mean_return': float(np.mean(oos_returns)),
        'oos_sharpe': float(np.mean(oos_sharpes)),
        'oos_mdd': float(worst_dd),
        'oos_trades': int(oos_trades),
        'oos_turnover': float(np.mean(oos_turnovers)),
        'oos_consistency': positive / len(oos_returns),
        'cost_multiplier': cost_multiplier,
        'chosen_params': chosen_all,
        'fold_picks': fold_picks,
    }


def _stage1_job(args):
    tf, regime, strategy_name, symbol, confirm, cost = args
    try:
        df = load_candles(symbol, tf, is_futures=True)
        r = walk_forward_restricted(df, strategy_name, regime, symbol=symbol,
                                    confirm=confirm, cost_multiplier=cost)
        if r is None:
            return None
        r['timeframe'] = tf
        return r
    except Exception as e:
        return {'timeframe': tf, 'regime': regime, 'strategy': strategy_name,
                'cost_multiplier': cost, 'error': f"{type(e).__name__}: {e}"}


def run_stage1(timeframes, symbol, workers, costs, confirm):
    jobs = [(tf, reg, name, symbol, confirm, c)
            for c in costs for tf in timeframes
            for reg in REGIMES for name in SEARCHABLE]

    print(f"1단계: 타임프레임 {len(timeframes)} × 국면 3 × 전략 {len(SEARCHABLE)} "
          f"× 비용 {len(costs)} = {len(jobs)}작업 / 워커 {workers}개", flush=True)

    results, errors = [], 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_stage1_job, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r is None:
                continue
            if 'error' in r:
                errors += 1
                if errors <= 5:
                    print(f"  [{i}/{len(jobs)}] ❌ {r['timeframe']} {r['regime']} "
                          f"{r['strategy']}: {r['error']}", flush=True)
                continue
            results.append(r)
            if i % 25 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  [{i}/{len(jobs)}] {el/60:.1f}분 경과, "
                      f"남은 예상 {el/i*(len(jobs)-i)/60:.1f}분", flush=True)

    out = REPORTS / f"regime_search_stage1_{symbol.replace('/', '-')}.json"
    out.write_text(json.dumps({'symbol': symbol, 'timeframes': timeframes,
                               'costs': costs, 'confirm': confirm,
                               'results': results}, ensure_ascii=False,
                              indent=2, default=str))
    print(f"\n저장: {out}  (성공 {len(results)} / 오류 {errors})")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2단계 — 조합 재검증
# ─────────────────────────────────────────────────────────────────────────────

def pick_best_per_regime(results, timeframe, cost=1.0):
    """1단계 결과에서 타임프레임별·국면별 1위를 고른다."""
    best = {}
    for reg in REGIMES:
        cands = [r for r in results
                 if r['timeframe'] == timeframe and r['regime'] == reg
                 and r['cost_multiplier'] == cost and r['oos_trades'] >= 5]
        if not cands:
            continue
        # 위험조정 수익 우선, 동률이면 일관성
        cands.sort(key=lambda r: (r['oos_sharpe'], r['oos_consistency']), reverse=True)
        best[reg] = cands[0]
    return best


def consensus(param_dicts):
    from collections import Counter
    if not param_dicts:
        return {}
    keys = set().union(*(set(d) for d in param_dicts))
    out = {}
    for k in keys:
        vals = [d[k] for d in param_dicts if k in d]
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(nums) == len(vals) and nums:
            med = float(np.median(nums))
            out[k] = min(nums, key=lambda v: abs(v - med))
        else:
            out[k] = Counter(vals).most_common(1)[0][0]
    return out


def build_config(best, symbol, timeframe, confirm):
    """국면별 승자로 regime_config 딕셔너리를 만든다."""
    regime_strategies = {}
    for reg in REGIMES:
        if reg not in best:
            return None
        r = best[reg]
        params = consensus(r['chosen_params'])
        regime_strategies[reg] = {
            'strategy_name': r['strategy'],
            'strategy_params': params,
        }
    return {
        'symbol': symbol,
        'is_futures': True,
        'timeframe': timeframe,
        'regime_confirm_candles': confirm,
        'regime_strategies': regime_strategies,
    }


def _stage2_job(args):
    """조합된 설정을 전 구간 워크포워드로 재검증한다 (파라미터 재탐색 없음)."""
    cfg, symbol, tf, cost, confirm = args
    try:
        df = load_candles(symbol, tf, is_futures=True)
        folds = make_folds(df)
        if not folds:
            return None

        rets, sharpes, turns = [], [], []
        trades_n = 0
        worst = 0.0
        for (_a, is_end, oos_end) in folds:
            m = _eval_window(
                df, is_end, oos_end,
                lambda: RegimeSwitchingStrategy(
                    regime_strategies=cfg['regime_strategies'],
                    regime_confirm_candles=confirm),
                symbol, cost, confirm)
            if m is None:
                continue
            rets.append(m['total_return'])
            sharpes.append(m['sharpe_ratio'])
            turns.append(m.get('annual_turnover', 0.0))
            trades_n += m['total_trades']
            worst = min(worst, m['max_drawdown'])

        if not rets:
            return None
        pos = sum(1 for r in rets if r > 0)
        return {
            'timeframe': tf, 'cost_multiplier': cost,
            'config': cfg,
            'folds': len(rets),
            'oos_total_return': float(np.prod([1 + r for r in rets]) - 1),
            'oos_sharpe': float(np.mean(sharpes)),
            'oos_mdd': float(worst),
            'oos_trades': trades_n,
            'oos_turnover': float(np.mean(turns)),
            'oos_consistency': pos / len(rets),
        }
    except Exception as e:
        return {'timeframe': tf, 'cost_multiplier': cost,
                'error': f"{type(e).__name__}: {e}"}


def run_stage2(symbol, workers, costs, confirm):
    src = REPORTS / f"regime_search_stage1_{symbol.replace('/', '-')}.json"
    if not src.exists():
        raise SystemExit(f"1단계 결과가 없습니다: {src}")
    data = json.loads(src.read_text())
    results = data['results']
    timeframes = data['timeframes']

    jobs = []
    for tf in timeframes:
        best = pick_best_per_regime(results, tf, cost=1.0)
        cfg = build_config(best, symbol, tf, confirm)
        if cfg is None:
            print(f"  {tf}: 국면별 후보가 부족해 건너뜁니다")
            continue
        print(f"  {tf}: " + " / ".join(
            f"{reg[:4]}={best[reg]['strategy']}" for reg in REGIMES))
        for c in costs:
            jobs.append((cfg, symbol, tf, c, confirm))

    print(f"\n2단계: {len(jobs)}작업 / 워커 {workers}개", flush=True)
    out_results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_stage2_job, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            if r is None:
                continue
            if 'error' in r:
                print(f"  ❌ {r['timeframe']} x{r['cost_multiplier']}: {r['error']}", flush=True)
                continue
            out_results.append(r)
            print(f"  {r['timeframe']:<4} 비용x{r['cost_multiplier']}: "
                  f"OOS {r['oos_total_return']:+.2%} | 샤프 {r['oos_sharpe']:+.2f} | "
                  f"MDD {r['oos_mdd']:.1%} | 일관성 {r['oos_consistency']:.0%} | "
                  f"회전 {r['oos_turnover']:.0f}", flush=True)

    out = REPORTS / f"regime_search_stage2_{symbol.replace('/', '-')}.json"
    out.write_text(json.dumps({'symbol': symbol, 'results': out_results},
                              ensure_ascii=False, indent=2, default=str))
    print(f"\n저장: {out}")
    return out_results



# ─────────────────────────────────────────────────────────────────────────────
# 3단계 — 중첩 워크포워드 (정직한 아웃샘플)
# ─────────────────────────────────────────────────────────────────────────────
#
# 2단계에는 두 가지 누수가 있다.
#
#   (1) 파라미터 누수 — 국면별 파라미터를 **전 폴드의 인샘플**에서 합의로 뽑은 뒤
#       모든 폴드의 아웃샘플에 적용했다. 16번 폴드의 정보가 3번 폴드의 평가에
#       들어가는 미래 참조다.
#   (2) 선택 편향 — "어느 전략을 쓸지"를 1단계의 **아웃샘플 성적**을 보고 골랐다.
#       국면마다 15개 중 최고를 아웃샘플 기준으로 뽑아 놓고 그 아웃샘플 성적을
#       그대로 보고하면, 뽑기 운이 성과로 둔갑한다.
#
# 3단계는 폴드마다 처음부터 다시 고른다.
#   각 폴드에서 → 그 폴드의 **인샘플만** 보고 국면별 전략·파라미터 선택
#               → 조합해서 그 폴드의 아웃샘플에서 평가
# 이렇게 하면 선택에 쓰인 정보와 평가에 쓰인 정보가 겹치지 않는다.


def _stage3_job(args):
    """타임프레임 하나에 대한 중첩 워크포워드."""
    tf, symbol, confirm, cost, picks_by_key = args
    try:
        df = load_candles(symbol, tf, is_futures=True)
        folds = make_folds(df)
        if not folds:
            return None

        rets, sharpes, turns = [], [], []
        trades_n = 0
        worst = 0.0
        per_fold = []

        for fold_idx, (_is_start, is_end, oos_end) in enumerate(folds):
            # 이 폴드의 인샘플 점수만 보고 국면별 승자를 고른다
            chosen = {}
            ok = True
            for reg in REGIMES:
                best = None
                for name in SEARCHABLE:
                    picks = picks_by_key.get((tf, reg, name))
                    if not picks:
                        continue
                    fp = next((x for x in picks if x['fold'] == fold_idx), None)
                    if fp is None:
                        continue
                    if best is None or fp['is_score'] > best[1]:
                        best = (name, fp['is_score'], fp['params'])
                if best is None:
                    ok = False
                    break
                chosen[reg] = {'strategy_name': best[0], 'strategy_params': best[2]}
            if not ok:
                continue

            cfg_rs = chosen
            m = _eval_window(
                df, is_end, oos_end,
                lambda c=cfg_rs: RegimeSwitchingStrategy(
                    regime_strategies=c, regime_confirm_candles=confirm),
                symbol, cost, confirm)
            if m is None:
                continue

            rets.append(m['total_return'])
            sharpes.append(m['sharpe_ratio'])
            turns.append(m.get('annual_turnover', 0.0))
            trades_n += m['total_trades']
            worst = min(worst, m['max_drawdown'])
            per_fold.append({
                'fold': fold_idx,
                'oos_return': m['total_return'],
                'picks': {r: chosen[r]['strategy_name'] for r in REGIMES},
            })

        if not rets:
            return None
        pos = sum(1 for r in rets if r > 0)
        return {
            'timeframe': tf, 'cost_multiplier': cost,
            'folds': len(rets),
            'oos_total_return': float(np.prod([1 + r for r in rets]) - 1),
            'oos_sharpe': float(np.mean(sharpes)),
            'oos_mdd': float(worst),
            'oos_trades': trades_n,
            'oos_turnover': float(np.mean(turns)),
            'oos_consistency': pos / len(rets),
            'per_fold': per_fold,
        }
    except Exception as e:
        return {'timeframe': tf, 'cost_multiplier': cost,
                'error': f"{type(e).__name__}: {e}"}


def run_stage3(symbol, workers, costs, confirm):
    src = REPORTS / f"regime_search_stage1_{symbol.replace('/', '-')}.json"
    if not src.exists():
        raise SystemExit(f"1단계 결과가 없습니다: {src}")
    data = json.loads(src.read_text())
    timeframes = data['timeframes']

    picks_by_key = {}
    missing = 0
    for r in data['results']:
        if r.get('cost_multiplier') != 1.0:
            continue
        fp = r.get('fold_picks')
        if not fp:
            missing += 1
            continue
        picks_by_key[(r['timeframe'], r['regime'], r['strategy'])] = fp

    if missing:
        raise SystemExit(
            f"fold_picks가 없는 결과 {missing}건. 1단계를 다시 돌려야 합니다 "
            f"(폴드별 인샘플 점수를 기록하도록 바뀌었습니다).")

    jobs = [(tf, symbol, confirm, c, picks_by_key) for tf in timeframes for c in costs]
    print(f"3단계(중첩 워크포워드): {len(jobs)}작업 / 워커 {workers}개", flush=True)

    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_stage3_job, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            if r is None:
                continue
            if 'error' in r:
                print(f"  ❌ {r['timeframe']} x{r['cost_multiplier']}: {r['error']}", flush=True)
                continue
            out.append(r)
            print(f"  {r['timeframe']:<4} 비용x{r['cost_multiplier']}: "
                  f"OOS {r['oos_total_return']:+.2%} | 샤프 {r['oos_sharpe']:+.2f} | "
                  f"MDD {r['oos_mdd']:.1%} | 일관성 {r['oos_consistency']:.0%} | "
                  f"회전 {r['oos_turnover']:.0f}", flush=True)

    path = REPORTS / f"regime_search_stage3_{symbol.replace('/', '-')}.json"
    path.write_text(json.dumps({'symbol': symbol, 'results': out},
                               ensure_ascii=False, indent=2, default=str))
    print(f"\n저장: {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['stage1', 'stage2', 'stage3'])
    ap.add_argument('--symbol', default='BTC/USDT')
    ap.add_argument('--timeframes', default='1h,2h,4h,6h,8h,12h,1d,3d')
    ap.add_argument('--workers', type=int, default=12)
    # 1단계는 비용 1배만 돌린다. 후보를 좁히는 단계라 2배까지 돌리면 계산이
    # 두 배가 되는데, 정작 비용 강건성은 조합을 확정한 2단계에서 판정한다.
    ap.add_argument('--costs', default=None,
                    help='쉼표 구분. 기본값은 stage1=1.0, stage2=1.0,2.0')
    ap.add_argument('--confirm', type=int, default=2)
    a = ap.parse_args()

    if a.costs:
        costs = [float(c) for c in a.costs.split(',')]
    else:
        costs = [1.0] if a.stage == 'stage1' else [1.0, 2.0]

    if a.stage == 'stage1':
        run_stage1(a.timeframes.split(','), a.symbol, a.workers, costs, a.confirm)
    elif a.stage == 'stage2':
        run_stage2(a.symbol, a.workers, costs, a.confirm)
    else:
        run_stage3(a.symbol, a.workers, costs, a.confirm)


if __name__ == '__main__':
    main()
