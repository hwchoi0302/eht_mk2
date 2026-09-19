"""
research/report_regime_search.py — 전수 탐색 결과를 Markdown으로 정리한다.

1단계(국면별)와 2단계(조합 재검증) 결과를 함께 읽어,
"어느 타임프레임의 어느 조합을 쓸 것인가"를 판단할 수 있는 표를 만든다.

    python research/report_regime_search.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.paths import REPORTS

REGIMES = ('BULL', 'BEAR', 'SIDEWAYS')


def pct(x):
    return f"{x:+.1%}"


def load(symbol, stage):
    p = REPORTS / f"regime_search_{stage}_{symbol.replace('/', '-')}.json"
    return json.loads(p.read_text()) if p.exists() else None


def true_buy_hold(symbol, timeframe):
    """아웃샘플 전 구간을 통으로 보유했을 때의 수익률과 최대낙폭."""
    from research.strategy_lab import load_candles, make_folds
    try:
        df = load_candles(symbol, timeframe, True)
    except Exception:
        return None, None
    folds = make_folds(df)
    if not folds:
        return None, None
    seg = df[(df['datetime'] >= folds[0][1]) & (df['datetime'] < folds[-1][2])]
    if seg.empty:
        return None, None
    roll = seg['close'].cummax()
    return (float(seg['close'].iloc[-1] / seg['close'].iloc[0] - 1),
            float(((seg['close'] - roll) / roll).min()))


def build(symbol='BTC/USDT'):
    s1 = load(symbol, 'stage1')
    s2 = load(symbol, 'stage2')
    if not s1:
        raise SystemExit("1단계 결과가 없습니다.")

    r1 = s1['results']
    timeframes = s1['timeframes']

    L = []
    L.append(f"# 타임프레임 × 국면 × 전략 전수 탐색 — {symbol}")
    L.append("")
    L.append("인샘플 6개월 → 아웃샘플 2개월 롤링. **모든 수치는 아웃샘플이다.**")
    L.append("")
    L.append("2단계로 나눴다. 전략 15종을 세 국면에 독립 배치하면 15³ = 3,375가지라")
    L.append("전수 탐색이 불가능하다. RegimeSwitchingStrategy에서 각 하위 전략은")
    L.append("자기 국면에서만 거래하므로, 국면별로 따로 고른 뒤(15 × 3 = 45) 조합해")
    L.append("재검증하는 방식을 썼다.")
    L.append("")

    # ── 1단계: 국면별 상위 전략 ──
    L.append("## 1단계 — 국면별 최적 전략 (타임프레임별)")
    L.append("")
    L.append("각 칸은 해당 국면에서만 거래하도록 제한했을 때의 1위 전략이다.")
    L.append("괄호는 (아웃샘플 수익률 / 샤프).")
    L.append("")
    L.append("| TF | BULL | BEAR | SIDEWAYS |")
    L.append("|---|---|---|---|")
    for tf in timeframes:
        row = [tf]
        for reg in REGIMES:
            c = [r for r in r1 if r['timeframe'] == tf and r['regime'] == reg
                 and r['cost_multiplier'] == 1.0 and r['oos_trades'] >= 5]
            if not c:
                row.append("—")
                continue
            c.sort(key=lambda r: (r['oos_sharpe'], r['oos_consistency']), reverse=True)
            b = c[0]
            row.append(f"{b['strategy']}<br>({pct(b['oos_total_return'])} / {b['oos_sharpe']:+.2f})")
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # ── 국면별 전략 순위 (전 타임프레임 평균) ──
    L.append("### 국면별 전략 순위 (전 타임프레임 평균 샤프)")
    L.append("")
    for reg in REGIMES:
        L.append(f"**{reg}**")
        L.append("")
        agg = {}
        for r in r1:
            if r['regime'] != reg or r['cost_multiplier'] != 1.0:
                continue
            agg.setdefault(r['strategy'], []).append(r)
        rows = []
        for name, rs in agg.items():
            n = len(rs)
            rows.append((
                sum(x['oos_sharpe'] for x in rs) / n,
                name, n,
                sum(x['oos_total_return'] for x in rs) / n,
                sum(x['oos_consistency'] for x in rs) / n,
                sum(x['oos_turnover'] for x in rs) / n,
            ))
        rows.sort(reverse=True)
        L.append("| 전략 | 평균 샤프 | 평균 수익률 | 평균 일관성 | 평균 회전율 |")
        L.append("|---|---|---|---|---|")
        for sh, name, n, ret, cons, turn in rows[:6]:
            L.append(f"| {name} | {sh:+.2f} | {pct(ret)} | {cons:.0%} | {turn:.0f} |")
        L.append("")

    # ── 2단계 ──
    if s2 and s2.get('results'):
        L.append("## 2단계 — 조합 재검증 (최종 판단 근거)")
        L.append("")
        L.append("1단계 승자를 국면별로 조합해 전 구간 워크포워드로 다시 돌린 결과다.")
        L.append("1단계는 국면을 분리해 평가하지만 실제로는 국면 전환이 포지션을")
        L.append("강제 청산시키는 상호작용이 있으므로, **이 표가 최종 근거다.**")
        L.append("")
        L.append("| TF | OOS 수익률 | 샤프 | MDD | 일관성 | 회전율 | 비용2배 | B&H | B&H MDD |")
        L.append("|---|---|---|---|---|---|---|---|---|")

        by_tf = {}
        for r in s2['results']:
            by_tf.setdefault(r['timeframe'], {})[r['cost_multiplier']] = r

        ordered = sorted(by_tf.items(),
                         key=lambda kv: kv[1].get(1.0, {}).get('oos_sharpe', -99),
                         reverse=True)
        for tf, byc in ordered:
            base = byc.get(1.0)
            if not base:
                continue
            st = byc.get(2.0)
            bh, bhmdd = true_buy_hold(symbol, tf)
            L.append(
                f"| {tf} | {pct(base['oos_total_return'])} | {base['oos_sharpe']:+.2f} | "
                f"{base['oos_mdd']:.1%} | {base['oos_consistency']:.0%} | "
                f"{base['oos_turnover']:.0f} | "
                f"{pct(st['oos_total_return']) if st else '—'} | "
                f"{pct(bh) if bh is not None else '—'} | "
                f"{f'{bhmdd:.1%}' if bhmdd is not None else '—'} |")
        L.append("")

        # 합격선
        L.append("### 합격선")
        L.append("")
        L.append("- 아웃샘플 수익률 > 0 · 샤프 > 0 · 일관성 ≥ 50%")
        L.append("- **비용 2배에서도 수익률 > 0**")
        L.append("")
        winners = []
        for tf, byc in ordered:
            b, st = byc.get(1.0), byc.get(2.0)
            if (b and st and b['oos_total_return'] > 0 and b['oos_sharpe'] > 0
                    and b['oos_consistency'] >= 0.5 and st['oos_total_return'] > 0):
                winners.append((tf, b, st))
        if winners:
            L.append(f"통과: {len(winners)}개")
            L.append("")
            for tf, b, st in winners:
                L.append(f"#### {tf}")
                L.append("")
                for reg in REGIMES:
                    blk = b['config']['regime_strategies'][reg]
                    ps = ", ".join(f"{k}={v}" for k, v in sorted(blk['strategy_params'].items()))
                    L.append(f"- **{reg}**: {blk['strategy_name']} — {ps}")
                L.append("")
                L.append(f"  아웃샘플 {pct(b['oos_total_return'])} / 샤프 {b['oos_sharpe']:+.2f} / "
                         f"MDD {b['oos_mdd']:.1%} / 일관성 {b['oos_consistency']:.0%} / "
                         f"비용2배 {pct(st['oos_total_return'])}")
                L.append("")
        else:
            L.append("**통과한 조합이 없다.**")
            L.append("")
    else:
        L.append("## 2단계 — 아직 결과 없음")
        L.append("")

    out = REPORTS / f"regime_search_{symbol.replace('/', '-')}.md"
    out.write_text("\n".join(L))
    print(f"저장: {out}")
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='BTC/USDT')
    build(ap.parse_args().symbol)
