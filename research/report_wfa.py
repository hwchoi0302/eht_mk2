"""
research/report_wfa.py — run_wfa.py 결과를 읽어 Markdown 리포트를 만든다.

    python research/report_wfa.py
    python research/report_wfa.py --symbol ETH/USDT --timeframe 4h
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.paths import REPORTS


def fmt_pct(x):
    return f"{x:+.2%}" if x is not None else "—"


def true_buy_hold(symbol, timeframe):
    """아웃샘플 전 구간을 통으로 보유했을 때의 수익률과 최대낙폭.

    구간별 수익률을 복리로 이어붙이면 안 된다. 폴드 k의 마지막 봉과 폴드 k+1의
    첫 봉 사이 공백(4h면 4시간, 1d면 하루)이 통째로 빠지기 때문이다. 1d에서는
    그 공백이 16번 쌓여 B&H가 +72.6%가 아니라 +53.0%로 나왔다 — 전략이 B&H를
    이긴 것처럼 보이게 만드는 크기의 오차다. 여기서는 첫 아웃샘플 시작부터
    마지막 아웃샘플 끝까지 가격을 그대로 본다.
    """
    from research.strategy_lab import load_candles, make_folds

    df = load_candles(symbol, timeframe, True)
    folds = make_folds(df)
    if not folds:
        return None, None, None, None
    start, end = folds[0][1], folds[-1][2]
    seg = df[(df['datetime'] >= start) & (df['datetime'] < end)]
    if seg.empty:
        return None, None, None, None
    ret = float(seg['close'].iloc[-1] / seg['close'].iloc[0] - 1)
    roll = seg['close'].cummax()
    mdd = float(((seg['close'] - roll) / roll).min())
    return ret, mdd, start, end


def build(symbol='BTC/USDT', timeframe='4h'):
    tag = f"{symbol.replace('/', '-')}_{timeframe}"
    src = REPORTS / f"wfa_{tag}.json"
    if not src.exists():
        raise SystemExit(f"결과 파일이 없습니다: {src}\n먼저 research/run_wfa.py 를 돌리세요.")

    data = json.loads(src.read_text())
    results = data['results']

    by_cost = {}
    for r in results:
        by_cost.setdefault(r['cost_multiplier'], []).append(r)

    base = sorted(by_cost.get(1.0, []), key=lambda r: r['oos_sharpe'], reverse=True)
    stress = {r['strategy']: r for r in by_cost.get(2.0, [])}

    bh, bh_mdd, oos_start, oos_end = true_buy_hold(symbol, timeframe)

    lines = []
    lines.append(f"# 워크포워드 분석 결과 — {symbol} {timeframe}")
    lines.append("")
    lines.append(f"생성: {data['generated_at'][:19]}")
    lines.append("")
    lines.append("인샘플 6개월에서 파라미터를 고르고, 뒤따르는 아웃샘플 2개월 성과만 집계했다.")
    lines.append("아래 수치는 **전부 아웃샘플**이다. 인샘플 성과는 보고하지 않는다.")
    lines.append("")
    lines.append(f"아웃샘플 전 구간: **{oos_start.date()} ~ {oos_end.date()}** "
                 f"({(oos_end - oos_start).days}일, {len(base[0]['fold_returns']) if base else 0}개 구간)")
    lines.append("")
    lines.append(f"같은 구간을 그냥 들고 있었을 때 (Buy & Hold): "
                 f"**{fmt_pct(bh)}**, 최대낙폭 **{bh_mdd:.1%}**")
    lines.append("")
    lines.append("> 이 B&H는 구간별 수익률을 복리로 이어붙인 값이 아니라, 첫 아웃샘플")
    lines.append("> 시작가에서 마지막 아웃샘플 종가까지의 실제 가격 변화다. 구간별로")
    lines.append("> 이어붙이면 구간 사이 공백이 빠져 값이 과소평가된다.")
    lines.append("")
    lines.append("## 전략별 순위 (아웃샘플 샤프 기준)")
    lines.append("")
    lines.append("| # | 전략 | OOS 수익률 | 샤프 | MDD | 일관성 | 거래 | 회전율/년 | 비용2배 수익률 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(base, 1):
        s = stress.get(r['strategy'])
        stress_ret = fmt_pct(s['oos_total_return']) if s else "—"
        lines.append(
            f"| {i} | {r['strategy']} | {fmt_pct(r['oos_total_return'])} | "
            f"{r['oos_sharpe']:+.2f} | {r['oos_mdd']:.1%} | "
            f"{r['oos_consistency']:.0%} | {r['oos_trades']} | "
            f"{r['oos_turnover']:.0f} | {stress_ret} |"
        )
    lines.append("")

    # 합격선을 넘는 전략
    survivors = [
        r for r in base
        if r['oos_total_return'] > 0
        and r['oos_sharpe'] > 0
        and r['oos_consistency'] >= 0.5
        and stress.get(r['strategy'], {}).get('oos_total_return', -1) > 0
    ]

    lines.append("## 합격선")
    lines.append("")
    lines.append("아래를 **모두** 만족해야 실계좌 후보로 본다.")
    lines.append("")
    lines.append("- 아웃샘플 수익률 > 0")
    lines.append("- 아웃샘플 샤프 > 0")
    lines.append("- 구간 일관성 ≥ 50% (수익 구간이 절반 이상)")
    lines.append("- **비용 2배에서도 수익률 > 0**")
    lines.append("")
    if survivors:
        lines.append(f"통과: {len(survivors)}개")
        lines.append("")
        for r in survivors:
            s = stress[r['strategy']]
            lines.append(f"### {r['strategy']}")
            lines.append("")
            lines.append(f"- 아웃샘플 {fmt_pct(r['oos_total_return'])} "
                         f"(샤프 {r['oos_sharpe']:+.2f}, MDD {r['oos_mdd']:.1%})")
            lines.append(f"- 비용 2배: {fmt_pct(s['oos_total_return'])} "
                         f"(샤프 {s['oos_sharpe']:+.2f})")
            lines.append(f"- 구간 {r['folds']}개 중 {r['oos_positive_folds']}개 수익 "
                         f"({r['oos_consistency']:.0%})")
            lines.append(f"- 최악 구간 {fmt_pct(r['oos_worst_fold'])} / "
                         f"최고 구간 {fmt_pct(r['oos_best_fold'])}")
            lines.append(f"- 연 회전율 {r['oos_turnover']:.0f}회")
            lines.append("")
    else:
        lines.append("**통과한 전략이 없다.**")
        lines.append("")
        lines.append("전 구간 최적화로 보면 좋아 보이던 전략들이, 파라미터를 과거에서만 고르고")
        lines.append("미래 구간에서 평가하자 대부분 손실로 돌아섰다. 비용을 2배로 놓으면")
        lines.append("남는 것이 더 줄어든다. 이 결과는 \"어느 전략을 쓸까\"가 아니라")
        lines.append("\"이 전략군으로는 비용을 넘는 알파가 없다\"는 쪽을 가리킨다.")
        lines.append("")

    # 비용 민감도 요약
    lines.append("## 비용 민감도")
    lines.append("")
    lines.append("비용을 2배로 올렸을 때 수익률이 얼마나 무너지는지. 회전율이 높을수록 크게 밀린다.")
    lines.append("")
    lines.append("| 전략 | 비용 1배 | 비용 2배 | 차이 | 회전율/년 |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(base, key=lambda r: -r['oos_turnover']):
        s = stress.get(r['strategy'])
        if not s:
            continue
        delta = s['oos_total_return'] - r['oos_total_return']
        lines.append(f"| {r['strategy']} | {fmt_pct(r['oos_total_return'])} | "
                     f"{fmt_pct(s['oos_total_return'])} | {delta:+.1%} | "
                     f"{r['oos_turnover']:.0f} |")
    lines.append("")

    out = REPORTS / f"wfa_{tag}.md"
    out.write_text("\n".join(lines))
    print(f"저장: {out}")
    return out, base, stress, survivors, bh


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='BTC/USDT')
    ap.add_argument('--timeframe', default='4h')
    a = ap.parse_args()
    build(a.symbol, a.timeframe)
