"""
report_generator.py — 한국어 마크다운 보고서 생성 모듈

결과 딕셔너리 구조 (all_results 의 각 원소):
{
    'symbol':       str,        # e.g. "BTC/USDT"
    'timeframe':    str,        # e.g. "4h"
    'strategy_name':str,        # 한국어 전략명
    'period_name':  str,        # e.g. "6개월"
    'period_days':  int,
    'is_futures':   bool,
    'start_date':   str,
    'end_date':     str,
    'is_end_date':  str,        # IS 훈련 기간 종료일
    'oos_start_date':str,       # OOS 검증 기간 시작일
    'best_params':  dict,
    'score':        float,      # Optuna 최고 목표함수값 (IS 기준)
    'is_metrics':   dict,
    'oos_metrics':  dict,
    'full_metrics': dict,
    'regime_attribution': dict, # {'BULL': {...}, 'BEAR': {...}, 'SIDEWAYS': {...}}
    'error':        str | None, # 오류 발생 시 메시지
}
"""

import os
import json
from datetime import datetime


# ─────────────────────────────────────────────
#  유틸리티 함수
# ─────────────────────────────────────────────

def _pct(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v)*100:.2f}%"
    except Exception:
        return "N/A"


def _f2(v, digits=4) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "N/A"


def _market_label(is_futures: bool) -> str:
    return "선물" if is_futures else "현물"


def _overfitting_verdict(is_sharpe, oos_sharpe) -> str:
    try:
        is_s = float(is_sharpe)
        oos_s = float(oos_sharpe)
    except Exception:
        return "⚠️ 측정 불가"
    if oos_s < 0:
        return "❌ 실패 (OOS 손실)"
    if is_s <= 0:
        return "⚠️ IS 거래 없음"
    ratio = oos_s / is_s
    if ratio >= 0.6:
        return f"✅ 통과 (OOS/IS = {ratio:.1%})"
    elif ratio >= 0.4:
        return f"⚠️ 주의 (OOS/IS = {ratio:.1%})"
    else:
        return f"❌ 과최적화 의심 (OOS/IS = {ratio:.1%})"


def _regime_emoji(regime: str) -> str:
    return {"BULL": "🟢", "BEAR": "🔴", "SIDEWAYS": "🔵"}.get(regime, "⚪")


def _regime_label(regime: str) -> str:
    return {"BULL": "상승장", "BEAR": "하락장", "SIDEWAYS": "횡보장"}.get(regime, regime)


def _robustness_score(r: dict) -> float:
    """결과 딕셔너리에서 강건성 점수를 계산합니다 (IS와 OOS 조화평균)."""
    try:
        is_s = float(r['is_metrics'].get('sharpe_ratio', 0) or 0)
        oos_s = float(r['oos_metrics'].get('sharpe_ratio', 0) or 0)
        # 두 값 모두 양수여야 의미있는 점수
        if is_s <= 0 or oos_s <= 0:
            return max(is_s, oos_s, 0)
        # 조화평균
        return 2 * is_s * oos_s / (is_s + oos_s)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
#  보고서 생성기
# ─────────────────────────────────────────────

class ReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        self.detail_dir = os.path.join(output_dir, "전략별_상세")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.detail_dir, exist_ok=True)

    def _write(self, filename: str, content: str):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    # ── 공개 API ──────────────────────────────

    def generate_all_reports(self, all_results: list, sweep_config: dict = None):
        """모든 한국어 마크다운 보고서를 생성합니다."""
        if not all_results:
            print("⚠️  결과가 없어 보고서를 생성할 수 없습니다.")
            return

        # 오류 없는 결과만 필터
        valid = [r for r in all_results if not r.get('error')]
        # 강건성 점수 추가
        for r in valid:
            r['robustness_score'] = _robustness_score(r)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        paths = []
        paths.append(self._generate_leaderboard(valid, now_str))
        paths.append(self._generate_best_summary(valid, now_str))
        paths.append(self._generate_period_comparison(valid, now_str))
        paths.append(self._generate_regime_analysis(valid, now_str))
        paths.append(self._generate_live_recommendations(valid, now_str))
        self._generate_strategy_details(valid, now_str)
        
        # 시장 국면 봇용 동적 전략 설정 파일 생성
        self._generate_regime_configs(valid)

        print(f"\n{'='*60}")
        print(f"📁 보고서 생성 완료: {self.output_dir}/")
        for p in paths:
            if p:
                print(f"  ✅ {os.path.basename(p)}")
        print(f"  ✅ 전략별_상세/ ({len(set(r['strategy_name'] for r in valid))}개 파일)")
        print(f"{'='*60}\n")

    # ── 1. 전체 리더보드 ─────────────────────────

    def _generate_leaderboard(self, valid: list, now_str: str) -> str:
        sorted_all = sorted(valid, key=lambda r: r['robustness_score'], reverse=True)

        lines = [
            "# 📊 전체 전략 성과 리더보드\n",
            f"> 최종 업데이트: {now_str}  \n",
            f"> 총 분석 조합: {len(valid)}개\n\n",
            "---\n\n",
            "## 🏆 종합 랭킹 (강건성 점수 기준)\n\n",
            "> **강건성 점수** = IS 샤프비율과 OOS 샤프비율의 조화평균 (과최적화 저항력 반영)\n\n",
            "| 순위 | 전략 | 종목 | TF | 기간 | 마켓 | 강건성 | IS 수익률 | OOS 수익률 | 최대낙폭 | 과최적화 |\n",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n",
        ]

        for rank, r in enumerate(sorted_all[:30], 1):
            ism = r['is_metrics']
            oosm = r['oos_metrics']
            verdict = _overfitting_verdict(
                ism.get('sharpe_ratio', 0),
                oosm.get('sharpe_ratio', 0)
            )
            lines.append(
                f"| {rank} | {r['strategy_name']} | {r['symbol']} | {r['timeframe']} "
                f"| {r['period_name']} | {_market_label(r['is_futures'])} "
                f"| {_f2(r['robustness_score'], 3)} "
                f"| {_pct(ism.get('total_return'))} "
                f"| {_pct(oosm.get('total_return'))} "
                f"| {_pct(r['full_metrics'].get('max_drawdown'))} "
                f"| {verdict} |\n"
            )

        # 기간별 분리 랭킹
        lines.append("\n---\n\n")
        period_names = sorted(set(r['period_name'] for r in valid),
                              key=lambda x: ['90일','6개월','1년','3년'].index(x)
                              if x in ['90일','6개월','1년','3년'] else 99)

        for period in period_names:
            subset = sorted(
                [r for r in valid if r['period_name'] == period],
                key=lambda r: r['robustness_score'], reverse=True
            )
            if not subset:
                continue
            lines.append(f"## ⏱️ {period} — 상위 10개\n\n")
            lines.append("| 순위 | 전략 | 종목 | TF | 마켓 | 강건성 | IS 샤프 | OOS 샤프 | 총수익률 |\n")
            lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            for rank, r in enumerate(subset[:10], 1):
                lines.append(
                    f"| {rank} | {r['strategy_name']} | {r['symbol']} | {r['timeframe']} "
                    f"| {_market_label(r['is_futures'])} "
                    f"| {_f2(r['robustness_score'], 3)} "
                    f"| {_f2(r['is_metrics'].get('sharpe_ratio'), 3)} "
                    f"| {_f2(r['oos_metrics'].get('sharpe_ratio'), 3)} "
                    f"| {_pct(r['full_metrics'].get('total_return'))} |\n"
                )
            lines.append("\n")

        return self._write("전체_리더보드.md", "".join(lines))

    # ── 2. 최적 전략 요약 ─────────────────────────

    def _generate_best_summary(self, valid: list, now_str: str) -> str:
        sorted_all = sorted(valid, key=lambda r: r['robustness_score'], reverse=True)
        top5 = sorted_all[:5]

        lines = [
            "# 🏆 최적 전략 상세 분석\n\n",
            f"> 최종 업데이트: {now_str}\n\n",
            "---\n\n",
        ]

        for rank, r in enumerate(top5, 1):
            ism = r['is_metrics']
            oosm = r['oos_metrics']
            fullm = r['full_metrics']
            verdict = _overfitting_verdict(
                ism.get('sharpe_ratio', 0),
                oosm.get('sharpe_ratio', 0)
            )
            lines.append(f"## {rank}위: {r['strategy_name']} — {r['symbol']} {r['timeframe']} {_market_label(r['is_futures'])}\n\n")
            lines.append(f"- **분석 기간**: {r['period_name']} ({r['start_date']} ~ {r['end_date']})\n")
            lines.append(f"- **강건성 점수**: {_f2(r['robustness_score'], 3)}\n")
            lines.append(f"- **과최적화 검증**: {verdict}\n\n")

            lines.append("### 성과 비교표 (IS 훈련 / OOS 검증 / 전체)\n\n")
            lines.append("| 지표 | 훈련 기간 (IS) | 검증 기간 (OOS) | 전체 기간 |\n")
            lines.append("|:---|:---:|:---:|:---:|\n")
            for metric_key, metric_label in [
                ('sharpe_ratio', '샤프비율'),
                ('sortino_ratio', '소르티노비율'),
                ('total_return', '총수익률'),
                ('cagr', '연환산수익률 (CAGR)'),
                ('max_drawdown', '최대낙폭 (MDD)'),
                ('win_rate', '승률'),
                ('total_trades', '총거래수'),
                ('profit_factor', '이익계수'),
            ]:
                is_val = ism.get(metric_key)
                oos_val = oosm.get(metric_key)
                full_val = fullm.get(metric_key)
                # 퍼센트 포맷 여부
                if metric_key in ('total_return', 'cagr', 'max_drawdown', 'win_rate'):
                    lines.append(f"| {metric_label} | {_pct(is_val)} | {_pct(oos_val)} | {_pct(full_val)} |\n")
                elif metric_key == 'total_trades':
                    lines.append(f"| {metric_label} | {int(is_val or 0)} | {int(oos_val or 0)} | {int(full_val or 0)} |\n")
                else:
                    lines.append(f"| {metric_label} | {_f2(is_val, 3)} | {_f2(oos_val, 3)} | {_f2(full_val, 3)} |\n")

            lines.append("\n### 시장국면별 성과\n\n")
            lines.append("| 시장 국면 | 거래수 | 승률 | 누적 손익 |\n")
            lines.append("|:---:|:---:|:---:|:---:|\n")
            regime_attr = r.get('regime_attribution', {})
            for reg in ['BULL', 'BEAR', 'SIDEWAYS']:
                info = regime_attr.get(reg, {})
                emoji = _regime_emoji(reg)
                label = _regime_label(reg)
                trades = info.get('trades', 0)
                wr = info.get('win_rate', 0)
                pnl = info.get('pnl', 0)
                sign = "+" if pnl >= 0 else ""
                lines.append(f"| {emoji} {label} | {trades}건 | {_pct(wr)} | {sign}${pnl:.2f} |\n")

            lines.append("\n### 최적 파라미터\n\n")
            lines.append("```json\n")
            lines.append(json.dumps(r['best_params'], ensure_ascii=False, indent=2))
            lines.append("\n```\n\n---\n\n")

        return self._write("최적_전략_요약.md", "".join(lines))

    # ── 3. 기간별 성과 비교 ─────────────────────

    def _generate_period_comparison(self, valid: list, now_str: str) -> str:
        lines = [
            "# ⏱️ 기간별 성과 비교\n\n",
            f"> 최종 업데이트: {now_str}\n\n",
            "> 동일한 전략도 분석 기간에 따라 성과가 달라질 수 있습니다.  \n",
            "> 여러 기간에서 일관되게 좋은 성과를 보이는 전략을 선택하세요.\n\n",
            "---\n\n",
        ]

        # 각 전략에 대해 기간별 성과 테이블 생성
        strategy_names = sorted(set(r['strategy_name'] for r in valid))
        periods = ['90일', '6개월', '1년', '3년']

        for strat in strategy_names:
            strat_results = [r for r in valid if r['strategy_name'] == strat]
            if not strat_results:
                continue

            lines.append(f"## {strat}\n\n")
            lines.append("| 기간 | 종목 | TF | 마켓 | 강건성 | 총수익률 | MDD | 샤프 | 거래수 |\n")
            lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

            for period in periods:
                period_r = sorted(
                    [r for r in strat_results if r['period_name'] == period],
                    key=lambda r: r['robustness_score'], reverse=True
                )
                for r in period_r[:3]:  # 기간당 상위 3개
                    lines.append(
                        f"| {r['period_name']} | {r['symbol']} | {r['timeframe']} "
                        f"| {_market_label(r['is_futures'])} "
                        f"| {_f2(r['robustness_score'], 3)} "
                        f"| {_pct(r['full_metrics'].get('total_return'))} "
                        f"| {_pct(r['full_metrics'].get('max_drawdown'))} "
                        f"| {_f2(r['full_metrics'].get('sharpe_ratio'), 3)} "
                        f"| {int(r['full_metrics'].get('total_trades', 0))} |\n"
                    )

            lines.append("\n")

        return self._write("기간별_성과_비교.md", "".join(lines))

    # ── 4. 시장국면별 성과 ───────────────────────

    def _generate_regime_analysis(self, valid: list, now_str: str) -> str:
        lines = [
            "# 📈 시장국면별 성과 분석\n\n",
            f"> 최종 업데이트: {now_str}\n\n",
            "> **상승장**: 현재 시장이 상승 추세일 때 어떤 전략이 효과적인지 확인하세요.\n",
            "> **하락장**: 하락 추세에서 가장 손실이 적은 전략을 확인하세요.\n",
            "> **횡보장**: 방향성 없는 시장에서 안정적인 전략을 확인하세요.\n\n",
            "---\n\n",
        ]

        for reg in ['BULL', 'BEAR', 'SIDEWAYS']:
            emoji = _regime_emoji(reg)
            label = _regime_label(reg)
            lines.append(f"## {emoji} {label}\n\n")

            # 해당 국면에서 수익이 가장 좋은 전략 순
            def regime_pnl(r):
                attr = r.get('regime_attribution', {})
                return attr.get(reg, {}).get('pnl', 0)

            sorted_by_regime = sorted(valid, key=regime_pnl, reverse=True)

            lines.append("| 순위 | 전략 | 종목 | TF | 기간 | 마켓 | 국면 손익 | 국면 거래수 | 국면 승률 |\n")
            lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

            for rank, r in enumerate(sorted_by_regime[:15], 1):
                info = r.get('regime_attribution', {}).get(reg, {})
                pnl = info.get('pnl', 0)
                trades = info.get('trades', 0)
                wr = info.get('win_rate', 0)
                sign = "+" if pnl >= 0 else ""
                lines.append(
                    f"| {rank} | {r['strategy_name']} | {r['symbol']} | {r['timeframe']} "
                    f"| {r['period_name']} | {_market_label(r['is_futures'])} "
                    f"| {sign}${pnl:.2f} | {trades}건 | {_pct(wr)} |\n"
                )
            lines.append("\n")

        return self._write("시장국면별_성과.md", "".join(lines))

    # ── 5. 실거래 추천 설정 ──────────────────────

    def _generate_live_recommendations(self, valid: list, now_str: str) -> str:
        sorted_all = sorted(valid, key=lambda r: r['robustness_score'], reverse=True)
        top10 = sorted_all[:10]

        lines = [
            "# 🚀 실거래 추천 설정\n\n",
            f"> 최종 업데이트: {now_str}\n\n",
            "> 아래 설정은 백테스팅 결과 기준으로 자동 선정된 것입니다.  \n",
            "> **반드시 모의 투자로 충분히 검증한 후 실거래에 적용하세요.**\n\n",
            "---\n\n",
        ]

        for rank, r in enumerate(top10, 1):
            market_flag = "--futures" if r['is_futures'] else "--spot"
            verdict = _overfitting_verdict(
                r['is_metrics'].get('sharpe_ratio', 0),
                r['oos_metrics'].get('sharpe_ratio', 0)
            )
            lines.append(f"## {rank}위: {r['strategy_name']}\n\n")
            lines.append(f"- **종목**: {r['symbol']} ({_market_label(r['is_futures'])})\n")
            lines.append(f"- **타임프레임**: {r['timeframe']}\n")
            lines.append(f"- **분석 기간**: {r['period_name']}\n")
            lines.append(f"- **강건성 점수**: {_f2(r['robustness_score'], 3)}\n")
            lines.append(f"- **과최적화 검증**: {verdict}\n")
            lines.append(f"- **전체 수익률**: {_pct(r['full_metrics'].get('total_return'))}\n")
            lines.append(f"- **최대낙폭**: {_pct(r['full_metrics'].get('max_drawdown'))}\n\n")

            lines.append("**▶ 모의 투자 실행 명령어:**\n")
            lines.append("```bash\n")
            lines.append(
                f'python run_bot.py --symbol "{r["symbol"]}" --timeframe {r["timeframe"]} '
                f'--strategy "{r["strategy_name"]}" {market_flag} --testnet\n'
            )
            lines.append("```\n\n")

            lines.append("**▶ 전략 파라미터 (JSON):**\n")
            lines.append("```json\n")
            config = {
                "strategy_name": r['strategy_name'],
                "symbol": r['symbol'],
                "timeframe": r['timeframe'],
                "is_futures": r['is_futures'],
                "strategy_params": r['best_params'],
            }
            lines.append(json.dumps(config, ensure_ascii=False, indent=2))
            lines.append("\n```\n\n---\n\n")

        return self._write("실거래_추천_설정.md", "".join(lines))

    # ── 6. 전략별 상세 ──────────────────────────

    def _generate_strategy_details(self, valid: list, now_str: str):
        strategy_names = sorted(set(r['strategy_name'] for r in valid))

        for strat in strategy_names:
            strat_results = sorted(
                [r for r in valid if r['strategy_name'] == strat],
                key=lambda r: r['robustness_score'], reverse=True
            )

            safe_name = strat.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
            lines = [
                f"# {strat} — 상세 분석\n\n",
                f"> 최종 업데이트: {now_str}  \n",
                f"> 총 분석 조합: {len(strat_results)}개\n\n",
                "---\n\n",
                "## 전체 성과 요약\n\n",
                "| 종목 | TF | 기간 | 마켓 | 강건성 | IS 수익률 | OOS 수익률 | MDD | 거래수 | 과최적화 |\n",
                "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n",
            ]

            for r in strat_results:
                verdict = _overfitting_verdict(
                    r['is_metrics'].get('sharpe_ratio', 0),
                    r['oos_metrics'].get('sharpe_ratio', 0)
                )
                lines.append(
                    f"| {r['symbol']} | {r['timeframe']} | {r['period_name']} "
                    f"| {_market_label(r['is_futures'])} "
                    f"| {_f2(r['robustness_score'], 3)} "
                    f"| {_pct(r['is_metrics'].get('total_return'))} "
                    f"| {_pct(r['oos_metrics'].get('total_return'))} "
                    f"| {_pct(r['full_metrics'].get('max_drawdown'))} "
                    f"| {int(r['full_metrics'].get('total_trades', 0))} "
                    f"| {verdict} |\n"
                )

            # 최고 조합의 파라미터
            if strat_results:
                best = strat_results[0]
                lines.append(f"\n## 1위 조합 파라미터 ({best['period_name']} · {best['symbol']} {best['timeframe']})\n\n")
                lines.append("```json\n")
                lines.append(json.dumps(best['best_params'], ensure_ascii=False, indent=2))
                lines.append("\n```\n")

            path = os.path.join(self.detail_dir, f"{safe_name}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("".join(lines))

    # ── 7. 시장국면별 봇 최적 설정 생성 ──────────

    def _generate_regime_configs(self, valid: list) -> list:
        """
        (symbol, is_futures, timeframe)별로 그룹화하여,
        각 국면(BULL, BEAR, SIDEWAYS)에서 성과(PnL)가 가장 우수한 전략과 파라미터를 선별해
        JSON 설정 파일을 자동 생성합니다.
        """
        groups = {}
        for r in valid:
            key = (r['symbol'], r['is_futures'], r['timeframe'])
            if key not in groups:
                groups[key] = []
            groups[key].append(r)

        generated_paths = []

        for (symbol, is_futures, tf), results in groups.items():
            regime_strategies = {}

            for regime in ['BULL', 'BEAR', 'SIDEWAYS']:
                def get_regime_pnl(res):
                    attr = res.get('regime_attribution', {})
                    return attr.get(regime, {}).get('pnl', -999999.0)

                sorted_res = sorted(results, key=get_regime_pnl, reverse=True)

                best_res = None
                if sorted_res:
                    best_res = sorted_res[0]
                    if get_regime_pnl(best_res) == -999999.0:
                        # 해당 국면의 PnL 데이터가 아예 없는 경우 전체 강건성 1위 대용
                        best_res = sorted(results, key=lambda x: x['robustness_score'], reverse=True)[0]

                if best_res:
                    regime_strategies[regime] = {
                        "strategy_name": best_res['strategy_name'],
                        "strategy_params": best_res['best_params']
                    }
                else:
                    regime_strategies[regime] = {
                        "strategy_name": "EMA 크로스오버",
                        "strategy_params": {}
                    }

            config = {
                "symbol": symbol,
                "is_futures": is_futures,
                "timeframe": tf,
                "regime_strategies": regime_strategies,
                "generated_at": datetime.now().isoformat()
            }

            market_label = "futures" if is_futures else "spot"
            clean_symbol = symbol.replace("/", "-")
            filename = f"regime_config_{clean_symbol}_{market_label}_{tf}.json"
            path = os.path.join(self.output_dir, filename)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            generated_paths.append(path)

        return generated_paths
