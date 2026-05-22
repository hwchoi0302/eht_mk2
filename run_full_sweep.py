"""
run_full_sweep.py — 전체 스윕 실행 진입점

사용법:
    python run_full_sweep.py                        # 기본 설정으로 실행
    python run_full_sweep.py --trials 30            # Optuna 시도 횟수 변경
    python run_full_sweep.py --symbols BTC/USDT     # 특정 종목만
    python run_full_sweep.py --output my_reports    # 출력 폴더 변경
    python run_full_sweep.py --skip-download        # 데이터 다운로드 건너뜀

총 조합:
    2종목 × 2마켓(현물/선물) × 3타임프레임 × 13전략 × 4기간 = 624개
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from itertools import product

from data_manager import DataManager
from bulk_optimizer import BulkOptimizer
from report_generator import ReportGenerator
from strategies import ALL_STRATEGY_NAMES


# ─────────────────────────────────────────────
#  스윕 설정
# ─────────────────────────────────────────────

DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT"]
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
DEFAULT_PERIODS = [
    {"name": "90일",  "days": 90},
    {"name": "6개월", "days": 180},
    {"name": "1년",   "days": 365},
    {"name": "3년",   "days": 1095},
]
IS_FUTURES_OPTIONS = [False, True]  # 현물, 선물


# ─────────────────────────────────────────────
#  데이터 사전 다운로드
# ─────────────────────────────────────────────

def download_all_data(symbols, timeframes, is_futures_options, max_days=1095):
    """스윕 시작 전 모든 필요 데이터를 캐시에 저장합니다."""
    dm = DataManager()
    end_str = datetime.now().strftime("%Y-%m-%d")
    start_str = (datetime.now() - timedelta(days=max_days)).strftime("%Y-%m-%d")

    combos = list(product(symbols, timeframes, is_futures_options))
    total = len(combos)

    print(f"\n{'='*60}")
    print(f"📥 데이터 사전 다운로드 ({total}개 조합, 최대 {max_days}일)")
    print(f"   기간: {start_str} ~ {end_str}")
    print(f"{'='*60}")

    for i, (symbol, tf, is_futures) in enumerate(combos, 1):
        market = "선물" if is_futures else "현물"
        print(f"  [{i}/{total}] {symbol} ({tf}) {market} ...", end=" ", flush=True)
        try:
            df = dm.get_candles(symbol, tf, start_str, end_str, is_futures)
            if df is not None and not df.empty:
                print(f"✅ {len(df):,}개 캔들")
            else:
                print("⚠️  데이터 없음")
        except Exception as e:
            print(f"❌ 오류: {e}")

    print()


# ─────────────────────────────────────────────
#  조합 생성
# ─────────────────────────────────────────────

def build_combinations(symbols, timeframes, is_futures_options, strategy_names, periods):
    """모든 (종목 × 마켓 × 타임프레임 × 전략 × 기간) 조합을 생성합니다."""
    combos = []
    for symbol, tf, is_futures, strat, period in product(
        symbols, timeframes, is_futures_options, strategy_names, periods
    ):
        # 현물 거래에서는 숏 포지션이 불가하므로 선물 전략과 동일하게 취급
        combos.append({
            'symbol': symbol,
            'timeframe': tf,
            'is_futures': is_futures,
            'strategy_name': strat,
            'period_name': period['name'],
            'period_days': period['days'],
        })
    return combos


# ─────────────────────────────────────────────
#  메인 진입점
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="자동 전략 스윕 백테스팅 시스템",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--trials", type=int, default=20,
        help="각 조합당 Optuna 최적화 시도 횟수 (기본: 20)"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_SYMBOLS,
        help=f"분석할 종목 (기본: {DEFAULT_SYMBOLS})"
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES,
        help=f"타임프레임 (기본: {DEFAULT_TIMEFRAMES})"
    )
    parser.add_argument(
        "--strategies", nargs="+", default=ALL_STRATEGY_NAMES,
        help="분석할 전략 이름 목록"
    )
    parser.add_argument(
        "--periods", nargs="+", default=None,
        help="분석 기간 (예: '90일' '1년'). 기본: 90일 6개월 1년 3년"
    )
    parser.add_argument(
        "--output", type=str, default="reports",
        help="보고서 출력 폴더 (기본: reports/)"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="병렬 처리에 사용할 프로세스 수 (-1 이면 CPU 코어 수 - 1) (기본: -1)"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="데이터 다운로드 단계를 건너뜁니다"
    )
    parser.add_argument(
        "--spot-only", action="store_true",
        help="현물 거래만 분석합니다 (선물 제외)"
    )
    parser.add_argument(
        "--futures-only", action="store_true",
        help="선물 거래만 분석합니다 (현물 제외)"
    )
    args = parser.parse_args()

    # 마켓 타입 결정
    if args.spot_only:
        markets = [False]
    elif args.futures_only:
        markets = [True]
    else:
        markets = IS_FUTURES_OPTIONS

    # 기간 필터
    if args.periods:
        periods = [p for p in DEFAULT_PERIODS if p['name'] in args.periods]
        if not periods:
            print(f"❌ 유효하지 않은 기간: {args.periods}")
            sys.exit(1)
    else:
        periods = DEFAULT_PERIODS

    # ── 시작 배너 ──
    print(f"\n{'='*60}")
    print("🚀 자동 전략 스윕 백테스팅 시작")
    print(f"{'='*60}")
    print(f"  📅 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  📊 종목: {args.symbols}")
    print(f"  ⏱️  타임프레임: {args.timeframes}")
    print(f"  🔬 전략 수: {len(args.strategies)}개")
    print(f"  📆 분석 기간: {[p['name'] for p in periods]}")
    print(f"  🏦 마켓: {'선물' if markets == [True] else '현물' if markets == [False] else '현물+선물'}")
    print(f"  🔁 Optuna 시도 횟수: {args.trials}회 / 조합")
    print(f"  ⚡ 병렬 프로세스 수: {args.n_jobs if args.n_jobs != -1 else 'CPU 코어 수 - 1'}")

    # 총 조합 수 계산
    total_combos = (
        len(args.symbols) * len(args.timeframes) * len(markets) *
        len(args.strategies) * len(periods)
    )
    print(f"  🎯 총 분석 조합: {total_combos:,}개")
    print(f"  📁 보고서 폴더: {args.output}/")
    print(f"{'='*60}\n")

    # ── 데이터 다운로드 ──
    if not args.skip_download:
        max_days = max(p['days'] for p in periods)
        download_all_data(args.symbols, args.timeframes, markets, max_days=max_days)

    # ── 조합 생성 ──
    combinations = build_combinations(
        symbols=args.symbols,
        timeframes=args.timeframes,
        is_futures_options=markets,
        strategy_names=args.strategies,
        periods=periods,
    )

    print(f"\n{'='*60}")
    print(f"🔬 최적화 스윕 시작 (총 {len(combinations):,}개 조합)")
    print(f"{'='*60}")

    # ── 스윕 실행 ──
    sweep_engine = BulkOptimizer(n_trials=args.trials)
    all_results = sweep_engine.run_sweep(combinations, n_jobs=args.n_jobs)

    # ── 보고서 생성 ──
    print(f"\n{'='*60}")
    print(f"📝 한국어 마크다운 보고서 생성 중...")
    print(f"{'='*60}")

    generator = ReportGenerator(output_dir=args.output)
    sweep_config = {
        'symbols': args.symbols,
        'timeframes': args.timeframes,
        'strategies': args.strategies,
        'periods': periods,
        'markets': markets,
        'n_trials': args.trials,
        'run_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_combos': len(combinations),
        'success_count': sum(1 for r in all_results if not r.get('error')),
        'error_count': sum(1 for r in all_results if r.get('error')),
    }
    generator.generate_all_reports(all_results, sweep_config)

    # ── 최종 요약 ──
    valid = [r for r in all_results if not r.get('error')]
    if valid:
        sorted_valid = sorted(valid, key=lambda r: r.get('robustness_score', 0), reverse=True)
        best = sorted_valid[0]
        market_label = "선물" if best['is_futures'] else "현물"
        print(f"\n{'='*60}")
        print(f"🏆 전체 1위 전략")
        print(f"{'='*60}")
        print(f"  전략명:   {best['strategy_name']}")
        print(f"  종목:     {best['symbol']}")
        print(f"  타임프레임: {best['timeframe']}")
        print(f"  마켓:     {market_label}")
        print(f"  기간:     {best['period_name']}")
        print(f"  강건성점수: {best.get('robustness_score', 0):.4f}")
        print(f"  IS 수익률: {best['is_metrics'].get('total_return', 0)*100:.2f}%")
        print(f"  OOS 수익률: {best['oos_metrics'].get('total_return', 0)*100:.2f}%")
        print(f"  최대낙폭:  {best['full_metrics'].get('max_drawdown', 0)*100:.2f}%")

    print(f"\n{'='*60}")
    print(f"✅ 스윕 완료!")
    print(f"   성공: {sweep_config['success_count']}개 / 오류: {sweep_config['error_count']}개")
    print(f"   보고서 위치: {os.path.abspath(args.output)}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
