"""
tools/verify.py — 설치 상태와 모듈 배선을 자가 진단한다.

폴더를 개편한 뒤 임포트가 깨지지 않았는지 빠르게 확인하는 용도.

    python tools/verify.py
    python tools/verify.py --offline    # 거래소 요청 없이 임포트만 확인
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

FAILURES = []


def check(label, fn, optional=False):
    try:
        result = fn()
        print(f"   ✅ {label}" + (f" — {result}" if result else ""))
        return True
    except Exception as e:
        mark = "⚠️ " if optional else "❌"
        print(f"   {mark} {label}: {type(e).__name__}: {e}")
        if not optional:
            FAILURES.append(label)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--offline', action='store_true', help='거래소 요청을 건너뛴다')
    args = ap.parse_args()

    print("1. 필수 패키지")

    def _ccxt():
        import ccxt
        return f"ccxt {ccxt.__version__}"

    def _pandas():
        import pandas, numpy
        return f"pandas {pandas.__version__} / numpy {numpy.__version__}"

    check("ccxt", _ccxt)
    check("pandas / numpy", _pandas)
    check("python-dotenv", lambda: __import__('dotenv') and "")

    print("\n2. 선택 패키지 (리서치 전용 — 없어도 봇은 돈다)")

    def _optuna():
        import optuna
        return f"optuna {optuna.__version__}"

    check("optuna (research/optimizer.py)", _optuna, optional=True)

    print("\n3. 프로젝트 모듈")

    def _paths():
        from core.paths import ROOT, DB_PATH, CONFIG
        return f"루트 {ROOT.name}, DB {'있음' if DB_PATH.exists() else '없음'}"

    def _core():
        from core.indicators import add_all_indicators, confirm_regimes, classify_market_regime
        from core.strategies import ALL_STRATEGY_NAMES, get_strategy_by_name
        return f"전략 {len(ALL_STRATEGY_NAMES)}종"

    def _research():
        from research.backtester import Backtester
        from research.binance_env import TAKER_FEE, get_spec, liquidation_price
        from research.strategy_lab import load_candles, walk_forward
        return f"taker {TAKER_FEE:.2%}"

    def _live():
        from live.trader import LiveTrader
        from live.regime_bot import RegimeLiveTrader
        return ""

    check("core.paths", _paths)
    check("core.indicators / core.strategies", _core)
    check("research.*", _research)
    check("live.*", _live)

    print("\n4. 전략 인스턴스화")

    def _all_strategies():
        from core.strategies import ALL_STRATEGY_NAMES, get_strategy_by_name
        bad = []
        for name in ALL_STRATEGY_NAMES:
            try:
                get_strategy_by_name(name)
            except Exception as e:
                bad.append(f"{name}({type(e).__name__})")
        if bad:
            raise RuntimeError("생성 실패: " + ", ".join(bad))
        return f"{len(ALL_STRATEGY_NAMES)}종 모두 생성 가능"

    check("전 전략 생성", _all_strategies)

    print("\n5. 라이브 설정")

    def _configs():
        import json
        from core.paths import CONFIG
        files = sorted(CONFIG.glob("regime_config_*.json"))
        if not files:
            raise RuntimeError("config/ 에 설정 파일이 없습니다")
        for f in files:
            json.load(open(f))
        return f"{len(files)}개 파싱 성공"

    check("config/*.json", _configs)

    print("\n6. 백테스트 한 바퀴")

    def _backtest():
        from research.strategy_lab import load_candles, run_backtest
        df = load_candles('BTC/USDT', '4h', True)
        df = df.tail(600).reset_index(drop=True)
        m, _, _ = run_backtest(df, 'EMA 크로스오버')
        return f"{len(df)}봉 / 거래 {m['total_trades']}건 / 수익률 {m['total_return']:+.2%}"

    check("백테스트 실행", _backtest)

    if not args.offline:
        print("\n7. 거래소 연결 (공개 API)")

        def _exchange():
            import ccxt
            ex = ccxt.binanceusdm({'enableRateLimit': True})
            candles = ex.fetch_ohlcv('BTC/USDT', '4h', limit=3)
            return f"캔들 {len(candles)}개 수신, 최근 종가 {candles[-1][4]:,.1f}"

        check("바이낸스 OHLCV", _exchange, optional=True)

    print("\n" + "=" * 52)
    if FAILURES:
        print(f"❌ 실패 {len(FAILURES)}건: {', '.join(FAILURES)}")
        return 1
    print("✅ 모든 필수 점검 통과")
    return 0


if __name__ == '__main__':
    sys.exit(main())
