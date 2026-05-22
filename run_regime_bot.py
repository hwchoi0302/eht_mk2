"""
run_regime_bot.py — 시장 상황(Regime)별 동적 전략 전환 거래 봇 실행 엔진

이 봇은 최근 캔들 데이터 기반으로 상승장(BULL), 하락장(BEAR), 횡보장(SIDEWAYS)을 판별하고,
각 시장 상황에 최적화된 전략으로 즉시 교체(기존 포지션 전량 청산)하며 자동 거래를 수행합니다.

사용법:
    python run_regime_bot.py reports/regime_config_BTC-USDT_spot_1h.json --dry-run
    python run_regime_bot.py reports/regime_config_BTC-USDT_futures_4h.json --testnet
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# 기존 라이브 트레이더 임포트
from live_trader import LiveTrader, start_bot
from indicators import add_all_indicators
from strategies import get_strategy_by_name

# 로그 설정
logging.basicConfig(
    filename='regime_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class RegimeLiveTrader(LiveTrader):
    def __init__(self, api_key, secret_key, config, use_testnet=True, dry_run=False):
        self.config = config
        self.symbol = config['symbol']
        self.timeframe = config['timeframe']
        self.is_futures = config['is_futures']
        self.regime_strategies = config['regime_strategies']
        self.dry_run = dry_run

        self.api_key = api_key
        self.secret_key = secret_key
        self.use_testnet = use_testnet

        self.current_regime = None
        self.strategy = None
        self.strategy_name = None
        self.strategy_params = None
        self.running = False

        if not self.dry_run:
            self._init_exchange()
            self._init_db()
        else:
            print("⚡ [DRY-RUN] 모드로 봇을 실행합니다. 실제 API 요청 및 거래는 생략됩니다.")

    def name_to_strategy_class(self, name):
        # HeikinAshiTrendStrategy 매핑 추가
        name_lower = name.lower()
        if "하이킨" in name_lower or "heikin" in name_lower:
            return "HeikinAshiTrendStrategy"
        return super().name_to_strategy_class(name)

    def switch_strategy_for_regime(self, new_regime):
        """시장 국면에 따라 활성 전략을 동적으로 변경합니다."""
        if new_regime not in self.regime_strategies:
            print(f"⚠️ 경고: 알 수 없는 국면 '{new_regime}'. 스위칭을 건너뜁니다.")
            return

        strat_info = self.regime_strategies[new_regime]
        new_strat_name = strat_info['strategy_name']
        new_strat_params = strat_info['strategy_params']

        # 레버리지 제한 적용
        if 'leverage' in new_strat_params:
            new_strat_params['leverage'] = min(int(new_strat_params['leverage']), 3)

        print(f"\n🔄 [국면 전환 감지] {self.current_regime} -> {new_regime}")
        print(f"   👉 신규 활성화 전략: {new_strat_name}")
        print(f"   👉 파라미터 설정: {new_strat_params}")

        # 1. 기존 전략의 포지션 전량 즉시 시장가 청산
        if self.current_regime is not None and not self.dry_run:
            try:
                pos_size, entry_price = self.get_position()
                if pos_size != 0:
                    print(f"   🚨 기존 포지션 ({pos_size}) 즉시 청산 (국면 이탈)")
                    self.close_position(pos_size, entry_price, "REGIME_CHANGE_EXIT")
            except Exception as e:
                print(f"   ❌ 포지션 청산 실패: {e}")

        # 2. 새 전략 설정
        self.current_regime = new_regime
        self.strategy_name = new_strat_name
        self.strategy_params = new_strat_params

        # 인스턴스 생성
        self.strategy = get_strategy_by_name(self.name_to_strategy_class(new_strat_name), **new_strat_params)

        if not self.dry_run:
            self.set_leverage()

        log_msg = f"Regime Strategy Switched to {new_strat_name} for regime {new_regime}."
        logging.info(log_msg)
        print(f"   ✅ {new_regime} 국면 최적 전략 '{new_strat_name}' 적용 완료!\n")

    def run_once(self):
        """동적으로 국면을 감지하여 전략을 교체하고, 최신 캔들에 기반해 주문을 실행합니다."""
        try:
            # 1. 최근 캔들 데이터 Fetch
            if self.dry_run:
                df = self._generate_dry_run_candles()
            else:
                candles = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=200)
                if not candles:
                    logging.warning("Failed to fetch candles from exchange.")
                    return
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            # 2. 모든 기술적 지표 및 시장 국면 계산
            df_ind = add_all_indicators(df)

            # 마지막 마감 캔들(즉, 2번째 전 행) 기준으로 국면 판정
            # 마지막 인덱스는 현재 실시간으로 형성 중인 캔들이므로 리페인팅을 방지하기 위함입니다.
            last_closed_candle = df_ind.iloc[-2]
            detected_regime = last_closed_candle['regime']

            # 3. 국면 변경 시 전략 스위칭
            if self.current_regime is None or detected_regime != self.current_regime:
                self.switch_strategy_for_regime(detected_regime)

            # 로그 정보
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"현재 국면: {self.current_regime} | 최근 종가: {last_closed_candle['close']:.2f}")

            # 4. 거래 실행
            if self.dry_run:
                # 드라이런인 경우 가상 신호만 출력
                signals = self.strategy.generate_signals(df_ind)
                signal = signals.iloc[-2]
                print(f"   [DRY-RUN] 신호 계산 완료: {signal} (1=롱, -1=숏, 0=대기)")
            else:
                # CCXT 통신을 통한 실제 거래 (LiveTrader의 run_once)
                super().run_once()

        except Exception as e:
            logging.error(f"Error in RegimeLiveTrader run_once loop: {e}", exc_info=True)
            print(f"❌ 루프 실행 중 에러 발생: {e}")

    def _generate_dry_run_candles(self):
        """드라이런 테스트를 위해 모의 캔들 데이터를 생성합니다."""
        np.random.seed(int(time.time()) % 10000)
        dates = pd.date_range(end=datetime.now(), periods=200, freq=self.timeframe)
        prices = 60000 + np.cumsum(np.random.normal(0, 150, 200))
        df = pd.DataFrame({
            'timestamp': [int(d.timestamp() * 1000) for d in dates],
            'open': prices - 20,
            'high': prices + 50,
            'low': prices - 40,
            'close': prices,
            'volume': np.random.randint(100, 1000, 200)
        })
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df


def main():
    parser = argparse.ArgumentParser(description="시장 상황별 동적 전략 전환 거래 봇")
    parser.add_argument("config_path", help="regime_config_*.json 파일 경로")
    parser.add_argument("--testnet", action="store_true", help="Binance Testnet Sandbox 사용")
    parser.add_argument("--dry-run", action="store_true", help="실제 거래소 API 요청 없는 드라이런 테스트 모드")
    args = parser.parse_args()

    # 설정 파일 로드
    try:
        with open(args.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패 ({args.config_path}): {e}")
        sys.exit(1)

    print(f"\n==============================================")
    print(f"🤖 시장국면 동적 전략 전환 봇 시동")
    print(f"==============================================")
    print(f"  📊 대상 종목: {config['symbol']}")
    print(f"  ⏱️  타임프레임: {config['timeframe']}")
    print(f"  🏦 마켓 종류: {'선물' if config['is_futures'] else '현물'}")
    print(f"  ⚙️  설정 파일: {os.path.basename(args.config_path)}")
    print(f"==============================================\n")

    # API key 로드 (드라이런이 아닐 때만 필수)
    api_key = ""
    secret_key = ""
    if not args.dry_run:
        # env 파일 등에서 키 로드 시도
        try:
            from dotenv import load_dotenv
            # 스크립트 실행 경로 기준 .env 로드
            script_dir = os.path.dirname(os.path.abspath(__file__))
            load_dotenv(os.path.join(script_dir, '.env'))
        except ImportError:
            pass
        
        if args.testnet:
            api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
            secret_key = os.getenv("BINANCE_TESTNET_SECRET_KEY", "")
            if not api_key:
                api_key = os.getenv("BINANCE_API_KEY", "")
                secret_key = os.getenv("BINANCE_SECRET_KEY", "")
        else:
            api_key = os.getenv("BINANCE_REAL_API_KEY", "")
            secret_key = os.getenv("BINANCE_REAL_SECRET_KEY", "")
            if not api_key:
                api_key = os.getenv("BINANCE_API_KEY", "")
                secret_key = os.getenv("BINANCE_SECRET_KEY", "")

        if not api_key or not secret_key:
            mode_str = "TESTNET" if args.testnet else "실거래(REAL)"
            print(f"⚠️ {mode_str}용 API 키 환경 변수가 설정되지 않았습니다.")
            print(f"   .env 파일에 BINANCE_{'TESTNET' if args.testnet else 'REAL'}_API_KEY 및 SECRET_KEY가 정의되어 있는지 확인하세요.")

    bot = RegimeLiveTrader(
        api_key=api_key,
        secret_key=secret_key,
        config=config,
        use_testnet=args.testnet,
        dry_run=args.dry_run
    )

    try:
        bot.run_loop()
    except KeyboardInterrupt:
        print("\n👋 봇이 사용자에 의해 수동 종료되었습니다.")
        bot.running = False


if __name__ == "__main__":
    main()
