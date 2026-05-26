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

# 파일 락을 위한 배타적 핸들 보관 변수
lock_fp = None

def acquire_lock(lock_file="bot.lock"):
    global lock_fp
    try:
        import fcntl
        lock_fp = open(lock_file, 'w')
        fcntl.flock(lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fp.write(str(os.getpid()))
        lock_fp.flush()
        return True
    except ImportError:
        # fcntl 모듈이 없는 환경 (Windows 등)
        return True
    except IOError:
        return False

def release_lock(lock_file="bot.lock"):
    global lock_fp
    if lock_fp:
        try:
            import fcntl
            fcntl.flock(lock_fp, fcntl.LOCK_UN)
            lock_fp.close()
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except Exception:
            pass

# 로그 설정: 10MB마다 회전, 최대 5개 백업 유지
from logging.handlers import RotatingFileHandler as _RegimeRotatingHandler
_regime_log_handler = _RegimeRotatingHandler(
    'regime_bot.log',
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
_regime_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# live_trader 임포트 시 이미 root logger에 핸들러가 추가됐을 수 있으므로 기존 핸들러 제거 후 교체
logging.getLogger().handlers.clear()
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(_regime_log_handler)


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
                # fetch한 df_ind를 그대로 재사용하여 신호 생성
                # (super().run_once() 호출 시 발생하는 이중 fetch 제거)
                signals = self.strategy.generate_signals(df_ind)
                signal = signals.iloc[-2]

                logging.info(f"Checking state. Last closed candle price: {last_closed_candle['close']}. Signal: {signal}")

                # 현재 포지션 조회
                pos_size, entry_price = self.get_position()
                pos_dir = 1 if pos_size > 0 else (-1 if pos_size < 0 else 0)

                # SL/TP 동적 체크
                if pos_dir != 0:
                    current_price = df_ind.iloc[-1]['close']
                    stop_loss_pct = self.strategy.stop_loss_pct
                    take_profit_pct = self.strategy.take_profit_pct

                    if hasattr(self.strategy, 'get_dynamic_risk') and 'regime' in last_closed_candle:
                        regime = last_closed_candle['regime']
                        dyn_risk = self.strategy.get_dynamic_risk(regime)
                        stop_loss_pct = dyn_risk.get('stop_loss_pct', stop_loss_pct)
                        take_profit_pct = dyn_risk.get('take_profit_pct', take_profit_pct)

                    trigger_exit = False
                    exit_reason = ""

                    if pos_dir == 1:
                        sl_price = entry_price * (1 - stop_loss_pct)
                        tp_price = entry_price * (1 + take_profit_pct)
                        if current_price <= sl_price:
                            trigger_exit = True
                            exit_reason = "STOP_LOSS"
                        elif current_price >= tp_price:
                            trigger_exit = True
                            exit_reason = "TAKE_PROFIT"
                    elif pos_dir == -1:
                        sl_price = entry_price * (1 + stop_loss_pct)
                        tp_price = entry_price * (1 - take_profit_pct)
                        if current_price >= sl_price:
                            trigger_exit = True
                            exit_reason = "STOP_LOSS"
                        elif current_price <= tp_price:
                            trigger_exit = True
                            exit_reason = "TAKE_PROFIT"

                    if trigger_exit:
                        logging.info(f"Risk trigger: {exit_reason} at {current_price}. Closing position.")
                        self.close_position(pos_size, current_price, exit_reason)
                        return

                # 신호 기반 주문 실행
                if signal == 1 and pos_dir != 1:
                    if pos_dir == -1:
                        self.close_position(pos_size, last_closed_candle['close'], "SIGNAL_REVERSAL")
                    self.open_position("BUY", last_closed_candle['close'])

                elif signal == -1 and pos_dir != -1:
                    if not self.is_futures:
                        logging.info("Short signal ignored. Spot market does not support short positions.")
                        return
                    if pos_dir == 1:
                        self.close_position(pos_size, last_closed_candle['close'], "SIGNAL_REVERSAL")
                    self.open_position("SELL", last_closed_candle['close'])

                elif signal == 0 and pos_dir != 0:
                    self.close_position(pos_size, last_closed_candle['close'], "SIGNAL_EXIT")

        except Exception as e:
            err_msg = str(e).lower()
            if "too many requests" in err_msg or "429" in err_msg or "ddos" in err_msg:
                print("\n🚨 [Rate Limit 감지] 바이낸스 요청 제한을 초과했습니다.")
                print("   IP 영구 차단을 방지하기 위해 5분(300초) 동안 쿨다운 대기에 들어갑니다...")
                logging.warning(f"Rate Limit 429 detected. Sleeping for 300s. Error: {e}")
                time.sleep(300)
            else:
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
        # 1. 스크립트 실행 경로 기준 .env 로드 (직접 파일 파싱으로 100% 보장)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(script_dir, '.env')
        if os.path.exists(env_path):
            try:
                with open(env_path, 'r', encoding='utf-8') as env_f:
                    for line in env_f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            k, v = line.split('=', 1)
                            # 값 앞뒤의 따옴표 및 공백 제거
                            val = v.strip().strip("'").strip('"')
                            os.environ[k.strip()] = val
            except Exception as e:
                print(f"⚠️ .env 직접 파싱 중 오류 발생: {e}")

        # 2. 추가적으로 dotenv 모듈이 설치되어 있다면 로드 (보안/보완용)
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
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

    if not acquire_lock():
        print("❌ 에러: 이미 다른 봇 인스턴스가 구동 중입니다. (bot.lock이 잠겨 있음)", flush=True)
        print("   기존에 띄워둔 봇을 종료하거나 'pkill -f run_regime_bot.py' 명령으로 정리 후 실행하세요.", flush=True)
        sys.exit(1)

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
    finally:
        release_lock()


if __name__ == "__main__":
    main()
