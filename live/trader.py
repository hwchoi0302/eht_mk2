import os
import time
import json
import sqlite3
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
import pandas as pd
import ccxt

from core.paths import DB_PATH, STATUS_FILE as STATUS_FILE_PATH, TRADER_LOG, HEARTBEAT_FILE
from core.data_manager import DataManager
from core.indicators import add_all_indicators, REGIME_LOOKBACK
from core.strategies import get_strategy_by_name

# Configure logging with rotation (10MB per file, keep 5 backups)
# 핸들러 중복 방지: 이 모듈이 임포트될 때마다 중복 추가되지 않도록 조건 확인
if not logging.getLogger().handlers:
    _log_handler = RotatingFileHandler(
        str(TRADER_LOG),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    _log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(_log_handler)

DB_NAME = str(DB_PATH)
STATUS_FILE = str(STATUS_FILE_PATH)

class LiveTrader:
    # ── 클래스 수준 기본값 ────────────────────────────────────────────────────
    # RegimeLiveTrader는 super().__init__()을 호출하지 않고 상태를 직접 세팅한다.
    # 그래서 여기 __init__에만 새 속성을 추가하면 하위 클래스에서 누락되고,
    # 처음 참조하는 순간 AttributeError로 루프가 죽는다 (실제로 그렇게 죽었다).
    # 클래스 속성으로 두면 어느 생성 경로를 타든 기본값이 보장된다.
    _ticker_fail_count = 0        # 티커 조회 연속 실패 횟수
    _protected_position = None    # SL/TP를 걸어 둔 포지션 식별자(수량)
    _protected_at = 0.0           # 마지막으로 보호 주문을 건 시각
    _exchange_stops_unsupported = False   # 거래소가 조건부 주문을 만들지 않는 환경인가
    _warned_no_exchange_stops = False     # 위 경고를 이미 냈는가

    def __init__(self, api_key, secret_key, symbol, timeframe, is_futures, strategy_name, strategy_params, use_testnet=True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol
        self.timeframe = timeframe
        self.is_futures = is_futures
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params
        self.use_testnet = use_testnet
        
        self.exchange = None
        self.strategy = None
        self.running = False
        self._order_timeout_count = 0      # Circuit breaker: 연속 타임아웃 횟수
        self._last_order_timeout_ts = 0   # 마지막 타임아웃 발생 시각 (epoch)
        self._ticker_fail_count = 0   # 티커 조회 연속 실패 횟수
        self._protected_position = None  # SL/TP를 걸어 둔 포지션 식별자
        self._protected_at = 0.0         # 마지막으로 보호 주문을 건 시각
        self.last_exit_candle_timestamp = None
        

        self._init_exchange()
        self._init_strategy()
        self._init_db()

    def _init_exchange(self):
        """Initializes ccxt connection to Binance with correct mode and credentials."""
        exchange_class = ccxt.binance
        
        # Configure market type
        options = {}
        if self.is_futures:
            options['defaultType'] = 'future'
        else:
            options['defaultType'] = 'spot'
            
        self.exchange = exchange_class({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'enableRateLimit': True,
            'timeout': 20000,  # 20초 클라이언트 측 요청 타임아웃
            'options': options
        })
        
        if self.use_testnet:
            try:
                # CCXT의 새로운 Binance Demo Trading 지원 메서드 우선 시도 (선물/현물 통합)
                self.exchange.enable_demo_trading(True)
                logging.info("Initialized CCXT exchange with enable_demo_trading(True).")
            except Exception as e:
                # 백업용으로 기존 set_sandbox_mode 시도
                logging.warning(f"enable_demo_trading failed: {e}. Falling back to set_sandbox_mode.")
                self.exchange.set_sandbox_mode(True)
            logging.info("Initialized CCXT exchange in DEMO/MOCK TRADING mode.")
        else:
            logging.info("Initialized CCXT exchange in REAL/PRODUCTION mode.")

    def _init_strategy(self):
        """Instantiates the selected strategy."""
        # Enforce leverage limit of 3x
        if 'leverage' in self.strategy_params:
            self.strategy_params['leverage'] = min(int(self.strategy_params['leverage']), 3)
        self.strategy = get_strategy_by_name(self.name_to_strategy_class(self.strategy_name), **self.strategy_params)
        logging.info(f"Initialized strategy {self.strategy.name} with params: {self.strategy_params}")

    def _init_db(self):
        """Initializes SQLite database for trading logs."""
        with sqlite3.connect(DB_NAME, timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    datetime TEXT,
                    symbol TEXT,
                    direction TEXT,
                    action TEXT,
                    price REAL,
                    amount REAL,
                    pnl REAL,
                    is_futures INTEGER,
                    environment TEXT
                )
            """)
            conn.commit()

    def name_to_strategy_class(self, name):
        # Maps user-friendly names to internal class names
        name_lower = name.lower()
        if "ema" in name_lower or "cross" in name_lower:
            return "EMACrossStrategy"
        elif "rsi" in name_lower or "bb" in name_lower:
            return "RSIBBStrategy"
        elif "vol" in name_lower or "breakout" in name_lower:
            return "VolatilityBreakoutStrategy"
        elif "adaptive" in name_lower or "regime" in name_lower:
            return "AdaptiveRegimeStrategy"
        return name

    def log_trade(self, action, direction, price, amount, pnl=0.0):
        """Saves a trade action to database and logs it."""
        ts = int(time.time() * 1000)
        dt_str = datetime.now().isoformat()
        env_str = "TESTNET" if self.use_testnet else "REAL"
        is_futures_int = 1 if self.is_futures else 0
        
        # with 컨텍스트 매니저를 사용해 예외 발생 시에도 연결이 반드시 닫히도록 보장
        with sqlite3.connect(DB_NAME, timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trade_logs (timestamp, datetime, symbol, direction, action, price, amount, pnl, is_futures, environment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ts, dt_str, self.symbol, direction, action, price, amount, pnl, is_futures_int, env_str))
            conn.commit()
        
        log_msg = f"TRADE EVENT: {action} {direction} {amount} {self.symbol} at {price} (PnL: {pnl}) [{env_str}]"
        logging.info(log_msg)
        print(log_msg)

    def get_balance(self):
        """Fetches account balance (quote currency, e.g. USDT)."""
        balance = self.exchange.fetch_balance()
        if self.is_futures:
            # For futures, get USDT margin/equity balance
            return float(balance['total'].get('USDT', 0.0))
        else:
            return float(balance['total'].get('USDT', 0.0))

    def get_position(self):
        """
        Fetches current position size and entry price.
        Returns:
            position_size (float): positive for Long, negative for Short, 0.0 for Flat.
            entry_price (float): entry price.
        """
        if self.is_futures:
            positions = self.exchange.fetch_positions(symbols=[self.symbol])
            if positions:
                pos = positions[0]
                size = float(pos.get('contracts', 0.0)) # contracts is quantity
                side = pos.get('side', '') # 'long' or 'short'
                entry = float(pos.get('entryPrice', 0.0))
                if side == 'short':
                    return -size, entry
                return size, entry
            return 0.0, 0.0
        else:
            # Spot: check base currency balance (e.g. BTC)
            base_currency = self.symbol.split('/')[0]
            balance = self.exchange.fetch_balance()
            size = float(balance['total'].get(base_currency, 0.0))
            # Get latest price to approximate entry price or let's just use ticker
            ticker = self.exchange.fetch_ticker(self.symbol)
            close = float(ticker['last'])
            # In spot, we treat positive balance as position if it exceeds threshold
            min_val = 10.0 / close # At least $10 worth
            if size > min_val:
                return size, close
            return 0.0, 0.0

    def set_leverage(self):
        """Configures futures leverage on the exchange."""
        if self.is_futures:
            try:
                # CCXT standard method to set leverage
                leverage = min(self.strategy.leverage, 3) # Force maximum 3x
                self.exchange.set_leverage(leverage, self.symbol)
                logging.info(f"Set leverage to {leverage}x on exchange for {self.symbol}.")
            except Exception as e:
                logging.warning(f"Could not set leverage: {e}. It might already be set or not supported on this asset.")

    # ── 시장가 전용 주문 경로 ────────────────────────────────────────────────
    # 예전에는 지정가로 진입한 뒤 4회까지 가격을 추격하고 그래도 안 되면 시장가로
    # 폴백했다. 체결까지 최대 4루프(약 2분)가 걸렸고, 그 사이 신호가 만료되거나
    # 폴백 경로에서 log_trade("OPEN", ...)이 누락돼 DB에 구멍이 생겼다.
    # 이제 진입·청산 모두 단일 시장가 주문이다. 즉시 체결되고 기록 경로도 하나다.

    def _current_price(self):
        """현재가를 얻는다. 얻지 못하면 None을 돌려준다 (예외를 던지지 않는다).

        예전 코드는 `float(ticker['bid'])`로 bid가 None이면 TypeError를 던졌고,
        이 예외가 루프를 죽여 6/24~7/27 약 한 달간 신규 진입이 막혔다
        (regime_bot.log에 같은 TypeError가 65,900회). 여기서는 사이클을 건너뛴다.
        """
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
        except Exception as e:
            logging.warning(f"[_current_price] 티커 조회 실패: {e}")
            self._ticker_fail_count += 1
            return None

        for key in ('last', 'close', 'bid', 'ask'):
            value = ticker.get(key)
            if value is not None:
                try:
                    price = float(value)
                except (TypeError, ValueError):
                    continue
                if price > 0:
                    self._ticker_fail_count = 0
                    return price

        # bid/ask/last/close가 전부 비어 있는 경우 — 죽지 말고 건너뛴다
        self._ticker_fail_count += 1
        logging.warning(
            f"[_current_price] 티커에 유효한 가격이 없습니다 "
            f"(연속 {self._ticker_fail_count}회). 이번 사이클을 건너뜁니다."
        )
        if self._ticker_fail_count >= 10:
            logging.error(
                f"🚨 티커 조회가 연속 {self._ticker_fail_count}회 실패했습니다. "
                f"거래소 연결 상태를 확인하세요."
            )
        return None

    def _has_live_conditional_orders(self):
        """거래소에 살아 있는 조건부(STOP/TAKE_PROFIT) 주문이 실제로 있는지 확인한다.

        생성 응답의 주문 ID도, -4130("이미 존재") 응답도 신뢰할 수 없다.
        바이낸스 데모 트레이딩은 조건부 주문 요청을 받아 ID까지 돌려주면서
        실제로는 만들지 않는다. 전체 주문 이력을 조회해 보면 STOP_MARKET /
        TAKE_PROFIT_MARKET 이 **한 건도** 남지 않는다 (MARKET/LIMIT만 있다).
        openOrders도 빈 배열을 돌려주므로, 유일하게 확실한 근거가 주문 이력이다.
        """
        try:
            market = self.exchange.market(self.symbol)
            rows = self.exchange.fapiPrivateGetAllOrders(
                {'symbol': market['id'], 'limit': 50})
        except Exception as e:
            logging.warning(f"주문 이력 조회 실패, 보호 상태를 확인할 수 없습니다: {e}")
            return False
        for o in rows:
            otype = str(o.get('type', ''))
            if ('STOP' in otype or 'TAKE_PROFIT' in otype) and \
                    o.get('status') in ('NEW', 'PARTIALLY_FILLED'):
                return True
        return False

    def cancel_protective_orders(self):
        """걸려 있는 SL/TP 주문을 전부 취소한다.

        ⚠️ `fetch_open_orders()`는 `closePosition=True` 조건부 주문을 **돌려주지
        않는다.** 거래소는 같은 주문을 또 걸면 -4130 ("An open stop or take profit
        order with GTE and closePosition in the direction is existing")으로
        거절하면서도, 목록 조회에는 0건으로 나온다. 그래서 주문을 하나씩 찾아
        취소하는 방식은 동작하지 않았다.

        심볼 단위 전량 취소를 쓴다. 이 봇은 진입을 시장가로만 하므로 미체결로
        남아 있을 수 있는 주문은 SL/TP뿐이고, 전량 취소해도 잃을 것이 없다.
        """
        if not self.is_futures:
            return
        try:
            self.exchange.cancel_all_orders(self.symbol)
            logging.info("보호 주문 전량 취소 완료.")
        except Exception as e:
            logging.warning(f"보호 주문 취소 실패: {e}")
        finally:
            self._protected_position = None

    def place_protective_orders(self, direction, entry_price, position_key=None):
        """진입 직후 거래소에 SL/TP를 STOP_MARKET / TAKE_PROFIT_MARKET으로 올린다.

        예전에는 봇이 30초 루프를 돌며 직접 가격을 보고 청산했다. 봇이 죽어 있으면
        SL/TP가 아예 관리되지 않았고(7/29~9/14 47일 방치), 살아 있어도 최대 30초
        늦게 반응했다. 거래소에 주문을 올려두면 봇 상태와 무관하게 체결된다.
        백테스트가 트리거 가격 체결을 가정하는 것과도 이제 일치한다.
        """
        if not self.is_futures:
            return

        # 먼저 기존 보호 주문을 걷어낸다. 남아 있으면 -4130으로 거절당하고,
        # 조회로는 존재를 확인할 수 없으므로 '취소 후 재등록'이 유일하게
        # 확실한 경로다.
        self.cancel_protective_orders()

        sl_pct = self.strategy.stop_loss_pct
        tp_pct = self.strategy.take_profit_pct
        sign = 1 if direction == 'BUY' else -1

        sl_price = entry_price * (1 - sl_pct * sign)
        tp_price = entry_price * (1 + tp_pct * sign)
        close_side = 'sell' if direction == 'BUY' else 'buy'
        requested = 0

        for order_type, stop_price, label in (
            ('STOP_MARKET', sl_price, 'SL'),
            ('TAKE_PROFIT_MARKET', tp_price, 'TP'),
        ):
            try:
                stop_str = self.exchange.price_to_precision(self.symbol, stop_price)
                order = self.exchange.create_order(
                    self.symbol, order_type, close_side, None, None,
                    params={
                        'stopPrice': float(stop_str),
                        'closePosition': True,
                        'workingType': 'MARK_PRICE',
                        'recvWindow': 10000,
                    }
                )
                # ⚠️ 생성 응답만 믿지 않는다. 바이낸스 데모 트레이딩은 조건부 주문을
                # 받아 주문 ID까지 돌려주지만 **실제로는 만들지 않는다**
                # (openOrders/allOrders/fetch_order 어디에도 없다).
                # 검증 없이 성공으로 처리하다가, 포지션이 스탑 없이 방치되는데도
                # 로그에는 "SL 주문 등록"이 찍히는 상태로 오래 돌았다.
                logging.info(f"{label} 주문 요청 전송: {order_type} @ {stop_str} "
                             f"(id={order.get('id')})")
                requested += 1
            except Exception as e:
                msg = str(e)
                if '-4130' in msg:
                    # "이미 존재한다"는 응답. 실제로 존재하는지는 아래에서 이력으로 확인한다.
                    logging.info(f"{label}: 거래소가 이미 존재한다고 응답(-4130).")
                    requested += 1
                else:
                    logging.error(f"{label} 보호 주문 등록 실패 ({order_type} @ {stop_price}): {e}")

        # 요청이 통했다고 끝이 아니다. 이력으로 실제 존재를 확인한다.
        placed = 2 if (requested == 2 and self._has_live_conditional_orders()) else 0
        if placed == 2:
            logging.info("보호 주문 존재 확인됨 (주문 이력 대조).")
            self._protected_position = position_key
            self._protected_at = time.time()
        else:
            if requested == 2:
                self._exchange_stops_unsupported = True
            self._protected_position = None
            if self._exchange_stops_unsupported:
                # 거래소 스탑을 못 쓰는 환경이다. 봇 내부 폴링 SL/TP가 유일한
                # 보호 수단이므로, 봇이 죽으면 포지션은 무방비다.
                # (run_once의 SL/TP 검사와 tools/check_heartbeat.py가 그 역할)
                if not self._warned_no_exchange_stops:
                    logging.error(
                        "🚨 이 거래소 환경은 조건부 주문(STOP_MARKET/TAKE_PROFIT_MARKET)을 "
                        "생성하지 않습니다. SL/TP는 봇 내부 폴링으로만 관리됩니다. "
                        "봇이 정지하면 포지션이 무방비 상태가 되므로 하트비트 감시가 필수입니다."
                    )
                    self._warned_no_exchange_stops = True
            else:
                logging.warning(f"보호 주문 요청이 완전하지 않습니다 ({requested}/2). 다음 루프에서 재시도합니다.")

    # 보호 주문을 다시 확인/재등록하는 주기(초). 조회로 존재를 확인할 수 없으므로
    # 외부에서 취소되는 경우에 대비해 느리게 재확인한다. 30분이면 API 비용은
    # 무시할 수준이고, 스탑 없이 방치되는 최대 시간도 그만큼으로 제한된다.
    PROTECTIVE_REFRESH_SEC = 1800

    def ensure_protective_orders(self, pos_size, entry_price):
        """포지션에 SL/TP가 걸려 있도록 보장한다.

        **거래소 조회로는 확인할 수 없다.** `fetch_open_orders()`가
        `closePosition=True` 조건부 주문을 돌려주지 않기 때문이다(0건으로 나오지만
        같은 주문을 걸면 -4130으로 거절당한다). 처음엔 조회 결과를 믿고 매 루프
        재등록을 시도했는데, 30초마다 -4130 에러만 쌓였다.

        그래서 조회 대신 **로컬 상태**로 추적한다.
          - 어떤 포지션에 대해 보호 주문을 걸었는지 `_protected_position`에 기록
          - 포지션이 바뀌었거나(재시작 포함) 기록이 없으면 취소 후 재등록
          - 기록이 맞아도 PROTECTIVE_REFRESH_SEC마다 한 번은 다시 걸어,
            외부에서 취소된 경우 스스로 복구한다
        """
        if not self.is_futures or pos_size == 0:
            self._protected_position = None
            return

        if self._exchange_stops_unsupported:
            # 이 환경은 조건부 주문을 만들지 못한다. 매 루프 헛되이 요청하지 않는다.
            # SL/TP는 run_once의 폴링 검사가 담당한다.
            return

        # 포지션 식별은 **수량(부호 포함)만** 쓴다.
        # 진입가를 키에 넣었더니 매 루프 불일치가 났다 — 주문 체결가(order.average,
        # 예: 80854.4)와 거래소가 보고하는 포지션 평균단가(예: 80856.93)가 미세하게
        # 다르기 때문이다. 수량은 정확히 일치한다. 청산 시 플래그를 비우므로
        # 같은 수량으로 새 포지션을 잡아도 open_position 경로에서 다시 설정된다.
        key = round(float(pos_size), 8)

        if self._protected_position == key:
            age = time.time() - self._protected_at
            if age < self.PROTECTIVE_REFRESH_SEC:
                return
            logging.info(
                f"보호 주문 주기 재확인 ({age/60:.0f}분 경과). 취소 후 다시 겁니다."
            )
        else:
            logging.info(
                f"보호 주문 기록 없음 (포지션 {pos_size} @ {entry_price:.2f}). 등록합니다."
            )

        direction = 'BUY' if pos_size > 0 else 'SELL'
        self.place_protective_orders(direction, entry_price, position_key=key)

    def run_once(self):
        """Runs a single iteration of fetching candles, computing signals, and placing trades."""
        try:
            # 1. 캔들 수집.
            # REGIME_LOOKBACK(=1000)봉을 받는다. 예전에는 200봉만 받았는데
            # EMA200 워밍업에 200봉으로는 턱없이 부족해서, 백테스트(전 이력)와
            # 지표값 자체가 달라졌다.
            candles = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=REGIME_LOOKBACK)
            if not candles:
                logging.warning("Failed to fetch candles from exchange.")
                return
                
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 2. Add indicators and get signals
            signals = self.strategy.generate_signals(df)
            
            # Use second-to-last candle for signals to avoid repainting (last closed candle)
            last_closed_candle = df.iloc[-2]
            signal = signals.iloc[-2]
            
            logging.info(f"Checking state. Last closed candle price: {last_closed_candle['close']}. Signal: {signal}")
            
            # 3. Get current position details
            pos_size, entry_price = self.get_position()
            pos_dir = 1 if pos_size > 0 else (-1 if pos_size < 0 else 0)
            
            # 4. Check dynamic Stop Loss / Take Profit for open positions
            if pos_dir != 0:
                self.ensure_protective_orders(pos_size, entry_price)
                current_price = df.iloc[-1]['close']
                # Determine current SL/TP levels based on strategy rules
                stop_loss_pct = self.strategy.stop_loss_pct
                take_profit_pct = self.strategy.take_profit_pct
                
                # Check for dynamic adjustment if adaptive
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
                    self.last_exit_candle_timestamp = last_closed_candle['timestamp']
                    return
            
            # Check for SL/TP cooldown on the current candle
            current_candle_ts = last_closed_candle['timestamp']
            is_cooldown = (self.last_exit_candle_timestamp is not None and current_candle_ts == self.last_exit_candle_timestamp)

            # 5. Order execution based on signals
            if signal == 1 and pos_dir != 1:
                # Settle current opposite position first
                if pos_dir == -1:
                    self.close_position(pos_size, last_closed_candle['close'], "SIGNAL_REVERSAL")
                    
                # Open Long
                if not is_cooldown:
                    self.open_position("BUY", last_closed_candle['close'])
                else:
                    logging.info("Skipping Long entry due to SL/TP cooldown on the current candle.")
                
            elif signal == -1 and pos_dir != -1:
                if not self.is_futures:
                    logging.info("Short signal ignored. Spot market does not support short positions.")
                    return
                    
                # Settle current opposite position first
                if pos_dir == 1:
                    self.close_position(pos_size, last_closed_candle['close'], "SIGNAL_REVERSAL")
                    
                # Open Short
                if not is_cooldown:
                    self.open_position("SELL", last_closed_candle['close'])
                else:
                    logging.info("Skipping Short entry due to SL/TP cooldown on the current candle.")
                
            elif signal == 0 and pos_dir != 0:
                # Close current position
                self.close_position(pos_size, last_closed_candle['close'], "SIGNAL_EXIT")
                
        except Exception as e:
            logging.error(f"Error in run_once loop: {e}", exc_info=True)
    def open_position(self, direction, est_price):
        """시장가로 포지션을 열고, 거래소에 SL/TP 보호 주문을 건다."""
        try:
            # ── Circuit Breaker ────────────────────────────────────────────────
            # 연속 5회 이상 타임아웃 발생 시 5분간 주문 건너뜀 (서버 보호)
            elapsed = time.time() - self._last_order_timeout_ts
            if self._order_timeout_count >= 5 and elapsed < 300:
                remaining = int(300 - elapsed)
                logging.warning(
                    f"[Circuit Breaker] {self._order_timeout_count}회 연속 타임아웃. "
                    f"서버 보호를 위해 주문을 건너뜁니다. (잔여 제한시간: {remaining}초)"
                )
                return
            elif self._order_timeout_count >= 5 and elapsed >= 300:
                logging.info("[Circuit Breaker] 제한시간 해제. 주문 시도를 재개합니다.")
                self._order_timeout_count = 0
            # ──────────────────────────────────────────────────────────────────

            price = self._current_price()
            if price is None:
                # 가격을 못 얻으면 주문하지 않는다. 다음 루프에서 다시 시도한다.
                return

            balance = self.get_balance()
            max_alloc = self.strategy.max_allocation_pct
            leverage = min(self.strategy.leverage if self.is_futures else 1, 3)

            trade_value = balance * max_alloc * leverage
            amount = trade_value / price

            self.exchange.load_markets()
            amount_float_rounded = float(self.exchange.amount_to_precision(self.symbol, amount))

            if amount_float_rounded <= 0:
                logging.warning(f"주문 수량이 너무 작습니다: {amount_float_rounded}")
                return

            # 최소명목가 검사. BTC/USDT 선물은 50 USDT이고, 미달이면 거래소가 거절한다.
            market = self.exchange.market(self.symbol)
            min_notional = (market.get('limits', {}).get('cost', {}) or {}).get('min')
            notional = amount_float_rounded * price
            if min_notional and notional < float(min_notional):
                logging.warning(
                    f"주문 명목가 {notional:.2f} USDT가 최소 {min_notional} USDT 미만입니다. "
                    f"진입을 건너뜁니다. (잔고 {balance:.2f}, 배분 {max_alloc:.0%}, 레버리지 {leverage}x)"
                )
                return

            logging.info(
                f"{direction} 시장가 진입 시도: {amount_float_rounded} {self.symbol} "
                f"@ ~{price} (명목가 ~{notional:.2f} USDT)"
            )

            self.set_leverage()

            side = 'buy' if direction == 'BUY' else 'sell'
            order = self.exchange.create_market_order(
                self.symbol, side, amount_float_rounded,
                params={'recvWindow': 10000}
            )

            fill_price = float(order.get('average') or order.get('price') or price)
            filled_amount = float(order.get('filled') or amount_float_rounded)

            # 기록은 모든 진입 경로에서 반드시 남는다. 예전 지정가/폴백 경로에서는
            # 이게 빠져 trade_logs에 구멍이 생겼다 (7/28 진입 건 누락).
            self.log_trade("OPEN", direction, fill_price, filled_amount)
            logging.info(f"시장가 체결 완료: {filled_amount} @ {fill_price}")

            # 진입 즉시 SL/TP를 거래소에 올린다
            pos_key = round(filled_amount if direction == 'BUY' else -filled_amount, 8)
            self.place_protective_orders(direction, fill_price, position_key=pos_key)

            self._order_timeout_count = 0

        except ccxt.RequestTimeout as e:
            # -1007: 주문 상태 불명 → 포지션 조회로 실제 체결 여부만 확인
            # ⚠️ 즉시 재시도 금지: demo 서버가 408을 리턴하는 상황에서 재시도는 반드시 실패함
            self._order_timeout_count += 1
            self._last_order_timeout_ts = time.time()
            logging.warning(
                f"[open_position] RequestTimeout #{self._order_timeout_count} (-1007). "
                f"포지션 상태 확인 중... (다음 루프에서 재시도)"
            )
            time.sleep(3)
            try:
                pos_size, pos_entry = self.get_position()
                expected_dir = 1 if direction == 'BUY' else -1
                already_filled = (expected_dir == 1 and pos_size > 0) or (expected_dir == -1 and pos_size < 0)
                if already_filled:
                    self._order_timeout_count = 0
                    logging.info(f"[Timeout Recovery] 포지션 체결 확인 ({pos_size}). 성공으로 처리합니다.")
                    self.log_trade("OPEN", direction, pos_entry, abs(pos_size))
                    self.place_protective_orders(direction, pos_entry)
                else:
                    msg = (
                        f"[Timeout Recovery] 포지션 미확인 (count={self._order_timeout_count}). "
                        f"다음 루프(약 30초 후)에 재시도합니다."
                    )
                    if self._order_timeout_count >= 5:
                        msg += (
                            f" ⚠️ 연속 {self._order_timeout_count}회 타임아웃 — "
                            f"Circuit Breaker 활성화. 향후 5분간 주문을 일시 중단합니다."
                        )
                    logging.warning(msg)
            except Exception as ve:
                logging.error(f"[Timeout Recovery] 포지션 확인 실패: {ve}", exc_info=True)

        except Exception as e:
            logging.error(f"포지션 진입 실패: {e}", exc_info=True)


    def close_position(self, current_size, est_price, reason):
        """Places an order to close current position."""
        try:
            if current_size == 0:
                return
                
            # For spot, size is positive. To close spot, we sell all base asset.
            # For futures, size is positive (Long) or negative (Short).
            # To close futures, we place order of opposite side.
            side = 'sell' if current_size > 0 else 'buy'
            abs_size = abs(current_size)
            
            # Load markets to format precision
            self.exchange.load_markets()
            abs_size = float(self.exchange.amount_to_precision(self.symbol, abs_size))
            
            logging.info(f"Closing position of size {abs_size} {self.symbol}. Reason: {reason}")

            # 남아 있는 SL/TP 주문을 먼저 걷어낸다. 안 그러면 다음 포지션에 대고
            # 엉뚱하게 발동한다.
            self.cancel_protective_orders()
            
            # recvWindow 확장으로 타임아웃 빈도 감소
            order = self.exchange.create_market_order(
                self.symbol, side, abs_size,
                params={'recvWindow': 10000}
            )
            
            fill_price = order.get('price') or order.get('average') or est_price
            fill_price = float(fill_price)
            filled_amount = order.get('filled') or abs_size
            filled_amount = float(filled_amount)
            
            pnl = 0.0
            self._order_timeout_count = 0  # 성공 시 카운터 리셋
            self.log_trade(f"CLOSE_{reason}", "SELL" if current_size > 0 else "BUY", fill_price, filled_amount, pnl)

        except ccxt.RequestTimeout as e:
            # 청산 타임아웃: 포지션이 이미 청산됐는지 확인 (즉시 재시도 금지)
            self._order_timeout_count += 1
            self._last_order_timeout_ts = time.time()
            logging.warning(
                f"[close_position] RequestTimeout #{self._order_timeout_count} (-1007). "
                f"포지션 상태 확인 중... (다음 루프에서 재시도)"
            )
            time.sleep(3)
            try:
                pos_size_after, _ = self.get_position()
                already_closed = (
                    pos_size_after == 0
                    or (current_size > 0 and pos_size_after <= 0)
                    or (current_size < 0 and pos_size_after >= 0)
                )
                if already_closed:
                    # 이미 청산됨 → 성공으로 처리 후 카운터 리셋
                    self._order_timeout_count = 0
                    logging.info(f"[Timeout Recovery] 청산 확인 (pos={pos_size_after}). 성공으로 처리합니다.")
                    self.log_trade(f"CLOSE_{reason}", "SELL" if current_size > 0 else "BUY", est_price, abs_size, 0.0)
                else:
                    # 포지션 잔존 → 즉시 재시도 없이 다음 루프에 위임
                    logging.warning(
                        f"[Timeout Recovery] 포지션 잔존 (count={self._order_timeout_count}). "
                        f"다음 루프(약 30초 후)에 재청산을 시도합니다."
                    )
            except Exception as ve:
                logging.error(f"[Timeout Recovery] 포지션 확인 실패: {ve}", exc_info=True)

        except Exception as e:
            logging.error(f"Failed to close position: {e}", exc_info=True)

    def run_loop(self):
        """Continuous execution loop."""
        self.running = True
        self.save_status()
        logging.info("Starting live trading bot loop.")
        
        while self.running:
            # Check if status has been set to stopped externally
            if not self.check_status_active():
                logging.info("Stopping bot loop based on external status file.")
                self.running = False
                break
                
            self.run_once()
            self.write_heartbeat()

            # Sleep based on timeframe.
            # E.g. check every 30 seconds
            # In production, we'd sleep until next candle close + 5s.
            time.sleep(30)
            
        self.running = False
        self.save_status()
        logging.info("Live trading bot loop terminated.")

    def write_heartbeat(self):
        """마지막으로 루프를 돈 시각을 파일에 남긴다.

        봇이 7/29~9/14 47일간 죽어 있었는데 아무도 몰랐다. 외부 감시(cron 등)가
        이 파일의 나이를 보고 정체를 감지할 수 있게 한다.
        """
        try:
            HEARTBEAT_FILE.write_text(json.dumps({
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'strategy': self.strategy_name,
            }))
        except Exception as e:
            logging.warning(f"하트비트 기록 실패: {e}")

    def save_status(self):
        """Saves current running status to a JSON file."""
        status = {
            'running': self.running,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'is_futures': self.is_futures,
            'strategy_name': self.strategy_name,
            'strategy_params': self.strategy_params,
            'use_testnet': self.use_testnet,
            'last_check': datetime.now().isoformat()
        }
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=4)

    def check_status_active(self):
        """Checks if bot is allowed to continue running according to status file."""
        if not os.path.exists(STATUS_FILE):
            return True
        try:
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f)
            return status.get('running', True)
        except Exception:
            return True

def start_bot(api_key, secret_key, symbol, timeframe, is_futures, strategy_name, strategy_params, use_testnet):
    """Entry point to launch the bot."""
    bot = LiveTrader(api_key, secret_key, symbol, timeframe, is_futures, strategy_name, strategy_params, use_testnet)
    bot.run_loop()

def stop_bot():
    """Modifies the status file to request the bot to stop."""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f)
            status['running'] = False
            with open(STATUS_FILE, 'w') as f:
                json.dump(status, f, indent=4)
            print("Stop signal sent to bot.")
            return True
        except Exception as e:
            print(f"Error sending stop signal: {e}")
    return False

if __name__ == "__main__":
    # Standard dummy launch for script testing if API keys are mocked
    pass
