import os
import time
import json
import sqlite3
import logging
from datetime import datetime
import pandas as pd
import ccxt

from data_manager import DataManager
from indicators import add_all_indicators
from strategies import get_strategy_by_name

# Configure logging
logging.basicConfig(
    filename='live_trader.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DB_NAME = "trading_data.db"
STATUS_FILE = "bot_status.json"

class LiveTrader:
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
            logging.info("Initialized CCXT exchange in TESTNET/SANDBOX mode.")
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

    def run_once(self):
        """Runs a single iteration of fetching candles, computing signals, and placing trades."""
        try:
            # 1. Fetch latest candles (warmup + extra)
            # Fetch 200 candles to calculate indicators accurately
            candles = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=200)
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
                    return
            
            # 5. Order execution based on signals
            if signal == 1 and pos_dir != 1:
                # Settle current opposite position first
                if pos_dir == -1:
                    self.close_position(pos_size, last_closed_candle['close'], "SIGNAL_REVERSAL")
                    
                # Open Long
                self.open_position("BUY", last_closed_candle['close'])
                
            elif signal == -1 and pos_dir != -1:
                if not self.is_futures:
                    logging.info("Short signal ignored. Spot market does not support short positions.")
                    return
                    
                # Settle current opposite position first
                if pos_dir == 1:
                    self.close_position(pos_size, last_closed_candle['close'], "SIGNAL_REVERSAL")
                    
                # Open Short
                self.open_position("SELL", last_closed_candle['close'])
                
            elif signal == 0 and pos_dir != 0:
                # Close current position
                self.close_position(pos_size, last_closed_candle['close'], "SIGNAL_EXIT")
                
        except Exception as e:
            logging.error(f"Error in run_once loop: {e}", exc_info=True)

    def open_position(self, direction, est_price):
        """Calculates size and places an order to open a position."""
        try:
            balance = self.get_balance()
            max_alloc = self.strategy.max_allocation_pct
            leverage = min(self.strategy.leverage if self.is_futures else 1, 3) # Force 3x cap
            
            # Sizing calculation
            allocated_usdt = balance * max_alloc
            trade_value = allocated_usdt * leverage
            
            # Convert to asset units
            amount = trade_value / est_price
            
            # Fetch market details to round amount and price correctly
            markets = self.exchange.load_markets()
            market = markets[self.symbol]
            amount = self.exchange.amount_to_precision(self.symbol, amount)
            amount_float = float(amount)
            
            if amount_float <= 0:
                logging.warning(f"Calculated trade size is too small: {amount_float}")
                return
                
            logging.info(f"Attempting to open {direction} position of size {amount_float} {self.symbol} (value ~{trade_value} USDT)")
            
            # Set leverage before trade
            self.set_leverage()
            
            # Place market order
            side = 'buy' if direction == 'BUY' else 'sell'
            order = self.exchange.create_market_order(self.symbol, side, amount_float)
            
            fill_price = float(order.get('price', order.get('average', est_price)))
            filled_amount = float(order.get('filled', amount_float))
            
            self.log_trade("OPEN", direction, fill_price, filled_amount)
            
        except Exception as e:
            logging.error(f"Failed to open position: {e}", exc_info=True)

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
            
            order = self.exchange.create_market_order(self.symbol, side, abs_size)
            
            fill_price = float(order.get('price', order.get('average', est_price)))
            filled_amount = float(order.get('filled', abs_size))
            
            # Calculate PnL (for log)
            # In live, we can fetch fill details, but let's approximate PnL
            pnl = 0.0
            # For simple logging:
            self.log_trade(f"CLOSE_{reason}", "SELL" if current_size > 0 else "BUY", fill_price, filled_amount, pnl)
            
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
            
            # Sleep based on timeframe.
            # E.g. check every 30 seconds
            # In production, we'd sleep until next candle close + 5s.
            time.sleep(30)
            
        self.running = False
        self.save_status()
        logging.info("Live trading bot loop terminated.")

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
