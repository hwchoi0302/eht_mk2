import os
import sys
import ccxt
import sqlite3
import pandas as pd
from dotenv import load_dotenv

# Load .env from parent directory
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_path)
load_dotenv(dotenv_path=os.path.join(workspace_path, ".env"))

api_key = os.getenv("BINANCE_TESTNET_API_KEY")
secret_key = os.getenv("BINANCE_TESTNET_SECRET_KEY")

if not api_key or not secret_key:
    print("Error: Binance Testnet API credentials not found in .env")
    sys.exit(1)

exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret_key,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

try:
    exchange.enable_demo_trading(True)
except Exception as e:
    exchange.set_sandbox_mode(True)

print("=== 1. Futures Account Balance ===")
try:
    balance = exchange.fetch_balance()
    # Total margin balance, equity, and available
    usdt_info = balance.get('info', {}).get('assets', [])
    usdt_asset = next((asset for asset in usdt_info if asset['asset'] == 'USDT'), None)
    
    usdt_total = float(balance['total'].get('USDT', 0.0))
    usdt_free = float(balance['free'].get('USDT', 0.0))
    usdt_used = float(balance['used'].get('USDT', 0.0))
    
    print(f"Total Equity: {usdt_total:,.4f} USDT")
    print(f"Available Balance: {usdt_free:,.4f} USDT")
    print(f"Used Margin: {usdt_used:,.4f} USDT")
    
    if usdt_asset:
        print(f"Wallet Balance: {float(usdt_asset.get('walletBalance', 0.0)):,.4f} USDT")
        print(f"Unrealized PnL: {float(usdt_asset.get('unrealizedProfit', 0.0)):,.4f} USDT")
except Exception as e:
    print(f"Error fetching balance: {e}")

print("\n=== 2. Current Positions ===")
try:
    positions = exchange.fetch_positions(symbols=['BTC/USDT', 'ETH/USDT'])
    active_positions = []
    for pos in positions:
        # ccxt는 값이 없는 필드를 키째로 None으로 채워 보내므로 .get의 기본값이 먹지 않는다
        num = lambda v: float(v) if v is not None else 0.0
        size = num(pos.get('contracts'))
        if size > 0:
            active_positions.append(pos)
            side = (pos.get('side') or '').upper()
            entry = num(pos.get('entryPrice'))
            mark = num(pos.get('markPrice'))
            unpnl = num(pos.get('unrealizedPnl'))
            liq = num(pos.get('liquidationPrice'))
            leverage = pos.get('leverage', 1)
            print(f"Symbol: {pos['symbol']}")
            print(f"  Side: {side} (Leverage: {leverage}x)")
            print(f"  Size: {size}")
            print(f"  Entry Price: {entry:,.2f}")
            print(f"  Mark Price: {mark:,.2f}")
            print(f"  Unrealized PnL: {unpnl:+,.4f} USDT")
            print(f"  Liquidation Price: {liq:,.2f}")
    if not active_positions:
        print("No active positions.")
except Exception as e:
    print(f"Error fetching positions: {e}")

print("\n=== 3. Database Trade Logs ===")
try:
    from core.paths import DB_PATH
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query("SELECT * FROM trade_logs ORDER BY datetime DESC LIMIT 15", conn)
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("Database trade logs table is empty.")
    conn.close()
except Exception as e:
    print(f"Error reading DB: {e}")

print("\n=== 4. Recent Trades from Exchange ===")
try:
    trades = exchange.fetch_my_trades(symbol='BTC/USDT', limit=10)
    if trades:
        for t in trades:
            print(f"[{t['datetime']}] {t['side'].upper()} {t['amount']} BTC at {t['price']} (Cost: {t['cost']} USDT)")
    else:
        print("No recent trades found on exchange.")
except Exception as e:
    print(f"Error fetching recent trades: {e}")
