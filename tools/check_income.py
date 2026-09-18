import os
import sys
import ccxt
from dotenv import load_dotenv

# Load .env from parent directory
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(workspace_path, ".env"))

api_key = os.getenv("BINANCE_TESTNET_API_KEY")
secret_key = os.getenv("BINANCE_TESTNET_SECRET_KEY")

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

print("=== Fetching Income History ===")
try:
    # fapiPrivateGetIncome is a binance-specific endpoint in ccxt
    income = exchange.fapiPrivateGetIncome()
    print(f"Total income records: {len(income)}")
    # Print first few and last few
    print("\nFirst 5 records (oldest):")
    for item in income[:5]:
        print(item)
    print("\nLast 10 records (newest):")
    for item in income[-10:]:
        import datetime
        dt = datetime.datetime.fromtimestamp(int(item['time'])/1000.0)
        print(f"[{dt}] {item['incomeType']}: {item['income']} {item['asset']} (Symbol: {item.get('symbol')})")
except Exception as e:
    print(f"Error: {e}")
