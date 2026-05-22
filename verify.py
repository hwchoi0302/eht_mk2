import sys
from datetime import datetime, timedelta
import pandas as pd

print("1. Testing imports...")
try:
    import ccxt
    import optuna
    import streamlit
    import plotly
    print(f"   ccxt version: {ccxt.__version__}")
    print(f"   optuna version: {optuna.__version__}")
except Exception as e:
    print(f"   Import failed: {e}")
    sys.exit(1)

print("\n2. Importing custom modules...")
try:
    from data_manager import DataManager
    from indicators import add_all_indicators
    from strategies import EMACrossStrategy, RSIBBStrategy, VolatilityBreakoutStrategy, AdaptiveRegimeStrategy
    from backtester import Backtester
    from optimizer import StrategyOptimizer
    print("   All custom modules imported successfully!")
except Exception as e:
    print(f"   Custom import failed: {e}")
    sys.exit(1)

print("\n3. Testing DataManager with real exchange query (Binance)...")
try:
    dm = DataManager()
    # Download 5 days of hourly spot data for BTC/USDT (to keep it small and fast)
    start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"   Fetching BTC/USDT Spot from {start_date} to {end_date}...")
    df = dm.get_candles("BTC/USDT", "1h", start_date, end_date, is_futures=False, force_download=True)
    print(f"   Fetched {len(df)} candles.")
    if df.empty:
        print("   Warning: DataFrame is empty!")
    else:
        print(f"   Columns: {list(df.columns)}")
        print(f"   First row close: {df['close'].iloc[0]}, Last row close: {df['close'].iloc[-1]}")
except Exception as e:
    print(f"   DataManager test failed: {e}")
    sys.exit(1)

print("\n4. Testing Indicators and Market Regime Classifier...")
try:
    df_ind = add_all_indicators(df)
    print(f"   Indicators calculated successfully. Added columns: {[col for col in df_ind.columns if col not in df.columns]}")
    print(f"   Market Regimes distribution:")
    print(df_ind['regime'].value_counts())
except Exception as e:
    print(f"   Indicators test failed: {e}")
    sys.exit(1)

print("\n5. Testing Backtester with EMA Cross Strategy...")
try:
    bt = Backtester(initial_capital=10000.0)
    strategy = EMACrossStrategy(fast_period=10, slow_period=30, leverage=1)
    metrics, df_eq, trades = bt.run(df, strategy, is_futures=False)
    print("   Backtest run completed!")
    print(f"   Initial Capital: ${metrics['initial_capital']:.2f}")
    print(f"   Final Equity: ${metrics['final_equity']:.2f}")
    print(f"   Total Return: {metrics['total_return']*100:.2f}%")
    print(f"   Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"   Total Trades: {metrics['total_trades']}")
except Exception as e:
    print(f"   Backtester test failed: {e}")
    sys.exit(1)

print("\n6. Testing StrategyOptimizer (Optuna)...")
try:
    opt = StrategyOptimizer()
    print("   Running optimization with 5 trials to test functionality...")
    best_params, best_val, study = opt.optimize(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date_str=start_date,
        end_date_str=end_date,
        strategy_name="EMA_Cross",
        is_futures=False,
        target_metric="sharpe_ratio",
        n_trials=5
    )
    print("   Optimization successful!")
    print(f"   Best Value (Sharpe): {best_val:.4f}")
    print(f"   Best Parameters: {best_params}")
except Exception as e:
    print(f"   Optimizer test failed: {e}")
    sys.exit(1)

print("\n=== ALL TESTS PASSED SUCCESSFULLY! ===")
