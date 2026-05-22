import pandas as pd
import numpy as np

def calculate_sma(series, period):
    return series.rolling(window=period).mean()

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero
    rs = gain / np.where(loss == 0, 1e-10, loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = calculate_ema(series, fast_period)
    slow_ema = calculate_ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(series, period=20, num_std=2):
    sma = calculate_sma(series, period)
    std = series.rolling(window=period).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    return upper_band, sma, lower_band

def calculate_atr(df, period=14):
    """Calculates the Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_adx(df, period=14):
    """Calculates the Average Directional Index (ADX)."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high.diff()
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth True Range and Directional Moves
    tr_smooth = tr.rolling(window=period).sum()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).sum()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).sum()
    
    plus_di = 100 * (plus_dm_smooth / np.where(tr_smooth == 0, 1e-10, tr_smooth))
    minus_di = 100 * (minus_dm_smooth / np.where(tr_smooth == 0, 1e-10, tr_smooth))
    
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100 * (di_diff / np.where(di_sum == 0, 1e-10, di_sum))
    
    adx = pd.Series(dx).rolling(window=period).mean()
    # Align index with original DataFrame
    adx.index = df.index
    return adx

def calculate_heikin_ashi(df):
    """
    Calculates Heikin-Ashi OHLC values.
    Returns a DataFrame with columns: ['ha_open', 'ha_high', 'ha_low', 'ha_close']
    """
    ha_df = pd.DataFrame(index=df.index)
    close = df['close']
    open_val = df['open']
    high = df['high']
    low = df['low']
    
    ha_close = (open_val + high + low + close) / 4
    
    ha_open = np.zeros(len(df))
    ha_open[0] = (open_val.iloc[0] + close.iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
        
    ha_df['ha_open'] = ha_open
    ha_df['ha_close'] = ha_close
    ha_df['ha_high'] = np.maximum(high, np.maximum(ha_open, ha_close))
    ha_df['ha_low'] = np.minimum(low, np.minimum(ha_open, ha_close))
    
    return ha_df

def add_all_indicators(df):
    """Calculates and appends all indicators to the DataFrame."""
    df = df.copy()
    close = df['close']
    
    df['sma_20'] = calculate_sma(close, 20)
    df['ema_50'] = calculate_ema(close, 50)
    df['ema_200'] = calculate_ema(close, 200)
    
    df['rsi_14'] = calculate_rsi(close, 14)
    
    macd_l, signal_l, hist = calculate_macd(close)
    df['macd'] = macd_l
    df['macd_signal'] = signal_l
    df['macd_hist'] = hist
    
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_mid
    df['bb_lower'] = bb_lower
    
    df['atr_14'] = calculate_atr(df, 14)
    df['adx_14'] = calculate_adx(df, 14)
    
    # Add Heikin-Ashi columns
    ha_df = calculate_heikin_ashi(df)
    df['ha_open'] = ha_df['ha_open']
    df['ha_high'] = ha_df['ha_high']
    df['ha_low'] = ha_df['ha_low']
    df['ha_close'] = ha_df['ha_close']
    
    # Add market regime classification
    df['regime'] = classify_market_regime(df)
    
    return df

def classify_market_regime(df):
    """
    Classifies the market regime for each row in the DataFrame.
    - BULL: Price above 50 EMA and 50 EMA > 200 EMA (or ADX shows strong trend & EMA trend is up)
    - BEAR: Price below 50 EMA and 50 EMA < 200 EMA (or ADX shows strong trend & EMA trend is down)
    - SIDEWAYS: Price crossing EMA, ADX is low (< 22), or tight Bollinger Bands.
    """
    close = df['close']
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    adx = calculate_adx(df, 14)
    
    regimes = []
    for i in range(len(df)):
        # Handle warm-up periods where EMA or ADX might be NaN
        if pd.isna(ema_200.iloc[i]) or pd.isna(adx.iloc[i]):
            regimes.append('SIDEWAYS')
            continue
            
        cur_close = close.iloc[i]
        cur_ema_50 = ema_50.iloc[i]
        cur_ema_200 = ema_200.iloc[i]
        cur_adx = adx.iloc[i]
        
        # Trend indicators
        trend_up = cur_close > cur_ema_50 and cur_ema_50 > cur_ema_200
        trend_down = cur_close < cur_ema_50 and cur_ema_50 < cur_ema_200
        
        if cur_adx > 22:
            if trend_up:
                regimes.append('BULL')
            elif trend_down:
                regimes.append('BEAR')
            else:
                regimes.append('SIDEWAYS')
        else:
            # Low ADX -> Range bound
            regimes.append('SIDEWAYS')
            
    return pd.Series(regimes, index=df.index)

# Quick debug check
if __name__ == "__main__":
    # Create dummy data to verify calculations
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=250, freq="h")
    prices = 20000 + np.cumsum(np.random.normal(0, 100, 250))
    dummy_df = pd.DataFrame({
        'open': prices - 10,
        'high': prices + 20,
        'low': prices - 20,
        'close': prices,
        'volume': np.random.randint(100, 1000, 250)
    }, index=dates)
    
    df_with_ind = add_all_indicators(dummy_df)
    print("Indicators calculated successfully. Columns:")
    print(df_with_ind.columns)
    print("\nRegime Distribution:")
    print(df_with_ind['regime'].value_counts())
