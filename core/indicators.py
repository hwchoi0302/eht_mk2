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
    각 봉의 시장 국면을 판정한다.

    - BULL:     ADX > 22 이고 종가 > EMA50 > EMA200
    - BEAR:     ADX > 22 이고 종가 < EMA50 < EMA200
    - SIDEWAYS: 그 외 전부 (ADX가 낮거나, EMA 배열이 엇갈리거나, 워밍업 구간)

    예전에는 파이썬 for 루프로 한 봉씩 돌았다. 워크포워드 탐색은 이 함수를
    수만 번 호출하므로(조합 × 폴드) 그 루프가 전체 탐색 시간을 지배했다.
    아래는 같은 판정을 벡터화한 것이다 — 출력은 루프 버전과 완전히 동일하다.
    """
    close = df['close']
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    adx = calculate_adx(df, 14)

    # 워밍업 구간(EMA200/ADX가 NaN)은 판정하지 않고 SIDEWAYS로 둔다
    valid = ema_200.notna() & adx.notna()
    strong_trend = valid & (adx > 22)

    trend_up = (close > ema_50) & (ema_50 > ema_200)
    trend_down = (close < ema_50) & (ema_50 < ema_200)

    regimes = np.where(
        strong_trend & trend_up, 'BULL',
        np.where(strong_trend & trend_down, 'BEAR', 'SIDEWAYS')
    )
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


# ─────────────────────────────────────────────────────────────────────────────
# 국면 확정 (라이브 / 백테스트 공용)
# ─────────────────────────────────────────────────────────────────────────────
#
# 이전에는 이 로직이 run_regime_bot.py 안에 인라인으로만 있었고 백테스터는
# 원시(raw) regime을 그대로 썼다. 국면 확정은 경로 의존적이라 두 경로의 결과가
# 어긋났다. 이제 양쪽 모두 아래 confirm_regimes()를 호출한다.

# 라이브에서 캔들을 몇 개 받아올지. EMA200 + ADX 워밍업에 200봉으로는 부족해서
# 백테스트(전 이력)와 지표값 자체가 달라졌다. 넉넉히 받는다. (바이낸스 상한 1500)
REGIME_LOOKBACK = 1000


def confirm_regimes(raw_regimes, confirm_candles=1):
    """휩소 방지용 확정 국면 시계열을 만든다.

    새 국면이 연속 `confirm_candles`개 캔들 동안 유지되어야 전환을 인정한다.
    경로 의존적이므로 라이브와 백테스트가 반드시 같은 입력 길이 위에서
    같은 함수를 돌려야 결과가 일치한다.

    Args:
        raw_regimes: classify_market_regime()이 낸 'BULL'/'BEAR'/'SIDEWAYS' 시리즈
        confirm_candles: 전환에 필요한 연속 캔들 수 (1이면 확정 없이 그대로)

    Returns:
        pd.Series: 확정된 국면 (입력과 같은 index)
    """
    raw = pd.Series(raw_regimes)
    if len(raw) == 0:
        return raw.copy()
    if confirm_candles is None or confirm_candles <= 1:
        return raw.copy()

    values = raw.to_numpy()
    confirmed = np.empty(len(values), dtype=object)

    current = values[0]
    candidate = current
    streak = 0

    for i, raw_reg in enumerate(values):
        if raw_reg == current:
            candidate = current
            streak = 0
        else:
            if raw_reg == candidate:
                streak += 1
            else:
                candidate = raw_reg
                streak = 1
            if streak >= confirm_candles:
                current = candidate
                streak = 0
        confirmed[i] = current

    return pd.Series(confirmed, index=raw.index)


def add_confirmed_regime(df, confirm_candles=1):
    """df에 'regime_confirmed' 컬럼을 붙여 돌려준다 (원본 비파괴)."""
    df = df.copy()
    if 'regime' not in df.columns:
        df['regime'] = classify_market_regime(df)
    df['regime_confirmed'] = confirm_regimes(df['regime'], confirm_candles)
    return df
