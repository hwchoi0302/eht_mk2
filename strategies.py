import pandas as pd
import numpy as np
from indicators import add_all_indicators


class BaseStrategy:
    def __init__(self, name="기본전략", **kwargs):
        self.name = name
        self.parameters = kwargs
        # 리스크 관리 파라미터 (전략별 개별 설정)
        self.leverage = min(int(kwargs.get('leverage', 1)), 3)  # 최대 3배
        self.stop_loss_pct = float(kwargs.get('stop_loss_pct', 0.02))
        self.take_profit_pct = float(kwargs.get('take_profit_pct', 0.04))
        self.max_allocation_pct = float(kwargs.get('max_allocation_pct', 0.20))

    def generate_signals(self, df):
        """
        각 캔들에 대한 매매 시그널 생성.
        반환값:
            signals (pd.Series):
                1  → 매수(롱)
               -1  → 매도(숏)
                0  → 중립(포지션 없음)
        """
        raise NotImplementedError("전략은 generate_signals()를 구현해야 합니다")


# ─────────────────────────────────────────────
#  기존 전략 (하위 호환)
# ─────────────────────────────────────────────

class EMACrossStrategy(BaseStrategy):
    """EMA 크로스오버 — 단기/장기 EMA 교차 추세추종"""
    def __init__(self, **kwargs):
        params = {
            'fast_period': 9,
            'slow_period': 21,
            'leverage': 1,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05,
            'max_allocation_pct': 0.50,
        }
        params.update(kwargs)
        super().__init__(name="EMA 크로스오버", **params)

    def generate_signals(self, df):
        close = df['close']
        fast = close.ewm(span=self.parameters['fast_period'], adjust=False).mean()
        slow = close.ewm(span=self.parameters['slow_period'], adjust=False).mean()
        signals = pd.Series(0, index=df.index)
        signals[fast > slow] = 1
        signals[fast < slow] = -1
        warmup = max(self.parameters['fast_period'], self.parameters['slow_period'])
        signals.iloc[:warmup] = 0
        return signals


class RSIBBStrategy(BaseStrategy):
    """RSI + 볼린저 밴드 — 과매도/과매수 평균회귀"""
    def __init__(self, **kwargs):
        params = {
            'rsi_period': 14,
            'rsi_lower': 30,
            'rsi_upper': 70,
            'bb_period': 20,
            'bb_std': 2.0,
            'leverage': 1,
            'stop_loss_pct': 0.015,
            'take_profit_pct': 0.03,
            'max_allocation_pct': 0.30,
        }
        params.update(kwargs)
        super().__init__(name="RSI + 볼린저 밴드", **params)

    def generate_signals(self, df):
        close = df['close']
        rsi_p = self.parameters['rsi_period']
        rsi_lower = self.parameters['rsi_lower']
        rsi_upper = self.parameters['rsi_upper']
        bb_p = self.parameters['bb_period']
        bb_std_val = self.parameters['bb_std']

        bb_mid = close.rolling(bb_p).mean()
        bb_std = close.rolling(bb_p).std()
        bb_upper = bb_mid + bb_std_val * bb_std
        bb_lower = bb_mid - bb_std_val * bb_std

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_p).mean()
        rs = gain / np.where(loss == 0, 1e-10, loss)
        rsi = 100 - (100 / (1 + rs))

        long_cond = (close < bb_lower) & (rsi < rsi_lower)
        short_cond = (close > bb_upper) & (rsi > rsi_upper)

        signals = pd.Series(0, index=df.index)
        current = 0
        for i in range(len(df)):
            if long_cond.iloc[i]:
                current = 1
            elif short_cond.iloc[i]:
                current = -1
            elif current == 1 and not pd.isna(bb_mid.iloc[i]) and close.iloc[i] >= bb_mid.iloc[i]:
                current = 0
            elif current == -1 and not pd.isna(bb_mid.iloc[i]) and close.iloc[i] <= bb_mid.iloc[i]:
                current = 0
            signals.iloc[i] = current

        signals.iloc[:max(rsi_p, bb_p)] = 0
        return signals


class VolatilityBreakoutStrategy(BaseStrategy):
    """변동성 돌파 — ATR 기반 오늘의 시가 돌파"""
    def __init__(self, **kwargs):
        params = {
            'lookback_period': 20,
            'k': 0.7,
            'leverage': 2,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.06,
            'max_allocation_pct': 0.40,
        }
        params.update(kwargs)
        super().__init__(name="변동성 돌파", **params)

    def generate_signals(self, df):
        close = df['close']
        high = df['high']
        low = df['low']
        lb = self.parameters['lookback_period']
        k = self.parameters['k']

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(lb).mean()

        signals = pd.Series(0, index=df.index)
        for i in range(lb, len(df)):
            cur_open = df['open'].iloc[i]
            cur_atr = atr.iloc[i - 1]
            if close.iloc[i] > cur_open + k * cur_atr:
                signals.iloc[i] = 1
            elif close.iloc[i] < cur_open - k * cur_atr:
                signals.iloc[i] = -1
            else:
                signals.iloc[i] = signals.iloc[i - 1]

        signals.iloc[:lb] = 0
        return signals


class AdaptiveRegimeStrategy(BaseStrategy):
    """적응형 시장국면 — 상승/하락장: 추세추종, 횡보장: 평균회귀"""
    def __init__(self, **kwargs):
        params = {
            'ema_fast': 9,
            'ema_slow': 21,
            'rsi_period': 14,
            'rsi_lower': 30,
            'rsi_upper': 70,
            'bb_period': 20,
            'bb_std': 2.0,
            'leverage': 1,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'max_allocation_pct': 0.30,
        }
        params.update(kwargs)
        super().__init__(name="적응형 시장국면", **params)

    def generate_signals(self, df):
        df_ind = add_all_indicators(df)
        regimes = df_ind['regime']
        close = df_ind['close']

        ema_fast = close.ewm(span=self.parameters['ema_fast'], adjust=False).mean()
        ema_slow = close.ewm(span=self.parameters['ema_slow'], adjust=False).mean()
        trend_sig = pd.Series(0, index=df.index)
        trend_sig[ema_fast > ema_slow] = 1
        trend_sig[ema_fast < ema_slow] = -1

        bb_mid = close.rolling(self.parameters['bb_period']).mean()
        bb_std = close.rolling(self.parameters['bb_period']).std()
        bb_upper = bb_mid + self.parameters['bb_std'] * bb_std
        bb_lower = bb_mid - self.parameters['bb_std'] * bb_std

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.parameters['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.parameters['rsi_period']).mean()
        rsi = 100 - (100 / (1 + gain / np.where(loss == 0, 1e-10, loss)))

        mr_sig = pd.Series(0, index=df.index)
        cur_mr = 0
        for i in range(len(df)):
            long_c = (close.iloc[i] < bb_lower.iloc[i]) and (rsi.iloc[i] < self.parameters['rsi_lower'])
            short_c = (close.iloc[i] > bb_upper.iloc[i]) and (rsi.iloc[i] > self.parameters['rsi_upper'])
            if long_c:
                cur_mr = 1
            elif short_c:
                cur_mr = -1
            elif cur_mr == 1 and not pd.isna(bb_mid.iloc[i]) and close.iloc[i] >= bb_mid.iloc[i]:
                cur_mr = 0
            elif cur_mr == -1 and not pd.isna(bb_mid.iloc[i]) and close.iloc[i] <= bb_mid.iloc[i]:
                cur_mr = 0
            mr_sig.iloc[i] = cur_mr

        signals = pd.Series(0, index=df.index)
        for i in range(len(df)):
            if regimes.iloc[i] in ['BULL', 'BEAR']:
                signals.iloc[i] = trend_sig.iloc[i]
            else:
                signals.iloc[i] = mr_sig.iloc[i]

        warmup = max(self.parameters['ema_slow'], self.parameters['bb_period'], self.parameters['rsi_period'])
        signals.iloc[:warmup] = 0
        return signals

    def get_dynamic_risk(self, current_regime):
        if current_regime in ['BULL', 'BEAR']:
            return {
                'leverage': self.leverage,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct * 1.5,
                'max_allocation_pct': min(self.max_allocation_pct * 1.5, 0.8),
            }
        else:
            return {
                'leverage': 1,
                'stop_loss_pct': self.stop_loss_pct * 0.75,
                'take_profit_pct': self.take_profit_pct * 0.75,
                'max_allocation_pct': self.max_allocation_pct * 0.5,
            }


# ─────────────────────────────────────────────
#  신규 전략 9개
# ─────────────────────────────────────────────

class MACDStrategy(BaseStrategy):
    """MACD 히스토그램 — MACD선이 시그널선 위/아래로 교차할 때 진입"""
    def __init__(self, **kwargs):
        params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'leverage': 1,
            'stop_loss_pct': 0.025,
            'take_profit_pct': 0.06,
            'max_allocation_pct': 0.40,
        }
        params.update(kwargs)
        super().__init__(name="MACD 히스토그램", **params)

    def generate_signals(self, df):
        close = df['close']
        fast = close.ewm(span=self.parameters['fast_period'], adjust=False).mean()
        slow = close.ewm(span=self.parameters['slow_period'], adjust=False).mean()
        macd = fast - slow
        signal_line = macd.ewm(span=self.parameters['signal_period'], adjust=False).mean()

        signals = pd.Series(0, index=df.index)
        signals[macd > signal_line] = 1
        signals[macd < signal_line] = -1

        warmup = self.parameters['slow_period'] + self.parameters['signal_period']
        signals.iloc[:warmup] = 0
        return signals


class StochRSIStrategy(BaseStrategy):
    """스토캐스틱 RSI — RSI에 스토캐스틱을 적용한 과매수/과매도"""
    def __init__(self, **kwargs):
        params = {
            'rsi_period': 14,
            'stoch_period': 14,
            'k_smooth': 3,
            'oversold': 20,
            'overbought': 80,
            'leverage': 1,
            'stop_loss_pct': 0.015,
            'take_profit_pct': 0.04,
            'max_allocation_pct': 0.30,
        }
        params.update(kwargs)
        super().__init__(name="스토캐스틱 RSI", **params)

    def generate_signals(self, df):
        close = df['close']
        rsi_p = self.parameters['rsi_period']
        stoch_p = self.parameters['stoch_period']
        k_sm = self.parameters['k_smooth']
        oversold = self.parameters['oversold']
        overbought = self.parameters['overbought']

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_p).mean()
        rsi = 100 - (100 / (1 + gain / np.where(loss == 0, 1e-10, loss)))

        rsi_min = rsi.rolling(stoch_p).min()
        rsi_max = rsi.rolling(stoch_p).max()
        rng = rsi_max - rsi_min
        stoch_rsi = 100 * (rsi - rsi_min) / np.where(rng == 0, 1e-10, rng)
        k = stoch_rsi.rolling(k_sm).mean()

        signals = pd.Series(0, index=df.index)
        current = 0
        for i in range(len(df)):
            if pd.isna(k.iloc[i]):
                signals.iloc[i] = 0
                continue
            if k.iloc[i] < oversold:
                current = 1
            elif k.iloc[i] > overbought:
                current = -1
            elif current == 1 and k.iloc[i] > 50:
                current = 0
            elif current == -1 and k.iloc[i] < 50:
                current = 0
            signals.iloc[i] = current

        signals.iloc[:rsi_p + stoch_p + k_sm] = 0
        return signals


class TripleEMAStrategy(BaseStrategy):
    """삼중 EMA — 빠른/중간/느린 EMA 세 선이 정렬될 때 진입"""
    def __init__(self, **kwargs):
        params = {
            'fast_period': 8,
            'mid_period': 21,
            'slow_period': 55,
            'leverage': 1,
            'stop_loss_pct': 0.025,
            'take_profit_pct': 0.07,
            'max_allocation_pct': 0.40,
        }
        params.update(kwargs)
        super().__init__(name="삼중 EMA", **params)

    def generate_signals(self, df):
        close = df['close']
        fast = close.ewm(span=self.parameters['fast_period'], adjust=False).mean()
        mid = close.ewm(span=self.parameters['mid_period'], adjust=False).mean()
        slow = close.ewm(span=self.parameters['slow_period'], adjust=False).mean()

        signals = pd.Series(0, index=df.index)
        signals[(fast > mid) & (mid > slow)] = 1   # 강세 정렬
        signals[(fast < mid) & (mid < slow)] = -1  # 약세 정렬

        signals.iloc[:self.parameters['slow_period']] = 0
        return signals


class DonchianChannelStrategy(BaseStrategy):
    """도니안 채널 돌파 — N기간 최고/최저 채널 돌파 시 진입"""
    def __init__(self, **kwargs):
        params = {
            'channel_period': 20,
            'exit_period': 10,
            'leverage': 2,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.08,
            'max_allocation_pct': 0.40,
        }
        params.update(kwargs)
        super().__init__(name="도니안 채널 돌파", **params)

    def generate_signals(self, df):
        close = df['close']
        high = df['high']
        low = df['low']
        period = self.parameters['channel_period']
        exit_p = self.parameters['exit_period']

        upper = high.rolling(period).max().shift(1)
        lower = low.rolling(period).min().shift(1)
        exit_lower = low.rolling(exit_p).min().shift(1)
        exit_upper = high.rolling(exit_p).max().shift(1)

        signals = pd.Series(0, index=df.index)
        current = 0
        for i in range(len(df)):
            if pd.isna(upper.iloc[i]) or pd.isna(lower.iloc[i]):
                signals.iloc[i] = 0
                continue
            if close.iloc[i] > upper.iloc[i]:
                current = 1
            elif close.iloc[i] < lower.iloc[i]:
                current = -1
            elif current == 1 and not pd.isna(exit_lower.iloc[i]) and close.iloc[i] < exit_lower.iloc[i]:
                current = 0
            elif current == -1 and not pd.isna(exit_upper.iloc[i]) and close.iloc[i] > exit_upper.iloc[i]:
                current = 0
            signals.iloc[i] = current

        signals.iloc[:period] = 0
        return signals


class MFIStrategy(BaseStrategy):
    """머니플로우 지수 (MFI) — 거래량 가중 RSI 기반 과매수/과매도"""
    def __init__(self, **kwargs):
        params = {
            'mfi_period': 14,
            'oversold': 20,
            'overbought': 80,
            'leverage': 1,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05,
            'max_allocation_pct': 0.30,
        }
        params.update(kwargs)
        super().__init__(name="머니플로우 지수 (MFI)", **params)

    def generate_signals(self, df):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        period = self.parameters['mfi_period']

        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        tp_shift = typical_price.shift(1)

        pos_flow = money_flow.where(typical_price > tp_shift, 0).rolling(period).sum()
        neg_flow = money_flow.where(typical_price < tp_shift, 0).rolling(period).sum()
        mfi = 100 - (100 / (1 + pos_flow / np.where(neg_flow == 0, 1e-10, neg_flow)))

        oversold = self.parameters['oversold']
        overbought = self.parameters['overbought']
        signals = pd.Series(0, index=df.index)
        current = 0
        for i in range(len(df)):
            if pd.isna(mfi.iloc[i]):
                signals.iloc[i] = 0
                continue
            if mfi.iloc[i] < oversold:
                current = 1
            elif mfi.iloc[i] > overbought:
                current = -1
            elif current == 1 and mfi.iloc[i] > 50:
                current = 0
            elif current == -1 and mfi.iloc[i] < 50:
                current = 0
            signals.iloc[i] = current

        signals.iloc[:period] = 0
        return signals


class WilliamsRStrategy(BaseStrategy):
    """윌리엄스 %R — 0~-100 범위 오실레이터 과매수/과매도"""
    def __init__(self, **kwargs):
        params = {
            'period': 14,
            'oversold': -80,   # 이하면 과매도(매수)
            'overbought': -20, # 이상이면 과매수(매도)
            'leverage': 1,
            'stop_loss_pct': 0.015,
            'take_profit_pct': 0.04,
            'max_allocation_pct': 0.25,
        }
        params.update(kwargs)
        super().__init__(name="윌리엄스 %R", **params)

    def generate_signals(self, df):
        high = df['high']
        low = df['low']
        close = df['close']
        period = self.parameters['period']

        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        wr = -100 * (hh - close) / np.where((hh - ll) == 0, 1e-10, (hh - ll))

        oversold = self.parameters['oversold']
        overbought = self.parameters['overbought']
        signals = pd.Series(0, index=df.index)
        current = 0
        for i in range(len(df)):
            if pd.isna(wr.iloc[i]):
                signals.iloc[i] = 0
                continue
            if wr.iloc[i] < oversold:
                current = 1
            elif wr.iloc[i] > overbought:
                current = -1
            elif current == 1 and wr.iloc[i] > -50:
                current = 0
            elif current == -1 and wr.iloc[i] < -50:
                current = 0
            signals.iloc[i] = current

        signals.iloc[:period] = 0
        return signals


class IchimokuStrategy(BaseStrategy):
    """이치모쿠 구름 — 구름 위/아래 위치와 전환선/기준선 교차 확인"""
    def __init__(self, **kwargs):
        params = {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_b_period': 52,
            'leverage': 1,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.08,
            'max_allocation_pct': 0.35,
        }
        params.update(kwargs)
        super().__init__(name="이치모쿠 구름", **params)

    def generate_signals(self, df):
        high = df['high']
        low = df['low']
        close = df['close']
        tp = self.parameters['tenkan_period']
        kp = self.parameters['kijun_period']
        sbp = self.parameters['senkou_b_period']

        tenkan = (high.rolling(tp).max() + low.rolling(tp).min()) / 2
        kijun = (high.rolling(kp).max() + low.rolling(kp).min()) / 2
        senkou_a = (tenkan + kijun) / 2
        senkou_b = (high.rolling(sbp).max() + low.rolling(sbp).min()) / 2

        signals = pd.Series(0, index=df.index)
        for i in range(len(df)):
            if any(pd.isna(v) for v in [tenkan.iloc[i], kijun.iloc[i],
                                         senkou_a.iloc[i], senkou_b.iloc[i]]):
                signals.iloc[i] = 0
                continue
            cloud_top = max(senkou_a.iloc[i], senkou_b.iloc[i])
            cloud_bot = min(senkou_a.iloc[i], senkou_b.iloc[i])
            cur = close.iloc[i]
            if cur > cloud_top and tenkan.iloc[i] > kijun.iloc[i]:
                signals.iloc[i] = 1
            elif cur < cloud_bot and tenkan.iloc[i] < kijun.iloc[i]:
                signals.iloc[i] = -1
            else:
                signals.iloc[i] = 0

        signals.iloc[:sbp] = 0
        return signals


class DualMomentumStrategy(BaseStrategy):
    """듀얼 모멘텀 — 절대 모멘텀(과거 대비 상승) + 장기 추세 필터 복합"""
    def __init__(self, **kwargs):
        params = {
            'lookback_period': 30,
            'trend_period': 100,
            'leverage': 1,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.10,
            'max_allocation_pct': 0.50,
        }
        params.update(kwargs)
        super().__init__(name="듀얼 모멘텀", **params)

    def generate_signals(self, df):
        close = df['close']
        lb = self.parameters['lookback_period']
        tp = self.parameters['trend_period']

        momentum = close / close.shift(lb) - 1
        trend_sma = close.rolling(tp).mean()

        signals = pd.Series(0, index=df.index)
        for i in range(len(df)):
            if pd.isna(momentum.iloc[i]) or pd.isna(trend_sma.iloc[i]):
                signals.iloc[i] = 0
                continue
            if momentum.iloc[i] > 0 and close.iloc[i] > trend_sma.iloc[i]:
                signals.iloc[i] = 1
            elif momentum.iloc[i] < 0 and close.iloc[i] < trend_sma.iloc[i]:
                signals.iloc[i] = -1
            else:
                signals.iloc[i] = 0

        signals.iloc[:max(lb, tp)] = 0
        return signals


class ZScoreMeanReversionStrategy(BaseStrategy):
    """Z-Score 평균회귀 — 가격이 평균에서 표준편차 N배 벗어나면 진입"""
    def __init__(self, **kwargs):
        params = {
            'period': 20,
            'z_threshold': 2.0,
            'leverage': 1,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05,
            'max_allocation_pct': 0.30,
        }
        params.update(kwargs)
        super().__init__(name="Z-Score 평균회귀", **params)

    def generate_signals(self, df):
        close = df['close']
        period = self.parameters['period']
        z_thr = self.parameters['z_threshold']

        mean = close.rolling(period).mean()
        std = close.rolling(period).std()
        z_score = (close - mean) / np.where(std == 0, 1e-10, std)

        signals = pd.Series(0, index=df.index)
        current = 0
        for i in range(len(df)):
            if pd.isna(z_score.iloc[i]):
                signals.iloc[i] = 0
                continue
            if z_score.iloc[i] < -z_thr:
                current = 1
            elif z_score.iloc[i] > z_thr:
                current = -1
            elif current == 1 and z_score.iloc[i] > 0:
                current = 0
            elif current == -1 and z_score.iloc[i] < 0:
                current = 0
            signals.iloc[i] = current

        signals.iloc[:period] = 0
        return signals


class HeikinAshiTrendStrategy(BaseStrategy):
    """하이킨아시 추세추종 — 하이킨아시 캔들 색상 연속 및 EMA 필터링 기반 추세 추종 전략"""
    def __init__(self, **kwargs):
        params = {
            'ha_ema_period': 20,
            'consecutive_candles': 3,
            'leverage': 1,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.06,
            'max_allocation_pct': 0.30,
        }
        params.update(kwargs)
        super().__init__(name="하이킨아시 추세추종", **params)

    def generate_signals(self, df):
        if 'ha_open' not in df.columns:
            from indicators import calculate_heikin_ashi
            ha_df = calculate_heikin_ashi(df)
            df = df.copy()
            df['ha_open'] = ha_df['ha_open']
            df['ha_high'] = ha_df['ha_high']
            df['ha_low'] = ha_df['ha_low']
            df['ha_close'] = ha_df['ha_close']
            
        ha_open = df['ha_open']
        ha_close = df['ha_close']
        
        ema_p = self.parameters['ha_ema_period']
        ha_close_ema = ha_close.ewm(span=ema_p, adjust=False).mean()
        
        consecutive = self.parameters['consecutive_candles']
        
        is_bull_candle = ha_close > ha_open
        is_bear_candle = ha_close < ha_open
        
        bull_streak = is_bull_candle.rolling(consecutive).sum() == consecutive
        bear_streak = is_bear_candle.rolling(consecutive).sum() == consecutive
        
        signals = pd.Series(0, index=df.index)
        current = 0
        
        for i in range(len(df)):
            if pd.isna(ha_close_ema.iloc[i]) or i < consecutive:
                signals.iloc[i] = 0
                continue
                
            cur_ha_close = ha_close.iloc[i]
            cur_ema = ha_close_ema.iloc[i]
            
            if bull_streak.iloc[i] and cur_ha_close > cur_ema:
                current = 1
            elif bear_streak.iloc[i] and cur_ha_close < cur_ema:
                current = -1
            elif current == 1 and (is_bear_candle.iloc[i] or cur_ha_close < cur_ema):
                current = 0
            elif current == -1 and (is_bull_candle.iloc[i] or cur_ha_close > cur_ema):
                current = 0
                
            signals.iloc[i] = current
            
        signals.iloc[:max(ema_p, consecutive)] = 0
        return signals


# ─────────────────────────────────────────────
#  전략 레지스트리
# ─────────────────────────────────────────────

# 전략 한국어 이름 목록 (전체 스윕 시 사용)
ALL_STRATEGY_NAMES = [
    "EMA 크로스오버",
    "RSI + 볼린저 밴드",
    "변동성 돌파",
    "적응형 시장국면",
    "MACD 히스토그램",
    "스토캐스틱 RSI",
    "삼중 EMA",
    "도니안 채널 돌파",
    "머니플로우 지수 (MFI)",
    "윌리엄스 %R",
    "이치모쿠 구름",
    "듀얼 모멘텀",
    "Z-Score 평균회귀",
    "하이킨아시 추세추종",
]

STRATEGY_REGISTRY = {
    # 한국어 이름
    "EMA 크로스오버": EMACrossStrategy,
    "RSI + 볼린저 밴드": RSIBBStrategy,
    "변동성 돌파": VolatilityBreakoutStrategy,
    "적응형 시장국면": AdaptiveRegimeStrategy,
    "MACD 히스토그램": MACDStrategy,
    "스토캐스틱 RSI": StochRSIStrategy,
    "삼중 EMA": TripleEMAStrategy,
    "도니안 채널 돌파": DonchianChannelStrategy,
    "머니플로우 지수 (MFI)": MFIStrategy,
    "윌리엄스 %R": WilliamsRStrategy,
    "이치모쿠 구름": IchimokuStrategy,
    "듀얼 모멘텀": DualMomentumStrategy,
    "Z-Score 평균회귀": ZScoreMeanReversionStrategy,
    "하이킨아시 추세추종": HeikinAshiTrendStrategy,
    # 영어 이름 (이전 버전 호환)
    "EMA Crossover": EMACrossStrategy,
    "RSI + Bollinger Bands": RSIBBStrategy,
    "Volatility Breakout": VolatilityBreakoutStrategy,
    "Adaptive Regime Strategy": AdaptiveRegimeStrategy,
    # 클래스명 직접 매핑 (국면 봇 등에서 사용)
    "EMACrossStrategy": EMACrossStrategy,
    "RSIBBStrategy": RSIBBStrategy,
    "VolatilityBreakoutStrategy": VolatilityBreakoutStrategy,
    "AdaptiveRegimeStrategy": AdaptiveRegimeStrategy,
    "MACDStrategy": MACDStrategy,
    "StochRSIStrategy": StochRSIStrategy,
    "TripleEMAStrategy": TripleEMAStrategy,
    "DonchianChannelStrategy": DonchianChannelStrategy,
    "MFIStrategy": MFIStrategy,
    "WilliamsRStrategy": WilliamsRStrategy,
    "IchimokuStrategy": IchimokuStrategy,
    "DualMomentumStrategy": DualMomentumStrategy,
    "ZScoreMeanReversionStrategy": ZScoreMeanReversionStrategy,
    "HeikinAshiTrendStrategy": HeikinAshiTrendStrategy,
}


def get_strategy_by_name(name: str, **kwargs):
    """전략 이름으로 전략 인스턴스를 반환하는 팩토리 함수"""
    if name in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[name](**kwargs)
    # 소문자 부분 매칭 (관대한 매칭)
    name_lower = name.lower()
    for key, cls in STRATEGY_REGISTRY.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return cls(**kwargs)
    raise ValueError(
        f"알 수 없는 전략: '{name}'\n"
        f"사용 가능한 전략: {ALL_STRATEGY_NAMES}"
    )
