import numpy as np
import pandas as pd
import optuna
from data_manager import DataManager
from backtester import Backtester
from strategies import get_strategy_by_name

# Optuna 로그 최소화
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_params(trial: optuna.Trial, strategy_name: str, is_futures: bool) -> dict:
    """전략 이름에 따라 Optuna 파라미터 탐색 공간을 구성합니다."""
    params = {}
    n = strategy_name.lower()

    # ── 공통 리스크 파라미터 ──
    params['stop_loss_pct'] = trial.suggest_float('stop_loss_pct', 0.005, 0.05, step=0.001)
    params['take_profit_pct'] = trial.suggest_float('take_profit_pct', 0.01, 0.15, step=0.005)
    params['max_allocation_pct'] = trial.suggest_float('max_allocation_pct', 0.10, 0.50, step=0.05)
    if is_futures:
        params['leverage'] = trial.suggest_int('leverage', 1, 3)
    else:
        params['leverage'] = 1

    # ── 전략별 파라미터 ──
    if 'ema 크로스오버' in n or 'ema crossover' in n:
        params['fast_period'] = trial.suggest_int('fast_period', 5, 50)
        params['slow_period'] = trial.suggest_int('slow_period', params['fast_period'] + 5, 150)

    elif 'rsi + 볼린저' in n or 'rsi + bollinger' in n:
        params['rsi_period'] = trial.suggest_int('rsi_period', 7, 25)
        params['rsi_lower'] = trial.suggest_int('rsi_lower', 15, 40)
        params['rsi_upper'] = trial.suggest_int('rsi_upper', 60, 85)
        params['bb_period'] = trial.suggest_int('bb_period', 10, 40)
        params['bb_std'] = trial.suggest_float('bb_std', 1.5, 3.0, step=0.1)

    elif '변동성 돌파' in n or 'volatility breakout' in n:
        params['lookback_period'] = trial.suggest_int('lookback_period', 10, 40)
        params['k'] = trial.suggest_float('k', 0.2, 1.5, step=0.05)

    elif '적응형' in n or 'adaptive' in n:
        params['ema_fast'] = trial.suggest_int('ema_fast', 5, 30)
        params['ema_slow'] = trial.suggest_int('ema_slow', params['ema_fast'] + 5, 100)
        params['rsi_period'] = trial.suggest_int('rsi_period', 7, 25)
        params['rsi_lower'] = trial.suggest_int('rsi_lower', 15, 40)
        params['rsi_upper'] = trial.suggest_int('rsi_upper', 60, 85)
        params['bb_period'] = trial.suggest_int('bb_period', 10, 40)
        params['bb_std'] = trial.suggest_float('bb_std', 1.5, 3.0, step=0.1)

    elif 'macd' in n:
        params['fast_period'] = trial.suggest_int('fast_period', 5, 20)
        params['slow_period'] = trial.suggest_int('slow_period', params['fast_period'] + 5, 40)
        params['signal_period'] = trial.suggest_int('signal_period', 3, 15)

    elif '스토캐스틱' in n or 'stoch' in n:
        params['rsi_period'] = trial.suggest_int('rsi_period', 7, 21)
        params['stoch_period'] = trial.suggest_int('stoch_period', 7, 21)
        params['k_smooth'] = trial.suggest_int('k_smooth', 2, 5)
        params['oversold'] = trial.suggest_int('oversold', 10, 30)
        params['overbought'] = trial.suggest_int('overbought', 70, 90)

    elif '삼중 ema' in n or 'triple ema' in n:
        params['fast_period'] = trial.suggest_int('fast_period', 5, 20)
        params['mid_period'] = trial.suggest_int('mid_period', params['fast_period'] + 5, 50)
        params['slow_period'] = trial.suggest_int('slow_period', params['mid_period'] + 5, 120)

    elif '도니안' in n or 'donchian' in n:
        # exit_period < channel_period 보장: channel 최소 15, exit 최대 14
        params['channel_period'] = trial.suggest_int('channel_period', 15, 55)
        params['exit_period'] = trial.suggest_int('exit_period', 5, 14)

    elif '머니플로우' in n or 'mfi' in n:
        params['mfi_period'] = trial.suggest_int('mfi_period', 7, 25)
        params['oversold'] = trial.suggest_int('oversold', 10, 30)
        params['overbought'] = trial.suggest_int('overbought', 70, 90)

    elif '윌리엄스' in n or 'williams' in n:
        params['period'] = trial.suggest_int('period', 7, 25)
        # Optuna는 양수를 제안하고 음수로 변환
        params['oversold'] = -trial.suggest_int('oversold_abs', 70, 90)   # -90 ~ -70
        params['overbought'] = -trial.suggest_int('overbought_abs', 10, 30) # -30 ~ -10

    elif '이치모쿠' in n or 'ichimoku' in n:
        params['tenkan_period'] = trial.suggest_int('tenkan_period', 5, 15)
        params['kijun_period'] = trial.suggest_int('kijun_period', 15, 40)
        params['senkou_b_period'] = trial.suggest_int('senkou_b_period', 40, 70)

    elif '듀얼 모멘텀' in n or 'dual momentum' in n:
        params['lookback_period'] = trial.suggest_int('lookback_period', 10, 60)
        params['trend_period'] = trial.suggest_int('trend_period', 50, 150)

    elif 'z-score' in n or '평균회귀' in n:
        params['period'] = trial.suggest_int('period', 10, 40)
        params['z_threshold'] = trial.suggest_float('z_threshold', 1.0, 3.0, step=0.1)

    elif '하이킨아시' in n or 'heikin' in n:
        params['ha_ema_period'] = trial.suggest_int('ha_ema_period', 10, 50)
        params['consecutive_candles'] = trial.suggest_int('consecutive_candles', 2, 5)

    elif '동적결합' in n or 'switching' in n:
        params['regime_confirm_candles'] = trial.suggest_int('regime_confirm_candles', 1, 5)
        # BULL (Dual Momentum) params
        params['bull_lookback'] = trial.suggest_int('bull_lookback', 10, 60)
        params['bull_trend'] = trial.suggest_int('bull_trend', 50, 150)
        # BEAR (Triple EMA) params
        params['bear_fast'] = trial.suggest_int('bear_fast', 5, 20)
        params['bear_mid'] = trial.suggest_int('bear_mid', params['bear_fast'] + 5, 50)
        params['bear_slow'] = trial.suggest_int('bear_slow', params['bear_mid'] + 5, 120)
        # SIDEWAYS (Z-Score) params
        params['side_period'] = trial.suggest_int('side_period', 10, 40)
        params['side_z_threshold'] = trial.suggest_float('side_z_threshold', 1.0, 3.0, step=0.1)

    return params


class StrategyOptimizer:
    def __init__(self, db_path="trading_data.db"):
        self.data_manager = DataManager(db_path=db_path)
        self.backtester = Backtester()

    def optimize(self, symbol: str, timeframe: str, start_date_str: str, end_date_str: str,
                 strategy_name: str, is_futures: bool = False,
                 target_metric: str = "sharpe_ratio", n_trials: int = 20):
        """
        Optuna를 사용해 주어진 전략과 기간에 최적 파라미터를 탐색합니다.

        target_metric: "sharpe_ratio" | "sortino_ratio" | "total_return" |
                       "profit_factor" | "win_rate"
        """
        df_candles = self.data_manager.get_candles(
            symbol, timeframe, start_date_str, end_date_str, is_futures
        )
        if df_candles is None or df_candles.empty:
            raise ValueError(f"데이터 없음: {symbol} ({timeframe}) {start_date_str}~{end_date_str}")

        def objective(trial: optuna.Trial) -> float:
            try:
                params = _build_params(trial, strategy_name, is_futures)
                strategy = get_strategy_by_name(strategy_name, **params)
                metrics, _, _ = self.backtester.run(df_candles, strategy, is_futures=is_futures)

                # 최대낙폭 25% 초과 시 페널티
                if abs(metrics['max_drawdown']) > 0.25:
                    return -999.0

                val = metrics.get(target_metric, 0.0)
                if pd.isna(val) or np.isinf(val):
                    return -999.0
                return float(val)

            except Exception:
                return -999.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return study.best_params, study.best_value, study
