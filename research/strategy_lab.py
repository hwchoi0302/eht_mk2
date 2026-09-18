"""
research/strategy_lab.py — 전략 재탐색 도구.

계획 문서 5.1의 원칙을 코드로 옮긴 것이다.

  * 워크포워드(WFA): 인샘플 6개월에서 파라미터를 고르고, 뒤따르는 아웃샘플
    2개월 성과만 집계한다. 전 구간 최적화 후 전 구간 성과를 보는 방식은
    과최적화를 성과로 착각하게 만든다.
  * 파라미터 고원: 단일 최고점이 아니라 이웃 파라미터에서도 성과가 유지되는
    영역을 고른다.
  * 비용 스트레스: 수수료·슬리피지 2배에서도 살아남는 설정만 후보로 둔다.
  * 회전율 패널티: 비슷한 성과면 회전율이 낮은 쪽을 고른다.

평가 지표는 총수익률이 아니라 아웃샘플 샤프 · MDD · PF · 구간 일관성이다.
"""

import itertools
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from core.paths import DB_PATH
from core.indicators import add_all_indicators
from core.strategies import get_strategy_by_name
from research.backtester import Backtester
from research.binance_env import build_funding_lookup


# ─────────────────────────────────────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────────────────────────────────────

def load_candles(symbol='BTC/USDT', timeframe='4h', is_futures=True,
                 start=None, end=None, with_indicators=True):
    """DB에서 캔들을 읽어 지표까지 붙여 돌려준다.

    지표 워밍업 때문에 start 이전 데이터도 함께 읽은 뒤 잘라내야 하지만,
    여기서는 전체를 읽고 지표를 계산한 다음 슬라이스한다. 그래야 EMA200 같은
    장기 지표 값이 구간 경계에서 튀지 않는다.
    """
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume FROM candles "
        "WHERE symbol=? AND timeframe=? AND is_futures=? ORDER BY timestamp",
        conn, params=(symbol, timeframe, 1 if is_futures else 0),
    )
    conn.close()

    if df.empty:
        raise RuntimeError(f"{symbol} {timeframe} (futures={is_futures}) 캔들이 DB에 없습니다.")

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    if with_indicators:
        df = add_all_indicators(df)

    # 지표 계산 후에 자른다 — 워밍업을 구간 밖 데이터로 채우기 위해서다
    if start:
        df = df[df['datetime'] >= pd.Timestamp(start)]
    if end:
        df = df[df['datetime'] < pd.Timestamp(end)]

    return df.reset_index(drop=True)


def slice_period(df, start, end):
    """지표가 이미 붙은 df를 기간으로 자른다 (재계산 없음)."""
    mask = (df['datetime'] >= pd.Timestamp(start)) & (df['datetime'] < pd.Timestamp(end))
    return df[mask].reset_index(drop=True)


# 전략이 신호를 내기 시작하기까지 필요한 봉 수의 상한.
# 가장 긴 것이 장기 추세 필터의 EMA200과 삼중 EMA의 slow_period(=120)다.
WARMUP_BARS = 260


def slice_with_warmup(df, start, end, warmup_bars=WARMUP_BARS):
    """평가 구간 앞에 워밍업 봉을 붙여 자른다.

    전략들은 `signals.iloc[:warmup] = 0`으로 초기 구간의 신호를 죽인다.
    4h 봉에서는 6개월이 약 1,100봉이라 워밍업 96봉이 묻혔지만, 1d 봉에서는
    6개월이 약 180봉뿐이라 워밍업이 창 전체를 잡아먹어 **거래가 한 건도
    나오지 않았다**. 평가 구간 앞에 워밍업을 따로 붙여야 한다.

    Returns:
        (확장된 df, 평가 시작 위치 index)
    """
    after = df.index[df['datetime'] >= pd.Timestamp(start)]
    start_i = int(after[0]) if len(after) else 0
    before_end = df.index[df['datetime'] < pd.Timestamp(end)]
    end_i = int(before_end[-1]) if len(before_end) else len(df) - 1

    lead_i = max(0, start_i - warmup_bars)
    window = df.iloc[lead_i:end_i + 1].reset_index(drop=True)
    return window, start_i - lead_i


def restrict_metrics(metrics, equity, trades, eval_start_i, initial_capital=10000.0):
    """워밍업 구간을 제외하고 평가 구간만의 성과를 다시 계산한다."""
    eq = equity.iloc[eval_start_i:].reset_index(drop=True)
    if len(eq) < 2:
        return None

    base = eq['equity'].iloc[0]
    if base <= 0:
        return None

    total_return = eq['equity'].iloc[-1] / base - 1
    roll = eq['equity'].cummax()
    mdd = float(((eq['equity'] - roll) / roll).min())

    returns = eq['equity'].pct_change().fillna(0.0)
    bar_hours = (eq['datetime'].iloc[1] - eq['datetime'].iloc[0]).total_seconds() / 3600.0
    bars_per_year = (365.25 * 24.0) / max(bar_hours, 0.001)
    std = returns.std()
    sharpe = float((returns.mean() / std) * np.sqrt(bars_per_year)) if std > 0 else 0.0

    start_ms = int(eq['timestamp'].iloc[0])
    kept = [t for t in trades if t.get('entry_time', 0) >= start_ms]
    wins = [t for t in kept if t.get('pnl', 0) > 0]
    gp = sum(t['pnl'] for t in wins)
    gl = abs(sum(t['pnl'] for t in kept if t.get('pnl', 0) <= 0))

    days = max((eq['datetime'].iloc[-1] - eq['datetime'].iloc[0]).total_seconds() / 86400, 1.0)
    volume = sum(t['size'] * t['entry_price'] * 2 for t in kept)

    out = dict(metrics)
    out.update({
        'total_return': float(total_return),
        'max_drawdown': mdd,
        'sharpe_ratio': sharpe,
        'total_trades': len(kept),
        'win_rate': len(wins) / len(kept) if kept else 0.0,
        'profit_factor': (gp / gl) if gl > 0 else (gp if gp > 0 else 0.0),
        'annual_turnover': (volume / initial_capital) * (365.25 / days),
    })
    return out


def evaluate_window(df, strategy_name, params, start, end, symbol='BTC/USDT',
                    cost_multiplier=1.0, initial_capital=10000.0,
                    regime_confirm_candles=2, is_futures=True):
    """워밍업을 앞에 붙여 돌린 뒤, 평가 구간만의 성과를 돌려준다."""
    window, eval_i = slice_with_warmup(df, start, end)
    if len(window) - eval_i < 10:
        return None
    metrics, equity, trades = run_backtest(
        window, strategy_name, params, symbol=symbol,
        cost_multiplier=cost_multiplier, initial_capital=initial_capital,
        regime_confirm_candles=regime_confirm_candles, is_futures=is_futures)

    restricted = restrict_metrics(metrics, equity, trades, eval_i, initial_capital)
    if restricted is None:
        return None
    px = window['close'].iloc[eval_i:]
    restricted['buy_hold_return'] = float(px.iloc[-1] / px.iloc[0] - 1)
    return restricted


# ─────────────────────────────────────────────────────────────────────────────
# 단일 백테스트
# ─────────────────────────────────────────────────────────────────────────────

_FUNDING_CACHE = {}


def get_funding(symbol='BTC/USDT'):
    if symbol not in _FUNDING_CACHE:
        _FUNDING_CACHE[symbol] = build_funding_lookup(symbol)
    return _FUNDING_CACHE[symbol]


def run_backtest(df, strategy_name, params=None, symbol='BTC/USDT',
                 is_futures=True, cost_multiplier=1.0, initial_capital=10000.0,
                 apply_funding=True, regime_confirm_candles=2):
    """전략 하나를 df 위에서 돌린다."""
    params = params or {}
    strategy = get_strategy_by_name(strategy_name, **params)
    bt = Backtester(
        initial_capital=initial_capital,
        symbol=symbol,
        cost_multiplier=cost_multiplier,
        funding_lookup=get_funding(symbol) if apply_funding else None,
        apply_funding=apply_funding,
        regime_confirm_candles=regime_confirm_candles,
    )
    metrics, equity, trades = bt.run(df, strategy, is_futures=is_futures)
    return metrics, equity, trades


# ─────────────────────────────────────────────────────────────────────────────
# 파라미터 그리드
# ─────────────────────────────────────────────────────────────────────────────
# 전략별로 의미 있는 범위를 좁게 잡는다. 넓은 그리드는 고원이 아니라 잡음에서
# 최고점을 찾아낼 뿐이다.

# 손절/익절 폭도 탐색 대상이다. 예전 그리드는 3~5% 손절 / 4~8% 익절로만 좁게
# 잡혀 있었는데, 그러면 추세추종 계열이 추세를 타기도 전에 손절로 털린다.
# 저회전 전략에 공정한 기회를 주려면 넓은 손절도 후보에 있어야 한다.
RISK_GRID = {
    'stop_loss_pct': [0.03, 0.05, 0.08],
    'take_profit_pct': [0.05, 0.10, 0.20],
}

# 전략 성격상 기본 그리드가 맞지 않는 경우의 개별 지정.
STRATEGY_RISK_GRIDS = {
    # 장기 추세를 타는 것이 목적이므로 손절은 넓고 익절은 사실상 두지 않는다.
    # 방향 전환은 EMA 교차(신호)로 하지 익절로 하지 않는다.
    '장기 추세 필터': {
        'stop_loss_pct': [0.10, 0.15, 0.25],
        'take_profit_pct': [0.50, 1.00],
    },
}

STRATEGY_GRIDS = {
    'EMA 크로스오버': {'fast_period': [9, 12, 20], 'slow_period': [21, 50, 60]},
    'RSI + 볼린저 밴드': {'rsi_period': [14, 21], 'bb_period': [20, 30]},
    '변동성 돌파': {'k': [0.3, 0.5, 0.7]},
    'MACD 히스토그램': {'fast_period': [12, 19], 'slow_period': [26, 39]},
    '스토캐스틱 RSI': {'rsi_period': [14, 21], 'stoch_period': [14, 21]},
    '삼중 EMA': {'fast_period': [10, 20], 'mid_period': [30, 49], 'slow_period': [96, 120]},
    '도니안 채널 돌파': {'period': [20, 40, 55]},
    '머니플로우 지수 (MFI)': {'period': [14, 21]},
    '윌리엄스 %R': {'period': [14, 21]},
    '이치모쿠 구름': {},
    '듀얼 모멘텀': {'lookback_period': [30, 46, 60], 'trend_period': [94, 120]},
    'Z-Score 평균회귀': {'period': [20, 30], 'z_threshold': [1.5, 2.0, 2.5]},
    '하이킨아시 추세추종': {},
    '적응형 시장국면': {},
    '장기 추세 필터': {'ema_period': [100, 150, 200], 'allow_short': [False, True]},
    # 국면별 하위 전략의 기간까지 탐색한다. 축을 늘리면 조합이 곱으로 커지므로
    # 각 하위 전략에서 성과에 가장 민감한 축 하나씩만 연다.
    '시장국면 동적결합': {
        'bull_lookback': [30, 46, 60],
        'bear_mid': [30, 49],
        'side_z_threshold': [1.5, 2.0, 2.5],
    },
}

# 최대 거래 레버리지. 계획 문서 기준 라이브는 1~2배다.
DEFAULT_RISK = {'leverage': 2, 'max_allocation_pct': 0.35}


def expand_grid(grid, include_risk=True, strategy_name=None):
    """dict of lists -> list of dicts. 빈 그리드면 기본값 하나."""
    combined = dict(grid)
    if include_risk:
        combined.update(STRATEGY_RISK_GRIDS.get(strategy_name, RISK_GRID))
    if not combined:
        return [dict(DEFAULT_RISK)]
    keys = list(combined)
    out = []
    for values in itertools.product(*(combined[k] for k in keys)):
        params = dict(DEFAULT_RISK)
        params.update(dict(zip(keys, values)))
        out.append(params)
    return out


def valid_params(strategy_name, params):
    """말이 안 되는 조합을 걸러낸다 (빠른 기간이 느린 기간보다 크다든지)."""
    if 'fast_period' in params and 'slow_period' in params:
        if params['fast_period'] >= params['slow_period']:
            return False
    if 'mid_period' in params:
        if not (params.get('fast_period', 0) < params['mid_period'] < params.get('slow_period', 10**9)):
            return False
    if 'lookback_period' in params and 'trend_period' in params:
        if params['lookback_period'] >= params['trend_period']:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 평가 점수
# ─────────────────────────────────────────────────────────────────────────────

def score(metrics, min_trades=5):
    """파라미터 선택용 단일 점수.

    총수익률이 아니라 위험조정 수익 기준이다. 회전율에 패널티를 준다 —
    연 100회전은 비용 가정이 조금만 틀려도 성과가 뒤집힌다.
    거래 수가 너무 적으면 통계적 의미가 없으므로 제외한다.
    """
    if metrics['total_trades'] < min_trades:
        return -99.0
    sharpe = metrics['sharpe_ratio']
    mdd = abs(metrics['max_drawdown'])
    turnover = metrics.get('annual_turnover', 0.0)

    # 회전율 50회를 넘어가는 만큼 샤프에서 깎는다
    turnover_penalty = max(0.0, (turnover - 50.0) / 50.0) * 0.15
    # MDD 25%를 넘어가는 만큼 추가 패널티
    mdd_penalty = max(0.0, (mdd - 0.25)) * 2.0

    return sharpe - turnover_penalty - mdd_penalty


# ─────────────────────────────────────────────────────────────────────────────
# 워크포워드
# ─────────────────────────────────────────────────────────────────────────────

def make_folds(df, is_months=6, oos_months=2):
    """(인샘플 구간, 아웃샘플 구간) 목록을 만든다. 롤링 윈도우."""
    t0 = df['datetime'].iloc[0]
    t_end = df['datetime'].iloc[-1]

    folds = []
    is_start = t0
    while True:
        is_end = is_start + pd.DateOffset(months=is_months)
        oos_end = is_end + pd.DateOffset(months=oos_months)
        if oos_end > t_end:
            break
        folds.append((is_start, is_end, oos_end))
        is_start = is_start + pd.DateOffset(months=oos_months)
    return folds


def plateau_pick(results, top_frac=0.3):
    """고원에서 파라미터를 고른다.

    상위 성과 구간의 파라미터들을 모아 각 축의 중앙값을 취한다. 단일 최고점은
    거의 항상 이웃이 나쁜 뾰족한 봉우리이고, 다음 구간에서 무너진다.

    Args:
        results: [(params, score), ...]
    Returns:
        dict: 고른 파라미터
    """
    ranked = sorted(results, key=lambda r: r[1], reverse=True)
    ranked = [r for r in ranked if r[1] > -99]
    if not ranked:
        return dict(DEFAULT_RISK)

    n_top = max(1, int(len(ranked) * top_frac))
    top = [r[0] for r in ranked[:n_top]]

    picked = {}
    for key in top[0]:
        values = [p[key] for p in top if key in p]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            median = float(np.median(values))
            # 실제 그리드에 있는 값 중 중앙값에 가장 가까운 것으로 스냅
            picked[key] = min(values, key=lambda v: abs(v - median))
        else:
            picked[key] = max(set(values), key=values.count)
    return picked


def walk_forward(df, strategy_name, symbol='BTC/USDT', is_months=6, oos_months=2,
                 cost_multiplier=1.0, verbose=False):
    """전략 하나에 대한 워크포워드 분석.

    Returns:
        dict: 아웃샘플만 이어붙인 성과 + 구간별 기록
    """
    grid = STRATEGY_GRIDS.get(strategy_name, {})
    combos = [p for p in expand_grid(grid, strategy_name=strategy_name)
              if valid_params(strategy_name, p)]
    folds = make_folds(df, is_months, oos_months)

    if not folds:
        return None

    fold_records = []
    oos_returns = []      # 구간별 수익률 (복리로 이어붙인다)
    oos_trades = 0
    oos_sharpes = []
    oos_turnovers = []
    worst_dd = 0.0

    for (is_start, is_end, oos_end) in folds:
        # 인샘플에서 파라미터 탐색.
        # evaluate_window()가 평가 구간 앞에 워밍업 봉을 따로 붙인다 — 이걸
        # 안 하면 1d 봉처럼 창이 짧을 때 워밍업이 창을 다 잡아먹어 거래가
        # 한 건도 나오지 않는다.
        scored = []
        for params in combos:
            try:
                m = evaluate_window(df, strategy_name, params, is_start, is_end,
                                    symbol=symbol, cost_multiplier=cost_multiplier)
                if m is None:
                    continue
                scored.append((params, score(m)))
            except Exception:
                continue

        if not scored:
            continue

        chosen = plateau_pick(scored)

        # 아웃샘플에서 그 파라미터로 평가 — 여기 성과만 집계한다
        try:
            m_oos = evaluate_window(df, strategy_name, chosen, is_end, oos_end,
                                    symbol=symbol, cost_multiplier=cost_multiplier)
        except Exception:
            continue
        if m_oos is None:
            continue

        oos_returns.append(m_oos['total_return'])
        oos_trades += m_oos['total_trades']
        oos_sharpes.append(m_oos['sharpe_ratio'])
        oos_turnovers.append(m_oos.get('annual_turnover', 0.0))
        worst_dd = min(worst_dd, m_oos['max_drawdown'])

        fold_records.append({
            'is_start': is_start, 'is_end': is_end, 'oos_end': oos_end,
            'params': chosen,
            'oos_return': m_oos['total_return'],
            'oos_sharpe': m_oos['sharpe_ratio'],
            'oos_mdd': m_oos['max_drawdown'],
            'oos_trades': m_oos['total_trades'],
            'oos_pf': m_oos['profit_factor'],
            'buy_hold': m_oos['buy_hold_return'],
        })

        if verbose:
            print(f"    {is_end.date()}~{oos_end.date()}: "
                  f"{m_oos['total_return']:+.2%} (B&H {m_oos['buy_hold_return']:+.2%}) "
                  f"거래 {m_oos['total_trades']}")

    if not fold_records:
        return None

    # 아웃샘플 구간들을 복리로 이어붙인다
    compounded = float(np.prod([1 + r for r in oos_returns]) - 1)
    positive = sum(1 for r in oos_returns if r > 0)

    return {
        'strategy': strategy_name,
        'folds': len(fold_records),
        'oos_total_return': compounded,
        'oos_mean_return': float(np.mean(oos_returns)),
        'oos_median_return': float(np.median(oos_returns)),
        'oos_worst_fold': float(min(oos_returns)),
        'oos_best_fold': float(max(oos_returns)),
        'oos_positive_folds': positive,
        'oos_consistency': positive / len(oos_returns),
        'oos_sharpe': float(np.mean(oos_sharpes)),
        'oos_mdd': worst_dd,
        'oos_trades': oos_trades,
        'oos_turnover': float(np.mean(oos_turnovers)),
        'buy_hold_compounded': float(np.prod([1 + f['buy_hold'] for f in fold_records]) - 1),
        'fold_records': fold_records,
        'cost_multiplier': cost_multiplier,
    }
