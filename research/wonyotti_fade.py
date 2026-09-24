"""
research/wonyotti_fade.py — 워뇨띠 스타일의 기계적 근사를 2022~2026(그의 공개 데이터 이후) 구간에서 검증한다.

공개 분석에서 확인된 특징:
  - 급등락으로 가격이 크게 이탈한 시점에 진입 (역추세)
  - 변동성이 클 때 규모를 키움 (저변동 구간에서는 대기)
  - 보유 시간 중앙값 26분, 다만 수일 보유도 있음
  - 지정가(메이커) 위주 체결

근사 규칙 (1h 봉):
  z = (최근 lookback봉 누적수익률) / (롤링 변동성 × sqrt(lookback))
  |z| > z_entry 이고 실현변동성이 자기 중앙값보다 높으면 → 반대 방향 진입
  |z| < z_exit 로 되돌아오면 청산, 또는 max_hold 봉 경과 시 청산
  손절은 ATR 배수

분할진입(물타기)은 엔진이 단일 진입만 지원해 넣지 않았다 — 그의 핵심 특징
하나를 빠뜨린 근사임을 감안해야 한다.

비용 시나리오:
  taker : 0.04% + ATR 슬리피지 (우리 봇의 실제 조건, 시장가 전용)
  maker : 0.02%, 슬리피지 0 — 낙관적 상한. 실제로는 지정가가 불리할 때만
          잘 체결되는 역선택이 있어 이보다 나쁘다.
"""
import itertools, sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from core.strategies import BaseStrategy
from research.backtester import Backtester
from research.strategy_lab import (load_candles, make_folds, slice_with_warmup,
                                   restrict_metrics, score, plateau_pick, get_funding)


class FadeStrategy(BaseStrategy):
    def __init__(self, **kw):
        p = dict(lookback=4, z_entry=2.5, z_exit=0.5, max_hold=24, vol_window=72,
                 vol_filter=True, leverage=2, stop_loss_pct=0.05,
                 take_profit_pct=1.0, max_allocation_pct=0.35, atr_sl_mult=3.0)
        p.update(kw)
        super().__init__(name='역추세 페이드', **p)
        self.atr_sl_mult = p['atr_sl_mult']

    def generate_signals(self, df):
        P = self.parameters
        c = df['close']
        r = np.log(c).diff()
        vol = r.rolling(P['vol_window']).std()
        cum = np.log(c / c.shift(P['lookback']))
        z = (cum / (vol * np.sqrt(P['lookback']))).to_numpy()
        hv = (vol > vol.rolling(P['vol_window'] * 4).median()).to_numpy() if P['vol_filter'] \
            else np.ones(len(df), bool)
        sig = np.zeros(len(df)); state = 0; held = 0
        for i in range(len(df)):
            zi = z[i]
            if not np.isfinite(zi):
                continue
            if state == 0:
                if hv[i] and zi > P['z_entry']:
                    state, held = -1, 0
                elif hv[i] and zi < -P['z_entry']:
                    state, held = 1, 0
            else:
                held += 1
                # 되돌림 완료(|z| 작음) · 반대쪽으로 넘어감(과잉 되돌림) · 시간초과 → 청산
                if abs(zi) < P['z_exit'] or np.sign(zi) == state or held >= P['max_hold']:
                    state = 0
            sig[i] = state
        sig[:P['vol_window'] * 4] = 0
        return pd.Series(sig, index=df.index)


GRID = {'lookback': [2, 4, 8], 'z_entry': [2.0, 2.5, 3.0], 'max_hold': [12, 48],
        'vol_filter': [True, False]}
COSTS = {'taker': dict(taker_fee=0.0004, slippage_base=0.0001, slippage_atr_coef=0.02),
         'maker': dict(taker_fee=0.0002, slippage_base=0.0, slippage_atr_coef=0.0)}


def ev(df, a, b, params, symbol, cost):
    w, i = slice_with_warmup(df, a, b, warmup_bars=400)
    if len(w) - i < 50:
        return None
    bt = Backtester(symbol=symbol, funding_lookup=get_funding(symbol), **COSTS[cost])
    m, e, t = bt.run(w, FadeStrategy(**params), is_futures=True)
    return restrict_metrics(m, e, t, i)


def wfa(symbol, cost, start='2022-01-01'):
    df = load_candles(symbol, '1h', True)
    df = df[df['datetime'] >= start].reset_index(drop=True)
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    rets, shs, turns, nt, worst = [], [], [], 0, 0.0
    for (a, b, c) in make_folds(df):
        sc = [(p, score(m, min_trades=5)) for p in combos for m in [ev(df, a, b, p, symbol, cost)] if m]
        if not sc:
            continue
        ch = plateau_pick(sc)
        m = ev(df, b, c, ch, symbol, cost)
        if not m:
            continue
        rets.append(m['total_return']); shs.append(m['sharpe_ratio'])
        turns.append(m['annual_turnover']); nt += m['total_trades']; worst = min(worst, m['max_drawdown'])
    yrs = len(rets) * 2 / 12
    tot = float(np.prod([1 + r for r in rets]) - 1)
    return dict(tot=tot, cagr=(1 + tot) ** (1 / yrs) - 1 if yrs else 0, sharpe=float(np.mean(shs)),
                mdd=worst, cons=sum(r > 0 for r in rets) / len(rets), turn=float(np.mean(turns)),
                trades=nt, folds=len(rets))


if __name__ == '__main__':
    print(f"{'종목':<10}{'비용':<7}{'연환산':>8}{'샤프':>7}{'MDD':>8}{'일관성':>7}{'회전':>7}{'거래':>6}")
    for sym in ('BTC/USDT', 'ETH/USDT'):
        for cost in ('taker', 'maker'):
            r = wfa(sym, cost)
            print(f"{sym:<10}{cost:<7}{r['cagr']:>8.1%}{r['sharpe']:>7.2f}{r['mdd']:>8.1%}"
                  f"{r['cons']:>7.0%}{r['turn']:>7.0f}{r['trades']:>6}", flush=True)
