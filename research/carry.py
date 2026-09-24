"""
research/carry.py — 델타중립 현물-선물 캐리(cash-and-carry) 백테스트.

현물을 사고 같은 수량의 무기한 선물을 숏한다. 가격 방향 노출은 상쇄되고,
남는 수익원은 **펀딩비**(콘탱고에서 숏이 받는다)와 베이시스 변화뿐이다.

방향성 전략(추세/평균회귀)은 실비용 아웃샘플에서 샤프 1 근처가 한계였다.
샤프 2 이상은 보통 이런 시장중립 전략에서 나온다 — 변동성이 가격이 아니라
베이시스에서만 오기 때문이다. 대신 수익률이 펀딩 수준(연 5~15%)에 묶인다.

비용·위험 모델:
- 현물 taker 0.10% (바이낸스 VIP0, BNB 할인 없음) — 선물(0.04%)보다 비싸다
- 선물 taker 0.04%, 양쪽 슬리피지 2bp
- 펀딩: 8시간 실제 이력. 요율이 음수면 숏이 **지불**한다
- 선물 숏은 레버리지 L로 증거금을 잡는다. 가격이 오르면 숏 쪽 증거금이
  깎이므로, 가격이 기준가 대비 rebalance_pct 이상 움직이면 현물 일부를 팔아
  증거금을 채우는(또는 반대) 리밸런싱을 하고 그 비용을 문다
- 리밸런싱 전에 청산가에 닿으면 강제청산: 숏 증거금 손실 + 청산수수료 0.5%
"""

import numpy as np
import pandas as pd

from research.strategy_lab import load_candles
from research.binance_env import load_funding_history

SPOT_FEE = 0.0010
PERP_FEE = 0.0004
SLIP = 0.0002
LIQ_FEE = 0.005
MMR = 0.004


def load_pair(symbol):
    """현물·선물 일봉과 일별 펀딩 합계를 같은 날짜축으로 맞춘다."""
    s = load_candles(symbol, '1d', is_futures=False, with_indicators=False)
    f = load_candles(symbol, '1d', is_futures=True, with_indicators=False)
    df = s[['timestamp', 'datetime', 'close', 'high']].rename(
        columns={'close': 'spot', 'high': 'spot_high'}).merge(
        f[['timestamp', 'close', 'high']].rename(
            columns={'close': 'perp', 'high': 'perp_high'}), on='timestamp')
    fh = load_funding_history(symbol)
    day = 86400000
    fh['day'] = (fh['timestamp'] - 1) // day * day   # 00:00 정산은 전날 몫으로
    daily = fh.groupby('day')['rate'].sum()
    df['funding'] = df['timestamp'].map(daily).fillna(0.0)
    return df.dropna().reset_index(drop=True)


def run_carry(df, leverage=1.0, rebalance_pct=0.25, entry_rule=None,
              cost_mult=1.0, capital=10000.0):
    """
    Args:
        leverage: 선물 숏 레버리지. 높을수록 자본 효율↑, 청산위험↑
        entry_rule: None이면 상시 보유. 아니면 (i, df) -> bool, True일 때만 보유
    Returns:
        (일별 자본 Series, 통계 dict)
    """
    sf, pf, sl = SPOT_FEE * cost_mult, PERP_FEE * cost_mult, SLIP * cost_mult
    n = len(df)
    spot, perp = df['spot'].to_numpy(), df['perp'].to_numpy()
    perp_high, fund = df['perp_high'].to_numpy(), df['funding'].to_numpy()

    cash = capital
    q = 0.0            # 보유 수량 (현물 롱 = 선물 숏)
    margin = 0.0       # 선물 증거금
    ref = 0.0          # 리밸런싱 기준가
    eq = np.empty(n)
    fees = funding_sum = 0.0
    min_cash = 0.0
    liqs = rebal = 0
    on = False

    def equity(i):
        return cash + q * spot[i] + margin + q * (ref_entry - perp[i]) if q else cash

    ref_entry = 0.0
    for i in range(n):
        want = True if entry_rule is None else bool(entry_rule(i, df))

        # 청산 (신호 off)
        if on and not want:
            proceeds = q * spot[i] * (1 - sf - sl)
            perp_pnl = q * (ref_entry - perp[i] * (1 + sl)) - q * perp[i] * pf
            fees += q * spot[i] * (sf + sl) + q * perp[i] * (pf + sl)
            cash += proceeds + margin + perp_pnl
            q = margin = 0.0; on = False

        # 진입
        if not on and want and i > 0:
            total = cash
            notional = total / (1 + 1 / leverage)
            q = notional / spot[i]
            cost = q * spot[i] * (sf + sl) + q * perp[i] * (pf + sl)
            fees += cost
            margin = notional / leverage
            cash = total - notional - margin - cost
            ref = ref_entry = perp[i]
            on = True

        if on:
            # 강제청산 확인: 숏 손실이 증거금을 넘으면
            liq_px = ref_entry + (margin / q) * (1 - MMR)
            if perp_high[i] >= liq_px:
                loss = margin
                fees += q * liq_px * LIQ_FEE
                # 숏이 날아가고 현물만 남는다 → 즉시 현물 매도, 재진입은 다음 날
                cash += q * spot[i] * (1 - sf - sl) - q * liq_px * LIQ_FEE
                q = margin = 0.0; on = False; liqs += 1
            else:
                # 펀딩 (양수면 숏이 받는다)
                f = q * perp[i] * fund[i]
                cash += f; funding_sum += f
                # 리밸런싱: 가격이 기준가 대비 많이 움직였으면 **현재 자본 기준으로
                # 포지션 전체를 다시 맞춘다** (현물 일부 매매 + 숏 일부 청산/추가).
                #
                # ⚠️ 초판은 수량 q를 그대로 둔 채 숏 손실을 현금에서 빼서 증거금을
                # 채웠다. 현금이 마이너스가 되어도 막지 않았으니 사실상 무이자
                # 차입이었고, BTC가 4배 오르는 동안 헤지 명목가도 4배가 돼 펀딩을
                # 자본의 4배 규모로 받았다(연 11% → 과대계상). 실제로는 선물
                # 증거금을 채우려면 현물을 팔아야 하고, 그만큼 헤지가 줄어든다.
                if abs(perp[i] / ref - 1) >= rebalance_pct:
                    eq_now = cash + q * spot[i] + margin + q * (ref_entry - perp[i])
                    q_new = (eq_now / (1 + 1 / leverage)) / spot[i]
                    dq = abs(q_new - q)
                    cost = dq * spot[i] * (sf + sl) + dq * perp[i] * (pf + sl)
                    fees += cost
                    q = q_new
                    margin = q * perp[i] / leverage
                    cash = eq_now - q * spot[i] - margin - cost
                    ref = ref_entry = perp[i]
                    rebal += 1

        min_cash = min(min_cash, cash)
        eq[i] = cash + (q * spot[i] + margin + q * (ref_entry - perp[i]) if on else 0.0)

    s = pd.Series(eq, index=pd.to_datetime(df['timestamp'], unit='ms'))
    r = s.pct_change().fillna(0)
    yrs = len(s) / 365.25
    roll = s.cummax()
    stats = {
        'total_return': s.iloc[-1] / capital - 1,
        'cagr': (s.iloc[-1] / capital) ** (1 / yrs) - 1,
        'sharpe': r.mean() / r.std() * np.sqrt(365.25) if r.std() > 0 else 0.0,
        'vol': r.std() * np.sqrt(365.25),
        'mdd': ((s - roll) / roll).min(),
        'fees': fees, 'funding': funding_sum, 'liquidations': liqs,
        'rebalances': rebal, 'min_cash': min_cash,
    }
    return s, stats


def trailing_funding_rule(window=7, threshold_annual=0.05):
    """직전 window일 펀딩 평균(연환산)이 임계값을 넘을 때만 보유."""
    def rule(i, df):
        if i < window:
            return False
        avg = df['funding'].iloc[i - window:i].mean() * 365
        return avg > threshold_annual
    return rule
