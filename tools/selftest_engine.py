"""
tools/selftest_engine.py — 백테스트 엔진의 계산을 손으로 검산한다.

엔진을 고칠 때마다 손익 계산이 조용히 틀어지는 걸 막는다. 인위적으로 만든
가격 데이터 위에서, 기대값을 직접 계산해 비교한다.

    python tools/selftest_engine.py
"""

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.strategies import BaseStrategy
from research.backtester import Backtester
from research.binance_env import liquidation_price, mmr_for_notional, _count_settlements

PASS, FAIL = [], []


def check(label, condition, detail=""):
    (PASS if condition else FAIL).append(label)
    print(f"   {'✅' if condition else '❌'} {label}" + (f" — {detail}" if detail else ""))


def make_df(prices, atr=0.0, regime='BULL'):
    prices = np.asarray(prices, dtype=float)
    return pd.DataFrame({
        'timestamp': [1700000000000 + i * 4 * 3600 * 1000 for i in range(len(prices))],
        'open': prices, 'high': prices * 1.001, 'low': prices * 0.999,
        'close': prices, 'volume': 1.0, 'atr_14': atr, 'regime': regime,
    })


class FixedSignal(BaseStrategy):
    """항상 같은 신호를 내는 전략. 엔진 검산용."""
    def __init__(self, signal=1, **kw):
        params = dict(leverage=1, stop_loss_pct=0.99, take_profit_pct=9.9,
                      max_allocation_pct=1.0)
        params.update(kw)
        super().__init__(name='fixed', **params)
        self.signal = signal

    def generate_signals(self, df):
        return pd.Series(self.signal, index=df.index)


def test_pnl_exact():
    print("\n1. 손익 계산 (수수료만, 슬리피지·펀딩 없음)")
    price = np.array([100.0 * (1.01 ** i) for i in range(11)])
    df = make_df(price)
    bt = Backtester(initial_capital=10000.0, taker_fee=0.0004,
                    slippage_base=0.0, slippage_atr_coef=0.0, apply_funding=False)
    m, _, trades = bt.run(df, FixedSignal(1), is_futures=True)

    qty = np.floor((10000.0 / price[1]) / 0.001) * 0.001
    expected = (10000.0
                - qty * price[1] * 0.0004
                + qty * (price[-1] - price[1])
                - qty * price[-1] * 0.0004)

    check("거래 1건 발생", m['total_trades'] == 1)
    check("진입가 = 다음 봉 시가 (룩어헤드 없음)",
          abs(trades[0]['entry_price'] - price[1]) < 1e-9,
          f"{trades[0]['entry_price']:.4f}")
    check("최종자본이 손계산과 일치",
          abs(expected - m['final_equity']) < 1e-6,
          f"기대 {expected:.4f} / 실제 {m['final_equity']:.4f}")


def test_min_notional():
    print("\n2. 최소명목가 가드 (BTC 선물 50 USDT)")
    df = make_df([100.0 * (1.01 ** i) for i in range(11)])
    bt = Backtester(initial_capital=40.0, slippage_base=0.0,
                    slippage_atr_coef=0.0, apply_funding=False)
    m, _, _ = bt.run(df, FixedSignal(1), is_futures=True)
    check("자본 40 USDT면 진입하지 않는다", m['total_trades'] == 0)

    bt2 = Backtester(initial_capital=10000.0, slippage_base=0.0,
                     slippage_atr_coef=0.0, apply_funding=False)
    m2, _, _ = bt2.run(df, FixedSignal(1), is_futures=True)
    check("자본 10,000 USDT면 진입한다", m2['total_trades'] == 1)


def test_stop_loss_on_entry_bar():
    print("\n3. 진입 봉에서의 SL 발동")
    # 봉 1에서 시가 100에 진입한 뒤 같은 봉에서 저가 90까지 빠진다
    df = make_df([100.0, 100.0, 100.0])
    df.loc[1, 'low'] = 90.0
    bt = Backtester(initial_capital=10000.0, slippage_base=0.0,
                    slippage_atr_coef=0.0, apply_funding=False)
    m, _, trades = bt.run(df, FixedSignal(1, stop_loss_pct=0.05), is_futures=True)
    check("진입한 봉에서도 SL이 발동한다",
          len(trades) > 0 and trades[0]['exit_reason'] == 'STOP_LOSS',
          trades[0]['exit_reason'] if trades else "거래 없음")
    if trades:
        check("SL 체결가 = 손절가", abs(trades[0]['exit_price'] - 95.0) < 1e-9,
              f"{trades[0]['exit_price']:.4f}")


def test_sl_before_tp():
    print("\n4. 한 봉에서 SL·TP 동시 도달 시 SL 우선 (보수적)")
    df = make_df([100.0, 100.0, 100.0])
    df.loc[1, 'low'] = 90.0      # SL(95) 도달
    df.loc[1, 'high'] = 110.0    # TP(105) 도 도달
    bt = Backtester(initial_capital=10000.0, slippage_base=0.0,
                    slippage_atr_coef=0.0, apply_funding=False)
    m, _, trades = bt.run(df, FixedSignal(1, stop_loss_pct=0.05, take_profit_pct=0.05),
                          is_futures=True)
    check("SL이 먼저 잡힌다",
          trades and trades[0]['exit_reason'] == 'STOP_LOSS',
          trades[0]['exit_reason'] if trades else "거래 없음")


def test_slippage_direction():
    print("\n5. 슬리피지는 항상 불리한 방향")
    df = make_df([100.0] * 5, atr=2.0)   # atr_pct = 2%
    bt = Backtester(initial_capital=10000.0, taker_fee=0.0,
                    slippage_base=0.001, slippage_atr_coef=0.0, apply_funding=False)
    _, _, long_trades = bt.run(df, FixedSignal(1), is_futures=True)
    _, _, short_trades = bt.run(df, FixedSignal(-1), is_futures=True)
    check("롱 진입가 > 시가", long_trades[0]['entry_price'] > 100.0,
          f"{long_trades[0]['entry_price']:.4f}")
    check("숏 진입가 < 시가", short_trades[0]['entry_price'] < 100.0,
          f"{short_trades[0]['entry_price']:.4f}")


def test_funding_sign():
    print("\n6. 펀딩비 부호 (요율 양수면 롱이 지불, 숏이 수취)")
    df = make_df([100.0] * 20)
    flat = lambda s, e, n, d: _count_settlements(s, e) * 0.0001 * n * d
    bt = Backtester(initial_capital=10000.0, taker_fee=0.0, slippage_base=0.0,
                    slippage_atr_coef=0.0, funding_lookup=flat)
    m_long, _, _ = bt.run(df, FixedSignal(1), is_futures=True)
    m_short, _, _ = bt.run(df, FixedSignal(-1), is_futures=True)
    check("롱은 펀딩을 지불한다 (total_funding > 0)", m_long['total_funding'] > 0,
          f"{m_long['total_funding']:.4f}")
    check("숏은 펀딩을 수취한다 (total_funding < 0)", m_short['total_funding'] < 0,
          f"{m_short['total_funding']:.4f}")


def test_liquidation_formula():
    print("\n7. 청산가 공식")
    # 진입 100, 수량 1, 증거금 50 (사실상 2배) → 롱 청산가는 50 근처
    liq_long = liquidation_price(100.0, 1.0, 1, 50.0)
    liq_short = liquidation_price(100.0, 1.0, -1, 50.0)
    check("롱 청산가가 진입가 아래", 45 < liq_long < 55, f"{liq_long:.4f}")
    check("숏 청산가가 진입가 위", 145 < liq_short < 155, f"{liq_short:.4f}")

    rate, ded = mmr_for_notional(10_000)
    check("명목가 1만이면 MMR 0.40%", abs(rate - 0.004) < 1e-9, f"{rate:.4%}")
    rate2, ded2 = mmr_for_notional(100_000)
    check("명목가 10만이면 MMR 0.50% / 공제 50", abs(rate2 - 0.005) < 1e-9 and ded2 == 50.0)


def test_settlement_count():
    print("\n8. 8시간 정산 시점 계산")
    h = 3600 * 1000
    check("0~8시간 구간에 1회", _count_settlements(0, 8 * h) == 1)
    check("0~24시간 구간에 3회", _count_settlements(0, 24 * h) == 3)
    check("1~7시간 구간에 0회", _count_settlements(1 * h, 7 * h) == 0)


def main():
    print("=" * 56)
    print("백테스트 엔진 자가 검산")
    print("=" * 56)

    test_pnl_exact()
    test_min_notional()
    test_stop_loss_on_entry_bar()
    test_sl_before_tp()
    test_slippage_direction()
    test_funding_sign()
    test_liquidation_formula()
    test_settlement_count()

    print("\n" + "=" * 56)
    if FAIL:
        print(f"❌ 실패 {len(FAIL)}건 / 통과 {len(PASS)}건")
        for f in FAIL:
            print(f"   - {f}")
        return 1
    print(f"✅ 전부 통과 ({len(PASS)}건)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
