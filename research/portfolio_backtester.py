"""
research/portfolio_backtester.py — 여러 종목 사이를 오가는 백테스트 엔진.

기존 Backtester는 한 종목만 본다. 이건 BTC와 ETH를 함께 놓고 매 봉마다
{현금, BTC 롱/숏, ETH 롱/숏} 중 하나를 고른다. "더 센 쪽에 붙는다"는
횡단면 모멘텀(cross-sectional momentum)이 핵심 아이디어다.

단일 종목 추세 전략이 잘 안 되는 이유 하나는, 방향이 애매한 구간에서도
그 종목만 쳐다보기 때문이다. 두 종목을 비교하면 "둘 다 애매하면 쉰다"와
"상대적으로 강한 쪽만 잡는다"가 가능해진다.

비용 모델은 단일 종목 엔진과 같다 (taker 0.04%, 실제 펀딩, ATR 슬리피지,
스텝/최소명목가). 종목을 갈아탈 때는 **청산 + 진입 두 번**의 비용을 문다 —
회전율이 올라가기 쉬운 구조라 이 점이 특히 중요하다.
"""

import numpy as np
import pandas as pd

from research.binance_env import (
    TAKER_FEE, SLIPPAGE_BASE, SLIPPAGE_ATR_COEF,
    get_spec, round_qty, is_tradable, slippage_frac, build_funding_lookup,
)


class PortfolioBacktester:
    """한 번에 한 포지션만 들되, 종목·방향을 자유롭게 바꾸는 엔진."""

    def __init__(self, initial_capital=10000.0, taker_fee=TAKER_FEE,
                 slippage_base=SLIPPAGE_BASE, slippage_atr_coef=SLIPPAGE_ATR_COEF,
                 cost_multiplier=1.0, leverage=2, max_allocation=0.5,
                 target_vol=None, vol_window=20, atr_sl_mult=None,
                 funding_lookups=None):
        self.initial_capital = initial_capital
        self.taker_fee = taker_fee * cost_multiplier
        self.slip_base = slippage_base * cost_multiplier
        self.slip_coef = slippage_atr_coef * cost_multiplier
        self.cost_multiplier = cost_multiplier
        self.leverage = leverage
        self.max_allocation = max_allocation
        self.target_vol = target_vol
        self.vol_window = vol_window
        self.atr_sl_mult = atr_sl_mult
        self.funding_lookups = funding_lookups or {}

    def run(self, frames, signal_fn):
        """
        Args:
            frames: {symbol: DataFrame} — 모두 같은 timestamp 축으로 정렬되어 있어야 한다
            signal_fn: (bar_index, frames_np) -> (symbol or None, direction)
                해당 봉 **종가까지의** 정보로 판단한다. 엔진이 i-1을 넘겨
                다음 봉 시가에 체결하므로 룩어헤드가 생기지 않는다.
        """
        symbols = list(frames)
        base = frames[symbols[0]]
        n = len(base)
        stamps = base['timestamp'].to_numpy(dtype='int64')
        tf_ms = int(stamps[1] - stamps[0]) if n > 1 else 86400000
        bars_per_year = (365.25 * 24 * 3600 * 1000) / max(tf_ms, 1)

        # 종목별 배열을 미리 뽑아 둔다
        A = {}
        for s in symbols:
            d = frames[s]
            rv = (d['close'].pct_change().rolling(self.vol_window).std()
                  * np.sqrt(bars_per_year)).to_numpy() if self.target_vol else None
            A[s] = {
                'open': d['open'].to_numpy(float), 'high': d['high'].to_numpy(float),
                'low': d['low'].to_numpy(float), 'close': d['close'].to_numpy(float),
                'atr': d['atr_14'].to_numpy(float) if 'atr_14' in d else np.full(n, np.nan),
                'rv': rv, 'spec': get_spec(s),
                'funding': self.funding_lookups.get(s, (lambda a, b, c, e: 0.0)),
            }

        balance = self.initial_capital
        pos_sym, pos_dir, qty, entry = None, 0, 0.0, 0.0
        sl_price = 0.0
        equity_hist = np.empty(n)
        trades = []
        cur = None
        fees = funding_total = volume = 0.0

        for i in range(n):
            # ⚠️ 신호는 **직전 봉**의 정보로만 만든다. signal_fn(i, ...)를 쓰면
            # i봉 종가로 만든 신호를 i봉 시가에 체결하는 미래 참조가 된다.
            # (처음에 그렇게 짰다가 샤프 4.26 / 일관성 100%라는 말도 안 되는
            #  결과가 나와서 잡았다. 단일 종목 엔진은 sig_arr[i-1]로 맞춰져 있다.)
            want_sym, want_dir = signal_fn(i - 1, A) if i > 0 else (None, 0)

            # ── 청산 판단: 신호가 바뀌었거나 없어졌으면 시가에 정리 ──
            if pos_sym is not None and (want_sym != pos_sym or want_dir != pos_dir):
                a = A[pos_sym]
                slip = self._slip(a, i)
                fill = a['open'][i] * (1 - slip * pos_dir)
                pnl = qty * (fill - entry) * pos_dir
                fee = qty * fill * self.taker_fee
                balance += pnl - fee
                fees += fee; volume += qty * fill
                cur.update({'exit_time': int(stamps[i]), 'exit_price': fill,
                            'pnl': pnl - fee, 'exit_reason': 'SWITCH'})
                trades.append(cur); cur = None
                pos_sym, pos_dir, qty, entry = None, 0, 0.0, 0.0

            # ── 진입 ──
            if pos_sym is None and want_sym is not None and want_dir != 0:
                a = A[want_sym]
                slip = self._slip(a, i)
                fill = a['open'][i] * (1 + slip * want_dir)
                notional = balance * self.max_allocation * self.leverage
                if a['rv'] is not None and np.isfinite(a['rv'][i]) and a['rv'][i] > 1e-9:
                    notional *= float(np.clip(self.target_vol / a['rv'][i], 0.2, 3.0))
                q = round_qty(notional / fill, a['spec'])
                if q > 0 and is_tradable(q, fill, a['spec']):
                    fee = q * fill * self.taker_fee
                    balance -= fee; fees += fee; volume += q * fill
                    pos_sym, pos_dir, qty, entry = want_sym, want_dir, q, fill
                    bar_atr = a['atr'][i]
                    if self.atr_sl_mult and np.isfinite(bar_atr):
                        sl_price = entry - want_dir * bar_atr * self.atr_sl_mult
                    else:
                        sl_price = entry * (1 - 0.15 * want_dir)
                    cur = {'symbol': want_sym, 'entry_time': int(stamps[i]),
                           'entry_price': entry, 'size': q,
                           'direction': 'LONG' if want_dir == 1 else 'SHORT'}

            # ── 봉 중 손절 ──
            if pos_sym is not None:
                a = A[pos_sym]
                hit = (a['low'][i] <= sl_price) if pos_dir == 1 else (a['high'][i] >= sl_price)
                if hit:
                    slip = self._slip(a, i)
                    fill = sl_price * (1 - slip * pos_dir)
                    pnl = qty * (fill - entry) * pos_dir
                    fee = qty * fill * self.taker_fee
                    balance += pnl - fee
                    fees += fee; volume += qty * fill
                    cur.update({'exit_time': int(stamps[i]), 'exit_price': fill,
                                'pnl': pnl - fee, 'exit_reason': 'STOP_LOSS'})
                    trades.append(cur); cur = None
                    pos_sym, pos_dir, qty, entry = None, 0, 0.0, 0.0

            # ── 펀딩 ──
            if pos_sym is not None:
                a = A[pos_sym]
                f = a['funding'](int(stamps[i]), int(stamps[i]) + tf_ms,
                                 qty * a['close'][i], pos_dir)
                balance -= f; funding_total += f

            unreal = (qty * (A[pos_sym]['close'][i] - entry) * pos_dir) if pos_sym else 0.0
            equity_hist[i] = balance + unreal

        if pos_sym is not None and cur is not None:
            a = A[pos_sym]
            fill = a['close'][-1]
            pnl = qty * (fill - entry) * pos_dir
            fee = qty * fill * self.taker_fee
            balance += pnl - fee
            cur.update({'exit_time': int(stamps[-1]), 'exit_price': fill,
                        'pnl': pnl - fee, 'exit_reason': 'END'})
            trades.append(cur)
            equity_hist[-1] = balance

        eq = pd.DataFrame({'timestamp': stamps, 'equity': equity_hist})
        eq['datetime'] = pd.to_datetime(eq['timestamp'], unit='ms')
        return self._metrics(eq, trades, bars_per_year, fees, funding_total, volume), eq, trades

    def _slip(self, a, i):
        atr = a['atr'][i]; c = a['close'][i]
        pct = (atr / c) if (np.isfinite(atr) and c > 0) else 0.0
        return slippage_frac(pct, self.slip_base, self.slip_coef)

    def _metrics(self, eq, trades, bars_per_year, fees, funding, volume):
        e = eq['equity']
        r = e.pct_change().fillna(0.0)
        final = float(e.iloc[-1])
        roll = e.cummax()
        mdd = float(((e - roll) / roll).min())
        sd = r.std()
        sharpe = float((r.mean() / sd) * np.sqrt(bars_per_year)) if sd > 0 else 0.0
        dn = r[r < 0].std()
        sortino = float((r.mean() / dn) * np.sqrt(bars_per_year)) if dn > 0 else 0.0
        days = max((eq['datetime'].iloc[-1] - eq['datetime'].iloc[0]).total_seconds() / 86400, 1)
        wins = [t for t in trades if t.get('pnl', 0) > 0]
        gl = abs(sum(t['pnl'] for t in trades if t.get('pnl', 0) <= 0))
        gp = sum(t['pnl'] for t in wins)
        return {
            'final_equity': final,
            'total_return': final / self.initial_capital - 1,
            'cagr': (final / self.initial_capital) ** (365.25 / days) - 1 if final > 0 else -1.0,
            'max_drawdown': mdd, 'sharpe_ratio': sharpe, 'sortino_ratio': sortino,
            'total_trades': len(trades),
            'win_rate': len(wins) / len(trades) if trades else 0.0,
            'profit_factor': (gp / gl) if gl > 0 else (gp if gp > 0 else 0.0),
            'total_fees': fees, 'total_funding': funding,
            'annual_turnover': (volume / self.initial_capital) * (365.25 / days),
        }
