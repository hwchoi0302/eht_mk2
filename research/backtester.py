"""
research/backtester.py — 바이낸스 USDⓈ-M 선물 환경에 맞춘 백테스트 엔진.

구버전 대비 바뀐 것:

  1. 봉 안에서의 사건 순서를 바로잡았다. 구버전은 SL/TP를 먼저 보고 그 다음에
     시가 체결 신호를 처리해서, 시간 순서가 뒤집혀 있었다. 이제
     [시가 체결 → 봉 중 고저로 청산/SL/TP] 순이다.
  2. 진입한 봉에서도 SL/TP를 검사한다. 구버전은 진입 봉을 건너뛰어 낙관적이었다.
  3. 펀딩비를 실제 이력 요율로 8시간마다 정산한다. 구버전은 아예 없었다.
  4. 청산가를 실제 유지증거금 브래킷으로 계산한다. 구버전의
     `명목가 × 0.05 / leverage` 근사는 부호 방향부터 틀렸다.
  5. 슬리피지가 ATR에 비례한다. 구버전은 장세와 무관한 고정 0.02%였다.
  6. 수량/가격을 스텝사이즈·틱사이즈로 반올림하고 최소명목가를 검사한다.
  7. 국면 확정을 라이브와 같은 core.indicators.confirm_regimes()로 한다.
  8. 신호 반전 시 라이브처럼 같은 봉에서 청산 후 즉시 반대 진입한다.
  9. SL/TP 청산 후 같은 봉 재진입 금지 — 라이브의 쿨다운과 맞춘다.

전 체결은 시장가(taker)다. 지정가 진입 경로는 라이브에서 제거했으므로
백테스트에도 없다.
"""

import numpy as np
import pandas as pd

from core.indicators import confirm_regimes
from research.binance_env import (
    TAKER_FEE,
    SLIPPAGE_BASE,
    SLIPPAGE_ATR_COEF,
    get_spec,
    round_qty,
    is_tradable,
    liquidation_price,
    slippage_frac,
    build_funding_lookup,
)


class Backtester:
    """시장가 전용 바이낸스 선물 백테스터."""

    def __init__(
        self,
        initial_capital=10000.0,
        taker_fee=TAKER_FEE,
        slippage_base=SLIPPAGE_BASE,
        slippage_atr_coef=SLIPPAGE_ATR_COEF,
        symbol='BTC/USDT',
        margin_mode='cross',
        leverage_cap=3,
        cost_multiplier=1.0,
        funding_lookup=None,
        apply_funding=True,
        regime_confirm_candles=1,
    ):
        """
        Args:
            cost_multiplier: 수수료·슬리피지에 일괄 곱하는 계수.
                비용 스트레스 테스트용 (2.0이면 비용 2배).
            funding_lookup: build_funding_lookup()이 만든 함수.
                None이면 심볼의 실제 이력으로 자동 생성.
            apply_funding: False면 펀딩비를 끈다 (기여도 분리용).
            regime_confirm_candles: 라이브 설정과 같은 값을 넣어야 한다.
        """
        self.initial_capital = initial_capital
        self.taker_fee = taker_fee * cost_multiplier
        self.slippage_base = slippage_base * cost_multiplier
        self.slippage_atr_coef = slippage_atr_coef * cost_multiplier
        self.symbol = symbol
        self.spec = get_spec(symbol)
        self.margin_mode = margin_mode
        self.leverage_cap = leverage_cap
        self.cost_multiplier = cost_multiplier
        self.regime_confirm_candles = regime_confirm_candles

        self.apply_funding = apply_funding
        if not apply_funding:
            self.funding_lookup = lambda s, e, n, d: 0.0
        elif funding_lookup is not None:
            self.funding_lookup = funding_lookup
        else:
            self.funding_lookup = build_funding_lookup(symbol)

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────────

    def _slip(self, atr_pct):
        return slippage_frac(atr_pct, self.slippage_base, self.slippage_atr_coef)

    @staticmethod
    def _fill(price, direction, slip, is_entry):
        """슬리피지는 항상 우리에게 불리한 쪽으로 민다."""
        if is_entry:
            # 롱 진입은 비싸게, 숏 진입은 싸게 체결된다
            return price * (1 + slip) if direction == 1 else price * (1 - slip)
        # 청산은 반대 방향
        return price * (1 - slip) if direction == 1 else price * (1 + slip)

    # ── 메인 루프 ────────────────────────────────────────────────────────────

    def run(self, df, strategy, is_futures=True):
        """OHLCV + 지표가 들어있는 df 위에서 백테스트를 돌린다.

        Returns:
            (metrics: dict, df_equity: DataFrame, trades: list)
        """
        df = df.copy().reset_index(drop=True)
        signals = strategy.generate_signals(df).reset_index(drop=True)

        # 국면 확정을 라이브와 동일 함수로. 동적 리스크 전략이 이 값을 쓴다.
        if 'regime' in df.columns:
            df['regime_confirmed'] = confirm_regimes(df['regime'], self.regime_confirm_candles)
        else:
            df['regime_confirmed'] = 'SIDEWAYS'

        # 기본 리스크 설정
        base_leverage = min(strategy.leverage if is_futures else 1, self.leverage_cap)
        base_sl = strategy.stop_loss_pct
        base_tp = strategy.take_profit_pct
        base_alloc = strategy.max_allocation_pct
        has_dynamic = hasattr(strategy, 'get_dynamic_risk')

        # 계좌 상태
        balance = self.initial_capital      # 실현 잔고
        qty = 0.0                            # 계약 수량 (항상 양수)
        direction = 0                        # 1=롱, -1=숏, 0=없음
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0
        liq_price = 0.0
        entry_leverage = base_leverage

        # 넘파이로 뽑아두면 iloc 반복보다 훨씬 빠르다
        opens = df['open'].to_numpy(dtype=float)
        highs = df['high'].to_numpy(dtype=float)
        lows = df['low'].to_numpy(dtype=float)
        closes = df['close'].to_numpy(dtype=float)
        stamps = df['timestamp'].to_numpy(dtype='int64')
        regimes = df['regime_confirmed'].to_numpy()
        sig_arr = signals.to_numpy()
        if 'atr_14' in df.columns:
            atr_arr = df['atr_14'].to_numpy(dtype=float)
        else:
            atr_arr = np.full(len(df), np.nan)

        tf_ms = int(stamps[1] - stamps[0]) if len(stamps) > 1 else 4 * 3600 * 1000
        bars_per_year = (365.25 * 24 * 3600 * 1000) / max(tf_ms, 1)

        # ── 선택 기능: 변동성 타겟팅 ──────────────────────────────────────────
        # 전략이 target_vol(연환산)을 노출하면 포지션 크기를 실현변동성에
        # 반비례시켜 위험 기여도를 일정하게 만든다. 고정 배분은 조용한 장에서
        # 위험을 덜 지고 급변동 장에서 과하게 지는데, 그게 샤프를 깎는다.
        target_vol = getattr(strategy, 'target_vol', None)
        if target_vol:
            vw = int(getattr(strategy, 'vol_window', 20))
            rv = (pd.Series(closes).pct_change().rolling(vw).std()
                  * np.sqrt(bars_per_year)).to_numpy()
        else:
            rv = None

        # ── 선택 기능: ATR 기반 손절/익절 ─────────────────────────────────────
        # 고정 퍼센트 손절은 변동성이 커지면 쉽게 털리고 작아지면 너무 멀다.
        atr_sl_mult = getattr(strategy, 'atr_sl_mult', None)
        atr_tp_mult = getattr(strategy, 'atr_tp_mult', None)

        equity_history = np.empty(len(df), dtype=float)
        trades = []
        current_trade = None

        total_fees = 0.0
        total_funding = 0.0
        total_slippage_cost = 0.0
        total_volume = 0.0        # 거래대금 (회전율 계산용)
        bars_in_position = 0

        # SL/TP 청산이 일어난 봉 index. 백테스트는 봉당 진입을 한 번만 시도하므로
        # (시가 체결 → 봉 중 SL/TP 순서) 실제로 이 가드가 걸리는 경우는 없다.
        # 라이브는 30초마다 폴링해서 같은 캔들 안에 여러 번 진입을 시도할 수 있고,
        # 거기서는 last_exit_candle_timestamp 쿨다운이 실제로 동작한다.
        # 두 경로의 결과를 같게 유지하기 위한 방어적 장치로 남겨 둔다.
        cooldown_bar = -1

        for i in range(len(df)):
            open_p = opens[i]
            high_p = highs[i]
            low_p = lows[i]
            close_p = closes[i]
            ts = int(stamps[i])

            atr = atr_arr[i]
            atr_pct = (atr / close_p) if (np.isfinite(atr) and close_p > 0) else 0.0
            slip = self._slip(atr_pct)

            # 이 봉의 리스크 파라미터 (동적 리스크 전략이면 국면별로 갈아끼운다)
            leverage, sl_pct, tp_pct, alloc = base_leverage, base_sl, base_tp, base_alloc
            if has_dynamic:
                dyn = strategy.get_dynamic_risk(regimes[i])
                leverage = min(dyn.get('leverage', base_leverage), self.leverage_cap)
                sl_pct = dyn.get('stop_loss_pct', base_sl)
                tp_pct = dyn.get('take_profit_pct', base_tp)
                alloc = dyn.get('max_allocation_pct', base_alloc)
            if not is_futures:
                leverage = 1

            # ── (A) 시가: 직전 봉 신호에 반응 ────────────────────────────────
            # 신호는 i-1 봉 종가에 확정되고 체결은 i 봉 시가에 난다 (룩어헤드 없음)
            signal = sig_arr[i - 1] if i > 0 else 0

            if direction != 0:
                should_close = (
                    signal == 0
                    or (signal == -1 and direction == 1)
                    or (signal == 1 and direction == -1)
                )
                if should_close:
                    fill = self._fill(open_p, direction, slip, is_entry=False)
                    pnl = qty * (fill - entry_price) * direction
                    fee = qty * fill * self.taker_fee
                    balance += pnl - fee
                    total_fees += fee
                    total_slippage_cost += qty * open_p * slip
                    total_volume += qty * fill

                    current_trade.update({
                        'exit_time': ts,
                        'exit_price': fill,
                        'pnl': pnl - fee,
                        'exit_reason': 'SIGNAL_EXIT',
                    })
                    trades.append(current_trade)
                    current_trade = None
                    qty, direction, entry_price = 0.0, 0, 0.0

            # 신호 진입 (반전이면 위에서 청산된 직후 같은 봉에서 바로 재진입 —
            # 라이브의 close_position → open_position 순서와 같다)
            if direction == 0 and i != cooldown_bar:
                want = 0
                if signal == 1:
                    want = 1
                elif signal == -1 and is_futures:
                    want = -1

                if want != 0:
                    equity = balance
                    notional = equity * alloc * leverage
                    if rv is not None and np.isfinite(rv[i]) and rv[i] > 1e-9:
                        # 위험 기여도를 일정하게. 과도한 레버리지를 막기 위해 제한.
                        notional *= float(np.clip(target_vol / rv[i], 0.2, 3.0))
                    fill = self._fill(open_p, want, slip, is_entry=True)
                    raw_qty = notional / fill
                    new_qty = round_qty(raw_qty, self.spec)

                    if new_qty > 0 and is_tradable(new_qty, fill, self.spec):
                        fee = new_qty * fill * self.taker_fee
                        balance -= fee
                        total_fees += fee
                        total_slippage_cost += new_qty * open_p * slip
                        total_volume += new_qty * fill

                        qty = new_qty
                        direction = want
                        entry_price = fill
                        entry_leverage = leverage
                        bar_atr = atr_arr[i] if np.isfinite(atr_arr[i]) else None
                        if atr_sl_mult and bar_atr:
                            sl_price = entry_price - direction * bar_atr * atr_sl_mult
                        else:
                            sl_price = entry_price * (1 - sl_pct * direction)
                        if atr_tp_mult and bar_atr:
                            tp_price = entry_price + direction * bar_atr * atr_tp_mult
                        else:
                            tp_price = entry_price * (1 + tp_pct * direction)

                        # 청산가: cross면 계좌 전체가 증거금, isolated면 배정분만
                        wb = balance if self.margin_mode == 'cross' else (qty * entry_price / leverage)
                        liq_price = liquidation_price(entry_price, qty, direction, wb, self.margin_mode)

                        current_trade = {
                            'entry_time': ts,
                            'entry_price': entry_price,
                            'direction': 'LONG' if direction == 1 else 'SHORT',
                            'size': qty,
                            'leverage': leverage,
                            'regime': regimes[i],
                            'entry_bar': i,
                        }

            # ── (B) 봉 진행 중: 청산 → SL → TP 순으로 검사 ──────────────────
            # 진입한 봉에서도 검사한다. 거래소에 STOP_MARKET / TAKE_PROFIT_MARKET
            # 주문이 체결 즉시 올라가므로 진입 봉부터 발동할 수 있다.
            if direction != 0:
                bars_in_position += 1
                exit_trigger = None
                trigger_price = 0.0

                hit_liq = (low_p <= liq_price) if direction == 1 else (high_p >= liq_price)
                hit_sl = (low_p <= sl_price) if direction == 1 else (high_p >= sl_price)
                hit_tp = (high_p >= tp_price) if direction == 1 else (low_p <= tp_price)

                if hit_liq:
                    exit_trigger, trigger_price = 'LIQUIDATION', liq_price
                elif hit_sl:
                    # 봉 안에서 SL·TP가 둘 다 닿으면 SL을 먼저 본다 (보수적)
                    exit_trigger, trigger_price = 'STOP_LOSS', sl_price
                elif hit_tp:
                    exit_trigger, trigger_price = 'TAKE_PROFIT', tp_price

                if exit_trigger:
                    # 스탑 주문은 트리거 후 시장가로 나가므로 슬리피지를 맞는다
                    fill = self._fill(trigger_price, direction, slip, is_entry=False)
                    pnl = qty * (fill - entry_price) * direction
                    fee = qty * fill * self.taker_fee

                    if exit_trigger == 'LIQUIDATION':
                        # 청산이면 증거금을 전부 잃는다 (잔고는 음수가 될 수 없다)
                        balance = max(0.0, balance + pnl - fee)
                    else:
                        balance += pnl - fee

                    total_fees += fee
                    total_slippage_cost += qty * trigger_price * slip
                    total_volume += qty * fill

                    current_trade.update({
                        'exit_time': ts,
                        'exit_price': fill,
                        'pnl': pnl - fee,
                        'exit_reason': exit_trigger,
                    })
                    trades.append(current_trade)
                    current_trade = None
                    qty, direction, entry_price = 0.0, 0, 0.0
                    cooldown_bar = i     # 같은 봉 재진입 금지 (위 주석 참고)

            # ── (C) 펀딩 정산 ────────────────────────────────────────────────
            # 이 봉 구간에 8시간 정산 시점이 들어 있으면 그때의 실제 요율로 지불/수취
            if direction != 0:
                notional = qty * close_p
                funding = self.funding_lookup(ts, ts + tf_ms, notional, direction)
                balance -= funding
                total_funding += funding

            # ── (D) 자본 곡선 기록 ──────────────────────────────────────────
            unrealized = qty * (close_p - entry_price) * direction if direction != 0 else 0.0
            equity_history[i] = balance + unrealized

        # 마지막에 포지션이 남아 있으면 종가로 청산해 기록에 남긴다
        if direction != 0 and current_trade is not None:
            last_close = closes[-1]
            slip = self._slip(atr_arr[-1] / last_close if np.isfinite(atr_arr[-1]) else 0.0)
            fill = self._fill(last_close, direction, slip, is_entry=False)
            pnl = qty * (fill - entry_price) * direction
            fee = qty * fill * self.taker_fee
            balance += pnl - fee
            total_fees += fee
            current_trade.update({
                'exit_time': int(stamps[-1]),
                'exit_price': fill,
                'pnl': pnl - fee,
                'exit_reason': 'END_OF_DATA',
            })
            trades.append(current_trade)
            equity_history[-1] = balance

        df_equity = pd.DataFrame({'timestamp': stamps, 'equity': equity_history})
        df_equity['datetime'] = pd.to_datetime(df_equity['timestamp'], unit='ms')

        metrics = self._calculate_metrics(df_equity, trades, df)
        metrics.update({
            'total_fees': total_fees,
            'total_funding': total_funding,
            'total_slippage_cost': total_slippage_cost,
            'total_volume': total_volume,
            'exposure_pct': bars_in_position / len(df) if len(df) else 0.0,
            'cost_multiplier': self.cost_multiplier,
        })

        # 연 회전율 = 연환산 거래대금 / 초기자본
        days = max((df_equity['datetime'].iloc[-1] - df_equity['datetime'].iloc[0]).total_seconds() / 86400, 1.0)
        metrics['annual_turnover'] = (total_volume / self.initial_capital) * (365.25 / days)

        return metrics, df_equity, trades

    # ── 성과 지표 ────────────────────────────────────────────────────────────

    def _calculate_metrics(self, df_equity, trades, df_ohlcv):
        equity = df_equity['equity']
        returns = equity.pct_change().fillna(0.0)
        df_equity['returns'] = returns

        final_equity = float(equity.iloc[-1])
        total_return = final_equity / self.initial_capital - 1

        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        df_equity['drawdown'] = drawdown
        max_drawdown = float(drawdown.min())

        if len(df_equity) > 1:
            span = df_equity['datetime'].iloc[-1] - df_equity['datetime'].iloc[0]
            days = max(span.total_seconds() / 86400, 1.0)
            # 자본이 0이면 CAGR은 -100%
            cagr = (final_equity / self.initial_capital) ** (365.25 / days) - 1 if final_equity > 0 else -1.0
            bar_hours = (df_equity['datetime'].iloc[1] - df_equity['datetime'].iloc[0]).total_seconds() / 3600.0
            bars_per_year = (365.25 * 24.0) / max(bar_hours, 0.001)
        else:
            cagr, bars_per_year = 0.0, 365.25

        mean_r, std_r = returns.mean(), returns.std()
        sharpe = float((mean_r / std_r) * np.sqrt(bars_per_year)) if std_r > 0 else 0.0

        downside = returns[returns < 0]
        dstd = downside.std()
        sortino = float((mean_r / dstd) * np.sqrt(bars_per_year)) if dstd > 0 else 0.0

        calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0

        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) <= 0]
        n = len(trades)
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

        # Buy & Hold 비교
        bh_return = float(df_ohlcv['close'].iloc[-1] / df_ohlcv['close'].iloc[0] - 1)

        # 청산 사유별 분포
        reasons = {}
        for t in trades:
            reasons[t.get('exit_reason', '?')] = reasons.get(t.get('exit_reason', '?'), 0) + 1

        avg_hold_hours = 0.0
        if trades:
            spans = [(t['exit_time'] - t['entry_time']) / 3600000.0 for t in trades if 'exit_time' in t]
            avg_hold_hours = float(np.mean(spans)) if spans else 0.0

        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'buy_hold_return': bh_return,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'total_trades': n,
            'win_rate': len(wins) / n if n else 0.0,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_hold_hours': avg_hold_hours,
            'exit_reasons': reasons,
        }


# 구버전 이름으로 임포트하던 코드 호환
BinanceFuturesBacktester = Backtester
