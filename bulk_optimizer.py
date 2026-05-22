"""
bulk_optimizer.py — 다중 전략·기간·시장 스윕 최적화 엔진

외부에서 직접 run_full_sweep.py 를 통해 호출합니다.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

from data_manager import DataManager
from backtester import Backtester
from indicators import add_all_indicators
from optimizer import StrategyOptimizer
from strategies import get_strategy_by_name


# ─────────────────────────────────────────────
#  병렬 처리 워커 헬퍼
# ─────────────────────────────────────────────

def _worker_run_single(db_path: str, n_trials: int, combo: dict) -> dict:
    """각 프로세스에서 호출되는 독립된 단일 최적화 실행 워커"""
    local_optimizer = BulkOptimizer(db_path=db_path, n_trials=n_trials)
    return local_optimizer._run_single(
        symbol=combo['symbol'],
        timeframe=combo['timeframe'],
        is_futures=combo['is_futures'],
        strategy_name=combo['strategy_name'],
        period_name=combo['period_name'],
        period_days=combo['period_days']
    )


class BulkOptimizer:
    """
    모든 (종목 × 타임프레임 × 마켓 × 전략 × 기간) 조합에 대해
    IS/OOS 분할 최적화 + 시장국면 귀속 분석을 수행합니다.
    """

    IS_RATIO = 0.70  # 훈련 기간 비율

    def __init__(self, db_path: str = "trading_data.db", n_trials: int = 20):
        self.db_path = db_path
        self.n_trials = n_trials
        self.data_manager = DataManager(db_path=db_path)
        self.backtester = Backtester()
        self.optimizer = StrategyOptimizer(db_path=db_path)

    # ─────────────────────────────────────────────
    #  공개 API
    # ─────────────────────────────────────────────

    def run_sweep(self, combinations: list, n_jobs: int = -1, progress_cb=None) -> list:
        """
        조합 리스트를 받아 멀티프로세스로 전체 스윕을 실행하고 결과 리스트를 반환합니다.
        n_jobs: 사용할 프로세스 수. -1 이면 (CPU Core - 1) 개를 사용합니다.
        """
        if n_jobs == -1:
            n_jobs = max(1, os.cpu_count() - 1)

        print(f"\n⚡ 병렬 최적화 스윕 시작 (프로세스 수: {n_jobs})")

        all_results = []
        total = len(combinations)

        # ProcessPoolExecutor로 작업을 분할 실행
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # combo dict 리스트를 맵핑하여 submit
            futures = {
                executor.submit(_worker_run_single, self.db_path, self.n_trials, combo): combo
                for combo in combinations
            }

            for idx, future in enumerate(as_completed(futures), 1):
                combo = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    import traceback
                    result = {
                        **combo,
                        'error': f"Process execution failed: {e}\n{traceback.format_exc()}",
                        'best_params': {},
                        'score': -999.0,
                        'is_metrics': {},
                        'oos_metrics': {},
                        'full_metrics': {},
                        'regime_attribution': {},
                    }

                all_results.append(result)

                # 로그 출력
                symbol = combo['symbol']
                tf = combo['timeframe']
                is_futures = combo['is_futures']
                strat_name = combo['strategy_name']
                period_name = combo['period_name']
                market_label = "선물" if is_futures else "현물"

                label = f"[{idx}/{total}] {strat_name} | {symbol} {tf} {market_label} | {period_name}"

                if result.get('error'):
                    err_first_line = result['error'].splitlines()[0] if result['error'] else "알 수 없는 오류"
                    print(f"  ❌ {label} -> 오류: {err_first_line}")
                else:
                    ism = result['is_metrics']
                    oosm = result['oos_metrics']
                    print(
                        f"  ✅ {label} -> "
                        f"IS 샤프={ism.get('sharpe_ratio', 0):.3f} "
                        f"OOS 샤프={oosm.get('sharpe_ratio', 0):.3f} "
                        f"IS 수익={ism.get('total_return', 0)*100:.1f}% "
                        f"OOS 수익={oosm.get('total_return', 0)*100:.1f}%"
                    )

                if progress_cb:
                    progress_cb(idx, total, label)

        return all_results

    # ─────────────────────────────────────────────
    #  내부 메서드
    # ─────────────────────────────────────────────

    def _run_single(self, symbol: str, timeframe: str, is_futures: bool,
                    strategy_name: str, period_name: str, period_days: int) -> dict:
        """단일 (전략, 기간) 조합에 대해 IS/OOS 최적화를 실행합니다."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # ── 전체 데이터 조회 ──
        df_full = self.data_manager.get_candles(symbol, timeframe, start_str, end_str, is_futures)
        if df_full is None or df_full.empty:
            raise ValueError(f"데이터 없음 ({symbol} {timeframe})")

        # ── IS / OOS 분할 ──
        n = len(df_full)
        if n < 50:
            raise ValueError(f"캔들 수 부족 ({n}개 < 50개)")

        is_end_idx = int(n * self.IS_RATIO)
        df_is = df_full.iloc[:is_end_idx].copy()
        df_oos = df_full.iloc[is_end_idx:].copy()

        if len(df_oos) < 20:
            raise ValueError(f"OOS 캔들 부족 ({len(df_oos)}개 < 20개)")

        is_end_str = pd.to_datetime(df_is['timestamp'].iloc[-1], unit='ms').strftime("%Y-%m-%d")
        oos_start_str = pd.to_datetime(df_oos['timestamp'].iloc[0], unit='ms').strftime("%Y-%m-%d")

        # ── IS 최적화 ──
        best_params, best_score, _ = self.optimizer.optimize(
            symbol=symbol,
            timeframe=timeframe,
            start_date_str=start_str,
            end_date_str=is_end_str,
            strategy_name=strategy_name,
            is_futures=is_futures,
            target_metric="sharpe_ratio",
            n_trials=self.n_trials,
        )

        # ── IS 백테스트 ──
        strategy_is = get_strategy_by_name(strategy_name, **best_params)
        df_is_ind = add_all_indicators(df_is)
        is_metrics, _, is_trades = self.backtester.run(df_is_ind, strategy_is, is_futures=is_futures)

        # ── OOS 백테스트 ──
        strategy_oos = get_strategy_by_name(strategy_name, **best_params)
        df_oos_ind = add_all_indicators(df_oos)
        oos_metrics, _, oos_trades = self.backtester.run(df_oos_ind, strategy_oos, is_futures=is_futures)

        # ── 전체 기간 백테스트 ──
        strategy_full = get_strategy_by_name(strategy_name, **best_params)
        df_full_ind = add_all_indicators(df_full)
        full_metrics, _, full_trades = self.backtester.run(df_full_ind, strategy_full, is_futures=is_futures)

        # ── 시장국면 귀속 분석 ──
        regime_attribution = self._calculate_regime_attribution(df_full_ind, full_trades)

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'strategy_name': strategy_name,
            'period_name': period_name,
            'period_days': period_days,
            'is_futures': is_futures,
            'start_date': start_str,
            'end_date': end_str,
            'is_end_date': is_end_str,
            'oos_start_date': oos_start_str,
            'best_params': best_params,
            'score': best_score,
            'is_metrics': is_metrics,
            'oos_metrics': oosm_metrics if 'oosm_metrics' in locals() else oos_metrics, # typo safety
            'full_metrics': full_metrics,
            'regime_attribution': regime_attribution,
            'error': None,
        }

    def _calculate_regime_attribution(self, df_ind: pd.DataFrame, trades: list) -> dict:
        """거래를 시장국면(BULL/BEAR/SIDEWAYS)으로 분류하여 성과를 귀속합니다."""
        attribution = {
            'BULL': {'pnl': 0.0, 'trades': 0, 'win_rate': 0.0},
            'BEAR': {'pnl': 0.0, 'trades': 0, 'win_rate': 0.0},
            'SIDEWAYS': {'pnl': 0.0, 'trades': 0, 'win_rate': 0.0},
        }

        if not trades or 'regime' not in df_ind.columns:
            return attribution

        # timestamp → regime 조회용 인덱스 생성
        ts_to_regime = dict(zip(df_ind['timestamp'], df_ind['regime']))

        regime_data = {reg: {'pnl': 0.0, 'wins': 0, 'total': 0}
                       for reg in ['BULL', 'BEAR', 'SIDEWAYS']}

        for trade in trades:
            entry_ts = trade.get('entry_time')
            if entry_ts is None:
                continue
            regime = ts_to_regime.get(entry_ts)
            if regime is None:
                all_ts = sorted(ts_to_regime.keys())
                diffs = [abs(t - entry_ts) for t in all_ts]
                if diffs:
                    nearest = all_ts[diffs.index(min(diffs))]
                    regime = ts_to_regime.get(nearest, 'SIDEWAYS')
                else:
                    regime = 'SIDEWAYS'

            if regime not in regime_data:
                regime = 'SIDEWAYS'

            pnl = trade.get('pnl', 0)
            regime_data[regime]['pnl'] += pnl
            regime_data[regime]['total'] += 1
            if pnl > 0:
                regime_data[regime]['wins'] += 1

        for reg, data in regime_data.items():
            total = data['total']
            attribution[reg] = {
                'pnl': round(data['pnl'], 4),
                'trades': total,
                'win_rate': data['wins'] / total if total > 0 else 0.0,
            }

        return attribution
