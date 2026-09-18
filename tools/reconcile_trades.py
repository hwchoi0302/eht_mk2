"""
tools/reconcile_trades.py — 거래소 체결 내역과 로컬 trade_logs를 대조한다.

`trading_data.db`의 `trade_logs`에 구멍이 있었다. 7/28 16:23 LONG 0.027 진입이
기록되지 않아 id 7 다음이 바로 id 8이었고, 하필 가장 큰 수익을 낸 포지션의
진입 기록이 통째로 빠져 있었다. 원인은 지정가 → 시장가 폴백 경로에서
log_trade("OPEN", ...)이 호출되지 않는 분기였다.

진입 경로를 시장가 하나로 정리하면서 기록 경로도 하나로 줄였지만, 앞으로
같은 일이 생기면 바로 알 수 있게 대조를 자동화한다. 주간 리포트에 넣어 쓴다.

    python tools/reconcile_trades.py
    python tools/reconcile_trades.py --days 90
"""

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime

import ccxt
import pandas as pd
from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
load_dotenv(dotenv_path=os.path.join(_ROOT, ".env"))

from core.paths import DB_PATH

WINDOW_MS = 7 * 24 * 60 * 60 * 1000
# 같은 체결로 볼 시간 오차. 봇의 기록 시각과 거래소 체결 시각은 조금 다르다.
MATCH_TOLERANCE_MS = 5 * 60 * 1000


def build_exchange():
    exchange = ccxt.binance({
        'apiKey': os.getenv("BINANCE_TESTNET_API_KEY"),
        'secret': os.getenv("BINANCE_TESTNET_SECRET_KEY"),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })
    try:
        exchange.enable_demo_trading(True)
    except Exception:
        exchange.set_sandbox_mode(True)
    return exchange


def fetch_exchange_fills(exchange, symbol, start_ms, end_ms):
    """거래소 체결 내역을 7일 창으로 페이징해 전부 가져온다."""
    fills = []
    window_start = start_ms
    while window_start < end_ms:
        window_end = min(window_start + WINDOW_MS, end_ms)
        batch = exchange.fetch_my_trades(symbol, since=window_start, limit=1000,
                                         params={'endTime': window_end})
        for t in batch:
            fills.append({
                'timestamp': int(t['timestamp']),
                'datetime': pd.to_datetime(t['timestamp'], unit='ms'),
                'side': t['side'].upper(),
                'price': float(t['price']),
                'amount': float(t['amount']),
            })
        window_start = window_end
        time.sleep(exchange.rateLimit / 1000.0)

    if not fills:
        return pd.DataFrame(columns=['timestamp', 'datetime', 'side', 'price', 'amount'])
    df = pd.DataFrame(fills).drop_duplicates(subset=['timestamp', 'side', 'price', 'amount'])
    return df.sort_values('timestamp').reset_index(drop=True)


def load_db_trades(start_ms, end_ms):
    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = pd.read_sql_query("SELECT * FROM trade_logs ORDER BY id", conn)
    finally:
        conn.close()
    if df.empty:
        return df

    # timestamp 컬럼 이름이 버전마다 다를 수 있어 유연하게 찾는다
    ts_col = next((c for c in ('timestamp', 'time', 'created_at') if c in df.columns), None)
    if ts_col is None:
        raise RuntimeError(f"trade_logs에 시각 컬럼이 없습니다. 컬럼: {list(df.columns)}")

    if pd.api.types.is_numeric_dtype(df[ts_col]):
        df['ts_ms'] = df[ts_col].astype('int64')
        if df['ts_ms'].max() < 1e12:      # 초 단위로 저장된 경우
            df['ts_ms'] *= 1000
    else:
        df['ts_ms'] = pd.to_datetime(df[ts_col]).astype('int64') // 10**6

    df['datetime'] = pd.to_datetime(df['ts_ms'], unit='ms')
    return df[(df['ts_ms'] >= start_ms) & (df['ts_ms'] <= end_ms)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=90)
    ap.add_argument('--symbol', default='BTC/USDT')
    args = ap.parse_args()

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - args.days * 24 * 60 * 60 * 1000

    print(f"대조 구간: 최근 {args.days}일 "
          f"({datetime.fromtimestamp(start_ms/1000).date()} ~ "
          f"{datetime.fromtimestamp(end_ms/1000).date()})\n")

    exchange = build_exchange()
    fills = fetch_exchange_fills(exchange, args.symbol, start_ms, end_ms)
    db = load_db_trades(start_ms, end_ms)

    print(f"거래소 체결 : {len(fills):,}건")
    print(f"로컬 기록   : {len(db):,}건\n")

    if fills.empty:
        print("거래소 체결이 없어 대조할 것이 없습니다.")
        return

    # 거래소 체결 하나하나에 대해 허용 오차 안의 DB 기록을 찾는다
    db_ts = db['ts_ms'].to_numpy() if not db.empty else []
    unmatched = []
    for _, fill in fills.iterrows():
        if len(db_ts) == 0:
            unmatched.append(fill)
            continue
        delta = abs(db_ts - fill['timestamp'])
        if delta.min() > MATCH_TOLERANCE_MS:
            unmatched.append(fill)

    if unmatched:
        print(f"⚠️ 로컬 기록이 없는 거래소 체결 {len(unmatched)}건:")
        for f in unmatched:
            print(f"   {f['datetime']}  {f['side']:<5} {f['amount']:>10.4f} @ {f['price']:>12,.2f}")
        print("\n   → live/trader.py의 주문 경로에서 log_trade() 누락 분기를 확인하세요.")
    else:
        print("✅ 모든 거래소 체결에 대응하는 로컬 기록이 있습니다.")

    # 반대 방향: DB에만 있는 기록 (주문이 실패했는데 기록만 남은 경우)
    if not db.empty:
        fill_ts = fills['timestamp'].to_numpy()
        orphans = []
        for _, row in db.iterrows():
            delta = abs(fill_ts - row['ts_ms'])
            if delta.min() > MATCH_TOLERANCE_MS:
                orphans.append(row)
        if orphans:
            print(f"\n⚠️ 거래소 체결이 없는 로컬 기록 {len(orphans)}건:")
            for r in orphans[:20]:
                action = r.get('action', '?')
                print(f"   {r['datetime']}  {action}")

    # id 연속성 (구멍 탐지)
    if not db.empty and 'id' in db.columns:
        ids = db['id'].to_numpy()
        gaps = [(int(ids[i]), int(ids[i + 1])) for i in range(len(ids) - 1)
                if ids[i + 1] != ids[i] + 1]
        if gaps:
            print(f"\n⚠️ trade_logs id가 건너뛴 지점 {len(gaps)}곳: {gaps[:10]}")
            print("   (구간 밖으로 잘렸을 수도 있으니 --days를 넓혀 확인하세요)")


if __name__ == '__main__':
    main()
