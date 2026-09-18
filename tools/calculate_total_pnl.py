"""
tools/calculate_total_pnl.py — 선물 계좌의 전 구간 손익을 원장에서 재집계한다.

구버전의 버그: `fapiPrivateGetIncome({'limit': 1000})` 처럼 기간 인자 없이
호출하면 바이낸스는 **최근 7일치만** 돌려준다. 그래서 출력되던 "누적 손익",
"Calculated Starting Balance", "Total Net Yield (ROI) since start"가 전부
7일 롤링 값이었고, 주간 리포트의 +7.45% / +7.66% / +8.68% 같은 숫자도 모두
같은 이유로 틀렸다.

여기서는 startTime/endTime을 7일 이하 창으로 밀어가며 전 구간을 페이징한다.
또 시작잔고를 현재 잔고에서 역산하지 않는다. 역산은 충전(TRANSFER)이 있으면
반드시 틀린다. 대신 TRANSFER를 따로 뽑아 충전 이력을 드러내고, ROI는
**마지막 충전 시점 이후**만 계산한다.

    python tools/calculate_total_pnl.py
    python tools/calculate_total_pnl.py --days 400
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import ccxt
import pandas as pd
from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
load_dotenv(dotenv_path=os.path.join(_ROOT, ".env"))

# 바이낸스가 한 번에 돌려주는 최대 구간. 이보다 넓게 요청하면 조용히 잘린다.
WINDOW_MS = 7 * 24 * 60 * 60 * 1000
PAGE_LIMIT = 1000


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


def fetch_all_income(exchange, start_ms, end_ms):
    """income 원장을 7일 창으로 페이징해 전부 가져온다.

    한 창 안에서도 1000건을 넘으면 잘리므로, 가득 찬 페이지는 마지막 건의
    시각부터 이어서 다시 요청한다.
    """
    rows = []
    window_start = start_ms

    while window_start < end_ms:
        window_end = min(window_start + WINDOW_MS, end_ms)
        cursor = window_start

        while True:
            batch = exchange.fapiPrivateGetIncome({
                'startTime': int(cursor),
                'endTime': int(window_end),
                'limit': PAGE_LIMIT,
            })
            if not batch:
                break
            rows.extend(batch)

            if len(batch) < PAGE_LIMIT:
                break
            # 페이지가 가득 찼다 = 더 있을 수 있다. 마지막 건 이후부터 이어받는다.
            last_time = int(float(batch[-1]['time']))
            if last_time <= cursor:
                break
            cursor = last_time + 1

        window_start = window_end
        time.sleep(exchange.rateLimit / 1000.0)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['income'] = df['income'].astype(float)
    df['time_ms'] = df['time'].astype(float).astype('int64')
    df['time'] = pd.to_datetime(df['time_ms'], unit='ms')
    # tranId 기준 중복 제거 (창 경계에서 겹칠 수 있다)
    if 'tranId' in df.columns:
        df = df.drop_duplicates(subset=['tranId'])
    return df.sort_values('time_ms').reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=400, help='거슬러 올라갈 일수')
    args = ap.parse_args()

    exchange = build_exchange()
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - args.days * 24 * 60 * 60 * 1000

    print(f"income 원장 수집: 최근 {args.days}일 "
          f"({datetime.fromtimestamp(start_ms/1000).date()} ~ "
          f"{datetime.fromtimestamp(end_ms/1000).date()})")
    print(f"  7일 창 {args.days // 7 + 1}개를 페이징합니다...\n")

    df = fetch_all_income(exchange, start_ms, end_ms)
    if df.empty:
        print("income 기록이 없습니다.")
        return

    print(f"총 {len(df):,}건 수집 "
          f"({df['time'].iloc[0]} ~ {df['time'].iloc[-1]})\n")

    print("=== 종류별 합계 (전 구간) ===")
    for inc_type, total in df.groupby('incomeType')['income'].sum().items():
        count = int((df['incomeType'] == inc_type).sum())
        print(f"  {inc_type:<18} {total:>+14,.4f} USDT  ({count:,}건)")

    # ── 충전 이력 ────────────────────────────────────────────────────────────
    transfers = df[df['incomeType'] == 'TRANSFER']
    print("\n=== 충전/출금 (TRANSFER) ===")
    if transfers.empty:
        print("  기록 없음. 계좌가 사전 충전되었거나 조회 구간 밖입니다.")
        roi_start_ms = start_ms
        roi_start_label = f"조회 시작({datetime.fromtimestamp(start_ms/1000).date()})"
        deposit_base = None
    else:
        for _, r in transfers.iterrows():
            print(f"  {r['time']}  {r['income']:>+12,.2f} USDT")
        print(f"  합계 {transfers['income'].sum():>+12,.2f} USDT")
        last = transfers.iloc[-1]
        roi_start_ms = int(last['time_ms'])
        roi_start_label = f"마지막 충전({last['time']})"
        deposit_base = float(last['income'])

    # ── ROI 기준 구간 ────────────────────────────────────────────────────────
    # 잔고에서 시작잔고를 역산하지 않는다. 충전이 섞이면 반드시 틀린다.
    since = df[df['time_ms'] > roi_start_ms]
    trading = since[since['incomeType'] != 'TRANSFER']

    realized = trading[trading['incomeType'] == 'REALIZED_PNL']['income'].sum()
    commission = trading[trading['incomeType'] == 'COMMISSION']['income'].sum()
    funding = trading[trading['incomeType'] == 'FUNDING_FEE']['income'].sum()
    net = trading['income'].sum()

    print(f"\n=== 손익 ({roi_start_label} 이후) ===")
    print(f"  실현손익   {realized:>+14,.4f} USDT")
    print(f"  수수료     {commission:>+14,.4f} USDT")
    print(f"  펀딩비     {funding:>+14,.4f} USDT")
    print(f"  ─────────────────────────────────")
    print(f"  순손익     {net:>+14,.4f} USDT")

    # ── 현재 계좌 상태 ───────────────────────────────────────────────────────
    balance = exchange.fetch_balance()
    assets = balance.get('info', {}).get('assets', [])
    usdt = next((a for a in assets if a['asset'] == 'USDT'), None)
    if not usdt:
        print("\nUSDT 자산 정보를 찾지 못했습니다.")
        return

    wallet = float(usdt.get('walletBalance', 0.0))
    unrealized = float(usdt.get('unrealizedProfit', 0.0))
    equity = float(usdt.get('marginBalance', 0.0))

    print(f"\n=== 현재 계좌 ===")
    print(f"  지갑잔고     {wallet:>14,.4f} USDT")
    print(f"  미실현손익   {unrealized:>+14,.4f} USDT")
    print(f"  총자산       {equity:>14,.4f} USDT")

    if deposit_base:
        # 정합성 검증: 마지막 충전액 + 이후 순손익 == 현재 지갑잔고 여야 한다
        expected = deposit_base + net
        drift = wallet - expected
        print(f"\n=== 정합성 검증 ===")
        print(f"  마지막 충전 {deposit_base:,.2f} + 이후 순손익 {net:+,.2f} = {expected:,.2f}")
        print(f"  실제 지갑잔고 {wallet:,.2f}  (차이 {drift:+,.4f})")
        if abs(drift) > 1.0:
            print(f"  ⚠️ 차이가 1 USDT를 넘습니다. 조회 구간 밖의 충전이 있을 수 있습니다.")

        print(f"\n=== ROI ({roi_start_label} 기준) ===")
        print(f"  실현 기준     {net / deposit_base * 100:+.2f}%")
        print(f"  미실현 포함   {(equity - deposit_base) / deposit_base * 100:+.2f}%")
    else:
        print(f"\n=== ROI ===")
        print(f"  ⚠️ 충전 이력이 조회 구간 안에 없어 기준 시작잔고를 확정할 수 없습니다.")
        print(f"     --days를 늘려 충전 시점까지 포함시키세요. "
              f"(잔고 역산은 충전이 섞이면 틀리므로 하지 않습니다)")


if __name__ == '__main__':
    main()
