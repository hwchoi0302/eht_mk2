"""
tools/reset_testnet.py — 모의(테스트넷) 계좌를 깨끗한 상태로 되돌린다.

전략을 새로 얹기 전에 계좌와 로컬 기록을 같은 시점에서 다시 출발시키기 위한
도구다. 기존 원장이 섞여 있으면 새 전략의 성과를 분리해서 볼 수 없다.

하는 일:
  1. 미체결 주문 전부 취소 (남은 SL/TP가 새 포지션에 발동하는 사고 방지)
  2. 열린 포지션 전부 시장가 청산
  3. 로컬 trade_logs 비우기 + 런타임 상태 파일 제거
  4. 정리 후 잔고 보고

하지 못하는 일:
  잔고를 원래 금액으로 되돌리는 것. 바이낸스 선물 테스트넷의 잔고 충전은
  API가 아니라 웹 UI(testnet.binancefuture.com)에서만 가능하다. 스크립트가
  끝나면 안내를 출력한다.

    python tools/reset_testnet.py              # 무엇을 할지 보여주기만 한다
    python tools/reset_testnet.py --execute    # 실제로 실행
"""

import argparse
import os
import sqlite3
import sys

import ccxt
from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
load_dotenv(dotenv_path=os.path.join(_ROOT, ".env"))

from core.paths import DB_PATH, STATUS_FILE, LOCK_FILE, HEARTBEAT_FILE

SYMBOL = 'BTC/USDT'


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


def show_balance(exchange, label):
    balance = exchange.fetch_balance()
    assets = balance.get('info', {}).get('assets', [])
    usdt = next((a for a in assets if a['asset'] == 'USDT'), None)
    if not usdt:
        print(f"  {label}: USDT 자산 정보 없음")
        return None
    wallet = float(usdt.get('walletBalance', 0.0))
    equity = float(usdt.get('marginBalance', 0.0))
    unreal = float(usdt.get('unrealizedProfit', 0.0))
    print(f"  {label}: 지갑 {wallet:,.2f} / 미실현 {unreal:+,.2f} / 총자산 {equity:,.2f} USDT")
    return wallet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true',
                    help='실제로 실행한다. 없으면 무엇을 할지 보여주기만 한다.')
    ap.add_argument('--keep-db', action='store_true',
                    help='로컬 trade_logs를 지우지 않는다.')
    args = ap.parse_args()

    dry = not args.execute
    if dry:
        print("=== 예행 연습 (--execute 를 붙여야 실제로 실행됩니다) ===\n")
    else:
        print("=== 테스트넷 계좌 초기화 실행 ===\n")

    exchange = build_exchange()

    print("[현재 상태]")
    show_balance(exchange, "잔고")

    # ── 1. 미체결 주문 취소 ──────────────────────────────────────────────────
    open_orders = exchange.fetch_open_orders(SYMBOL)
    print(f"\n[1/4] 미체결 주문 {len(open_orders)}건")
    for o in open_orders:
        print(f"   {o.get('type')} {o.get('side')} {o.get('amount')} "
              f"@ {o.get('price') or o.get('stopPrice')}")
        if not dry:
            try:
                exchange.cancel_order(o['id'], SYMBOL)
                print("      → 취소됨")
            except Exception as e:
                print(f"      → 취소 실패: {e}")
    if not open_orders:
        print("   없음")

    # ── 2. 포지션 청산 ───────────────────────────────────────────────────────
    positions = [p for p in exchange.fetch_positions([SYMBOL])
                 if abs(float(p.get('contracts') or 0)) > 0]
    print(f"\n[2/4] 열린 포지션 {len(positions)}건")
    for p in positions:
        size = float(p['contracts'])
        side = p.get('side', '')
        print(f"   {side.upper()} {size} @ {p.get('entryPrice')} "
              f"(미실현 {float(p.get('unrealizedPnl') or 0):+,.2f} USDT)")
        if not dry:
            try:
                close_side = 'sell' if side == 'long' else 'buy'
                exchange.create_market_order(
                    SYMBOL, close_side, abs(size),
                    params={'reduceOnly': True, 'recvWindow': 10000})
                print("      → 시장가 청산됨")
            except Exception as e:
                print(f"      → 청산 실패: {e}")
    if not positions:
        print("   없음")

    # ── 3. 로컬 상태 정리 ────────────────────────────────────────────────────
    print("\n[3/4] 로컬 기록")
    if args.keep_db:
        print("   trade_logs 유지 (--keep-db)")
    else:
        conn = sqlite3.connect(str(DB_PATH))
        try:
            n = conn.execute("SELECT COUNT(*) FROM trade_logs").fetchone()[0]
            print(f"   trade_logs {n}건")
            if not dry:
                conn.execute("DELETE FROM trade_logs")
                conn.execute("DELETE FROM sqlite_sequence WHERE name='trade_logs'")
                conn.commit()
                print("      → 비움 (candles 캔들 캐시는 그대로 둡니다)")
        finally:
            conn.close()

    for path in (STATUS_FILE, LOCK_FILE, HEARTBEAT_FILE):
        if path.exists():
            print(f"   {path.name}")
            if not dry:
                path.unlink()
                print("      → 삭제됨")

    # ── 4. 결과 ──────────────────────────────────────────────────────────────
    print("\n[4/4] 정리 후 상태")
    wallet = show_balance(exchange, "잔고")

    if dry:
        print("\n예행 연습이었습니다. 실제로 실행하려면 --execute 를 붙이세요.")
        return

    print("\n" + "=" * 62)
    print("계좌 정리 완료.")
    if wallet is not None:
        print(f"현재 지갑잔고: {wallet:,.2f} USDT")
    print()
    print("잔고를 특정 금액으로 맞추려면 웹에서 충전해야 합니다 —")
    print("테스트넷 잔고 충전은 API로 불가능합니다:")
    print("   https://testnet.binancefuture.com  →  로그인  →  잔고 리셋/충전")
    print()
    print("충전 후 봇을 다시 띄우세요:")
    print("   ./scripts/run_with_loop.sh config/regime_config_BTC-USDT_futures_4h.json")
    print("=" * 62)


if __name__ == '__main__':
    main()
