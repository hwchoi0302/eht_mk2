"""
download_historical.py — 역사적 OHLCV 데이터 사전 다운로드

스윕 실행 전 3년치 데이터를 모두 로컬 SQLite에 캐싱합니다.
(run_full_sweep.py가 자동으로 호출하므로 보통 직접 실행할 필요 없음)

단독 실행:
    python download_historical.py
    python download_historical.py --days 365     # 1년치만
"""

import argparse
from datetime import datetime, timedelta
from data_manager import DataManager


def download_all(days: int = 1095):
    dm = DataManager()

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    symbols = ["BTC/USDT", "ETH/USDT"]
    timeframes = ["1h", "4h", "1d"]
    market_types = [False, True]  # 현물, 선물

    total = len(symbols) * len(timeframes) * len(market_types)
    count = 0

    print(f"\n{'='*60}")
    print(f"📥 역사적 데이터 다운로드")
    print(f"   기간: {start_date} ~ {end_date} ({days}일)")
    print(f"   총 조합: {total}개")
    print(f"{'='*60}")

    for symbol in symbols:
        for tf in timeframes:
            for is_futures in market_types:
                count += 1
                market_label = "선물" if is_futures else "현물"
                print(f"  [{count}/{total}] {symbol} ({tf}) {market_label} ...", end=" ", flush=True)
                try:
                    df = dm.get_candles(
                        symbol, tf, start_date, end_date,
                        is_futures=is_futures, force_download=True
                    )
                    if df is not None and not df.empty:
                        print(f"✅ {len(df):,}개 캔들")
                    else:
                        print("⚠️  데이터 없음")
                except Exception as e:
                    print(f"❌ 오류: {e}")

    print(f"\n✅ 다운로드 완료!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="역사적 OHLCV 데이터 다운로드")
    parser.add_argument("--days", type=int, default=1095,
                        help="다운로드할 최대 기간 (일 단위, 기본: 1095 = 3년)")
    args = parser.parse_args()
    download_all(days=args.days)
