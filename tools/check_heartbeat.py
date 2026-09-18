"""
tools/check_heartbeat.py — 봇이 살아 있는지 확인한다.

봇이 7/29~9/14 **47일간** 죽어 있었는데 아무도 몰랐다. 그 사이 방치된 포지션이
우연히 큰 수익을 냈고, 그게 전략의 알파로 집계됐다. 두 번 다시 없어야 한다.

live/trader.py가 루프를 돌 때마다 data/heartbeat.json을 갱신한다.
이 스크립트는 그 파일의 나이를 보고, 임계값을 넘으면 종료코드 1로 끝난다.
cron에 걸어 두고 종료코드로 알림을 띄우면 된다.

    python tools/check_heartbeat.py --max-age 300
    */5 * * * * cd ~/workspace/eht_mk2 && venv/bin/python tools/check_heartbeat.py || notify ...
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.paths import HEARTBEAT_FILE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-age', type=int, default=300,
                    help='이 초를 넘겨 정체하면 실패로 본다 (기본 300초 = 루프 주기 30초의 10배)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    if not HEARTBEAT_FILE.exists():
        print(f"🚨 하트비트 파일이 없습니다: {HEARTBEAT_FILE}")
        print("   봇이 한 번도 기동하지 않았거나, 기동 직후 죽었습니다.")
        return 1

    try:
        hb = json.loads(HEARTBEAT_FILE.read_text())
    except Exception as e:
        print(f"🚨 하트비트 파일을 읽을 수 없습니다: {e}")
        return 1

    age = time.time() - float(hb.get('timestamp', 0))

    if age > args.max_age:
        print(f"🚨 봇이 정체되었습니다. 마지막 확인 {age/60:.1f}분 전 "
              f"({hb.get('datetime', '?')})")
        print(f"   임계값 {args.max_age/60:.1f}분 초과. "
              f"{hb.get('symbol')} {hb.get('timeframe')} / {hb.get('strategy')}")
        return 1

    if not args.quiet:
        print(f"✅ 정상. 마지막 확인 {age:.0f}초 전 ({hb.get('datetime', '?')})")
        print(f"   {hb.get('symbol')} {hb.get('timeframe')} / 전략 {hb.get('strategy')}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
