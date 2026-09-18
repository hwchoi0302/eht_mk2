#!/usr/bin/env python3
"""
main.py — 라이브 봇 실행 엔트리포인트.

사용법:
    python main.py config/regime_config_BTC-USDT_futures_4h.json --testnet
    python main.py config/regime_config_BTC-USDT_futures_4h.json --dry-run
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live.regime_bot import main

if __name__ == "__main__":
    main()
