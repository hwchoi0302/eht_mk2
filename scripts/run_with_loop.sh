#!/bin/bash
# 봇이 죽으면 자동 재기동한다.
# 로그의 'KST' 표기와 실제 시각을 일치시킨다 (시스템 TZ는 UTC).
export TZ=Asia/Seoul
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

# 기본값은 워크포워드 재탐색에서 유일하게 합격선을 통과한 설정이다.
# (BTC/USDT 1d 시장국면 동적결합 — reports/wfa_BTC-USDT_1d.md 참조)
# 4h 설정으로 돌리려면 인자로 경로를 넘긴다.
CONFIG="${1:-config/regime_config_BTC-USDT_futures_1d.json}"

echo "=================================================="
echo "Starting Bitcoin Regime Trading Bot (Auto-Restart Loop)"
echo "Config: $CONFIG"
echo "=================================================="

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Launching main.py..."
    "$ROOT/venv/bin/python" main.py "$CONFIG" --testnet
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Process exited (Code: $EXIT_CODE). Restarting in 10 seconds..."
    sleep 10
done
