#!/bin/bash
# 데스크탑에서 전수 탐색 파이프라인을 끝까지 돌린다.
#
# ⚠️ 반드시 tmux 안에서 실행할 것. WSL은 ssh 세션이 끊기면 백그라운드 프로세스를
#    정리해 버린다. nohup만으로는 살아남지 못한다 (실제로 다운로드가 그렇게 죽었다).
#
#   tmux new-session -d -s search 'bash scripts/desktop_search.sh'
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY=./venv/bin/python
mkdir -p logs data reports

log() { echo "[$(date '+%F %T')] $*"; }

log "=== 1) OHLCV 다운로드 ==="
$PY -u research/download_historical.py --days 1200 2>&1 | tail -40
log "OHLCV 완료 (exit=$?)"

log "=== 2) 펀딩 이력 선다운로드 ==="
# 워커 12개가 동시에 같은 캐시를 받으려 하면 경쟁 상태가 된다. 미리 받아 둔다.
$PY -u -c "
import sys; sys.path.insert(0,'.')
from research.binance_env import download_funding_history, load_funding_history
for s in ('BTC/USDT','ETH/USDT'):
    download_funding_history(s)
    print(f'  펀딩 {s}: {len(load_funding_history(s))}건', flush=True)
"
log "펀딩 완료"

log "=== 3) 1단계: 국면별 전수 탐색 ==="
$PY -u research/regime_search.py stage1 \
    --timeframes "${TIMEFRAMES:-1h,2h,4h,6h,8h,12h,1d,3d}" \
    --workers "${WORKERS:-12}"
log "1단계 완료 (exit=$?)"

log "=== 4) 2단계: 조합 재검증 (참고용, 누수 있음) ==="
$PY -u research/regime_search.py stage2 --workers "${WORKERS:-12}"
log "2단계 완료 (exit=$?)"

log "=== 5) 3단계: 중첩 워크포워드 (최종 근거) ==="
# 2단계는 전략 선택을 아웃샘플 성적으로 하고 파라미터도 전 폴드 합의로 뽑아서
# 미래 참조가 섞여 있다. 3단계는 폴드마다 인샘플만 보고 다시 고른다.
$PY -u research/regime_search.py stage3 --workers "${WORKERS:-12}"
log "3단계 완료 (exit=$?)"

log "=== 전체 파이프라인 종료 ==="
touch reports/SEARCH_DONE
