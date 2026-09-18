"""
core/paths.py — 저장소 내 모든 경로를 한 곳에서 정의한다.

이전에는 각 스크립트가 `"trading_data.db"` 같은 상대 경로를 하드코딩해서,
실행 위치가 바뀌면 다른 파일을 열거나 새로 만들어 버렸다.
경로가 필요한 모듈은 반드시 여기서 가져다 쓴다.
"""

from pathlib import Path
import sys

# 이 파일은 <repo>/core/paths.py 이므로 부모의 부모가 저장소 루트다.
ROOT = Path(__file__).resolve().parent.parent

CORE = ROOT / "core"
LIVE = ROOT / "live"
RESEARCH = ROOT / "research"
TOOLS = ROOT / "tools"
SCRIPTS = ROOT / "scripts"

CONFIG = ROOT / "config"
DATA = ROOT / "data"
LOGS = ROOT / "logs"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

# 런타임 산출물
DB_PATH = DATA / "trading_data.db"
FUNDING_DIR = DATA / "funding"
STATUS_FILE = DATA / "bot_status.json"
LOCK_FILE = DATA / "bot.lock"
HEARTBEAT_FILE = DATA / "heartbeat.json"

REGIME_LOG = LOGS / "regime_bot.log"
TRADER_LOG = LOGS / "live_trader.log"

for _d in (CONFIG, DATA, LOGS, REPORTS, FUNDING_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def bootstrap():
    """저장소 루트를 sys.path에 넣어 `core.*` / `research.*` 임포트가 되게 한다.

    research/ 나 tools/ 의 스크립트를 `python research/foo.py` 처럼 직접 실행할 때
    파일 맨 위에서 한 번 호출한다.
    """
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def config_path(symbol: str, market: str, timeframe: str) -> Path:
    """예: config_path("BTC-USDT", "futures", "4h")"""
    return CONFIG / f"regime_config_{symbol}_{market}_{timeframe}.json"
