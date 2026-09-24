"""
research/binance_env.py — 바이낸스 USDⓈ-M 선물의 체결 환경 모델.

백테스터가 실제 거래소와 어긋나던 지점들을 여기 모아 둔다.

  * 계약 스펙 (틱사이즈 / 스텝사이즈 / 최소명목가) — 실측값 사용
  * 유지증거금 브래킷 (MMR) 과 청산가 공식 — 기존의 0.05/leverage 근사 대체
  * 펀딩비 — 8시간마다 실제 이력 요율로 정산
  * 시장가 슬리피지 — 고정값이 아니라 ATR 비례

거래는 전량 시장가(taker)로 한다. 지정가 추격/미체결 경로는 제거했으므로
수수료는 진입·청산 모두 taker 단일 요율이다.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.paths import FUNDING_DIR

# ─────────────────────────────────────────────────────────────────────────────
# 수수료
# ─────────────────────────────────────────────────────────────────────────────
# 바이낸스 USDⓈ-M 선물 일반 등급(VIP0, BNB 할인 없음)
TAKER_FEE = 0.0004   # 0.04% — 데모 계좌의 실제 청구 요율 (보수적 기본 프로필)
MAKER_FEE = 0.0002   # 0.02% — 현재 전략은 쓰지 않는다(참고용)

# 실계좌(VIP0) 요율은 taker 0.05%다 (ccxt 공개 스펙, 2026-09 확인).
# 데모 계좌는 0.04%를 청구한다. 실계좌를 전제로 한 평가에는 아래 값을 쓴다.
REAL_TAKER_FEE = 0.0005


# ─────────────────────────────────────────────────────────────────────────────
# 계약 스펙 (2026-09-18 load_markets() 실측)
# ─────────────────────────────────────────────────────────────────────────────
# 주의: 최소명목가는 BTC 50 USDT 다. 계획 문서의 "5 USDT"는 틀린 값이었다.
CONTRACT_SPECS = {
    'BTC/USDT': {
        'tick_size': 0.1,
        'step_size': 0.001,
        'min_qty': 0.001,
        'min_notional': 50.0,
        'max_market_qty': 120.0,
    },
    'ETH/USDT': {
        'tick_size': 0.01,
        'step_size': 0.001,
        'min_qty': 0.001,
        'min_notional': 20.0,
        'max_market_qty': 2000.0,
    },
}

DEFAULT_SPEC = CONTRACT_SPECS['BTC/USDT']


def get_spec(symbol):
    return CONTRACT_SPECS.get(symbol.replace(':USDT', ''), DEFAULT_SPEC)


def round_qty(qty, spec):
    """스텝사이즈로 내림. 거래소는 초과 정밀도를 버린다."""
    step = spec['step_size']
    return float(np.floor(abs(qty) / step) * step) * (1 if qty >= 0 else -1)


def round_price(price, spec):
    """틱사이즈로 반올림."""
    tick = spec['tick_size']
    return float(round(price / tick) * tick)


def is_tradable(qty, price, spec):
    """거래소가 받아줄 주문인지. 최소수량·최소명목가를 모두 만족해야 한다."""
    qty = abs(qty)
    if qty < spec['min_qty']:
        return False
    if qty * price < spec['min_notional']:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 유지증거금 브래킷 (BTCUSDT USDⓈ-M)
# ─────────────────────────────────────────────────────────────────────────────
# (명목가 상한, 유지증거금률, 유지증거금 공제액) — 바이낸스 공개 브래킷 표
MMR_BRACKETS_BTC = [
    (50_000,        0.0040,         0.0),
    (600_000,       0.0050,        50.0),
    (3_000_000,     0.0100,      3_050.0),
    (12_000_000,    0.0200,     33_050.0),
    (70_000_000,    0.0500,    393_050.0),
    (100_000_000,   0.1000,  3_893_050.0),
    (230_000_000,   0.1250,  6_393_050.0),
    (480_000_000,   0.1500, 11_143_050.0),
    (600_000_000,   0.2500, 59_143_050.0),
    (float('inf'),  0.5000, 209_143_050.0),
]


def mmr_for_notional(notional):
    """명목가가 속한 브래킷의 (유지증거금률, 공제액)."""
    for cap, rate, deduction in MMR_BRACKETS_BTC:
        if notional <= cap:
            return rate, deduction
    return MMR_BRACKETS_BTC[-1][1], MMR_BRACKETS_BTC[-1][2]


def liquidation_price(entry_price, qty, direction, wallet_balance, margin_mode='cross'):
    """바이낸스 공식에 따른 청산가.

    기존 백테스터는 `유지증거금 = 명목가 × 0.05 / leverage` 라는 근거 없는 근사를
    썼다. 그건 레버리지가 높을수록 유지증거금이 *작아지는* 방향이라 부호부터
    틀렸다. 여기서는 실제 브래킷을 쓴다.

    Args:
        entry_price: 진입가
        qty: 계약 수량 (양수)
        direction: 1=롱, -1=숏
        wallet_balance: 증거금으로 잡히는 잔고.
            cross면 계좌 전체 잔고, isolated면 해당 포지션에 배정한 증거금.
        margin_mode: 'cross' | 'isolated'

    Returns:
        float: 청산가. 도달 불가하면 롱은 0.0, 숏은 inf.
    """
    qty = abs(qty)
    if qty <= 0:
        return 0.0 if direction == 1 else float('inf')

    notional = qty * entry_price
    mmr, deduction = mmr_for_notional(notional)

    # Liq = (WB + cumB - Side * Position * EntryPrice) / (Position * MMR - Side * Position)
    if direction == 1:
        numerator = wallet_balance + deduction - qty * entry_price
        denominator = qty * mmr - qty
    else:
        numerator = wallet_balance + deduction + qty * entry_price
        denominator = qty * mmr + qty

    if denominator == 0:
        return 0.0 if direction == 1 else float('inf')

    liq = numerator / denominator
    if direction == 1:
        return max(liq, 0.0)
    return liq if liq > 0 else float('inf')


# ─────────────────────────────────────────────────────────────────────────────
# 슬리피지 모델
# ─────────────────────────────────────────────────────────────────────────────
# 고정 0.02%는 조용한 장과 급변동 장을 같게 본다. 실제 시장가 체결 비용은
# (스프레드 절반) + (변동성에 비례하는 충격) 이다.
SLIPPAGE_BASE = 0.0001      # 1bp — BTCUSDT 퍼프의 반스프레드 + 상시 충격
SLIPPAGE_ATR_COEF = 0.02    # ATR이 가격의 2%면 +4bp → 합계 5bp

# ── 현실 프로필 ────────────────────────────────────────────────────────────────
# 2026-09 실측: BTCUSDT 선물 스프레드 0.01bp, 시장가 $5만 매수 시 충격 0.01bp,
# $50만도 0.10bp. 위 보수 프로필(1d 기준 편도 7.8bp)은 이 계좌 규모에서
# 수백 배 비관적이다. 남는 실제 비용은 신호 확인 후 30초 폴링 사이의 가격 변화와
# 봇측 손절의 체결 지연 정도라, 작은 ATR 비례항만 남긴다.
REALISTIC_SLIPPAGE_BASE = 0.0001    # 1bp
REALISTIC_SLIPPAGE_ATR_COEF = 0.005 # ATR 3%면 +1.5bp


def slippage_frac(atr_pct, base=SLIPPAGE_BASE, coef=SLIPPAGE_ATR_COEF):
    """해당 캔들에서 시장가 체결이 불리하게 밀리는 비율.

    Args:
        atr_pct: ATR / 가격 (예: 0.02 = 2%)
        base: 하한 (반스프레드)
        coef: ATR 비례 계수
    """
    if atr_pct is None or not np.isfinite(atr_pct) or atr_pct < 0:
        atr_pct = 0.0
    return base + coef * atr_pct


# ─────────────────────────────────────────────────────────────────────────────
# 펀딩비
# ─────────────────────────────────────────────────────────────────────────────
FUNDING_INTERVAL_MS = 8 * 60 * 60 * 1000


def funding_cache_path(symbol):
    return FUNDING_DIR / f"funding_{symbol.replace('/', '-').replace(':', '')}.json"


def download_funding_history(symbol='BTC/USDT', since_ms=None, exchange=None):
    """바이낸스에서 펀딩 요율 전 이력을 페이징으로 받아 캐시에 저장한다.

    income 조회를 기간 인자 없이 호출해 최근 7일만 받아오던 것과 같은 실수를
    막으려고, 여기서는 반드시 since를 밀어가며 끝까지 받는다.
    """
    import ccxt

    if exchange is None:
        exchange = ccxt.binanceusdm({'enableRateLimit': True})

    if since_ms is None:
        # 넉넉히 4년 전부터
        since_ms = int(time.time() * 1000) - 4 * 365 * 24 * 60 * 60 * 1000

    rows = []
    cursor = since_ms
    now_ms = int(time.time() * 1000)

    while cursor < now_ms:
        batch = exchange.fetch_funding_rate_history(symbol, since=cursor, limit=1000)
        if not batch:
            break
        for r in batch:
            rows.append({'timestamp': int(r['timestamp']), 'rate': float(r['fundingRate'])})
        last_ts = int(batch[-1]['timestamp'])
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        if len(batch) < 1000:
            break

    # 중복 제거 후 정렬
    dedup = {r['timestamp']: r['rate'] for r in rows}
    rows = [{'timestamp': ts, 'rate': dedup[ts]} for ts in sorted(dedup)]

    path = funding_cache_path(symbol)
    path.write_text(json.dumps(rows))
    return rows


def load_funding_history(symbol='BTC/USDT', auto_download=True):
    """캐시된 펀딩 이력을 DataFrame으로. 없으면 받아온다."""
    path = funding_cache_path(symbol)
    if not path.exists():
        if not auto_download:
            return pd.DataFrame(columns=['timestamp', 'rate'])
        download_funding_history(symbol)
    rows = json.loads(path.read_text())
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=['timestamp', 'rate'])
    return df.sort_values('timestamp').reset_index(drop=True)


def build_funding_lookup(symbol='BTC/USDT', flat_rate=None):
    """타임스탬프 구간 -> 펀딩 비용 계산기를 돌려준다.

    Args:
        flat_rate: None이 아니면 실제 이력 대신 이 고정 요율을 쓴다
            (민감도 분석용). 예: 0.0001 = 8시간당 0.01%

    Returns:
        callable(start_ms, end_ms, notional, direction) -> 펀딩 비용(양수=지불)
    """
    if flat_rate is not None:
        def flat_lookup(start_ms, end_ms, notional, direction):
            # 구간 안에 들어오는 8시간 정산 시점 수를 센다
            n = _count_settlements(start_ms, end_ms)
            return n * flat_rate * notional * direction
        return flat_lookup

    df = load_funding_history(symbol)
    if df.empty:
        return lambda s, e, n, d: 0.0

    ts = df['timestamp'].to_numpy()
    rates = df['rate'].to_numpy()

    def lookup(start_ms, end_ms, notional, direction):
        # (start, end] 구간에 정산된 요율들
        lo = np.searchsorted(ts, start_ms, side='right')
        hi = np.searchsorted(ts, end_ms, side='right')
        if hi <= lo:
            return 0.0
        # 롱(direction=1)은 요율이 양수면 지불, 숏은 수취
        return float(rates[lo:hi].sum()) * notional * direction

    return lookup


def _count_settlements(start_ms, end_ms):
    """(start, end] 안의 8시간 정산 시점 개수."""
    if end_ms <= start_ms:
        return 0
    first = (start_ms // FUNDING_INTERVAL_MS + 1) * FUNDING_INTERVAL_MS
    if first > end_ms:
        return 0
    return int((end_ms - first) // FUNDING_INTERVAL_MS) + 1
