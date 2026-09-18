# 🤖 시장국면 동적 전략 전환 자동매매 봇 (eht_mk2)

바이낸스 USDⓈ-M 선물에서 시장 국면(상승/하락/횡보)을 판별하고, 국면별 하위 전략으로
전환하며 매매하는 봇과, 그 전략을 찾기 위한 백테스트·워크포워드 연구 코드.

> **현재 상태: 페이퍼 트레이딩(테스트넷) 전용.**
> 실계좌에 붙이기 전에 `docs/REWORK_PLAN.md`의 검증 단계를 먼저 통과시킬 것.

---

## 📊 현재 권고 설정

**BTC/USDT 1d · 시장국면 동적결합** — `config/regime_config_BTC-USDT_futures_1d.json`

워크포워드 재탐색(인샘플 6개월 → 아웃샘플 2개월 롤링, 16개 구간)에서
합격선을 통과한 설정이다. 아래 수치는 **전부 아웃샘플**이다.

아웃샘플 전 구간: 2023-11-24 ~ 2026-07-24 (973일, 16개 구간)

| | 전략 | Buy & Hold |
|---|---|---|
| 누적 수익률 | +76.4% | +72.6% |
| **최대 낙폭** | **-17.1%** | **-53.0%** |
| 샤프 | +0.73 | — |
| 구간 일관성 | 16개 중 10개 수익 (62%) | — |
| 연 회전율 | 78회 | 0회 |
| 비용 2배에서 | **+45.1%** | +72.6% |

**수익률로는 Buy & Hold와 사실상 동률이다.** 차이는 낙폭에 있다 —
같은 수익을 **3분의 1 수준의 낙폭**으로 냈다. 이 전략을 쓸 이유는
"더 번다"가 아니라 "덜 깨진다"이다. 그래도 못 견디는 낙폭(-17%)이라면
이 봇을 돌릴 이유가 없다.

기존 4h 설정에서 바뀐 것은 두 가지뿐이다: **타임프레임 4h → 1d**,
**손절 4% → 8% / 익절 4% → 10%**. 국면별 하위 전략 구성은 그대로다.

같은 전략이 4h에서는 비용 2배에서 -28.6%로 뒤집힌다. 차이는 전략이 아니라
**거래 빈도**에서 온다 — 4h는 연 회전율이 100~360회라 왕복 8bp 비용이
연 8~29%의 드래그가 된다.

값어치는 **하락장 방어**에 있다. 2025~2026년 BTC가 -34.4% 빠지는 동안
+17.3%를 냈다. 반대로 강한 상승장에서는 Buy & Hold에 크게 뒤진다
(2023~2024년 +66.4% vs +125.4%).

⚠️ **주의**
- 하위 전략의 기간 파라미터(듀얼 모멘텀 46/94 등)는 4h 시절 값을 물려받았다.
  워크포워드에서 탐색한 것은 손절/익절/레버리지/배분뿐이다.
- ETH에는 전이되지 않는다 (비용 2배에서 -10.1%, MDD -48%). **BTC 전용.**
- 실계좌 전에 무중단 페이퍼 트레이딩으로 추적오차를 먼저 측정할 것.
- 3년 4개월 / 16개 구간은 통계적으로 넉넉하지 않다. 수익률이 B&H와 동률인
  상황에서 샤프 0.73의 표준오차를 감안하면, "낙폭이 낮다"는 것 외에는
  강하게 주장할 수 있는 게 많지 않다.

전체 결과는 `reports/wfa_BTC-USDT_1d.md`, 진단 기록은 `docs/REWORK_PLAN.md` 부록 A.

---

## 📁 디렉토리 구조

```
eht_mk2/
├── main.py                 라이브 봇 엔트리포인트
│
├── core/                   라이브와 리서치가 함께 쓰는 공용 모듈
│   ├── paths.py              모든 경로를 한 곳에서 정의 (하드코딩 금지)
│   ├── indicators.py         지표 계산 + 국면 판정/확정
│   ├── strategies.py         전략 클래스 16종 + 레지스트리
│   └── data_manager.py       OHLCV 수집 / SQLite 캐시
│
├── live/                   실거래 경로
│   ├── regime_bot.py         국면 감지 → 전략 스위칭 → 주문 조율
│   └── trader.py             거래소 통신, 시장가 주문, SL/TP 보호 주문
│
├── research/               백테스트·최적화 (라이브가 임포트하지 않음)
│   ├── binance_env.py        바이낸스 체결 환경 모델 (수수료/펀딩/청산/슬리피지)
│   ├── backtester.py         백테스트 엔진
│   ├── strategy_lab.py       워크포워드·파라미터 고원 탐색
│   ├── run_wfa.py            전 전략 워크포워드 스윕 실행기
│   ├── report_wfa.py         워크포워드 결과 → Markdown
│   ├── make_config.py        워크포워드 결과 → 라이브 설정 파일
│   ├── compare_timeframes.py 타임프레임별 결과 비교
│   ├── optimizer.py          Optuna 단일 전략 최적화
│   ├── bulk_optimizer.py     다중 조합 일괄 탐색
│   ├── run_full_sweep.py     대규모 스윕 + 리더보드
│   ├── report_generator.py   결과 → Markdown 리포트
│   └── download_historical.py 과거 데이터 사전 다운로드
│
├── tools/                  진단·운영 스크립트
│   ├── verify.py             설치·배선 자가 진단
│   ├── selftest_engine.py    백테스트 엔진 계산 검산
│   ├── check_heartbeat.py    봇 정체 감지 (cron용)
│   ├── reconcile_trades.py   거래소 체결 ↔ 로컬 DB 대조
│   ├── calculate_total_pnl.py 전 구간 손익 재집계
│   ├── reset_testnet.py      모의 계좌 초기화
│   ├── check_testnet_status.py 계좌 현황 요약
│   ├── backtest_live_config.py 라이브 설정 그대로 백테스트
│   └── backtest_sensitivity.py 비용 민감도 분석
│
├── config/                 regime_config_*.json (봇의 행동 지침서)
├── data/                   trading_data.db, 런타임 상태, 펀딩 캐시  [git 제외]
├── logs/                   회전 로그  [git 제외]
├── reports/                백테스트 산출물
├── scripts/run_with_loop.sh 봇 자동 재기동 래퍼
└── docs/REWORK_PLAN.md     재설계 계획 및 진단 기록
```

경로는 **절대 하드코딩하지 않는다.** 필요한 경로는 `core/paths.py`에서 가져온다.
예전에는 각 스크립트가 `"trading_data.db"` 같은 상대 경로를 직접 써서, 실행 위치가
바뀌면 다른 파일을 열거나 빈 DB를 새로 만들었다.

---

## 🚀 실행

```bash
# 1. 의존성
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt

# 2. API 키 (.env — 커밋 금지)
cat > .env << 'EOF'
BINANCE_TESTNET_API_KEY=...
BINANCE_TESTNET_SECRET_KEY=...
EOF

# 3. 드라이런으로 동작 확인 (거래소 주문 없음)
venv/bin/python main.py config/regime_config_BTC-USDT_futures_1d.json --dry-run

# 4. 테스트넷 구동 (인자를 생략하면 권고 설정인 1d를 쓴다)
./scripts/run_with_loop.sh
```

### 운영 점검

```bash
venv/bin/python tools/check_heartbeat.py     # 봇이 살아 있는가
venv/bin/python tools/check_testnet_status.py # 계좌 현황
venv/bin/python tools/reconcile_trades.py     # 체결 기록에 구멍이 없는가
venv/bin/python tools/calculate_total_pnl.py  # 전 구간 손익
venv/bin/python tools/verify.py               # 설치·배선 점검
venv/bin/python tools/selftest_engine.py      # 백테스트 엔진 검산
```

`check_heartbeat.py`를 cron에 걸어두면 봇이 멈춰도 바로 알 수 있다.
예전에 봇이 **47일간 죽어 있었는데 아무도 몰랐던** 적이 있다.

```cron
*/5 * * * * cd ~/workspace/eht_mk2 && venv/bin/python tools/check_heartbeat.py || <알림>
```

---

## 🔬 백테스트 / 전략 재탐색

```bash
# 과거 데이터 받기
venv/bin/python research/download_historical.py --days 1200

# 라이브 설정 그대로 검증
venv/bin/python tools/backtest_live_config.py

# 전 전략 워크포워드 스윕 (비용 1배 / 2배)
venv/bin/python research/run_wfa.py --symbol BTC/USDT --timeframe 1d --workers 4
venv/bin/python research/report_wfa.py --symbol BTC/USDT --timeframe 1d
venv/bin/python research/compare_timeframes.py
```

### 백테스트가 가정하는 것

`research/binance_env.py`에 모아 뒀다. 실제 거래소와 맞춰 둔 지점들:

| 항목 | 값 / 방식 |
|---|---|
| 수수료 | taker 0.04% (진입·청산 모두 시장가) |
| 펀딩비 | 8시간마다 **실제 이력 요율** 정산 (4년치 캐시) |
| 슬리피지 | 고정값이 아니라 ATR 비례 (`1bp + 0.02 × ATR%`) |
| 청산가 | 실제 유지증거금 브래킷 (BTCUSDT MMR 표) |
| 정밀도 | 틱 0.1 / 스텝 0.001 / **최소명목가 50 USDT** |
| SL/TP | 진입 봉부터 검사. 동시 도달 시 SL 우선(보수적) |
| 국면 판정 | 라이브와 **같은 함수** (`core.indicators.confirm_regimes`) |

마지막 항목이 중요하다. 예전에는 국면 확정 로직이 라이브·백테스터·전략 클래스에
**각각 따로** 있었고 기본값(3봉 vs 2봉)과 초기 상태가 달라서, 같은 데이터에서도
서로 다른 국면이 나왔다.

### 전략 선택 원칙 (`research/strategy_lab.py`)

- **워크포워드**: 인샘플 6개월에서 파라미터를 고르고, 뒤따르는 아웃샘플 2개월
  성과만 집계한다. 전 구간 최적화 후 전 구간 성과를 보는 건 과최적화를 성과로
  착각하는 것이다.
- **파라미터 고원**: 상위 30% 구간의 중앙값을 취한다. 단일 최고점은 거의 항상
  이웃이 나쁜 뾰족한 봉우리고, 다음 구간에서 무너진다.
- **비용 스트레스**: 비용 2배에서도 살아남는 설정만 후보로 둔다.
- **회전율 패널티**: 연 50회전을 넘는 만큼 점수에서 깎는다.
- 평가 지표는 총수익률이 아니라 **아웃샘플 샤프 · MDD · PF · 구간 일관성**.

---

## ⚠️ 주문 방식: 전량 시장가

진입·청산 모두 시장가(taker)다.

예전에는 지정가로 진입한 뒤 최대 4회 가격을 추격하고, 그래도 미체결이면 시장가로
폴백했다. 체결까지 최대 2분이 걸렸고, 폴백 분기에서 `log_trade("OPEN", ...)`이
빠져 DB에 기록 구멍이 생겼다. 경로를 하나로 줄여 두 문제를 함께 없앴다.

SL/TP는 진입 직후 거래소에 `STOP_MARKET` / `TAKE_PROFIT_MARKET`로 올린다.
봇이 30초 루프를 돌며 직접 감시하던 방식은 봇이 죽으면 아예 작동하지 않았다.

---

## 📌 알려진 한계

- 실계좌 검증 전이다. 무중단 페이퍼 트레이딩으로 백테스트 대비 추적오차를
  먼저 측정해야 한다.
- 봉 안에서 SL·TP가 동시에 닿았을 때의 실제 순서는 4h 봉으로는 알 수 없다.
  보수적으로 SL을 먼저 본다.
- 현재 국면 판정은 EMA50/EMA200/ADX 고정 규칙이다. 이 규칙 자체는 아직
  최적화 대상에 넣지 않았다.
- 3년 4개월 / 16개 구간은 통계적으로 넉넉하지 않다. 샤프 0.73의 표준오차가
  작지 않으니 단일 수치를 과신하지 말 것.
