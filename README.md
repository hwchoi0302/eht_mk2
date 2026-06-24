# 🤖 시장국면 동적 전략 전환 자동매매 봇 (eht_mk2)

CCXT 라이브러리를 활용하여 바이낸스(Binance) 선물시장에서 실시간으로 시장 국면(상승장, 하락장, 횡보장)을 동적으로 판별하고, 각 국면에 최적화된 하위 전략으로 스위칭하며 자동매매를 수행하는 선물 매매 엔진입니다.

---

## 🛠️ 최근 변경 및 개선 사항

1. **국면 전환 컨펌 버퍼 (regime_confirm_candles = 2)**
   * 시장 국면의 전환 기준(예: EMA 50 돌파 등)이 순간적으로 발생했다가 복귀하는 **휩소(Whipsaw)에 의한 잦은 스위칭 청산 및 수수료 낭비(Fee bleed)를 원천 차단**하기 위해 도입되었습니다.
   * 3개년 전체 역사적 데이터 시뮬레이션 결과, 2개 마감 봉(8시간) 동안 국면이 유지될 때 교체하는 **2봉 버퍼가 하락장 리스크 회피 속도와 수수료 방어 측면에서 가장 이상적인 밸런스(Sweet Spot)**로 정량 확인되어 반영되었습니다.

2. **횡보장 최적 평균회귀 전략 (Z-Score Mean Reversion)**
   * 기존 횡보장(SIDEWAYS)에서 부적합하게 구동되던 추세추종형(Heikin-Ashi) 전략을 전격 폐기하고, 가격 편차를 이용해 박스권 상/하단에서 역추세 진입하는 **Z-Score 평균회귀 전략**을 도입했습니다.
   * 횡보 구간의 승률을 **40.7%에서 52.4%로 대폭 향상**시켰습니다.

3. **지정가(Limit) 진입 및 5단계 시장가(Market) 강제 체결 보장**
   * 포지션 진입 시 슬리피지를 없애고 거래 수수료를 아끼기 위해 지정가로 우선 주문을 생성합니다.
   * 30초 단위로 미체결 물량을 확인하며 최적 호가로 갱신 주문(Price Chase)을 수행하다가, 최종 5회차(약 2분 경과)까지 미체결된 잔여량은 최종 시장가로 체결하여 포지션 진입 지연 문제를 해결했습니다. (안전을 위해 포지션 청산/손절/익절/시그널 반전 시에는 즉시 시장가 청산을 유지합니다.)

4. **동일 캔들 즉시 재진입 방지 (SL/TP Cooldown)**
   * 손절/익절(SL/TP)로 인해 포지션 강제 청산이 발생했을 때, 해당 캔들이 마감되기 전까지는 동일한 시그널이 유지되더라도 재진입을 막는 쿨다운 제어 로직을 적용하여 수수료 누수를 예방합니다.

5. **파싱 에러 해결 및 데이터베이스 로깅 안정화**
   * 주문 및 청산 시 CCXT 응답 데이터의 결측에 대응하는 예외 처리를 완료하여 예외로 인한 거래 중단 문제를 극복하고 SQLite DB(`trade_logs`)의 거래 기록 신뢰도를 확보했습니다.

---

## 📁 디렉토리 구조 및 파일 설명

### 1. 코어 매매 엔진 및 스크립트
*   [`run_regime_bot.py`](run_regime_bot.py): 실거래 자동매매 봇 실행의 메인 엔트리포인트. 30초 주기로 캔들을 가져와 기술 지표 계산, 국면 판정(2-캔들 버퍼 반영), 미체결 지정가 체결 모니터링을 실시간 조율합니다.
*   [`live_trader.py`](live_trader.py): 바이낸스 선물 API 통신, 레버리지 세팅, 지정가 주문 추적/취소/갱신, 손절/익절(SL/TP) 감지, 체결 거래 로깅 등 거래 집행의 세부 기능을 구현하는 추상화 계층입니다.
*   [`strategies.py`](strategies.py): 자동매매 전략 클래스들의 라이브러리.
    *   `RegimeSwitchingStrategy`: 국면 감지 버퍼 및 하위 전략들을 제어하는 전략 결합기.
    *   `DualMomentumStrategy`: 상승장(BULL)용 절대 모멘텀 + 추세 필터 전략.
    *   `TripleEMAStrategy`: 하락장(BEAR)용 단/중/장기 EMA 크로스오버 전략.
    *   `ZScoreMeanReversionStrategy`: 횡보장(SIDEWAYS)용 Z-Score 기반 평균회귀 전략.
*   [`backtester.py`](backtester.py): 오프라인 과거 캔들 데이터에 기술적 지표 및 전략 신호를 대입하여 수익률, 최대 낙폭(MDD), 샤프 지수, 지불 수수료 등을 가상 시뮬레이션하는 테스트 엔진입니다.
*   [`indicators.py`](indicators.py): EMA, ADX, RSI, 볼린저 밴드 등의 기술 지표 계산 기능 및 시장 상태 분류 로직(`classify_market_regime`)을 담당합니다.
*   [`data_manager.py`](data_manager.py): 바이낸스 API를 통한 과거 OHLCV 데이터 수집과 로컬 SQLite 캐시 저장을 대행합니다.
*   [`optimizer.py`](optimizer.py): Optuna를 기반으로 각 단일 전략의 하이퍼파라미터 탐색 및 최적화를 조력하는 인터페이스 클래스입니다.
*   [`bulk_optimizer.py`](bulk_optimizer.py) & [`run_full_sweep.py`](run_full_sweep.py): 다중 종목, 기간, 타임프레임, 전략들을 일괄로 탐색하고 리더보드를 생성하는 대규모 매개변수 스위핑 엔진입니다.
*   [`report_generator.py`](report_generator.py): 백테스팅 결과를 리포트(Markdown 등)로 변환해 저장하는 파일 생성 모듈입니다.

### 2. 유틸리티 도구 모음 (`tools/`)
*   [`tools/verify.py`](tools/verify.py): 필수 패키지 및 커밀 모듈의 임포트 동작 상태를 자가 진단하는 검증 스크립트.
*   [`tools/check_testnet_status.py`](tools/check_testnet_status.py): 현재 선물 거래 계정 지갑 잔고, 미실현 손익, 체결 완료된 포지션 내역 및 로컬 DB 로그 현황을 요약 출력하는 진단 툴.
*   [`tools/calculate_total_pnl.py`](tools/calculate_total_pnl.py): CCXT 계정 인컴 데이터를 직접 파싱하여 누적 실현 수익, 지불한 수수료, 펀딩비 집계 및 실질 ROI를 분석하는 성과 툴.
*   [`tools/check_income.py`](tools/check_income.py): 수수료 및 펀딩비 상세 영수증 로그를 날짜별로 요약 표시해 주는 조회 도구.

### 3. 리포트 및 설정 폴더 (`reports/`)
*   [`reports/regime_config_BTC-USDT_futures_4h.json`](reports/regime_config_BTC-USDT_futures_4h.json): 봇의 실제 거래 행동 지침서. 상승장 레버리지 2배 설정 및 횡보장 Z-Score 평균회귀 파라미터가 명시되어 있으며, **2-캔들 국면 버퍼(`"regime_confirm_candles": 2`)**가 최종 세팅되어 있습니다.

---

## 🚀 구동 및 실행 안내

1. **의존성 설치 및 가상환경 구동**
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **환경변수 설정**
   루트 경로 내 `.env` 파일을 생성하고 아래와 같이 바이낸스 API 자격 증명을 작성합니다:
   ```env
   BINANCE_TESTNET_API_KEY=your_binance_testnet_api_key
   BINANCE_TESTNET_SECRET_KEY=your_binance_testnet_secret_key
   ```

3. **거래소 테스트넷(Testnet Sandbox) 실행**
   ```bash
   python run_regime_bot.py reports/regime_config_BTC-USDT_futures_4h.json --testnet
   ```
