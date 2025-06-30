# /config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. 기본 경로 및 환경 설정
# -----------------------------------------------------------------------------
# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 프로젝트의 루트 디렉토리를 기준으로 모든 경로를 설정하여 OS 호환성을 확보합니다.
BASE_DIR = Path(__file__).resolve().parent # config.py가 루트에 있으므로 .parent 한번만

# 주요 디렉토리 경로 정의
LOG_DIR = BASE_DIR / "logs"
ML_MODELS_DIR = BASE_DIR / "ml_models"
RAW_DATA_DIR = BASE_DIR / "knowledge_base" / "data"
VECTOR_STORE_DIR = BASE_DIR / "knowledge_base" / "vector_store"

# 로그 디렉토리가 없으면 생성
LOG_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. 데이터베이스 설정
# -----------------------------------------------------------------------------
# .env 파일에서 데이터베이스 연결 문자열을 불러옵니다.
# 예: "postgresql+psycopg2://user:password@host:port/dbname"
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")


# -----------------------------------------------------------------------------
# 3. 듀얼 평가 시스템 파라미터 (분석 설정)
# -----------------------------------------------------------------------------

# --- 3.1. 동적 피어 그룹 생성 (머신러닝 클러스터링) 설정 ---
# 클러스터링에 사용할 기업의 재무적 DNA를 나타내는 특징(Feature) 목록
CLUSTERING_FEATURES = [
    'grossProfitMargin',
    'operatingMargin',
    'netProfitMargin',
    'debtEquityRatio',
    'rdToRevenue',        # R&D to Revenue
    'capexToRevenue'      # CAPEX to Revenue
]
# 엘보우 방법 등으로 미리 찾아낸 최적의 클러스터 개수
OPTIMAL_K_CLUSTERS = 8

# --- 3.2. 퀄리티 벤치마크 생성 설정 ---
# '꾸준한 성장 기업'의 평균 프로필을 계산할 때 사용할 재무 지표 목록
QUALITY_BENCHMARK_COLS = [
    'roe', 'roa', 'operatingMargin', 'netProfitMargin', 'debtToEquity'
]

# -----------------------------------------------------------------------------
# 4. 상대 가치 평가 및 랭킹 파라미터
# -----------------------------------------------------------------------------

# --- 4.1. 점수화에 사용될 재무 지표 목록 ---
# {지표명: True/False} (True: 높을수록 좋음, False: 낮을수록 좋음)
METRICS_FOR_SCORING = {
    # 수익성
    'roe': True, 'roa': True, 'operatingMargin': True,
    # 안정성
    'debtToEquity': False, 'interestCoverage': True,
    # 성장성
    'revenueGrowth': True, 'netIncomeGrowth': True,
    # 가치평가
    'peRatio': False, 'pbRatio': False, 'psRatio': False,
}

# --- 4.2. 점수화된 지표의 카테고리 분류 ---
METRIC_CATEGORIES = {
    'profitability': ['roe', 'roa', 'operatingMargin'],
    'stability': ['debtToEquity', 'interestCoverage'],
    'growth': ['revenueGrowth', 'netIncomeGrowth'],
    'valuation': ['peRatio', 'pbRatio', 'psRatio']
}

# --- 4.3. 펀더멘털 점수 산출을 위한 전략별 가중치 ---
STRATEGY_WEIGHTS = {
    'value': {
        'valuation': 0.40, 'profitability': 0.25, 'stability': 0.25, 'growth': 0.10
    },
    'growth': {
        'growth': 0.40, 'profitability': 0.30, 'valuation': 0.20, 'stability': 0.10
    },
    'blend': {
        'valuation': 0.30, 'profitability': 0.25, 'growth': 0.25, 'stability': 0.20
    }
}

# --- 4.4. 최종 종합 점수 산출을 위한 듀얼 평가 가중치 ---
# 상대 강도 점수 vs 절대 퀄리티 점수
FINAL_COMPOSITE_WEIGHTS = {
    'relative': 0.7, # 동적 피어 그룹 내 상대평가 점수 가중치
    'absolute': 0.3  # 퀄리티 벤치마크 기반 절대평가 점수 가중치
}

# -----------------------------------------------------------------------------
# 5. 기술적 분석 파라미터
# -----------------------------------------------------------------------------
TECHNICAL_ANALYSIS_PARAMS = {
    'sma_short': 20,
    'sma_long': 120,
    'ema_short': 12,
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bbands_period': 20,
    # Z-점수 계산 대상이 될 원본 지표 목록
    'z_score_columns': [
        f'RSI_{14}', # pandas-ta가 생성하는 컬럼명 형식
        f'MACD_{12}_{26}_{9}',
        # 기타 필요한 Z-점수 대상 지표 추가
    ]
}

# --- 5.1. 기술적 매력도 점수 산출을 위한 가중치 ---
TECHNICAL_SCORE_WEIGHTS = {
    'trend': 0.4,       # 추세 (예: 이동평균선 기반)
    'momentum': 0.4,    # 모멘텀 (예: RSI, MACD 기반)
    'volume': 0.2,      # 거래량 (예: OBV 기반)
}

# config.py 내에 상세 가중치 구조 정의 추가
INDICATOR_CATEGORIES = {
    'trend': [
        ('sma_signal_zscore', 1.0, True), # (Z-점수 컬럼명, 가중치, 높을수록 좋은지)
    ],
    'momentum': [
        ('RSI_14_zscore', 0.5, False), # RSI는 Z-점수가 너무 높으면 과매수이므로 False
        ('MACD_12_26_9_zscore', 0.5, True)
    ],
    'volume': [
        ('OBV_zscore', 1.0, True)
    ]
}

# -----------------------------------------------------------------------------
# 6. 리스크 평가 파라미터
# -----------------------------------------------------------------------------
RISK_SCORE_WEIGHTS = {
    'cvar': 0.40,
    'mdd': 0.30,
    'sharpe': 0.15,
    'macro_exposure': 0.15
}

# -----------------------------------------------------------------------------
# 7. 머신러닝 모델 파라미터
# -----------------------------------------------------------------------------
# 훈련된 모델이 예측 시 입력으로 사용할 특징(feature) 목록
MODEL_FEATURES = [
    'profitability_score',
    'stability_score',
    'growth_score',
    'valuation_score',
    'technical_score'
    # ... 기타 모델 학습에 사용된 특징들
]