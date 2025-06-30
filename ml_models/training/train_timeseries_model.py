# /ml_models/training/train_timeseries_model.py

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Darts: 시계열 예측을 위한 전문 라이브러리
from darts import TimeSeries
from darts.models import TFTModel, NBEATSModel # Transformer, N-BEATS 등 최신 모델 사용
from darts.dataprocessing.transformers import Scaler
# from darts.utils.timeseries_split import train_test_split

# 시스템의 다른 모듈에서 DB 연결 및 유틸리티 함수를 불러옵니다.
# import sys
# sys.path.append('.')
from data.database import get_database_engine, load_data_from_db
import config

# 로깅 기본 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def prepare_timeseries_data(db_engine, metric_name: str, group_by_col: str) -> pd.DataFrame:
    """
    모델 훈련에 필요한 시계열 데이터를 DB에서 불러오고 Darts 형식으로 변환합니다.
    """
    logger.info(f"'{group_by_col}'별 '{metric_name}' 시계열 데이터 준비를 시작합니다...")
    
    # 가정: 'historical_sector_metrics' 테이블에 날짜별, 그룹별 집계 지표가 저장되어 있음.
    # 이 테이블은 별도의 스크립트를 통해 주기적으로 생성되어야 합니다.
    query = f"""
    SELECT date, "{group_by_col}", "{metric_name}"
    FROM historical_sector_metrics
    ORDER BY date ASC;
    """
    
    df = load_data_from_db(query, db_engine)
    if df.empty:
        logger.error("시계열 훈련용 데이터를 찾을 수 없습니다.")
        return pd.DataFrame()

    # Darts TimeSeries 객체로 변환
    # 'group_by_col'을 기준으로 여러 시계열을 한 번에 처리
    series = TimeSeries.from_group_dataframe(
        df,
        time_col='date',
        group_cols=group_by_col,
        value_cols=metric_name,
        freq='MS' # 월초(Month Start) 데이터라고 가정
    )

    logger.info(f"총 {len(series)}개의 시계열 데이터를 Darts 형식으로 변환했습니다.")
    return series

def train_timeseries_model(target_metric: str = 'roe', group_by: str = 'cluster_label'):
    """
    시계열 예측 모델(Darts + PyTorch)의 전체 훈련 파이프라인.
    """
    logger.info(f"--- 시계열 예측 모델 훈련을 시작합니다 (Target: {target_metric}) ---")
    
    db_engine = get_database_engine()
    if db_engine is None: return

    # --- 1. 데이터 준비 ---
    all_series = prepare_timeseries_data(db_engine, target_metric, group_by)
    if not all_series: return

    # 데이터 정규화 (스케일링)
    scaler = Scaler()
    series_scaled = scaler.fit_transform(all_series)

    # 훈련/검증 데이터셋 분리 (마지막 12개월을 검증용으로 사용)
    train_set, val_set = [], []
    for s in series_scaled:
        try:
            train, val = s.split_after(pd.Timestamp.now() - pd.DateOffset(months=12))
            train_set.append(train)
            val_set.append(val)
        except ValueError: # 시계열이 너무 짧은 경우
            train_set.append(s)

    logger.info(f"데이터 분할 완료: 훈련 {len(train_set)}개, 검증 {len(val_set)}개 시계열")

    # --- 2. 모델 정의 (N-BEATS 모델 예시) ---
    # N-BEATS: 해석 가능하고, 사전 특징 공학이 거의 필요 없는 강력한 모델
    # Darts는 내부적으로 PyTorch를 사용합니다.
    model = NBEATSModel(
        input_chunk_length=24,  # 입력으로 사용할 과거 데이터 길이 (24개월)
        output_chunk_length=12, # 예측할 미래 데이터 길이 (12개월)
        n_epochs=50,
        random_state=42,
        generic_architecture=True, # 일반적인 시계열에 더 잘 작동
        num_stacks=10,
        num_blocks=1,
        num_layers=4,
        layer_widths=512,
        # GPU 사용 설정 (사용 가능한 경우)
        # pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]}
    )

    # --- 3. 모델 훈련 ---
    logger.info("N-BEATS 모델 훈련을 시작합니다...")
    # 여러 시계열을 한 번에 학습
    model.fit(
        series=train_set,
        val_series=val_set,
        verbose=True
    )
    logger.info("모델 훈련이 완료되었습니다.")
    
    # (선택사항) 검증 데이터에 대한 백테스트 및 성능 평가
    # backtest_results = model.historical_forecasts(val_set, start=0.5, forecast_horizon=12)
    # ... MAPE, sMAPE 등 성능 지표 계산 ...

    # --- 4. 훈련된 모델 저장 ---
    model_dir = Path(config.ML_MODELS_DIR) / 'saved_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_filename = f"timeseries_forecast_model_{target_metric}.pt"
    model_path = model_dir / model_filename
    
    try:
        model.save(model_path)
        logger.info(f"훈련된 시계열 모델을 성공적으로 저장했습니다: {model_path}")
    except Exception as e:
        logger.error(f"모델 저장 중 오류 발생: {e}")

    logger.info("--- 모든 시계열 모델 훈련 과정이 완료되었습니다. ---")

if __name__ == "__main__":
    # 예시: 클러스터 그룹별 평균 ROE를 예측하는 모델 훈련
    train_timeseries_model(target_metric='roe', group_by='cluster_label')
