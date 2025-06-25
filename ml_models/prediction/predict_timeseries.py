# /ml_models/prediction/predict_timeseries.py

import pandas as pd
import logging
import sys
from pathlib import Path
from typing import Optional, Any, List

# Darts: 시계열 예측을 위한 전문 라이브러리
from darts import TimeSeries
from darts.models.forecasting.nbeats import NBEATSModel # 훈련 시 사용했던 모델 클래스 임포트

# 시스템의 다른 모듈에서 DB 연결 및 유틸리티 함수를 불러옵니다.
# import sys
# sys.path.append('.')
from data.database import get_database_engine, load_data_from_db
import config

# 로깅 기본 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def load_timeseries_model(model_name: str) -> Optional[Any]:
    """
    저장된 Darts(PyTorch) 시계열 모델을 파일에서 로드합니다.

    :param model_name: 로드할 모델 파일의 이름 (예: timeseries_forecast_model_roe.pt).
    :return: 로드된 Darts 모델 객체. 실패 시 None.
    """
    model_path = Path(config.ML_MODELS_DIR) / 'saved_models' / model_name
    
    if not model_path.exists():
        logger.error(f"시계열 예측 모델을 찾을 수 없습니다: {model_path}")
        return None
        
    try:
        # Darts 모델은 해당 모델 클래스의 load 메서드를 사용해야 합니다.
        model = NBEATSModel.load(model_path)
        logger.info(f"시계열 예측 모델 '{model_name}'을 성공적으로 로드했습니다.")
        return model
    except Exception as e:
        logger.error(f"시계열 모델 로드 중 오류 발생: {e}")
        return None

def prepare_prediction_input_series(db_engine, metric_name: str, group_name: str) -> Optional[TimeSeries]:
    """
    예측을 수행할 특정 그룹의 최신 시계열 데이터를 DB에서 불러옵니다.
    모델이 예측을 위해 참고할 과거 데이터(input_chunk_length)를 준비합니다.
    """
    logger.info(f"'{group_name}' 그룹의 '{metric_name}'에 대한 예측용 입력 데이터를 준비합니다...")
    
    # 모델의 input_chunk_length를 config에서 가져옴 (훈련 시와 동일해야 함)
    input_length = getattr(config, 'TIMESERIES_INPUT_CHUNK_LENGTH', 24)
    
    # 가정: historical_sector_metrics 테이블에 데이터가 저장되어 있음
    query = f"""
    SELECT date, "{metric_name}"
    FROM historical_sector_metrics
    WHERE cluster_label = '{group_name}'  -- group_by 컬럼을 동적으로 변경할 수 있음
    ORDER BY date DESC
    LIMIT {input_length};
    """
    
    df = load_data_from_db(query, db_engine)
    
    if df.empty or len(df) < input_length:
        logger.warning(f"예측에 필요한 충분한 길이({input_length})의 시계열 데이터가 없습니다.")
        return None
        
    # 날짜를 오름차순으로 다시 정렬
    df = df.sort_values('date')

    # Darts TimeSeries 객체로 변환
    series = TimeSeries.from_dataframe(df, time_col='date', value_cols=metric_name, freq='MS')
    
    logger.info(f"{len(series)}개의 데이터 포인트로 예측 입력 시계열을 생성했습니다.")
    return series

def predict_future_metrics(model: Any, input_series: TimeSeries) -> Optional[pd.DataFrame]:
    """
    주어진 모델과 입력 시계열을 사용하여 미래 지표를 예측합니다.

    :param model: 훈련된 Darts 모델 객체.
    :param input_series: 예측에 사용할 과거 데이터 시계열.
    :return: 미래 예측값이 담긴 DataFrame.
    """
    logger.info("시계열 모델 예측을 시작합니다...")
    
    try:
        # 모델의 output_chunk_length만큼 미래를 예측
        prediction = model.predict(n=model.output_chunk_length, series=input_series)
        
        logger.info(f"{model.output_chunk_length} 기간에 대한 예측 생성이 완료되었습니다.")
        return prediction.pd_dataframe() # 결과를 pandas DataFrame으로 변환하여 반환

    except Exception as e:
        logger.error(f"시계열 모델 예측 중 오류 발생: {e}")
        return None


def main(target_metric: str = 'roe', group_name: str = '1'):
    """
    시계열 예측 파이프라인의 메인 실행 함수 (테스트용).
    """
    logger.info(f"--- '{group_name}' 그룹의 '{target_metric}'에 대한 미래 예측을 시작합니다. ---")
    
    # 1. 모델 로드
    model_filename = f"timeseries_forecast_model_{target_metric}.pt"
    model = load_timeseries_model(model_filename)
    if model is None: return

    # 2. 데이터 준비
    db_engine = get_database_engine()
    if db_engine is None: return
        
    input_series = prepare_prediction_input_series(db_engine, target_metric, group_name)
    if input_series is None: return

    # 3. 예측 실행 및 결과 확인
    forecast_df = predict_future_metrics(model, input_series)
    
    if forecast_df is not None:
        logger.info("--- 예측 결과 ---")
        print(forecast_df)
        
        # (선택사항) 예측 결과를 DB 테이블에 저장
        # forecast_df['group'] = group_name
        # save_df_to_db(...)

    logger.info("--- 시계열 예측 파이프라인이 완료되었습니다. ---")

if __name__ == "__main__":
    # 예시: 클러스터 그룹 '1'의 평균 ROE를 예측
    main(target_metric='roe', group_name='1')

