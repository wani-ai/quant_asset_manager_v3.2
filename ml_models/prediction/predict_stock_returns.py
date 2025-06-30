

import pandas as pd
import logging
import sys
import joblib
from pathlib import Path
from typing import Optional

# 시스템의 다른 모듈에서 DB 연결 및 유틸리티 함수를 불러옵니다.
# import sys
# sys.path.append('.')
from data.database import get_database_engine, load_data_from_db
import config

# 로깅 기본 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def load_prediction_model(model_name: str = "stock_ranking_model.pkl") -> Optional[any]:
    """
    저장된 머신러닝 모델을 파일에서 로드합니다.

    :param model_name: 로드할 모델 파일의 이름.
    :return: 로드된 모델 객체. 실패 시 None.
    """
    model_path = Path(config.ML_MODELS_DIR) / 'saved_models' / model_name
    
    if not model_path.exists():
        logger.error(f"예측 모델을 찾을 수 없습니다: {model_path}")
        return None
        
    try:
        model = joblib.load(model_path)
        logger.info(f"예측 모델 '{model_name}'을 성공적으로 로드했습니다.")
        return model
    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {e}")
        return None

def prepare_prediction_data(db_engine) -> Optional[pd.DataFrame]:
    """
    예측을 수행할 최신 특징(feature) 데이터를 DB에서 불러옵니다.
    """
    logger.info("예측용 최신 특징 데이터를 준비합니다...")
    
    # 가정: 'latest_features'는 각 티커의 가장 최신 특징 데이터를 담고 있는 뷰 또는 테이블.
    # 이 데이터는 analysis/ 모듈들이 주기적으로 계산하여 업데이트해야 합니다.
    query = "SELECT * FROM latest_features"
    
    df = load_data_from_db(query, db_engine)
    
    if df.empty:
        logger.error("예측용 특징 데이터를 찾을 수 없습니다. 'analysis' 관련 스크립트가 먼저 실행되어야 합니다.")
        return None
        
    logger.info(f"총 {len(df)}개 기업의 예측용 데이터를 로드했습니다.")
    return df.set_index('ticker')


def predict_stock_returns(model, features_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    주어진 모델과 특징 데이터를 사용하여 주식의 미래 성과(점수)를 예측합니다.

    :param model: 훈련된 머신러닝 모델 객체.
    :param features_df: 예측에 사용할 특징 데이터가 담긴 DataFrame.
    :return: 티커별 예측 점수가 담긴 DataFrame.
    """
    logger.info("모델 예측을 시작합니다...")
    
    # config 파일에 정의된, 모델 훈련 시 사용했던 특징 컬럼 목록을 가져옵니다.
    model_features = config.MODEL_FEATURES
    
    # 예측용 데이터에 필요한 모든 특징이 있는지 확인
    missing_features = [f for f in model_features if f not in features_df.columns]
    if missing_features:
        logger.error(f"예측에 필요한 특징 데이터가 부족합니다: {missing_features}")
        return None

    # 모델이 학습한 특징 순서와 동일하게 맞춰서 예측 수행
    X_predict = features_df[model_features]
    
    try:
        predictions = model.predict(X_predict)
        
        result_df = pd.DataFrame({
            'ticker': X_predict.index,
            'prediction_score': predictions
        })
        
        # 예측 점수를 기준으로 내림차순 정렬 및 랭킹 부여
        result_df['rank'] = result_df['prediction_score'].rank(ascending=False, method='min')
        result_df = result_df.sort_values('rank').set_index('ticker')
        
        logger.info("모델 예측 및 랭킹 생성이 완료되었습니다.")
        return result_df

    except Exception as e:
        logger.error(f"모델 예측 중 오류 발생: {e}")
        return None

def main():
    """
    모델 예측 파이프라인의 메인 실행 함수.
    """
    logger.info("--- 주식 수익률 예측 파이프라인을 시작합니다. ---")
    
    # 1. 모델 로드
    model = load_prediction_model()
    if model is None:
        logger.error("모델 로딩 실패. 파이프라인을 중단합니다.")
        return

    # 2. 데이터베이스 연결 및 데이터 준비
    db_engine = get_database_engine()
    if db_engine is None:
        logger.error("데이터베이스 연결 실패. 파이프라인을 중단합니다.")
        return
        
    features_df = prepare_prediction_data(db_engine)
    if features_df is None or features_df.empty:
        return

    # 3. 예측 실행 및 결과 확인
    prediction_results = predict_stock_returns(model, features_df)
    
    if prediction_results is not None:
        logger.info("--- 예측 결과 (상위 20개) ---")
        print(prediction_results.head(20))
        
        # (선택사항) 예측 결과를 DB 테이블에 저장
        # save_df_to_db(prediction_results, 'daily_predictions', db_engine, if_exists='replace')

    logger.info("--- 주식 수익률 예측 파이프라인이 완료되었습니다. ---")

if __name__ == "__main__":
    main()
