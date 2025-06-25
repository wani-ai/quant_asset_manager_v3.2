# /ml_models/training/train_ranking_model.py

import pandas as pd
import numpy as np
import logging
import sys
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# 시스템의 다른 모듈에서 DB 연결 및 유틸리티 함수를 불러옵니다.
# import sys
# sys.path.append('.')
from data.database import get_database_engine, load_data_from_db
import config

# 로깅 기본 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def prepare_training_data(db_engine) -> pd.DataFrame:
    """
    모델 훈련에 필요한 특징(X)과 목표(y) 데이터를 DB에서 불러오고 통합합니다.
    """
    logger.info("모델 훈련용 데이터셋 준비를 시작합니다...")
    
    # 이 쿼리는 여러 테이블(예: 상대가치 점수, 기술적 점수)을 조인하고,
    # 미래 수익률(target)을 계산하는 복잡한 SQL이 될 수 있습니다.
    # 여기서는 'ml_training_data'라는 가상의 뷰가 준비되어 있다고 가정합니다.
    # 이 뷰는 사전에 'ticker', 'date', 'feature_1', ..., 'target_1m_return' 컬럼을 가집니다.
    query = "SELECT * FROM ml_training_data"
    
    df = load_data_from_db(query, db_engine)
    
    if df.empty:
        logger.error("훈련용 데이터를 찾을 수 없습니다. 데이터 준비 과정을 확인해주세요.")
        return pd.DataFrame()
        
    logger.info(f"총 {len(df)}개의 훈련 샘플을 로드했습니다.")
    return df

def train_ranking_model():
    """
    주식 랭킹 예측 모델(XGBoost)의 전체 훈련 파이프라인.
    """
    logger.info("--- 주식 랭킹 모델 훈련을 시작합니다. ---")
    
    db_engine = get_database_engine()
    if db_engine is None:
        logger.error("데이터베이스 엔진 연결 실패. 훈련을 중단합니다.")
        return

    # --- 1. 데이터 준비 ---
    training_df = prepare_training_data(db_engine)
    if training_df.empty:
        return

    # 특징(X)과 목표(y) 변수 분리 (config.py에서 정의)
    feature_columns = config.MODEL_FEATURES
    target_column = config.MODEL_TARGET
    
    X = training_df[feature_columns]
    y = training_df[target_column]

    # 훈련/테스트 데이터셋 분리
    # 시계열 데이터이므로, 실제로는 시간 순서에 따라 분리하는 것이 더 바람직합니다. (예: shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"데이터 분할 완료: 훈련 {len(X_train)}개, 테스트 {len(X_test)}개")

    # --- 2. 모델 훈련 (XGBoost Regressor) ---
    logger.info("XGBoost 모델 훈련을 시작합니다...")
    
    # XGBoost 모델 인스턴스 생성
    # 하이퍼파라미터는 GridSearchCV나 Optuna 등으로 최적화하는 것이 좋습니다.
    # 여기서는 일반적인 기본값을 사용합니다.
    xgbr = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.005,
        random_state=42,
        n_jobs=-1
    )
    
    # 조기 종료(Early Stopping)를 사용하여 과적합 방지
    xgbr.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric='rmse',
        early_stopping_rounds=50,
        verbose=False
    )
    logger.info("모델 훈련이 완료되었습니다.")

    # --- 3. 모델 성능 평가 ---
    logger.info("테스트 데이터셋으로 모델 성능을 평가합니다...")
    y_pred = xgbr.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"성능 평가 결과: RMSE = {rmse:.4f}, R-squared = {r2:.4f}")

    # 특징 중요도(Feature Importance) 시각화
    fig, ax = plt.subplots(figsize=(10, len(feature_columns) * 0.4))
    xgb.plot_importance(xgbr, ax=ax, height=0.8)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    logger.info("특징 중요도 그래프를 'feature_importance.png' 파일로 저장했습니다.")

    # --- 4. 훈련된 모델 저장 ---
    model_dir = Path(config.ML_MODELS_DIR) / 'saved_models'
    model_dir.mkdir(parents=True, exist_ok=True) # 디렉토리가 없으면 생성
    
    model_filename = "stock_ranking_model.pkl"
    model_path = model_dir / model_filename
    
    try:
        joblib.dump(xgbr, model_path)
        logger.info(f"훈련된 모델을 성공적으로 저장했습니다: {model_path}")
    except Exception as e:
        logger.error(f"모델 저장 중 오류 발생: {e}")

    logger.info("--- 모든 모델 훈련 과정이 완료되었습니다. ---")

if __name__ == "__main__":
    train_ranking_model()
