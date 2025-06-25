# /tests/ml_models/prediction/test_predict_stock_returns.py

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from unittest.mock import MagicMock, patch

# 테스트할 대상 스크립트의 함수들을 불러옵니다.
from ml_models.prediction.predict_stock_returns import (
    load_prediction_model,
    prepare_prediction_data,
    predict_with_ranking_model
)

# --- 테스트에 사용할 가짜 데이터와 객체를 미리 만들어주는 Pytest Fixture ---

@pytest.fixture
def mock_db_engine():
    """가짜 데이터베이스 엔진을 생성합니다."""
    return MagicMock()

@pytest.fixture
def mock_config(tmp_path):
    """테스트용 가짜 config 객체를 생성합니다."""
    # tmp_path는 pytest가 제공하는 임시 디렉토리 fixture입니다.
    class MockConfig:
        MODEL_FEATURES = ['feature1', 'feature2']
        ML_MODELS_DIR = tmp_path
    return MockConfig()

@pytest.fixture
def sample_prediction_features() -> pd.DataFrame:
    """예측에 사용할 가짜 최신 특징 데이터 DataFrame을 생성합니다."""
    data = {
        'ticker': ['AAPL', 'MSFT', 'GOOG'],
        'feature1': [0.8, 0.5, 0.9],
        'feature2': [0.7, 0.6, 0.85],
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_model(mock_config):
    """
    예측을 수행하는 가짜 머신러닝 모델 객체를 생성하고,
    임시 파일로 저장하여 로딩 테스트에 사용합니다.
    """
    # .predict 메서드를 가진 가짜 모델 생성
    model = MagicMock()
    # 특정 입력에 대해 정해진 예측값을 반환하도록 설정
    model.predict.return_value = np.array([0.08, 0.03, 0.12]) # GOOG > AAPL > MSFT 순서

    # 임시 디렉토리에 모델 파일 저장
    model_dir = Path(mock_config.ML_MODELS_DIR) / 'saved_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "stock_ranking_model.pkl"
    joblib.dump(model, model_path)
    
    return model, model_path

# --- predict_stock_returns.py 스크립트를 위한 메인 테스트 클래스 ---

class TestPredictStockReturns:

    def test_load_prediction_model_success(self, mock_model, mock_config):
        """저장된 모델 파일을 성공적으로 로드하는지 테스트합니다."""
        # config의 모델 디렉토리 경로를 가짜 경로로 변경
        with patch('ml_models.prediction.predict_stock_returns.config', mock_config):
            loaded_model = load_prediction_model("stock_ranking_model.pkl")
        
        assert loaded_model is not None
        # 로드된 모델이 예측 기능을 가지고 있는지 확인
        assert hasattr(loaded_model, 'predict')

    def test_load_prediction_model_not_found(self, mock_config):
        """모델 파일이 없을 때 None을 반환하는지 테스트합니다."""
        with patch('ml_models.prediction.predict_stock_returns.config', mock_config):
            loaded_model = load_prediction_model("non_existent_model.pkl")
        
        assert loaded_model is None

    def test_prepare_prediction_data(self, mock_db_engine, sample_prediction_features, monkeypatch):
        """예측용 데이터를 DB에서 올바르게 불러오는지 테스트합니다."""
        monkeypatch.setattr("ml_models.prediction.predict_stock_returns.load_data_from_db", lambda q, e: sample_prediction_features)
        
        df = prepare_prediction_data(db_engine=mock_db_engine)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'AAPL' in df.index

    def test_predict_with_ranking_model(self, mock_model, sample_prediction_features, mock_config):
        """모델 예측 및 랭킹 생성 로직을 테스트합니다."""
        model, _ = mock_model
        # ticker를 인덱스로 설정하여 함수 입력 형식에 맞춤
        features_df = sample_prediction_features.set_index('ticker')
        
        with patch('ml_models.prediction.predict_stock_returns.config', mock_config):
            result_df = predict_with_ranking_model(model, features_df)
        
        assert isinstance(result_df, pd.DataFrame)
        assert not result_df.empty
        assert 'rank' in result_df.columns
        assert 'prediction_score' in result_df.columns
        
        # 랭킹이 예측 점수 순서와 일치하는지 확인 (GOOG > AAPL > MSFT)
        assert result_df.index.tolist() == ['GOOG', 'AAPL', 'MSFT']
        assert result_df.iloc[0]['rank'] == 1.0
        assert result_df.iloc[1]['rank'] == 2.0
        assert result_df.iloc[2]['rank'] == 3.0

    def test_predict_with_missing_features(self, mock_model, sample_prediction_features, mock_config):
        """예측에 필요한 특징이 부족할 때 None을 반환하는지 테스트합니다."""
        model, _ = mock_model
        # 'feature2' 컬럼을 일부러 삭제
        features_df = sample_prediction_features.drop(columns=['feature2']).set_index('ticker')
        
        with patch('ml_models.prediction.predict_stock_returns.config', mock_config):
            result_df = predict_with_ranking_model(model, features_df)
            
        assert result_df is None
