import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from unittest.mock import MagicMock, patch

# [수정] 프로젝트 루트를 기준으로 절대 경로를 사용하여 모듈을 임포트합니다.
# 이렇게 하면 pytest 실행 시 경로 문제가 발생하지 않습니다.
from ml_models.prediction.predict_stock_returns import (
    load_prediction_model,
    prepare_prediction_data,
    predict_stock_returns  # 실제 함수 이름으로 임포트
)

# --- 테스트에 사용할 가짜 데이터와 객체를 미리 만들어주는 Pytest Fixture ---

@pytest.fixture
def mock_db_engine():
    """가짜 데이터베이스 엔진을 생성합니다."""
    return MagicMock()

@pytest.fixture
def mock_config(tmp_path):
    """테스트용 가짜 config 객체를 생성합니다."""
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
    model = MagicMock()
    model.predict.return_value = np.array([0.08, 0.03, 0.12])  # GOOG > AAPL > MSFT 순서

    model_dir = Path(mock_config.ML_MODELS_DIR) / 'saved_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "stock_ranking_model.pkl"

  # joblib.dump(model, model_path
    
    return model, model_path

# --- predict_stock_returns.py 스크립트를 위한 메인 테스트 클래스 ---

class TestPredictStockReturns:

    def test_load_prediction_model_success(self, mock_model, mock_config):
        """저장된 모델 파일을 성공적으로 로드하는지 테스트합니다."""
        with patch('joblib.load', return_value=mock_model):
            with patch('ml_models.prediction.predict_stock_returns.config', mock_config):
                loaded_model = load_prediction_model("any_model_name.pkl")

        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')

    def test_load_prediction_model_not_found(self, mock_config):
        """모델 파일이 없을 때 None을 반환하는지 테스트합니다."""
        with patch('ml_models.prediction.predict_stock_returns.config', mock_config):
            loaded_model = load_prediction_model("non_existent_model.pkl")
        
        assert loaded_model is None

    def test_prepare_prediction_data(self, mock_db_engine, sample_prediction_features, monkeypatch):
        """예측용 데이터를 DB에서 올바르게 불러오는지 테스트합니다."""
        # data.database.load_data_from_db 함수를 가짜 데이터로 대체
        monkeypatch.setattr("data.database.load_data_from_db", lambda query, engine: sample_prediction_features)
        
        df = prepare_prediction_data(db_engine=mock_db_engine)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        # prepare_prediction_data 함수는 ticker를 인덱스로 설정하므로, 인덱스에 있는지 확인
        assert 'AAPL' in df.index

    def test_predict_stock_returns(self, mock_model, sample_prediction_features, mock_config):
        """모델 예측 및 랭킹 생성 로직을 테스트합니다."""
        model, _ = mock_model
        features_df = sample_prediction_features.set_index('ticker')
        
        with patch('ml_models.prediction.predict_stock_returns.config', mock_config):
            # [수정] 실제 함수 이름인 'predict_stock_returns'으로 호출합니다.
            result_df = predict_stock_returns(mock_model, features_df)
        
        assert isinstance(result_df, pd.DataFrame)
        assert not result_df.empty
        assert 'rank' in result_df.columns
        assert 'prediction_score' in result_df.columns
        
        # 랭킹이 예측 점수 순서와 일치하는지 확인 (GOOG > AAPL > MSFT)
        assert result_df.index.tolist() == ['GOOG', 'AAPL', 'MSFT']
        assert result_df.loc['GOOG', 'rank'] == 1.0
        assert result_df.loc['AAPL', 'rank'] == 2.0
        assert result_df.loc['MSFT', 'rank'] == 3.0

    def test_predict_with_missing_features(self, mock_model, sample_prediction_features, mock_config):
        """예측에 필요한 특징이 부족할 때 None을 반환하는지 테스트합니다."""
        model, _ = mock_model
        features_df = sample_prediction_features.drop(columns=['feature2']).set_index('ticker')
        
        with patch('ml_models.prediction.predict_stock_returns.config', mock_config):
            # [수정] 실제 함수 이름인 'predict_stock_returns'으로 호출합니다.
            result_df = predict_stock_returns(model, features_df)
            
        assert result_df is None