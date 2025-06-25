# /tests/ml_models/training/test_train_ranking_model.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# 테스트할 대상 스크립트의 함수들을 불러옵니다.
from ml_models.training.train_ranking_model import prepare_training_data, train_ranking_model

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
        MODEL_FEATURES = ['feature1', 'feature2', 'feature3']
        MODEL_TARGET = 'target_return'
        ML_MODELS_DIR = tmp_path
    return MockConfig()

@pytest.fixture
def sample_training_data() -> pd.DataFrame:
    """테스트용 가짜 훈련 데이터 DataFrame을 생성합니다."""
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100),
        'target_return': np.random.rand(100) * 0.1 - 0.05,
    }
    return pd.DataFrame(data)

# --- train_ranking_model.py 스크립트를 위한 메인 테스트 클래스 ---

class TestTrainRankingModel:

    def test_prepare_training_data_success(self, mock_db_engine, sample_training_data, monkeypatch):
        """데이터 준비 함수가 성공적으로 데이터를 로드하는지 테스트합니다."""
        # DB에서 데이터를 로드하는 함수를 가짜 함수로 대체
        monkeypatch.setattr("ml_models.training.train_ranking_model.load_data_from_db", lambda q, e: sample_training_data)
        
        df = prepare_training_data(db_engine=mock_db_engine)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) == 100
        assert 'target_return' in df.columns

    def test_prepare_training_data_empty(self, mock_db_engine, monkeypatch):
        """DB에서 데이터를 가져오지 못했을 때 빈 DataFrame을 반환하는지 테스트합니다."""
        monkeypatch.setattr("ml_models.training.train_ranking_model.load_data_from_db", lambda q, e: pd.DataFrame())
        
        df = prepare_training_data(db_engine=mock_db_engine)
        
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch('ml_models.training.train_ranking_model.joblib.dump')
    @patch('ml_models.training.train_ranking_model.plt.savefig')
    def test_full_training_pipeline(self, mock_savefig, mock_joblib_dump, mock_db_engine, mock_config, sample_training_data, monkeypatch):
        """
        전체 모델 훈련 파이프라인이 오류 없이 실행되고,
        최종 모델이 저장되는지 테스트합니다.
        """
        # 스크립트 내의 의존성을 모두 가짜(Mock) 객체로 대체
        monkeypatch.setattr("ml_models.training.train_ranking_model.prepare_training_data", lambda e: sample_training_data)
        monkeypatch.setattr("ml_models.training.train_ranking_model.config", mock_config)
        
        # 메인 훈련 함수 실행
        train_ranking_model()
        
        # 1. 모델 저장 함수(joblib.dump)가 호출되었는지 확인
        mock_joblib_dump.assert_called_once()
        
        # 2. 특징 중요도 그래프 저장 함수(plt.savefig)가 호출되었는지 확인
        mock_savefig.assert_called_once()
        
        # 3. 저장된 모델 파일이 실제로 생성되었는지 확인
        # (joblib.dump를 실제 함수로 두고, tmp_path를 사용하여 확인할 수도 있음)
        saved_model_path = mock_config.ML_MODELS_DIR / 'saved_models' / 'stock_ranking_model.pkl'
        # mock_joblib_dump.call_args[0][1]는 dump 함수의 두 번째 인자, 즉 저장 경로를 의미
        assert str(saved_model_path) in str(mock_joblib_dump.call_args[0][1])

