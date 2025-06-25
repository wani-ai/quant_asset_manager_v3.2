# /tests/ml_models/training/test_train_timeseries_model.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Darts 라이브러리 및 테스트할 대상 함수들을 불러옵니다.
from darts import TimeSeries
from darts.models import NBEATSModel
from ml_models.training.train_timeseries_model import prepare_timeseries_data, train_timeseries_model

# --- 테스트에 사용할 가짜 데이터와 객체를 미리 만들어주는 Pytest Fixture ---

@pytest.fixture
def mock_db_engine():
    """가짜 데이터베이스 엔진을 생성합니다."""
    return MagicMock()

@pytest.fixture
def mock_config(tmp_path):
    """테스트용 가짜 config 객체를 생성합니다."""
    class MockConfig:
        ML_MODELS_DIR = tmp_path
    return MockConfig()

@pytest.fixture
def sample_timeseries_training_data() -> pd.DataFrame:
    """테스트용 가짜 시계열 훈련 데이터 DataFrame을 생성합니다."""
    dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=36, freq='MS'))
    data = {
        'date': dates,
        'cluster_label': ['group1'] * 36,
        'roe': np.linspace(0.1, 0.25, 36)
    }
    return pd.DataFrame(data)

# --- train_timeseries_model.py 스크립트를 위한 메인 테스트 클래스 ---

class TestTrainTimeseriesModel:

    def test_prepare_timeseries_data(self, mock_db_engine, sample_timeseries_training_data, monkeypatch):
        """데이터 준비 함수가 Darts TimeSeries 객체를 올바르게 생성하는지 테스트합니다."""
        monkeypatch.setattr("ml_models.training.train_timeseries_model.load_data_from_db", lambda q, e: sample_timeseries_training_data)
        
        series_list = prepare_timeseries_data(mock_db_engine, 'roe', 'cluster_label')
        
        assert isinstance(series_list, list)
        assert len(series_list) == 1
        assert isinstance(series_list[0], TimeSeries)
        assert len(series_list[0]) == 36

    @patch('darts.models.NBEATSModel.save') # Darts 모델의 save 메서드를 가로챔
    @patch('darts.models.NBEATSModel.fit')  # 실제 훈련을 막기 위해 fit 메서드를 가로챔
    def test_full_timeseries_training_pipeline(
        self, mock_fit, mock_save, mock_db_engine, mock_config, sample_timeseries_training_data, monkeypatch
    ):
        """
        전체 시계열 모델 훈련 파이프라인이 오류 없이 실행되고,
        최종 모델 저장 함수가 호출되는지 테스트합니다.
        """
        # 스크립트 내의 의존성을 모두 가짜(Mock) 객체로 대체
        monkeypatch.setattr("ml_models.training.train_timeseries_model.prepare_timeseries_data", 
                            lambda e, m, g: [TimeSeries.from_dataframe(sample_timeseries_training_data, time_col='date', value_cols='roe')])
        monkeypatch.setattr("ml_models.training.train_timeseries_model.config", mock_config)
        
        # 메인 훈련 함수 실행
        train_timeseries_model(target_metric='roe', group_by='cluster_label')
        
        # 1. 모델 훈련 함수(fit)가 호출되었는지 확인
        mock_fit.assert_called_once()
        
        # 2. 모델 저장 함수(save)가 호출되었는지 확인
        mock_save.assert_called_once()
        
        # 3. 모델이 올바른 경로에 저장되려고 했는지 확인
        expected_path = mock_config.ML_MODELS_DIR / 'saved_models' / 'timeseries_forecast_model_roe.pt'
        # mock_save.call_args[0][0]는 save 함수의 첫 번째 인자, 즉 저장 경로를 의미
        assert str(expected_path) in str(mock_save.call_args[0][0])
