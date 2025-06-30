# /tests/ml_models/prediction/test_predict_timeseries.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Darts 라이브러리 및 테스트할 대상 함수들을 불러옵니다.
from darts import TimeSeries
from darts.models import NBEATSModel
from ml_models.prediction.predict_timeseries import (
    load_timeseries_model,
    prepare_prediction_input_series,
    predict_future_metrics
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
        TIMESERIES_INPUT_CHUNK_LENGTH = 12
        ML_MODELS_DIR = tmp_path
    return MockConfig()

@pytest.fixture
def sample_timeseries_data() -> pd.DataFrame:
    """예측에 사용할 가짜 시계열 데이터 DataFrame을 생성합니다."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=24, freq='MS'))
    data = {
        'date': dates,
        'roe': np.linspace(0.1, 0.2, 24)
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_timeseries_model(mock_config):
    """
    예측을 수행하는 가짜 Darts 모델 객체를 생성하고,
    임시 파일로 저장하여 로딩 테스트에 사용합니다.
    """
    # Darts 모델은 input/output 길이를 필요로 하므로, 간단한 모델을 직접 생성
    model = NBEATSModel(
        input_chunk_length=12,
        output_chunk_length=6,
        n_epochs=1, # 실제 훈련은 하지 않음
        random_state=42
    )

    # 임시 디렉토리에 모델 파일 저장
    model_dir = Path(mock_config.ML_MODELS_DIR) / 'saved_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "timeseries_forecast_model_roe.pt"
    model.save(str(model_path))
    
    return model, model_path

# --- predict_timeseries.py 스크립트를 위한 메인 테스트 클래스 ---

class TestPredictTimeseries:

    def test_load_timeseries_model_success(self, mock_timeseries_model, mock_config):
        """저장된 Darts 모델 파일을 성공적으로 로드하는지 테스트합니다."""
        with patch('ml_models.prediction.predict_timeseries.config', mock_config):
            loaded_model = load_timeseries_model("timeseries_forecast_model_roe.pt")
        
        assert loaded_model is not None
        assert isinstance(loaded_model, NBEATSModel)

    def test_prepare_prediction_input_series(self, mock_db_engine, sample_timeseries_data, mock_config, monkeypatch):
        """예측용 시계열 데이터를 DB에서 올바르게 불러오고 변환하는지 테스트합니다."""
        monkeypatch.setattr("ml_models.prediction.predict_timeseries.load_data_from_db", lambda q, e, params: sample_timeseries_data)
        
        with patch('ml_models.prediction.predict_timeseries.config', mock_config):
            series = prepare_prediction_input_series(mock_db_engine, 'roe', '1')
        
        assert isinstance(series, TimeSeries)
        assert len(series) == len(sample_timeseries_data)

    def test_prepare_prediction_input_series_insufficient_data(self, mock_db_engine, sample_timeseries_data, mock_config, monkeypatch):
        """예측에 필요한 데이터 길이가 부족할 때 None을 반환하는지 테스트합니다."""
        # 데이터 길이를 일부러 부족하게 만듦
        insufficient_data = sample_timeseries_data.head(5)
        monkeypatch.setattr("ml_models.prediction.predict_timeseries.load_data_from_db", lambda q, e, params: insufficient_data)
        
        with patch('ml_models.prediction.predict_timeseries.config', mock_config):
            series = prepare_prediction_input_series(mock_db_engine, 'roe', '1')
            
        assert series is None

    def test_predict_future_metrics(self, mock_timeseries_model):
        """모델 예측 로직을 테스트합니다."""
        model = mock_timeseries_model
        
        # 테스트용 입력 시계열 생성
        input_series = TimeSeries.from_values(np.random.rand(model.input_chunk_length))
        
        model.fit(input_series)
        result_df = predict_future_metrics(model, input_series)
        
        assert isinstance(result_df, pd.DataFrame)
        assert not result_df.empty
        # 예측 기간(output_chunk_length)만큼의 결과가 생성되었는지 확인
        assert len(result_df) == model.output_chunk_length

