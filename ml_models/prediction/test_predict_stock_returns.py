# /tests/ml_models/prediction/test_predict_stock_returns.py

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from unittest.mock import MagicMock

# 테스트할 대상 스크립트의 함수들을 불러옵니다.
from ml_models.prediction.predict_stock_returns import (
    load_prediction_model,
    prepare_prediction_data,
    predict_stock_returns
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

    def test_predict_stock_returns(self, mock_model, sample_prediction_features, mock_config):
        """모델 예측 및 랭킹 생성 로직을 테스트합니다."""
        model, _ = mock_model
        # ticker를 인덱스로 설정하여 함수 입력 형식에 맞춤
        features_df = sample_prediction_features.set_index('ticker')
        
        with patch('ml_models.prediction.predict_stock_returns.config', mock_config):
            result_df = predict_stock_returns(model, features_df)
        
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
            result_df = predict_stock_returns(model, features_df)
            
        assert result_df is None

# /tests/analysis/test_technical_scorer.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# 테스트할 대상 클래스를 불러옵니다.
from analysis.technical_scorer import TechnicalScorer

# --- 테스트에 사용할 가짜 데이터와 객체를 미리 만들어주는 Pytest Fixture ---

@pytest.fixture
def mock_data_manager():
    """가짜 SmartDataManager 객체를 생성합니다."""
    return MagicMock()

@pytest.fixture
def mock_config():
    """테스트용 가짜 config 객체를 생성합니다."""
    class MockConfig:
        TECHNICAL_ANALYSIS_PARAMS = {
            'sma_short': 10, 'sma_long': 50, 'ema_short': 12,
            'rsi_period': 14, 'macd_fast': 12, 'macd_slow': 26,
            'macd_signal': 9, 'bbands_period': 20,
            'z_score_columns': ['RSI_14', 'MACD_12_26_9']
        }
        TECHNICAL_SCORE_WEIGHTS = {'trend': 0.5, 'momentum': 0.5}
        INDICATOR_CATEGORIES = {
            'trend': [('sma_signal_zscore', 1.0, True)],
            'momentum': [('RSI_14_zscore', 0.5, False), ('MACD_12_26_9_zscore', 0.5, True)]
        }
    return MockConfig()

@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """테스트용 가짜 시계열 OHLCV 데이터 DataFrame을 생성합니다."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=252))
    close_prices = 100 + np.cumsum(np.random.randn(252))
    df = pd.DataFrame({
        'open': close_prices - np.random.rand(252),
        'high': close_prices + np.random.rand(252),
        'low': close_prices - np.random.rand(252),
        'close': close_prices,
        'volume': np.random.randint(100000, 500000, 252)
    }, index=dates)
    return df

# --- TechnicalScorer 클래스를 위한 메인 테스트 클래스 ---

class TestTechnicalScorer:

    def test_initialization(self, mock_data_manager, mock_config):
        """클래스가 올바르게 초기화되는지 테스트합니다."""
        scorer = TechnicalScorer(data_manager=mock_data_manager)
        scorer.config = mock_config # 가짜 config 주입
        assert scorer.data_manager is not None
        assert scorer.params['sma_short'] == 10

    def test_calculate_indicators(self, mock_data_manager, mock_config, sample_ohlcv_data):
        """기술적 지표 계산 로직을 테스트합니다."""
        scorer = TechnicalScorer(data_manager=mock_data_manager)
        scorer.config = mock_config
        
        result_df = scorer._calculate_indicators(sample_ohlcv_data)
        
        # 주요 지표 컬럼들이 생성되었는지 확인
        assert f"SMA_{mock_config.TECHNICAL_ANALYSIS_PARAMS['sma_long']}" in result_df.columns
        assert f"RSI_{mock_config.TECHNICAL_ANALYSIS_PARAMS['rsi_period']}" in result_df.columns
        assert f"MACD_{mock_config.TECHNICAL_ANALYSIS_PARAMS['macd_fast']}_{mock_config.TECHNICAL_ANALYSIS_PARAMS['macd_slow']}_{mock_config.TECHNICAL_ANALYSIS_PARAMS['macd_signal']}" in result_df.columns

    def test_calculate_z_scores(self, mock_data_manager, mock_config, sample_ohlcv_data):
        """Z-점수 계산 로직을 테스트합니다."""
        scorer = TechnicalScorer(data_manager=mock_data_manager)
        scorer.config = mock_config

        df_with_indicators = scorer._calculate_indicators(sample_ohlcv_data)
        df_with_zscores = scorer._calculate_z_scores(df_with_indicators)
        
        # Z-점수 컬럼들이 생성되었는지 확인
        for col in mock_config.TECHNICAL_ANALYSIS_PARAMS['z_score_columns']:
            assert f'{col}_zscore' in df_with_zscores.columns
        
        # Z-점수의 평균은 0에 가까워야 함 (롤링이므로 완벽히 0은 아님)
        assert abs(df_with_zscores[f"RSI_{mock_config.TECHNICAL_ANALYSIS_PARAMS['rsi_period']}_zscore"].mean()) < 0.5

    def test_get_scores_for_tickers(self, mock_data_manager, mock_config, sample_ohlcv_data):
        """
        전체 점수 산출 프로세스를 테스트합니다.
        """
        # data_manager의 메서드가 항상 우리의 샘플 데이터를 반환하도록 설정
        mock_data_manager.get_historical_prices.return_value = sample_ohlcv_data
        
        scorer = TechnicalScorer(data_manager=mock_data_manager)
        scorer.config = mock_config
        
        score_df = scorer.get_scores_for_tickers(['TEST1', 'TEST2'])
        
        # 결과물의 구조가 올바른지 확인
        assert isinstance(score_df, pd.DataFrame)
        assert not score_df.empty
        assert 'technical_score' in score_df.columns
        assert 'TEST1' in score_df.index
        
        # 점수가 0-100 사이의 값인지 확인
        score = score_df.loc['TEST1']['technical_score']
        assert 0 <= score <= 100

