# =============================================================================
# 수정된 tests/analysis/test_technical_scorer.py
# =============================================================================

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from analysis.technical_scorer import TechnicalScorer
except ImportError:
    try:
        from ...analysis.technical_scorer import TechnicalScorer
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "technical_scorer", 
            os.path.join(project_root, "analysis", "technical_scorer.py")
        )
        technical_scorer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(technical_scorer_module)
        TechnicalScorer = technical_scorer_module.TechnicalScorer

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
    np.random.seed(42)  # 재현 가능한 결과를 위해 시드 설정
    close_prices = 100 + np.cumsum(np.random.randn(252))
    df = pd.DataFrame({
        'open': close_prices - np.random.rand(252),
        'high': close_prices + np.random.rand(252),
        'low': close_prices - np.random.rand(252),
        'close': close_prices,
        'volume': np.random.randint(100000, 500000, 252)
    }, index=dates)
    return df

class TestTechnicalScorer:

    def test_initialization(self, mock_data_manager, mock_config):
        """클래스가 올바르게 초기화되는지 테스트합니다."""
        scorer = TechnicalScorer(data_manager=mock_data_manager)
        scorer.config = mock_config
        assert scorer.data_manager is not None
        assert scorer.params['sma_short'] == 10

    def test_calculate_indicators(self, mock_data_manager, mock_config, sample_ohlcv_data):
        """기술적 지표 계산 로직을 테스트합니다."""
        scorer = TechnicalScorer(data_manager=mock_data_manager)
        scorer.config = mock_config
        
        result_df = scorer._calculate_indicators(sample_ohlcv_data)
        
        assert f"SMA_{mock_config.TECHNICAL_ANALYSIS_PARAMS['sma_long']}" in result_df.columns
        assert f"RSI_{mock_config.TECHNICAL_ANALYSIS_PARAMS['rsi_period']}" in result_df.columns
        assert f"MACD_{mock_config.TECHNICAL_ANALYSIS_PARAMS['macd_fast']}_{mock_config.TECHNICAL_ANALYSIS_PARAMS['macd_slow']}_{mock_config.TECHNICAL_ANALYSIS_PARAMS['macd_signal']}" in result_df.columns

    def test_calculate_z_scores(self, mock_data_manager, mock_config, sample_ohlcv_data):
        """Z-점수 계산 로직을 테스트합니다."""
        scorer = TechnicalScorer(data_manager=mock_data_manager)
        scorer.config = mock_config

        df_with_indicators = scorer._calculate_indicators(sample_ohlcv_data)
        df_with_zscores = scorer._calculate_z_scores(df_with_indicators)
        
        for col in mock_config.TECHNICAL_ANALYSIS_PARAMS['z_score_columns']:
            assert f'{col}_zscore' in df_with_zscores.columns
        
        assert abs(df_with_zscores[f"RSI_{mock_config.TECHNICAL_ANALYSIS_PARAMS['rsi_period']}_zscore"].mean()) < 0.5

    def test_get_scores_for_tickers(self, mock_data_manager, mock_config, sample_ohlcv_data):
        """전체 점수 산출 프로세스를 테스트합니다."""
        mock_data_manager.get_historical_prices.return_value = sample_ohlcv_data
        
        scorer = TechnicalScorer(data_manager=mock_data_manager)
        scorer.config = mock_config
        
        score_df = scorer.get_scores_for_tickers(['TEST1', 'TEST2'])
        
        assert isinstance(score_df, pd.DataFrame)
        assert not score_df.empty
        assert 'technical_score' in score_df.columns
        assert 'TEST1' in score_df.index
        
        score = score_df.loc['TEST1']['technical_score']
        assert 0 <= score <= 100