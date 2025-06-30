
# =============================================================================
# 수정된 tests/analysis/test_relative_valuator.py
# =============================================================================

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from analysis.metrics.relative_valuator import RelativeValuator


@pytest.fixture
def sample_financial_data() -> pd.DataFrame:
    """테스트용 가짜 재무 데이터 DataFrame을 생성합니다."""
    np.random.seed(42)  # 재현 가능한 결과를 위해 시드 설정
    data = {
        'ticker': [f'TICKER{i}' for i in range(20)],
        'grossProfitMargin': np.random.rand(20) * 0.5 + 0.2,
        'operatingMargin': np.random.rand(20) * 0.3,
        'debtEquityRatio': np.random.rand(20) * 2.0,
        'rdToRevenue': np.random.rand(20) * 0.2,
        'capexToRevenue': np.random.rand(20) * 0.1,
        'roe': np.random.rand(20) * 0.3,
        'peRatio': np.random.rand(20) * 30 + 5,
        'revenueGrowth': np.random.rand(20) * 0.4 - 0.1,
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_db_engine():
    """실제 DB에 접속하지 않도록 가짜 데이터베이스 엔진을 생성합니다."""
    return MagicMock()

@pytest.fixture
def mock_config():
    """테스트용 가짜 config 객체를 생성합니다."""
    class MockConfig:
        CLUSTERING_FEATURES = ['grossProfitMargin', 'operatingMargin', 'debtEquityRatio', 'rdToRevenue', 'capexToRevenue']
        OPTIMAL_K_CLUSTERS = 3
        METRICS_FOR_SCORING = {'roe': True, 'debtEquityRatio': False, 'peRatio': False, 'revenueGrowth': True}
        METRIC_CATEGORIES = {'profitability': ['roe'], 'stability': ['debtEquityRatio'], 'growth': ['revenueGrowth'], 'valuation': ['peRatio']}
        STRATEGY_WEIGHTS = {'blend': { 'valuation': 0.3, 'profitability': 0.25, 'growth': 0.25, 'stability': 0.20 }}
        FINAL_COMPOSITE_WEIGHTS = { 'relative': 0.7, 'absolute': 0.3 }
    return MockConfig()

class TestRelativeValuator:
    
    def test_initialization(self, mock_db_engine):
        """클래스가 올바르게 초기화되는지 테스트합니다."""
        RelativeValuator._load_quality_benchmark = MagicMock(return_value=pd.Series({'avg_roe': 0.15}))
        
        analyzer = RelativeValuator(db_engine=mock_db_engine)
        assert analyzer.db_engine is not None
        assert analyzer.quality_benchmark is not None

    def test_create_dynamic_peer_groups(self, mock_db_engine, sample_financial_data, mock_config):
        """동적 피어 그룹 생성 로직을 테스트합니다."""
        analyzer = RelativeValuator(db_engine=mock_db_engine)
        analyzer.config = mock_config

        result_df = analyzer._create_dynamic_peer_groups(sample_financial_data)
        
        assert 'cluster_label' in result_df.columns
        assert result_df['cluster_label'].nunique() == mock_config.OPTIMAL_K_CLUSTERS
        assert len(result_df) == len(sample_financial_data)

    def test_calculate_relative_scores(self, mock_db_engine, sample_financial_data, mock_config):
        """Z-점수 계산 로직을 테스트합니다."""
        analyzer = RelativeValuator(db_engine=mock_db_engine)
        analyzer.config = mock_config
        
        clustered_df = analyzer._create_dynamic_peer_groups(sample_financial_data)
        scored_df = analyzer._calculate_relative_scores(clustered_df)
        
        for metric in mock_config.METRICS_FOR_SCORING.keys():
            assert f'{metric}_z_score' in scored_df.columns
        
        assert abs(scored_df['roe_z_score'].mean()) < 0.1

    def test_run_full_valuation(self, mock_db_engine, sample_financial_data, mock_config, monkeypatch):
        """전체 가치평가 프로세스를 테스트합니다."""
        
        # 데이터 로드 함수를 모킹
        def mock_load_data(query, engine):
            return sample_financial_data
        
        monkeypatch.setattr("analysis.metrics.relative_valuator.load_data_from_db", mock_load_data)
        
        RelativeValuator._load_quality_benchmark = MagicMock(return_value=pd.Series({
            'avg_roe': 0.15, 'avg_debt_to_equity': 1.0, 'avg_operating_margin': 0.1
        }))
        
        analyzer = RelativeValuator(db_engine=mock_db_engine)
        analyzer.config = mock_config
        
        final_df = analyzer.run_full_valuation(strategy='blend')
        
        assert isinstance(final_df, pd.DataFrame)
        assert not final_df.empty
        assert 'rank' in final_df.columns
        assert 'final_score' in final_df.columns
        assert final_df.iloc[0]['rank'] == 1.0
        assert final_df['rank'].is_monotonic_increasing