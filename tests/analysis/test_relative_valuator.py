# /tests/analysis/test_relative_valuator.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# 테스트할 대상 클래스를 불러옵니다.
from analysis.relative_valuator import RelativeValuator

# --- 테스트에 사용할 가짜 데이터와 객체를 미리 만들어주는 Pytest Fixture ---

@pytest.fixture
def sample_financial_data() -> pd.DataFrame:
    """테스트용 가짜 재무 데이터 DataFrame을 생성합니다."""
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

# --- RelativeValuator 클래스를 위한 메인 테스트 클래스 ---

class TestRelativeValuator:
    
    def test_initialization(self, mock_db_engine):
        """클래스가 올바르게 초기화되는지 테스트합니다."""
        # _load_quality_benchmark 메서드가 DB에 접근하지 않도록 가짜 함수로 대체
        RelativeValuator._load_quality_benchmark = MagicMock(return_value=pd.Series({'avg_roe': 0.15}))
        
        analyzer = RelativeValuator(db_engine=mock_db_engine)
        assert analyzer.db_engine is not None
        assert analyzer.quality_benchmark is not None

    def test_create_dynamic_peer_groups(self, mock_db_engine, sample_financial_data, mock_config):
        """동적 피어 그룹 생성 로직을 테스트합니다."""
        analyzer = RelativeValuator(db_engine=mock_db_engine)
        analyzer.config = mock_config # 가짜 config 주입

        result_df = analyzer._create_dynamic_peer_groups(sample_financial_data)
        
        # 'cluster_label' 컬럼이 추가되었는지 확인
        assert 'cluster_label' in result_df.columns
        # 생성된 클러스터의 개수가 예상과 같은지 확인
        assert result_df['cluster_label'].nunique() == mock_config.OPTIMAL_K_CLUSTERS
        # 모든 기업이 그대로 남아있는지 확인
        assert len(result_df) == len(sample_financial_data)

    def test_calculate_relative_scores(self, mock_db_engine, sample_financial_data, mock_config):
        """Z-점수 계산 로직을 테스트합니다."""
        analyzer = RelativeValuator(db_engine=mock_db_engine)
        analyzer.config = mock_config
        
        clustered_df = analyzer._create_dynamic_peer_groups(sample_financial_data)
        scored_df = analyzer._calculate_relative_scores(clustered_df)
        
        # Z-점수 컬럼들이 생성되었는지 확인
        for metric in mock_config.METRICS_FOR_SCORING.keys():
            assert f'{metric}_z_score' in scored_df.columns
        
        # Z-점수의 평균은 0에 가까워야 함 (간단한 검증)
        assert abs(scored_df['roe_z_score'].mean()) < 0.1

    def test_run_full_valuation(self, mock_db_engine, sample_financial_data, mock_config, monkeypatch):
        """전체 가치평가 프로세스를 테스트합니다."""
        
        # DB에서 데이터를 로드하는 함수를 가짜 함수로 대체하여, 항상 우리의 샘플 데이터를 반환하도록 설정
        monkeypatch.setattr("analysis.relative_valuator.load_data_from_db", lambda q, e: sample_financial_data)
        
        # 퀄리티 벤치마크 로딩도 가짜 함수로 대체
        RelativeValuator._load_quality_benchmark = MagicMock(return_value=pd.Series({
            'avg_roe': 0.15, 'avg_debt_to_equity': 1.0, 'avg_operating_margin': 0.1
        }))
        
        analyzer = RelativeValuator(db_engine=mock_db_engine)
        analyzer.config = mock_config
        
        final_df = analyzer.run_full_valuation(strategy='blend')
        
        # 최종 결과물의 구조가 올바른지 확인
        assert isinstance(final_df, pd.DataFrame)
        assert not final_df.empty
        assert 'rank' in final_df.columns
        assert 'final_score' in final_df.columns
        
        # 랭킹이 올바르게 매겨졌는지 확인 (1등은 1.0)
        assert final_df.iloc[0]['rank'] == 1.0
        
        # 순위 기준으로 정렬되었는지 확인
        assert final_df['rank'].is_monotonic_increasing
