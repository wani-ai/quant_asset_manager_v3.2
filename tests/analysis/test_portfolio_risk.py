# =============================================================================
# 수정된 tests/analysis/test_portfolio_risk.py
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
    from analysis.portfolio_risk import PortfolioRiskAnalyzer
except ImportError:
    # 상대 임포트로 시도
    try:
        from ...analysis.portfolio_risk import PortfolioRiskAnalyzer
    except ImportError:
        # 절대 경로로 시도
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "portfolio_risk", 
            os.path.join(project_root, "analysis", "portfolio_risk.py")
        )
        portfolio_risk_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(portfolio_risk_module)
        PortfolioRiskAnalyzer = portfolio_risk_module.PortfolioRiskAnalyzer

@pytest.fixture
def mock_db_engine():
    """가짜 데이터베이스 엔진을 생성합니다."""
    return MagicMock()

@pytest.fixture
def mock_data_manager(sample_price_data_for_risk):
    """가짜 SmartDataManager 객체를 생성합니다."""
    manager = MagicMock()
    def get_prices(ticker, period):
        return sample_price_data_for_risk[ticker]
    manager.get_historical_prices.side_effect = get_prices
    return manager

@pytest.fixture
def mock_config():
    """테스트용 가짜 config 객체를 생성합니다."""
    class MockConfig:
        RISK_SCORE_WEIGHTS = {'cvar': 0.4, 'mdd': 0.3, 'sharpe': 0.3}
    return MockConfig()

@pytest.fixture
def sample_price_data_for_risk() -> dict:
    """테스트용 가짜 시계열 가격 데이터를 생성합니다."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=252))
    np.random.seed(42)  # 재현 가능한 결과를 위해 시드 설정
    aapl_close = 150 + np.cumsum(np.random.randn(252) * 0.5 + 0.1)
    msft_close = 300 + np.cumsum(np.random.randn(252) * 1.2 + 0.05)
    
    data_feeds = {
        'AAPL': pd.DataFrame({'close': aapl_close}, index=dates),
        'MSFT': pd.DataFrame({'close': msft_close}, index=dates)
    }
    return data_feeds

class TestPortfolioRiskAnalyzer:

    def test_initialization(self, mock_db_engine, mock_data_manager):
        """클래스가 올바르게 초기화되는지 테스트합니다."""
        analyzer = PortfolioRiskAnalyzer(db_engine=mock_db_engine, data_manager=mock_data_manager)
        assert analyzer.db_engine is not None
        assert analyzer.data_manager is not None

    def test_get_portfolio_returns(self, mock_db_engine, mock_data_manager):
        """포트폴리오 수익률 계산 로직을 테스트합니다."""
        analyzer = PortfolioRiskAnalyzer(db_engine=mock_db_engine, data_manager=mock_data_manager)
        
        returns = analyzer._get_portfolio_returns(['AAPL', 'MSFT'])
        
        assert isinstance(returns, pd.Series)
        assert not returns.empty
        assert len(returns) == 250

    def test_calculate_risk_metrics(self, mock_db_engine, mock_data_manager):
        """CVaR, MDD, Sharpe Ratio 등 핵심 리스크 지표 계산을 테스트합니다."""
        analyzer = PortfolioRiskAnalyzer(db_engine=mock_db_engine, data_manager=mock_data_manager)
        returns = analyzer._get_portfolio_returns(['AAPL'])
        
        cvar = analyzer._calculate_cvar(returns)
        mdd = analyzer._calculate_max_drawdown(returns)
        sharpe = analyzer._calculate_sharpe_ratio(returns)
        
        assert isinstance(cvar, float)
        assert isinstance(mdd, float)
        assert isinstance(sharpe, float)
        assert cvar <= 0
        assert mdd <= 0

    def test_generate_full_report(self, mock_db_engine, mock_data_manager, mock_config):
        """전체 리스크 리포트 생성 프로세스를 테스트합니다."""
        analyzer = PortfolioRiskAnalyzer(db_engine=mock_db_engine, data_manager=mock_data_manager)
        analyzer.config = mock_config
        
        with patch.object(analyzer, '_calculate_macro_exposure', return_value={'interest_rate_beta': 0.5}):
            with patch.object(analyzer, '_calculate_fama_french_exposure', return_value={'alpha': 0.01}):
                report = analyzer.generate_full_report(['AAPL', 'MSFT'])

        assert 'error' not in report
        assert 'composite_risk_score' in report
        assert 'scores' in report
        assert 'raw_metrics' in report
        
        score = report['composite_risk_score']
        assert 0 <= score <= 100
        
        assert 'cvar_99' in report['raw_metrics']
        assert 'max_drawdown' in report['raw_metrics']
        assert 'macro_exposure' in report['raw_metrics']



