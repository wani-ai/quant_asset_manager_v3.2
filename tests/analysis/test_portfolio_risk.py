# /tests/analysis/test_portfolio_risk.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# 테스트할 대상 클래스를 불러옵니다.
from analysis.portfolio_risk import PortfolioRiskAnalyzer

# --- 테스트에 사용할 가짜 데이터와 객체를 미리 만들어주는 Pytest Fixture ---

@pytest.fixture
def mock_db_engine():
    """가짜 데이터베이스 엔진을 생성합니다."""
    return MagicMock()

@pytest.fixture
def mock_data_manager(sample_price_data_for_risk):
    """
    가짜 SmartDataManager 객체를 생성하고, get_historical_prices 메서드가
    항상 우리의 샘플 데이터를 반환하도록 설정합니다.
    """
    manager = MagicMock()
    # get_historical_prices가 호출될 때, 티커에 맞는 데이터를 반환하도록 설정
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
    """테스트용 가짜 시계열 가격 데이터 2개를 생성합니다."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=252))
    # AAPL은 꾸준히 상승, MSFT는 변동성 있는 상승
    aapl_close = 150 + np.cumsum(np.random.randn(252) * 0.5 + 0.1)
    msft_close = 300 + np.cumsum(np.random.randn(252) * 1.2 + 0.05)
    
    data_feeds = {
        'AAPL': pd.DataFrame({'close': aapl_close}, index=dates),
        'MSFT': pd.DataFrame({'close': msft_close}, index=dates)
    }
    return data_feeds

# --- PortfolioRiskAnalyzer 클래스를 위한 메인 테스트 클래스 ---

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
        # 252일 데이터에서 pct_change()로 1일 손실, dropna()로 1일 손실 -> 250일
        assert len(returns) == 250

    def test_calculate_risk_metrics(self, mock_db_engine, mock_data_manager):
        """CVaR, MDD, Sharpe Ratio 등 핵심 리스크 지표 계산을 테스트합니다."""
        analyzer = PortfolioRiskAnalyzer(db_engine=mock_db_engine, data_manager=mock_data_manager)
        returns = analyzer._get_portfolio_returns(['AAPL'])
        
        cvar = analyzer._calculate_cvar(returns)
        mdd = analyzer._calculate_max_drawdown(returns)
        sharpe = analyzer._calculate_sharpe_ratio(returns)
        
        # 계산 결과가 숫자인지 확인
        assert isinstance(cvar, float)
        assert isinstance(mdd, float)
        assert isinstance(sharpe, float)
        # 손실이므로 음수여야 함
        assert cvar <= 0
        assert mdd <= 0

    def test_generate_full_report(self, mock_db_engine, mock_data_manager, mock_config):
        """
        전체 리스크 리포트 생성 프로세스를 테스트합니다.
        """
        analyzer = PortfolioRiskAnalyzer(db_engine=mock_db_engine, data_manager=mock_data_manager)
        analyzer.config = mock_config # 가짜 config 주입
        
        # _calculate_macro_exposure와 _calculate_fama_french_exposure는 외부 데이터 의존성이 크므로,
        # 테스트 중에는 가짜 데이터를 반환하도록 patch 처리합니다.
        with patch.object(analyzer, '_calculate_macro_exposure', return_value={'interest_rate_beta': 0.5}):
            with patch.object(analyzer, '_calculate_fama_french_exposure', return_value={'alpha': 0.01}):
                report = analyzer.generate_full_report(['AAPL', 'MSFT'])

        # 최종 리포트의 구조가 올바른지 확인
        assert 'error' not in report
        assert 'composite_risk_score' in report
        assert 'scores' in report
        assert 'raw_metrics' in report
        
        # 종합 점수가 0-100 사이의 값인지 확인
        score = report['composite_risk_score']
        assert 0 <= score <= 100
        
        # 주요 지표가 포함되었는지 확인
        assert 'cvar_99' in report['raw_metrics']
        assert 'max_drawdown' in report['raw_metrics']
        assert 'macro_exposure' in report['raw_metrics']

