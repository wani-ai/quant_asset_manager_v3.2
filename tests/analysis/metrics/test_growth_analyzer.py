# /tests/analysis/metrics/test_growth_analyzer.py

import pytest
import pandas as pd
from unittest.mock import MagicMock

# 테스트할 대상 함수를 불러옵니다.
from analysis.metrics.growth_analyzer import evaluate_growth

# --- 테스트에 사용할 가짜 데이터와 객체를 미리 만들어주는 Pytest Fixture ---

@pytest.fixture
def mock_db_engine():
    """실제 DB에 접속하지 않도록 가짜 데이터베이스 엔진을 생성합니다."""
    return MagicMock()

@pytest.fixture
def high_growth_data() -> pd.DataFrame:
    """'성장성 우수' 시나리오를 위한 가짜 재무 데이터."""
    data = {
        'revenueGrowth': [0.25],    # 매출액 증가율 높음
        'netIncomeGrowth': [0.30],  # 순이익 증가율 높음
        'epsGrowth': [0.28],        # EPS 성장률 높음
    }
    return pd.DataFrame(data)

@pytest.fixture
def low_growth_data() -> pd.DataFrame:
    """'성장성 저조' 시나리오를 위한 가짜 재무 데이터."""
    data = {
        'revenueGrowth': [-0.05],   # 매출액 역성장
        'netIncomeGrowth': [-0.10], # 순이익 역성장
        'epsGrowth': [-0.12],       # EPS 역성장
    }
    return pd.DataFrame(data)

# --- evaluate_growth 함수를 위한 메인 테스트 클래스 ---

class TestEvaluateGrowth:

    def test_high_growth_scenario(self, mock_db_engine, high_growth_data, monkeypatch):
        """성장성이 높은 기업에 대해 올바른 평가가 내려지는지 테스트합니다."""
        monkeypatch.setattr("analysis.metrics.growth_analyzer.load_data_from_db", lambda q, e: high_growth_data)
        
        report = evaluate_growth(ticker="GROWTH_TICKER", db_engine=mock_db_engine)
        
        assert 'error' not in report
        
        # 각 지표의 평가가 '최우수'인지 확인
        assert "최우수" in report['revenue_growth']['evaluation']
        assert "최우수" in report['net_income_growth']['evaluation']
        assert "최우수" in report['eps_growth']['evaluation']
        
        # 종합 요약이 긍정적인지 확인
        assert "최우수" in report['summary']

    def test_low_growth_scenario(self, mock_db_engine, low_growth_data, monkeypatch):
        """성장성이 낮은 기업에 대해 올바른 평가가 내려지는지 테스트합니다."""
        monkeypatch.setattr("analysis.metrics.growth_analyzer.load_data_from_db", lambda q, e: low_growth_data)
        
        report = evaluate_growth(ticker="DECLINE_TICKER", db_engine=mock_db_engine)

        assert 'error' not in report
        
        # 각 지표의 평가가 '역성장', '주의 필요' 계열인지 확인
        assert "역성장" in report['revenue_growth']['evaluation']
        assert "주의 필요" in report['net_income_growth']['evaluation']
        assert "주의 필요" in report['eps_growth']['evaluation']
        
        # 종합 요약이 부정적인지 확인
        assert "주의 필요" in report['summary']

    def test_no_data_scenario(self, mock_db_engine, monkeypatch):
        """DB에 해당 기업의 데이터가 없을 경우를 테스트합니다."""
        monkeypatch.setattr("analysis.metrics.growth_analyzer.load_data_from_db", lambda q, e: pd.DataFrame())
        
        report = evaluate_growth(ticker="UNKNOWN_TICKER", db_engine=mock_db_engine)
        
        # 에러 메시지가 포함된 딕셔너리가 반환되어야 함
        assert 'error' in report
        assert report['error'] == "No financial metrics data found."
