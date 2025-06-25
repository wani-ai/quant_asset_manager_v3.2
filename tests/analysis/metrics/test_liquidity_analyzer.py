# /tests/analysis/metrics/test_liquidity_analyzer.py

import pytest
import pandas as pd
from unittest.mock import MagicMock

# 테스트할 대상 함수를 불러옵니다.
from analysis.metrics.liquidity_analyzer import evaluate_liquidity

# --- 테스트에 사용할 가짜 데이터와 객체를 미리 만들어주는 Pytest Fixture ---

@pytest.fixture
def mock_db_engine():
    """실제 DB에 접속하지 않도록 가짜 데이터베이스 엔진을 생성합니다."""
    return MagicMock()

@pytest.fixture
def high_liquidity_data() -> pd.DataFrame:
    """'유동성 우수' 시나리오를 위한 가짜 재무 데이터."""
    data = {
        'currentRatio': [2.5],      # 유동비율 높음
        'quickRatio': [1.5],        # 당좌비율 높음
        'cashRatio': [0.8],         # 현금비율 높음
        'operatingCashFlowToCurrentLiabilitiesRatio': [0.6], # OCF 비율 높음
    }
    return pd.DataFrame(data)

@pytest.fixture
def low_liquidity_data() -> pd.DataFrame:
    """'유동성 취약' 시나리오를 위한 가짜 재무 데이터."""
    data = {
        'currentRatio': [0.8],      # 유동비율 1 미만
        'quickRatio': [0.4],        # 당좌비율 0.5 미만
        'cashRatio': [0.1],         # 현금비율 낮음
        'operatingCashFlowToCurrentLiabilitiesRatio': [0.1], # OCF 비율 낮음
    }
    return pd.DataFrame(data)

# --- evaluate_liquidity 함수를 위한 메인 테스트 클래스 ---

class TestEvaluateLiquidity:

    def test_high_liquidity_scenario(self, mock_db_engine, high_liquidity_data, monkeypatch):
        """유동성이 높은 기업에 대해 올바른 평가가 내려지는지 테스트합니다."""
        monkeypatch.setattr("analysis.metrics.liquidity_analyzer.load_data_from_db", lambda q, e: high_liquidity_data)
        
        report = evaluate_liquidity(ticker="LIQUID_TICKER", db_engine=mock_db_engine)
        
        assert 'error' not in report
        
        # 각 지표의 평가가 '우수' 또는 '양호' 계열인지 확인
        assert "우수" in report['current_ratio']['evaluation']
        assert "우수" in report['quick_ratio']['evaluation']
        assert "최우수" in report['cash_ratio']['evaluation']
        assert "우수" in report['ocf_ratio']['evaluation']
        
        # 종합 요약이 긍정적인지 확인
        assert "최우수" in report['summary']

    def test_low_liquidity_scenario(self, mock_db_engine, low_liquidity_data, monkeypatch):
        """유동성이 낮은 기업에 대해 올바른 평가가 내려지는지 테스트합니다."""
        monkeypatch.setattr("analysis.metrics.liquidity_analyzer.load_data_from_db", lambda q, e: low_liquidity_data)
        
        report = evaluate_liquidity(ticker="ILLIQUID_TICKER", db_engine=mock_db_engine)

        assert 'error' not in report
        
        # 각 지표의 평가가 '위험', '주의 필요' 계열인지 확인
        assert "위험" in report['current_ratio']['evaluation']
        assert "주의 필요" in report['quick_ratio']['evaluation']
        
        # 종합 요약이 부정적인지 확인
        assert "주의 필요" in report['summary']

    def test_no_data_scenario(self, mock_db_engine, monkeypatch):
        """DB에 해당 기업의 데이터가 없을 경우를 테스트합니다."""
        monkeypatch.setattr("analysis.metrics.liquidity_analyzer.load_data_from_db", lambda q, e: pd.DataFrame())
        
        report = evaluate_liquidity(ticker="UNKNOWN_TICKER", db_engine=mock_db_engine)
        
        # 에러 메시지가 포함된 딕셔너리가 반환되어야 함
        assert 'error' in report
        assert report['error'] == "No financial metrics data found."

