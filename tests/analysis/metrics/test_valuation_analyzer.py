# /tests/analysis/metrics/test_valuation_analyzer.py

import pytest
import pandas as pd
from unittest.mock import MagicMock

# 테스트할 대상 함수를 불러옵니다.
from analysis.metrics.valuation_analyzer import evaluate_valuation

# --- 테스트에 사용할 가짜 데이터와 객체를 미리 만들어주는 Pytest Fixture ---

@pytest.fixture
def mock_db_engine():
    """실제 DB에 접속하지 않도록 가짜 데이터베이스 엔진을 생성합니다."""
    return MagicMock()

@pytest.fixture
def undervalued_data() -> pd.DataFrame:
    """'저평가' 시나리오를 위한 가짜 재무 데이터."""
    data = {
        'peRatio': [10.0],                      # PER 낮음
        'pbRatio': [0.8],                       # PBR 1 미만
        'psRatio': [0.7],                       # PSR 1 미만
        'enterpriseValueOverEBITDA': [8.0],     # EV/EBITDA 낮음
    }
    return pd.DataFrame(data)

@pytest.fixture
def overvalued_data() -> pd.DataFrame:
    """'고평가' 시나리오를 위한 가짜 재무 데이터."""
    data = {
        'peRatio': [50.0],                      # PER 높음
        'pbRatio': [8.0],                       # PBR 높음
        'psRatio': [10.0],                      # PSR 높음
        'enterpriseValueOverEBITDA': [25.0],    # EV/EBITDA 높음
    }
    return pd.DataFrame(data)

# --- evaluate_valuation 함수를 위한 메인 테스트 클래스 ---

class TestEvaluateValuation:

    def test_undervalued_scenario(self, mock_db_engine, undervalued_data, monkeypatch):
        """저평가된 기업에 대해 올바른 평가가 내려지는지 테스트합니다."""
        monkeypatch.setattr("analysis.metrics.valuation_analyzer.load_data_from_db", lambda q, e: undervalued_data)
        
        report = evaluate_valuation(ticker="UNDERVALUED_TICKER", db_engine=mock_db_engine)
        
        assert 'error' not in report
        
        # 각 지표의 평가가 '저평가' 계열인지 확인
        assert "저평가" in report['per']['evaluation']
        assert "저평가" in report['pbr']['evaluation']
        assert "저평가" in report['psr']['evaluation']
        assert "저평가" in report['ev_to_ebitda']['evaluation']
        
        # 종합 요약이 긍정적인지 확인
        assert "저평가" in report['summary']

    def test_overvalued_scenario(self, mock_db_engine, overvalued_data, monkeypatch):
        """고평가된 기업에 대해 올바른 평가가 내려지는지 테스트합니다."""
        monkeypatch.setattr("analysis.metrics.valuation_analyzer.load_data_from_db", lambda q, e: overvalued_data)
        
        report = evaluate_valuation(ticker="OVERVALUED_TICKER", db_engine=mock_db_engine)

        assert 'error' not in report
        
        # 각 지표의 평가가 '고평가' 계열인지 확인
        assert "고평가" in report['per']['evaluation']
        assert "고평가" in report['pbr']['evaluation']
        assert "고평가" in report['psr']['evaluation']
        assert "고평가" in report['ev_to_ebitda']['evaluation']
        
        # 종합 요약이 부정적인지 확인
        assert "고평가" in report['summary']

    def test_no_data_scenario(self, mock_db_engine, monkeypatch):
        """DB에 해당 기업의 데이터가 없을 경우를 테스트합니다."""
        monkeypatch.setattr("analysis.metrics.valuation_analyzer.load_data_from_db", lambda q, e: pd.DataFrame())
        
        report = evaluate_valuation(ticker="UNKNOWN_TICKER", db_engine=mock_db_engine)
        
        # 에러 메시지가 포함된 딕셔너리가 반환되어야 함
        assert 'error' in report
        assert report['error'] == "No financial metrics data found."
