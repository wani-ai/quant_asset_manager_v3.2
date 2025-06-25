# /tests/analysis/metrics/test_stability_analyzer.py

import pytest
import pandas as pd
from unittest.mock import MagicMock

# 테스트할 대상 함수를 불러옵니다.
from analysis.metrics.stability_analyzer import evaluate_stability

# --- 테스트에 사용할 가짜 데이터와 객체를 미리 만들어주는 Pytest Fixture ---

@pytest.fixture
def mock_db_engine():
    """실제 DB에 접속하지 않도록 가짜 데이터베이스 엔진을 생성합니다."""
    return MagicMock()

@pytest.fixture
def high_stability_data() -> pd.DataFrame:
    """'안정성 우수' 시나리오를 위한 가짜 재무 데이터."""
    data = {
        'debtEquityRatio': [0.3],      # 부채비율 낮음
        'interestCoverage': [10.0],    # 이자보상배율 높음
        'totalDebtToEbitda': [1.5],    # 총부채/EBITDA 낮음
    }
    return pd.DataFrame(data)

@pytest.fixture
def low_stability_data() -> pd.DataFrame:
    """'안정성 취약' 시나리오를 위한 가짜 재무 데이터."""
    data = {
        'debtEquityRatio': [3.0],      # 부채비율 높음
        'interestCoverage': [0.8],     # 이자보상배율 1 미만
        'totalDebtToEbitda': [6.0],    # 총부채/EBITDA 높음
    }
    return pd.DataFrame(data)

# --- evaluate_stability 함수를 위한 메인 테스트 클래스 ---

class TestEvaluateStability:

    def test_high_stability_scenario(self, mock_db_engine, high_stability_data, monkeypatch):
        """재무적으로 안정적인 기업에 대해 올바른 평가가 내려지는지 테스트합니다."""
        monkeypatch.setattr("analysis.metrics.stability_analyzer.load_data_from_db", lambda q, e: high_stability_data)
        
        report = evaluate_stability(ticker="STABLE_TICKER", db_engine=mock_db_engine)
        
        assert 'error' not in report
        
        # 각 지표의 평가가 '안전', '우수', '양호' 계열인지 확인
        assert "안전" in report['debt_to_equity']['evaluation']
        assert "우수" in report['interest_coverage']['evaluation']
        assert "양호" in report['total_debt_to_ebitda']['evaluation']
        
        # 종합 요약이 긍정적인지 확인
        assert "최우수" in report['summary']

    def test_low_stability_scenario(self, mock_db_engine, low_stability_data, monkeypatch):
        """재무적으로 불안정한 기업에 대해 올바른 평가가 내려지는지 테스트합니다."""
        monkeypatch.setattr("analysis.metrics.stability_analyzer.load_data_from_db", lambda q, e: low_stability_data)
        
        report = evaluate_stability(ticker="RISKY_TICKER", db_engine=mock_db_engine)

        assert 'error' not in report
        
        # 각 지표의 평가가 '위험', '주의 필요' 계열인지 확인
        assert "위험" in report['debt_to_equity']['evaluation']
        assert "위험" in report['interest_coverage']['evaluation']
        assert "위험" in report['total_debt_to_ebitda']['evaluation']
        
        # 종합 요약이 부정적인지 확인
        assert "주의 필요" in report['summary']

    def test_no_data_scenario(self, mock_db_engine, monkeypatch):
        """DB에 해당 기업의 데이터가 없을 경우를 테스트합니다."""
        monkeypatch.setattr("analysis.metrics.stability_analyzer.load_data_from_db", lambda q, e: pd.DataFrame())
        
        report = evaluate_stability(ticker="UNKNOWN_TICKER", db_engine=mock_db_engine)
        
        # 에러 메시지가 포함된 딕셔너리가 반환되어야 함
        assert 'error' in report
        assert report['error'] == "No financial metrics data found."
