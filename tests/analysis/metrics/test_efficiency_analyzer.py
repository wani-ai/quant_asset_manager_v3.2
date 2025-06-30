# /tests/analysis/metrics/test_efficiency_analyzer.py

import pytest
import pandas as pd
from unittest.mock import MagicMock

# 테스트할 대상 함수를 불러옵니다.
from analysis.metrics.efficiency_analyzer import evaluate_efficiency

# --- 테스트에 사용할 가짜 데이터와 객체를 미리 만들어주는 Pytest Fixture ---

@pytest.fixture
def mock_db_engine():
    """실제 DB에 접속하지 않도록 가짜 데이터베이스 엔진을 생성합니다."""
    return MagicMock()

@pytest.fixture
def high_efficiency_data() -> pd.DataFrame:
    """'효율성 우수' 시나리오를 위한 가짜 재무 데이터."""
    data = {
        'assetTurnover': [1.5],         # 총자산회전율 높음
        'inventoryTurnover': [12.0],    # 재고자산회전율 높음
        'receivables_turnover': [15.0],  # 매출채권회전율 높음
    }
    return pd.DataFrame(data)

@pytest.fixture
def low_efficiency_data() -> pd.DataFrame:
    """'효율성 저조' 시나리오를 위한 가짜 재무 데이터."""
    data = {
        'assetTurnover': [0.4],         # 총자산회전율 낮음
        'inventoryTurnover': [2.0],     # 재고자산회전율 낮음
        'receivablesTurnover': [3.0],   # 매출채권회전율 낮음
    }
    return pd.DataFrame(data)

# --- evaluate_efficiency 함수를 위한 메인 테스트 클래스 ---

class TestEvaluateEfficiency:

    def test_high_efficiency_scenario(self, mock_db_engine, high_efficiency_data, monkeypatch):
        """운영 효율성이 높은 기업에 대해 올바른 평가가 내려지는지 테스트합니다."""
        monkeypatch.setattr("analysis.metrics.efficiency_analyzer.load_data_from_db", lambda q, e: high_efficiency_data)
        
        report = evaluate_efficiency(ticker="EFFICIENT_TICKER", db_engine=mock_db_engine)
        
        assert 'error' not in report
        
        # 각 지표의 평가가 '우수' 또는 '양호' 계열인지 확인
        assert "우수" in report['asset_turnover']['evaluation']
        assert "최우수" in report['inventory_turnover']['evaluation']
        # 'cash_ratio' 키가 'receivables_turnover'로 수정되어야 함
        assert "최우수" in report['receivables_turnover']['evaluation'] 
        
        # 종합 요약이 긍정적인지 확인
        assert "최우수" in report['summary']

    def test_low_efficiency_scenario(self, mock_db_engine, low_efficiency_data, monkeypatch):
        """운영 효율성이 낮은 기업에 대해 올바른 평가가 내려지는지 테스트합니다."""
        monkeypatch.setattr("analysis.metrics.efficiency_analyzer.load_data_from_db", lambda q, e: low_efficiency_data)
        
        report = evaluate_efficiency(ticker="INEFFICIENT_TICKER", db_engine=mock_db_engine)

        assert 'error' not in report
        
        # 각 지표의 평가가 '저조', '주의 필요' 계열인지 확인
        assert "저조" in report['asset_turnover']['evaluation']
        assert "주의 필요" in report['inventory_turnover']['evaluation']
        assert "검토 필요" in report['receivables_turnover']['evaluation']
        
        # 종합 요약이 부정적인지 확인
        assert "개선 필요" in report['summary']

    def test_no_data_scenario(self, mock_db_engine, monkeypatch):
        """DB에 해당 기업의 데이터가 없을 경우를 테스트합니다."""
        monkeypatch.setattr("analysis.metrics.efficiency_analyzer.load_data_from_db", lambda q, e: pd.DataFrame())
        
        report = evaluate_efficiency(ticker="UNKNOWN_TICKER", db_engine=mock_db_engine)
        
        # 에러 메시지가 포함된 딕셔너리가 반환되어야 함
        assert 'error' in report
        assert report['error'] == "No financial metrics data found."
