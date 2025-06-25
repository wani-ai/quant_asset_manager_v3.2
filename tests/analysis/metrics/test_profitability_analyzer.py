# /tests/analysis/metrics/test_profitability_analyzer.py

import pytest
import pandas as pd
from unittest.mock import MagicMock

# 테스트할 대상 함수를 불러옵니다.
from analysis.metrics.profitability_analyzer import evaluate_profitability

# --- 테스트에 사용할 가짜 데이터와 객체를 미리 만들어주는 Pytest Fixture ---

@pytest.fixture
def mock_db_engine():
    """실제 DB에 접속하지 않도록 가짜 데이터베이스 엔진을 생성합니다."""
    return MagicMock()

@pytest.fixture
def high_profit_data() -> pd.DataFrame:
    """'수익성 우수' 시나리오를 위한 가짜 재무 데이터."""
    data = {
        'grossProfitMargin': [0.5],
        'operatingMargin': [0.2],
        'netProfitMargin': [0.15],
        'roe': [0.20],
        'roa': [0.10],
    }
    return pd.DataFrame(data)

@pytest.fixture
def low_profit_data() -> pd.DataFrame:
    """'수익성 저조' 시나리오를 위한 가짜 재무 데이터."""
    data = {
        'grossProfitMargin': [0.1],
        'operatingMargin': [0.01],
        'netProfitMargin': [0.02],
        'roe': [0.03],
        'roa': [0.01],
    }
    return pd.DataFrame(data)

# --- evaluate_profitability 함수를 위한 메인 테스트 클래스 ---

class TestEvaluateProfitability:

    def test_high_profitability_scenario(self, mock_db_engine, high_profit_data, monkeypatch):
        """수익성이 높은 기업에 대해 올바른 평가가 내려지는지 테스트합니다."""
        # DB에서 데이터를 로드하는 함수를 가짜 함수로 대체하여, 항상 우리의 샘플 데이터를 반환하도록 설정
        monkeypatch.setattr("analysis.metrics.profitability_analyzer.load_data_from_db", lambda q, e: high_profit_data)
        
        report = evaluate_profitability(ticker="GOOD_TICKER", db_engine=mock_db_engine)
        
        # 1. 에러가 없어야 함
        assert 'error' not in report
        
        # 2. 각 지표의 평가가 '우수' 또는 '양호' 계열인지 확인
        assert "우수" in report['gross_margin']['evaluation']
        assert "우수" in report['operating_margin']['evaluation']
        assert "우수" in report['net_margin']['evaluation']
        assert "우수" in report['roe']['evaluation']
        assert "우수" in report['roa']['evaluation']
        
        # 3. 종합 요약이 긍정적인지 확인
        assert "최우수" in report['summary'] or "양호" in report['summary']

    def test_low_profitability_scenario(self, mock_db_engine, low_profit_data, monkeypatch):
        """수익성이 낮은 기업에 대해 올바른 평가가 내려지는지 테스트합니다."""
        monkeypatch.setattr("analysis.metrics.profitability_analyzer.load_data_from_db", lambda q, e: low_profit_data)
        
        report = evaluate_profitability(ticker="POOR_TICKER", db_engine=mock_db_engine)

        assert 'error' not in report
        
        # 각 지표의 평가가 '저조' 또는 '검토 필요' 계열인지 확인
        assert "저조" in report['operating_margin']['evaluation']
        assert "저조" in report['net_margin']['evaluation']
        assert "저조" in report['roe']['evaluation']
        assert "검토 필요" in report['roa']['evaluation']
        
        # 종합 요약이 부정적인지 확인
        assert "검토 필요" in report['summary']

    def test_no_data_scenario(self, mock_db_engine, monkeypatch):
        """DB에 해당 기업의 데이터가 없을 경우를 테스트합니다."""
        # DB 조회 결과가 빈 DataFrame인 상황을 시뮬레이션
        monkeypatch.setattr("analysis.metrics.profitability_analyzer.load_data_from_db", lambda q, e: pd.DataFrame())
        
        report = evaluate_profitability(ticker="UNKNOWN_TICKER", db_engine=mock_db_engine)
        
        # 에러 메시지가 포함된 딕셔너리가 반환되어야 함
        assert 'error' in report
        assert report['error'] == "No financial metrics data found."
        
    def test_missing_some_metrics(self, mock_db_engine, monkeypatch):
        """일부 재무 지표가 누락된 경우를 테스트합니다."""
        # ROE와 ROA 데이터가 없는 가짜 데이터
        missing_data = pd.DataFrame({
            'grossProfitMargin': [0.5],
            'operatingMargin': [0.2],
            'netProfitMargin': [0.15],
        })
        monkeypatch.setattr("analysis.metrics.profitability_analyzer.load_data_from_db", lambda q, e: missing_data)
        
        report = evaluate_profitability(ticker="MISSING_TICKER", db_engine=mock_db_engine)
        
        assert 'error' not in report
        
        # 존재하는 지표는 올바르게 평가되어야 함
        assert "우수" in report['gross_margin']['evaluation']
        
        # 존재하지 않는 지표는 기본값(0)으로 계산되어 '저조' 또는 '검토 필요'로 평가되어야 함
        assert report['roe']['value'] == "0.00%"
        assert "저조" in report['roe']['evaluation']

