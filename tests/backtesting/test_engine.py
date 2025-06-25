# /tests/backtesting/test_engine.py

import pytest
import pandas as pd
import backtrader as bt
from unittest.mock import MagicMock, patch

# 테스트할 대상 클래스를 불러옵니다.
from backtesting.engine import BacktestRunner

# --- 테스트에 사용할 가짜 데이터와 객체를 미리 만들어주는 Pytest Fixture ---

@pytest.fixture
def mock_db_engine():
    """가짜 데이터베이스 엔진을 생성합니다."""
    return MagicMock()

@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """테스트용 가짜 시계열 가격 데이터 DataFrame을 생성합니다."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100))
    data = {
        'date': dates,
        'open': [i for i in range(100, 200)],
        'high': [i + 5 for i in range(100, 200)],
        'low': [i - 5 for i in range(100, 200)],
        'close': [i + 2 for i in range(100, 200)],
        'volume': [10000 * (i+1) for i in range(100)],
        'ticker': 'TEST'
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_prediction_data() -> pd.DataFrame:
    """테스트용 가짜 ML 모델 예측 데이터 DataFrame을 생성합니다."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=4, freq='MS'))
    data = {
        'date': dates,
        'ticker': ['TEST'] * 4,
        'prediction_score': [70, 85, 60, 95],
        'rank': [3, 2, 4, 1]
    }
    return pd.DataFrame(data)

# --- 테스트용 가짜 전략 클래스 ---

class DummyStrategy(bt.Strategy):
    """테스트를 위한 최소한의 기능을 가진 가짜 전략."""
    def __init__(self):
        self.order = None

    def next(self):
        # 간단한 매수/매도 로직
        if self.order:
            return
        if not self.position:
            self.order = self.buy(size=10)
        else:
            if len(self) >= (self.start_len + 5):
                self.order = self.close()

    def stop(self):
        pass # 특별한 동작 없음

# --- BacktestRunner 클래스를 위한 메인 테스트 클래스 ---

class TestBacktestRunner:

    def test_initialization(self, mock_db_engine):
        """클래스가 올바르게 초기화되는지 테스트합니다."""
        engine = BacktestRunner(db_engine=mock_db_engine)
        assert engine.db_engine is not None

    def test_prepare_data_feeds(self, mock_db_engine, sample_price_data, monkeypatch):
        """데이터 피드 준비 로직을 테스트합니다."""
        # DB에서 데이터를 로드하는 함수를 가짜 함수로 대체
        monkeypatch.setattr("backtesting.engine.load_data_from_db", lambda q, e: sample_price_data)
        
        engine = BacktestRunner(db_engine=mock_db_engine)
        data_feeds = engine._prepare_data_feeds(['TEST'], '2023-01-01', '2023-04-10')
        
        assert 'TEST' in data_feeds
        assert isinstance(data_feeds['TEST'], bt.feeds.PandasData)
        assert len(data_feeds['TEST']) == 100

    @patch('backtesting.engine.pf.create_full_tear_sheet') # pyfolio의 차트 생성을 막음
    def test_run_backtest(self, mock_create_tear_sheet, mock_db_engine, sample_price_data, sample_prediction_data, monkeypatch):
        """전체 백테스팅 실행 프로세스를 테스트합니다."""
        
        # DB 의존성을 모두 가짜 함수로 대체
        def mock_load_data(query, engine):
            if "daily_prices" in query:
                return sample_price_data
            elif "ml_predictions" in query:
                return sample_prediction_data
            return pd.DataFrame()
            
        monkeypatch.setattr("backtesting.engine.load_data_from_db", mock_load_data)
        
        engine = BacktestRunner(db_engine=mock_db_engine)
        
        # run_backtest는 내부에서 결과를 출력하므로, 반환값이 없어도 실행되는지 확인
        # strategy_params는 DummyStrategy가 받지 않으므로 비워둠
        engine.run_backtest(
            strategy_class=DummyStrategy,
            start_date='2023-01-01',
            end_date='2023-04-10',
            tickers=['TEST'],
            initial_cash=100000,
            requires_prediction=True # 가짜 전략이지만, 예측 데이터 로딩 로직을 테스트하기 위해 True로 설정
        )
        
        # pyfolio의 리포트 생성 함수가 호출되었는지 확인
        mock_create_tear_sheet.assert_called_once()
