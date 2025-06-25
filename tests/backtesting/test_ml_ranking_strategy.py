# /tests/backtesting/test_ml_ranking_strategy.py

import pytest
import pandas as pd
import backtrader as bt
from unittest.mock import MagicMock

# 테스트할 대상 클래스를 불러옵니다.
from backtesting.strategies.ml_ranking_strategy import ML_Ranking_Strategy
from backtesting.engine import BacktestRunner

# --- 테스트에 사용할 가짜 데이터와 객체를 미리 만들어주는 Pytest Fixture ---

@pytest.fixture
def sample_strategy_data() -> dict:
    """테스트용 가짜 시계열 가격 데이터 3개를 생성합니다."""
    tickers = ['AAPL', 'MSFT', 'GOOG']
    data_feeds = {}
    for ticker in tickers:
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100))
        # 각 종목의 가격 패턴을 다르게 하여 테스트의 신뢰도를 높임
        if ticker == 'AAPL':
            close_prices = [150 + i for i in range(100)] # 꾸준히 상승
        elif ticker == 'MSFT':
            close_prices = [300 - i for i in range(100)] # 꾸준히 하락
        else:
            close_prices = [2000 + (i % 10) * (-1)**i for i in range(100)] # 횡보
            
        df = pd.DataFrame({
            'Open': [p - 1 for p in close_prices],
            'High': [p + 2 for p in close_prices],
            'Low': [p - 2 for p in close_prices],
            'Close': close_prices,
            'Volume': [100000] * 100
        }, index=dates)
        data_feeds[ticker] = bt.feeds.PandasData(dataname=df, name=ticker)
    return data_feeds

@pytest.fixture
def sample_strategy_predictions() -> pd.DataFrame:
    """테스트용 가짜 ML 모델 예측 데이터 DataFrame을 생성합니다."""
    # 2023-02-01 리밸런싱 시, AAPL과 GOOG이 top 2로 선정되어야 함
    data = [
        {'date': '2023-02-01', 'ticker': 'AAPL', 'prediction_score': 0.9, 'rank': 1},
        {'date': '2023-02-01', 'ticker': 'GOOG', 'prediction_score': 0.8, 'rank': 2},
        {'date': '2023-02-01', 'ticker': 'MSFT', 'prediction_score': 0.3, 'rank': 3},
        # 2023-03-01 리밸런싱 시, MSFT와 GOOG이 top 2로 선정되어야 함
        {'date': '2023-03-01', 'ticker': 'MSFT', 'prediction_score': 0.95, 'rank': 1},
        {'date': '2023-03-01', 'ticker': 'GOOG', 'prediction_score': 0.85, 'rank': 2},
        {'date': '2023-03-01', 'ticker': 'AAPL', 'prediction_score': 0.2, 'rank': 3},
    ]
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df

# --- ML_Ranking_Strategy 클래스를 위한 메인 테스트 함수 ---

def test_ml_ranking_strategy_logic(sample_strategy_data, sample_strategy_predictions):
    """
    ML_Ranking_Strategy의 핵심 리밸런싱 로직을 테스트합니다.
    - 첫 리밸런싱일에 올바른 종목을 매수하는가?
    - 두 번째 리밸런싱일에 포트폴리오를 올바르게 교체하는가?
    """
    cerebro = bt.Cerebro(stdstats=False)
    
    # 데이터 피드 추가
    for ticker, feed in sample_strategy_data.items():
        cerebro.adddata(feed, name=ticker)
        
    # 전략 추가 및 파라미터 전달
    cerebro.addstrategy(
        ML_Ranking_Strategy,
        prediction_data=sample_strategy_predictions,
        rebalance_monthday=1, # 매월 1일에 리밸런싱
        portfolio_size=2      # 상위 2개 종목만 보유
    )
    
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    # 백테스트 실행
    results = cerebro.run()
    strat = results[0]

    # --- 최종 상태 검증 ---
    final_portfolio = []
    for data in strat.datas:
        if strat.getposition(data).size > 0:
            final_portfolio.append(data._name)

    # 1. 최종 포트폴리오는 2023-03-01의 예측 결과와 일치해야 함 (MSFT, GOOG)
    assert len(final_portfolio) == 2
    assert 'MSFT' in final_portfolio
    assert 'GOOG' in final_portfolio
    assert 'AAPL' not in final_portfolio # AAPL은 매도되었어야 함
    
    # 2. 초기 자산보다 최종 자산이 변동되었는지 확인 (거래가 실제로 일어났는지)
    assert cerebro.broker.getvalue() != 100000.0
