# /backtesting/engine.py

import backtrader as bt
import pandas as pd
import pyfolio as pf
import logging
from sqlalchemy import Engine

from data.database import load_data_from_db

class BacktestRunner:
    """
    backtrader와 pyfolio를 사용하여 투자 전략의 과거 성과를 시뮬레이션하고,
    그 결과를 종합적으로 분석하는 백테스팅 엔진.
    """
    def __init__(self, db_engine: Engine):
        """
        BacktestRunner의 생성자.

        :param db_engine: 데이터베이스 연결을 위한 SQLAlchemy 엔진.
        """
        self.db_engine = db_engine
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("백테스팅 엔진이 초기화되었습니다.")

    def _prepare_data_feeds(self, tickers: list, start_date: str, end_date: str) -> dict:
        """
        PostgreSQL에서 여러 종목의 시계열 데이터를 가져와 backtrader 피드 형식으로 변환합니다.
        """
        data_feeds = {}
        for ticker in tickers:
            query = f"""
            SELECT date, open, high, low, close, volume, openinterest
            FROM daily_prices 
            WHERE ticker = '{ticker}' AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date ASC;
            """
            # 가정: daily_prices 테이블에 수정 주가 데이터가 저장되어 있음.
            # openinterest는 0으로 설정.
            try:
                df = load_data_from_db(query, self.db_engine)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    # backtrader가 인식하는 컬럼명으로 변경
                    df.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low', 
                        'close': 'Close', 'volume': 'Volume',
                        'openinterest': 'OpenInterest'
                    }, inplace=True)
                    data_feeds[ticker] = bt.feeds.PandasData(dataname=df, name=ticker)
            except Exception as e:
                self.logger.error(f"'{ticker}' 데이터 준비 중 오류 발생: {e}")
        
        return data_feeds

    def run_backtest(
        self, 
        strategy_class, 
        strategy_params: dict,
        tickers: list,
        start_date: str, 
        end_date: str, 
        initial_cash: float = 1_000_000.0
    ):
        """
        주어진 전략과 데이터로 백테스트를 실행하고, 상세한 성과 보고서를 생성합니다.

        :param strategy_class: 실행할 backtrader 전략 클래스.
        :param strategy_params: 전략에 전달할 파라미터 딕셔너리.
        :param tickers: 백테스팅에 사용할 전체 유니버스 티커 리스트.
        :param start_date: 백테스트 시작일 (YYYY-MM-DD).
        :param end_date: 백테스트 종료일 (YYYY-MM-DD).
        :param initial_cash: 초기 투자 자금.
        """
        self.logger.info(f"'{strategy_class.__name__}' 전략 백테스트를 시작합니다.")
        self.logger.info(f"기간: {start_date} ~ {end_date}, 초기 자금: ${initial_cash:,.2f}")

        # 1. Cerebro 엔진 초기화
        cerebro = bt.Cerebro()

        # 2. 전략 추가
        # signal_data 등 필요한 파라미터를 전략에 전달
        cerebro.addstrategy(strategy_class, **strategy_params)

        # 3. 데이터 피드 준비 및 추가
        self.logger.info(f"{len(tickers)}개 종목에 대한 데이터 피드를 준비합니다...")
        data_feeds = self._prepare_data_feeds(tickers, start_date, end_date)
        if not data_feeds:
            self.logger.error("백테스트를 위한 데이터가 없습니다. 중단합니다.")
            return

        for ticker, feed in data_feeds.items():
            cerebro.adddata(feed, name=ticker)

        # 4. 초기 자금 및 수수료 설정
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.001) # 매매 수수료 0.1%

        # 5. 핵심 분석기(Analyzer) 추가
        self.logger.info("성과 분석을 위한 분석기를 추가합니다...")
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=1, riskfreerate=0.0, annualize=True)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade')

        # 6. 백테스트 실행
        self.logger.info("백테스트 시뮬레이션을 시작합니다...")
        results = cerebro.run()
        self.logger.info("백테스트 시뮬레이션이 완료되었습니다.")
        
        # 7. 결과 분석 및 리포트 생성
        strat = results[0]
        pyfolio_analyzer = strat.analyzers.getbyname('pyfolio')
        returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()
        
        self.logger.info("Pyfolio를 사용하여 상세 성과 보고서를 생성합니다...")
        
        # Pyfolio를 사용하여 HTML 리포트 생성 또는 화면에 출력
        pf.create_full_tear_sheet(
            returns,
            positions=positions,
            transactions=transactions,
            live_start_date=None, # 실시간 거래 시작일 (백테스트에서는 None)
            round_trips=True
        )

        self._print_summary_stats(strat)

    def _print_summary_stats(self, strat):
        """콘솔에 핵심 성과 지표를 요약하여 출력합니다."""
        sharpe_analyzer = strat.analyzers.getbyname('sharpe')
        drawdown_analyzer = strat.analyzers.getbyname('drawdown')
        trade_analyzer = strat.analyzers.getbyname('trade')

        print("\n" + "="*50)
        print("백테스트 성과 요약")
        print("="*50)
        print(f"최종 포트폴리오 가치: ${strat.broker.getvalue():,.2f}")
        if sharpe_analyzer.get_analysis():
            print(f"연율화 샤프 지수: {sharpe_analyzer.get_analysis()['sharperatio']:.2f}")
        if drawdown_analyzer.get_analysis():
            print(f"최대 낙폭 (MDD): {drawdown_analyzer.get_analysis().max.drawdown:.2f}%")
        if trade_analyzer.get_analysis():
            print(f"총 거래 횟수: {trade_analyzer.get_analysis().total.total}")
            print(f"승률 (Win Rate): {trade_analyzer.get_analysis().won.total / trade_analyzer.get_analysis().total.total * 100:.2f}%" if trade_analyzer.get_analysis().total.total > 0 else "N/A")
            print(f"수익 팩터 (Profit Factor): {abs(trade_analyzer.get_analysis().won.pnl.total / trade_analyzer.get_analysis().lost.pnl.total):.2f}" if trade_analyzer.get_analysis().lost.pnl.total != 0 else "inf")
        print("="*50 + "\n")

