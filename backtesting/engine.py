# /backtesting/engine.py

import backtrader as bt
import pandas as pd
import pyfolio as pf
import logging
import sys
from sqlalchemy import Engine
from typing import List, Dict, Any, Type

# 시스템의 다른 모듈에서 DB 연결 및 유틸리티 함수를 불러옵니다.
# import sys
# sys.path.append('.')
from data.database import get_database_engine, load_data_from_db

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

    def _prepare_data_feeds(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, bt.feeds.PandasData]:
        """
        PostgreSQL에서 여러 종목의 시계열 데이터를 가져와 backtrader 피드 형식으로 변환합니다.
        """
        data_feeds = {}
        # 전체 기간에 대한 데이터를 한 번에 조회하여 효율성 향상
        tickers_tuple = tuple(tickers)
        query = f"""
        SELECT "date", "open", "high", "low", "close", "volume", "ticker"
        FROM daily_prices 
        WHERE ticker IN {tickers_tuple} AND "date" BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY "date" ASC;
        """
        # 가정: daily_prices 테이블에 수정 주가 데이터가 저장되어 있음.
        try:
            full_df = load_data_from_db(query, self.db_engine)
            if full_df.empty: return {}

            full_df['date'] = pd.to_datetime(full_df['date'])
            
            for ticker in tickers:
                df = full_df[full_df['ticker'] == ticker].copy()
                if not df.empty:
                    df = df.set_index('date')
                    # backtrader가 인식하는 컬럼명으로 변경
                    df.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low', 
                        'close': 'Close', 'volume': 'Volume'
                    }, inplace=True)
                    df['OpenInterest'] = 0 # openinterest 컬럼 추가
                    data_feeds[ticker] = bt.feeds.PandasData(dataname=df, name=ticker)
        except Exception as e:
            self.logger.error(f"데이터 피드 준비 중 오류 발생: {e}")
        
        return data_feeds

    def _load_prediction_data(self) -> pd.DataFrame:
        """
        ML 모델의 예측 결과(랭킹 데이터)를 데이터베이스에서 로드합니다.
        이는 선행 편향을 방지하는 핵심 단계입니다.
        """
        self.logger.info("ML 모델 예측 결과 데이터를 로드합니다...")
        # 가정: 'ml_predictions' 테이블에 날짜별, 티커별 예측 점수와 순위가 저장되어 있음.
        query = "SELECT date, ticker, prediction_score, rank FROM ml_predictions"
        df = load_data_from_db(query, self.db_engine)
        if not df.empty:
            # MultiIndex를 사용하여 날짜와 티커로 빠르게 조회할 수 있도록 준비
            df['date'] = pd.to_datetime(df['date'])
            return df.set_index(['date', 'ticker'])
        return pd.DataFrame()

    def run_backtest(
        self, 
        strategy_class: Type[bt.Strategy],
        start_date: str, 
        end_date: str, 
        initial_cash: float = 1_000_000.0,
        **strategy_params: Any
    ):
        """
        주어진 전략과 데이터로 백테스트를 실행하고, 상세한 성과 보고서를 생성합니다.
        """
        self.logger.info(f"'{strategy_class.__name__}' 전략 백테스트를 시작합니다.")
        self.logger.info(f"기간: {start_date} ~ {end_date}, 초기 자금: ${initial_cash:,.2f}")

        # --- 1. Cerebro 엔진 초기화 ---
        cerebro = bt.Cerebro(stdstats=False) # 기본 통계 출력은 비활성화

        # --- 2. 전략 추가 (ML 예측 데이터 주입) ---
        prediction_df = self._load_prediction_data()
        if prediction_df.empty and strategy_params.get('requires_prediction', False):
             self.logger.error("전략에 필요한 예측 데이터가 없습니다. 백테스트를 중단합니다.")
             return

        # 전략 파라미터에 예측 데이터 추가
        strategy_params['prediction_data'] = prediction_df
        cerebro.addstrategy(strategy_class, **strategy_params)

        # --- 3. 데이터 피드 준비 및 추가 ---
        # 예측 데이터에 존재하는 모든 티커를 투자 유니버스로 설정
        universe_tickers = prediction_df.reset_index()['ticker'].unique().tolist()
        self.logger.info(f"{len(universe_tickers)}개 종목에 대한 데이터 피드를 준비합니다...")
        data_feeds = self._prepare_data_feeds(universe_tickers, start_date, end_date)
        if not data_feeds:
            self.logger.error("백테스트를 위한 데이터가 없습니다. 중단합니다.")
            return

        for ticker, feed in data_feeds.items():
            cerebro.adddata(feed, name=ticker)

        # --- 4. 초기 자금, 수수료, 사이저 설정 ---
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.001) # 매매 수수료 0.1%
        cerebro.addsizer(bt.sizers.PercentSizer, percents=95) # 가용 현금의 95%를 투자에 사용

        # --- 5. 핵심 분석기(Analyzer) 추가 ---
        self.logger.info("성과 분석을 위한 분석기를 추가합니다...")
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe') # 연율화 샤프지수
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade')

        # --- 6. 백테스트 실행 ---
        self.logger.info("백테스트 시뮬레이션을 시작합니다...")
        results = cerebro.run()
        self.logger.info("백테스트 시뮬레이션이 완료되었습니다.")
        
        # --- 7. 결과 분석 및 리포트 생성 ---
        self._generate_report(results[0])

    def _generate_report(self, strat: bt.Strategy):
        """백테스트 결과를 바탕으로 상세 리포트를 생성하고 출력합니다."""
        pyfolio_analyzer = strat.analyzers.getbyname('pyfolio')
        returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()
        
        self.logger.info("Pyfolio를 사용하여 상세 성과 보고서를 생성합니다...")
        
        # Pyfolio를 사용하여 HTML 리포트 생성 또는 화면에 출력
        # benchmark_rets는 S&P500 등 벤치마크 수익률 (선택사항)
        pf.create_full_tear_sheet(
            returns,
            positions=positions,
            transactions=transactions,
            live_start_date=None, 
            round_trips=True
        )

        self._print_summary_stats(strat)

    def _print_summary_stats(self, strat: bt.Strategy):
        """콘솔에 핵심 성과 지표를 요약하여 출력합니다."""
        final_value = strat.broker.getvalue()
        sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        max_drawdown = strat.analyzers.drawdown.get_analysis().max.drawdown
        trade_info = strat.analyzers.trade.get_analysis()

        print("\n" + "="*60)
        print("백테스트 최종 성과 요약")
        print("="*60)
        print(f"최종 포트폴리오 가치: ${final_value:,.2f}")
        print(f"연율화 샤프 지수: {sharpe_ratio:.2f}")
        print(f"최대 낙폭 (MDD): {max_drawdown:.2f}%")
        print("-" * 60)
        print(f"총 거래 횟수: {trade_info.total.total}")
        print(f"수익 거래: {trade_info.won.total} / 손실 거래: {trade_info.lost.total}")
        print(f"승률 (Win Rate): {trade_info.won.total / trade_info.total.total * 100:.2f}%" if trade_info.total.total > 0 else "N/A")
        print(f"수익 팩터 (Profit Factor): {abs(trade_info.won.pnl.total / trade_info.lost.pnl.total):.2f}" if trade_info.lost.pnl.total != 0 else "inf")
        print("="*60 + "\n")

