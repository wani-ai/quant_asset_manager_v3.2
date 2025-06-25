# /backtesting/strategies/ml_ranking_strategy.py

import backtrader as bt
import pandas as pd
import math

class ML_Ranking_Strategy(bt.Strategy):
    """
    머신러닝 모델이 예측한 'prediction_score'를 기반으로,
    주기적으로 포트폴리오를 리밸런싱하는 투자 전략.
    """
    params = (
        ('prediction_data', None), # ML 모델의 예측 결과 DataFrame을 전달받음
        ('rebalance_monthday', 1), # 매월 리밸런싱을 실행할 날짜 (예: 1일)
        ('portfolio_size', 20),    # 포트폴리오에 편입할 상위 N개 종목 수
    )

    def __init__(self):
        """전략 생성자: 필요한 변수들을 초기화합니다."""
        # 타이머를 사용하여 리밸런싱 날짜를 추적
        self.add_timer(
            when=bt.Timer.SESSION_START,
            monthdays=[self.p.rebalance_monthday], 
            monthcarry=True, # 해당 날짜가 휴일이면 다음 거래일에 실행
        )
        
        # 전달받은 예측 데이터 DataFrame
        self.predictions = self.p.prediction_data
        if self.predictions is None or self.predictions.empty:
            raise ValueError("머신러닝 예측 데이터가 전달되지 않았습니다.")
            
        # 예측 데이터의 날짜를 파싱하여 빠르게 조회할 수 있도록 준비
        if not isinstance(self.predictions.index, pd.DatetimeIndex):
            self.predictions.index = pd.to_datetime(self.predictions.index)
        
        self.log("전략 초기화 완료.")

    def log(self, txt, dt=None):
        """로그 출력을 위한 헬퍼 함수"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        """주문 상태 변경 시 호출되는 메서드"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            self.log(
                f"ORDER EXECUTED: {order.getstatusname()}, Ticker: {order.data._name}, "
                f"Price: {order.executed.price:.2f}, Size: {order.executed.size}"
            )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER FAILED: {order.getstatusname()} for {order.data._name}")

    def notify_timer(self, timer, when, *args, **kwargs):
        """타이머 이벤트가 발생했을 때 호출됩니다 (리밸런싱 실행)"""
        self.log(f'리밸런싱 실행... (현재 자산: ${self.broker.getvalue():,.2f})')
        self.rebalance_portfolio()

    def rebalance_portfolio(self):
        """포트폴리오 리밸런싱의 핵심 로직"""
        # --- 1. 오늘 날짜의 최상위 랭킹 주식 선정 ---
        current_date = self.datas[0].datetime.date(0)
        
        # asof는 정확한 날짜가 없으면 가장 가까운 과거 날짜를 찾아줌 (선행 편향 방지)
        try:
            todays_ranks_df = self.predictions.loc[self.predictions.index <= str(current_date)]
            if todays_ranks_df.empty:
                self.log(f"{current_date}에 해당하는 예측 데이터가 없어 리밸런싱을 건너뜁니다.")
                return
            latest_date = todays_ranks_df.index.max()
            todays_ranks = todays_ranks_df[todays_ranks_df.index == latest_date]
        except Exception:
            self.log(f"{current_date} 예측 데이터 조회 중 오류 발생. 리밸런싱을 건너뜁니다.")
            return

        # 예측 점수(prediction_score)가 높은 순서대로 정렬하여 상위 N개 선택
        top_stocks = todays_ranks.sort_values(by='prediction_score', ascending=False).head(self.p.portfolio_size)
        
        if top_stocks.empty:
            self.log("선정된 상위 주식이 없습니다.")
            return

        target_tickers = top_stocks['ticker'].tolist()
        self.log(f"리밸런싱 목표 종목 ({len(target_tickers)}개): {', '.join(target_tickers)}")

        # --- 2. 기존 포지션 조정 (매도) ---
        # 현재 보유 중이지만, 새로운 목표 포트폴리오에는 없는 종목들을 매도
        for data in self.datas:
            ticker = data._name
            if self.getposition(data).size > 0 and ticker not in target_tickers:
                self.log(f"매도: {ticker}")
                self.close(data=data)

        # --- 3. 새로운 포트폴리오 구성 (매수) ---
        # 동일 가중으로 각 종목에 대한 목표 비중 계산
        target_percent = 1.0 / self.p.portfolio_size
        
        for ticker in target_tickers:
            data_feed = self.getdatabyname(ticker)
            if data_feed:
                # 목표 비중으로 주문 (매수 또는 비중 조절)
                self.order_target_percent(data=data_feed, target=target_percent)
    
    def stop(self):
        """백테스트 종료 시 호출되는 메서드"""
        self.log(f"(포트폴리오 크기: {self.p.portfolio_size}개) "
                 f"최종 자산 가치: ${self.broker.getvalue():,.2f}")

