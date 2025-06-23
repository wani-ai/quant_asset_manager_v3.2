# /app/__init__.py

import logging
import joblib
from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------------
# 시스템의 모든 핵심 모듈을 불러옵니다.
# -----------------------------------------------------------------------------
import config  # 시스템 전역 설정

# 데이터 계층
from data.connectors import SmartDataManager
from data.database import get_database_engine, load_data_from_db

# 분석 및 특징 공학 계층
from analysis.metrics.relative_valuator import RelativeValuator
from analysis.metrics.technical_scorer import TechnicalScorer
from analysis.metrics.portfolio_risk import PortfolioRiskAnalyzer
# 상세 지표 분석을 위한 개별 모듈 임포트 (가상)
# 실제 구현 시, 이 모듈들은 analysis/metrics/ 디렉토리 내에 생성되어야 합니다.
from analysis.metrics import (
    evaluate_profitability,
    evaluate_stability,
    evaluate_liquidity,
    evaluate_efficiency,
    evaluate_growth,
    evaluate_valuation
)

# 머신러닝 모델 계층
from ml_models.prediction.predict_stock_returns import predict_with_model

# -----------------------------------------------------------------------------
# 시스템의 모든 기능을 관장하는 메인 클래스
# -----------------------------------------------------------------------------
class QuantSystem:
    """
    퀀트 자산 관리 시스템의 메인 컨트롤러 클래스.
    데이터 수집, 분석, 모델 예측, 리포팅 등 모든 워크플로우를 관장합니다.
    """
    def __init__(self):
        """
        시스템 생성자: 모든 필수 구성 요소를 초기화하고 로드합니다.
        """
        self._setup_logging()
        self.logger.info("퀀트 자산 관리 시스템(QuantSystem) 초기화를 시작합니다.")

        # 설정 및 데이터베이스 엔진 로드
        self.config = config
        self.db_engine = get_database_engine()

        # 핵심 모듈 인스턴스화 (의존성 주입)
        self.data_manager = SmartDataManager() # API 키는 내부에서 .env를 통해 로드
        self.relative_valuator = RelativeValuator(db_engine=self.db_engine)
        self.technical_scorer = TechnicalScorer(data_manager=self.data_manager)
        self.portfolio_risk_analyzer = PortfolioRiskAnalyzer(db_engine=self.db_engine, data_manager=self.data_manager)
        
        # 훈련된 머신러닝 모델 로드
        self.models = self._load_models()
        
        self.logger.info("QuantSystem 초기화 완료. 시스템이 준비되었습니다.")

    def _setup_logging(self):
        """시스템 전반에 사용될 로거를 설정하는 내부 헬퍼 함수."""
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _load_models(self) -> dict:
        """'saved_models' 디렉토리에서 훈련된 모델 파일들을 로드합니다."""
        models = {}
        # config.py에 ML_MODELS_DIR 경로가 정의되어 있다고 가정
        model_path = Path(getattr(self.config, 'ML_MODELS_DIR', 'ml_models')) / 'saved_models'
        if not model_path.exists():
            self.logger.error(f"모델 디렉토리가 존재하지 않습니다: {model_path}")
            return models

        # 주식 랭킹 모델 로드 (예시)
        ranking_model_file = model_path / 'stock_ranking_model.pkl'
        if ranking_model_file.exists():
            try:
                models['ranking_model'] = joblib.load(ranking_model_file)
                self.logger.info(f"'{ranking_model_file.name}' 모델 로드 성공.")
            except Exception as e:
                self.logger.error(f"'{ranking_model_file.name}' 모델 로드 실패: {e}")
        return models

    def get_top_ranked_stocks(self, top_n: int = 100, strategy: str = 'blend') -> pd.DataFrame:
        """
        전체 투자 유니버스를 대상으로 종합 점수를 산출하고,
        상위 N개 주식의 랭킹을 반환합니다. 이 메소드는 시스템의 모든 분석 역량을 종합합니다.

        :param top_n: 상위 몇 개를 가져올지 결정합니다.
        :param strategy: 사용할 가중치 전략 ('value', 'growth', 'blend')
        :return: 상위 N개 주식의 정보가 담긴 DataFrame
        """
        self.logger.info(f"'{strategy}' 전략 기반 Top {top_n} 랭킹 분석을 시작합니다.")
        
        # 1. 상대 가치 평가 실행 (동적 피어 그룹 및 퀄리티 벤치마크 기반)
        # 이 모듈은 수익성, 안정성, 성장성, 가치평가 등 핵심 재무 지표를 기반으로
        # '상대 강도 점수'와 '절대 퀄리티 점수'를 계산하여 종합적인 펀더멘털 점수를 반환합니다.
        ranked_df = self.relative_valuator.run_full_valuation(strategy=strategy)

        # 2. 기술적 매력도 점수 추가
        # 'technical_scorer'는 MA, RSI, MACD 등을 종합하여 현재의 기술적 타이밍 점수를 계산합니다.
        tickers_to_score = ranked_df.index.tolist()
        technical_scores = self.technical_scorer.get_scores_for_tickers(tickers_to_score)
        ranked_df = ranked_df.join(technical_scores)

        # 3. ML 모델 예측 점수 추가 (가장 진보된 의사결정 보조)
        if 'ranking_model' in self.models:
            # config 파일에 모델이 학습한 특징(feature) 목록이 정의되어 있어야 합니다.
            # 이 특징들은 relative_valuator와 technical_scorer가 생성한 모든 점수를 포함합니다.
            model_features = getattr(self.config, 'MODEL_FEATURES', [])
            if set(model_features).issubset(ranked_df.columns):
                 ranked_df['ml_prediction_score'] = predict_with_model(
                    model=self.models['ranking_model'],
                    features=ranked_df[model_features]
                )
            else:
                self.logger.warning("ML 모델 예측에 필요한 모든 특징이 데이터에 없습니다.")


        # 4. 최종 종합 점수 계산 및 랭킹
        # 시스템의 최종 투자 철학을 반영하는 가중치 로직.
        # 예: 최종 점수 = 펀더멘털 점수 * 0.5 + 기술적 점수 * 0.3 + ML 예측 점수 * 0.2
        # 이 가중치는 config.py에서 관리합니다.
        weights = self.config.FINAL_SCORE_WEIGHTS.get(strategy, {'fundamental': 0.6, 'technical': 0.4})
        ranked_df['final_score'] = (
            ranked_df['fundamental_score'] * weights.get('fundamental', 0) +
            ranked_df.get('technical_score', 0) * weights.get('technical', 0) +
            ranked_df.get('ml_prediction_score', 0) * weights.get('ml', 0)
        )
        
        ranked_df = ranked_df.sort_values('final_score', ascending=False)
        
        return ranked_df.head(top_n)

    def get_stock_deep_dive_report(self, ticker: str) -> dict:
        """
        단일 종목에 대한 모든 평가 항목을 포함한 심층 분석 보고서를 생성합니다.
        사용자에게 특정 종목을 추천하는 '이유'를 상세히 설명하는 역할을 합니다.

        :param ticker: 분석할 주식 티커
        :return: 분석 결과를 담은 딕셔너리
        """
        self.logger.info(f"'{ticker}'에 대한 심층 분석을 시작합니다.")
        
        # DB에서 해당 티커의 최신 재무 데이터 로드
        # financial_data = load_data_from_db(f"SELECT * FROM financial_metrics WHERE ticker = '{ticker}'", self.db_engine)
        # market_data = self.data_manager.get_daily_ohlcv([ticker])
        
        # 각 분석 모듈을 순차적으로 호출
        report = {
            'ticker': ticker,
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # 1. 재무 분석 (6개 카테고리)
            'profitability': evaluate_profitability(ticker, self.db_engine),
            'stability': evaluate_stability(ticker, self.db_engine),
            'liquidity': evaluate_liquidity(ticker, self.db_engine),
            'efficiency': evaluate_efficiency(ticker, self.db_engine),
            'growth': evaluate_growth(ticker, self.db_engine),
            'valuation': evaluate_valuation(ticker, self.db_engine),

            # 2. 기술적 분석
            'technical_analysis': self.technical_scorer.get_summary_for_ticker(ticker),
            
            # 3. 리스크 평가 (개별 종목 관점)
            'volatility_analysis': self.portfolio_risk_analyzer.get_single_stock_volatility(ticker),
        }
        
        return report

    def get_portfolio_risk_dashboard(self, portfolio_tickers: list) -> dict:
        """
        주어진 포트폴리오에 대한 종합 리스크 대시보드 데이터를 생성합니다.
        포트폴리오의 '방패' 역할을 하는 핵심 기능입니다.

        :param portfolio_tickers: 포트폴리오를 구성하는 주식 티커 리스트
        :return: 리스크 대시보드 시각화에 필요한 데이터를 담은 딕셔너리
        """
        self.logger.info("포트폴리오 리스크 분석을 시작합니다.")
        
        # portfolio_risk_analyzer는 VaR, CVaR, Beta, Sharpe, MDD 등을 모두 계산
        risk_data = self.portfolio_risk_analyzer.generate_full_report(portfolio_tickers)
        
        return risk_data
